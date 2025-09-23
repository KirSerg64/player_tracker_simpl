#!/usr/bin/env bash

# Minimalistic parallel video processing script
# Usage: ./run_parallel.sh input_video.mp4 output_dir [config_name] [segment_duration] [overlap] [workers]

# Don't exit on error immediately - we want to handle job failures gracefully
set +e

# Enable debug output if DEBUG environment variable is set
if [ "${DEBUG:-0}" = "1" ]; then
    set -x
fi

# Parse arguments
INPUT_VIDEO="$1"
OUTPUT_DIR="$2"
CONFIG_NAME="${3:-main}"
SEGMENT_DURATION="${4:-300}"
OVERLAP="${5:-1.0}"
MAX_WORKERS="${6:-4}"

# Validate inputs
if [[ -z "$INPUT_VIDEO" || -z "$OUTPUT_DIR" ]]; then
    echo "Usage: $0 input_video.mp4 output_dir [config_name] [segment_duration] [overlap] [workers]"
    echo "Example: $0 video.mp4 ./output soccernet_test 300 1.0 4"
    exit 1
fi

# Validate input video exists
if [[ ! -f "$INPUT_VIDEO" ]]; then
    echo "Error: Input video '$INPUT_VIDEO' not found"
    exit 1
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

echo "=== Parallel Video Processing ==="
echo "Input video: $INPUT_VIDEO"
echo "Output directory: $OUTPUT_DIR"
echo "Config: $CONFIG_NAME"
echo "Segment duration: ${SEGMENT_DURATION}s"
echo "Overlap: ${OVERLAP}s"
echo "Max workers: $MAX_WORKERS"

# Step 1: Split video into segments
echo ""
echo "=== Step 1: Video Splitting ==="

SEGMENTS_DIR="$OUTPUT_DIR/segments"
mkdir -p "$SEGMENTS_DIR"

# Run Python video splitter
echo "Running: python tracker/utils/video_splitter.py \"$INPUT_VIDEO\" \"$SEGMENTS_DIR\" --segment_duration $SEGMENT_DURATION --overlap $OVERLAP"

python tracker/utils/video_splitter.py \
    "$INPUT_VIDEO" \
    "$SEGMENTS_DIR" \
    --segment_duration $SEGMENT_DURATION \
    --overlap $OVERLAP \
    --output_fps 15
    # --debug

if [ $? -ne 0 ]; then
    echo "Error: Video splitting failed"
    exit 1
fi

# Find created segments
SEGMENTS=($(find "$SEGMENTS_DIR" -name "*.mp4" -type f | sort))
SEGMENT_COUNT=${#SEGMENTS[@]}

if [ $SEGMENT_COUNT -eq 0 ]; then
    echo "Error: No video segments found in $SEGMENTS_DIR"
    exit 1
fi

echo "Created $SEGMENT_COUNT video segments"

# Step 2: Process segments in parallel
echo ""
echo "=== Step 2: Parallel Processing ==="

# Track job PIDs - using associative arrays for better reliability
declare -A JOB_PIDS
declare -A JOB_NAMES
RUNNING_JOBS=0
COMPLETED=0
FAILED=0

for segment in "${SEGMENTS[@]}"; do
    segment_name=$(basename "$segment")
    echo "Processing segment: $segment_name"
    
    # Check if this segment is already being processed
    for existing_job_id in "${!JOB_NAMES[@]}"; do
        if [ "${JOB_NAMES[$existing_job_id]}" = "$segment_name" ]; then
            echo "Warning: Segment $segment_name is already being processed (Job ID: $existing_job_id)"
        fi
    done
    
    # Wait if we have reached max workers
    while [ $RUNNING_JOBS -ge $MAX_WORKERS ]; do
        # Check for completed jobs - create a copy of keys to avoid modification during iteration
        job_keys=(${!JOB_PIDS[@]})
        for job_id in "${job_keys[@]}"; do
            pid=${JOB_PIDS[$job_id]}
            if ! kill -0 $pid 2>/dev/null; then
                # Job finished, get exit status
                wait $pid
                exit_status=$?
                
                segment_name=${JOB_NAMES[$job_id]}
                if [ $exit_status -eq 0 ]; then
                    echo "✓ Success: $segment_name"
                    ((COMPLETED++))
                else
                    echo "✗ Failed: $segment_name (exit code: $exit_status)"
                    ((FAILED++))
                fi
                
                # Remove from tracking
                unset JOB_PIDS[$job_id]
                unset JOB_NAMES[$job_id]
                ((RUNNING_JOBS--))
            fi
        done
        
        if [ $RUNNING_JOBS -ge $MAX_WORKERS ]; then
            sleep 1
        fi
    done
    
    # Start new job
    segment_name=$(basename "$segment")
    # Get the segment directory (parent of the video file)
    segment_dir=$(dirname "$segment")
    # Create results folder inside the segment directory
    segment_results_dir="$segment_dir"
    mkdir -p "$segment_results_dir"
    
    # Get absolute path to main.py to avoid path issues
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    
    # Run main.py in background
    (
        cd $SCRIPT_DIR
        set +e  # Don't exit on error in subshell
        python "video_processor.py" \
            --config-path "pkg://tracker.configs" \
            --config-name "$CONFIG_NAME" \
            video_path="$segment" \
            hydra.run.dir="$segment_results_dir"
            # > "$segment_results_dir/processing.log" 2>&1
        exit $?  # Preserve exit code
    ) &
    
    # Track the job with unique ID - fix race condition
    pid=$!
    job_id="${segment_name}_${pid}_$(date +%s%N)"  # Use segment name and PID for uniqueness
    JOB_PIDS[$job_id]=$pid
    JOB_NAMES[$job_id]="$segment_name"
    ((RUNNING_JOBS++))
    
    echo "Started job for: $segment_name (PID: $pid, Job ID: $job_id)"
done

echo "Processing $SEGMENT_COUNT segments with max $MAX_WORKERS workers"

# Wait for all remaining jobs to complete
while [ $RUNNING_JOBS -gt 0 ]; do
    echo "Waiting for $RUNNING_JOBS remaining jobs..."
    
    # Check all remaining jobs
    job_keys=(${!JOB_PIDS[@]})
    for job_id in "${job_keys[@]}"; do
        pid=${JOB_PIDS[$job_id]}
        if ! kill -0 $pid 2>/dev/null; then
            # Job finished
            wait $pid
            exit_status=$?
            
            segment_name=${JOB_NAMES[$job_id]}
            if [ $exit_status -eq 0 ]; then
                echo "✓ Success: $segment_name"
                ((COMPLETED++))
            else
                echo "✗ Failed: $segment_name (exit code: $exit_status)"
                ((FAILED++))
            fi
            
            # Remove from tracking
            unset JOB_PIDS[$job_id]
            unset JOB_NAMES[$job_id]
            ((RUNNING_JOBS--))
        fi
    done
    
    if [ $RUNNING_JOBS -gt 0 ]; then
        sleep 2
    fi
done

# Final results
echo ""
echo "=== Results ==="
echo "Successful: $COMPLETED/$SEGMENT_COUNT"
echo "Failed: $FAILED/$SEGMENT_COUNT"

if [ $COMPLETED -gt 0 ]; then
    success_rate=$((COMPLETED * 100 / SEGMENT_COUNT))
    echo "Success rate: ${success_rate}%"
    
    if [ $FAILED -eq 0 ]; then
        echo "All segments processed successfully!"
        
        # Step 3: Merge tracklet groups for each segment
        echo ""
        echo "=== Step 3: Merging Tracklet Groups ==="
        
        # Process each segment directory to merge tracklets
        for segment in "${SEGMENTS[@]}"; do
            segment_name=$(basename "$segment")
            segment_dir=$(dirname "$segment")
            segment_results_dir="$segment_dir"
            
            # Check if results directory exists and has tracklet files
            if [[ -d "$segment_results_dir" ]]; then
                tracklet_files=$(find "$segment_results_dir" -name "*video*.pkl" | head -1)
                if [[ -n "$tracklet_files" ]]; then
                    echo "Processing tracklet merging for: $segment_name"
                    
                    # Run merge_tracklets_groups.py for this segment
                    python tracker/utils/merge_tracklets_groups.py \
                        --track_base_dir "$segment_results_dir" \
                        --max_start_window 15 \
                        --max_end_window 15
                    
                    if [ $? -eq 0 ]; then
                        echo "✓ Tracklet merging successful for: $segment_name"
                    else
                        echo "✗ Tracklet merging failed for: $segment_name"
                    fi
                else
                    echo "⚠ No tracklet files found in: $segment_results_dir"
                fi
            else
                echo "⚠ Results directory not found: $segment_results_dir"
            fi
        done
        
        # Step 4: Create final combined video
        echo ""
        echo "=== Step 4: Creating Combined Video ==="
        
        # Create combined video from all processed segments
        FINAL_VIDEO_PATH="$OUTPUT_DIR/combined_video.mp4"
        
        echo "Creating combined video: $FINAL_VIDEO_PATH"
        python tracker/utils/create_video_from_tracklets.py \
            --mode multi \
            --base_dir "$OUTPUT_DIR" \
            --output_video_path "$FINAL_VIDEO_PATH" \
            --frame_rate 15
        
        if [ $? -eq 0 ]; then
            echo "✓ Combined video created successfully: $FINAL_VIDEO_PATH"
            exit 0
        else
            echo "✗ Combined video creation failed"
            exit 1
        fi
    else
        echo "Some segments failed. Skipping post-processing steps."
        exit 1
    fi
else
    echo "All segments failed. Check logs for details."
    exit 1
fi
