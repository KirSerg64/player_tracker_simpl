# Optimized single-stage Dockerfile for Football Player Tracker
# GPU/CUDA support with minimal size

FROM kvelertak/player_tracker:latest

# The base image already has:
# - Virtual environment at /opt/venv 
# - Working directory /app
# - User appuser created
# - All dependencies installed

# Use the existing virtual environment
ENV PATH="/opt/venv/bin:$PATH"

# Set working directory (should already be set in base image)
WORKDIR /app

# Copy dependency files (for reference, dependencies already installed in base image)
COPY pyproject.toml ./
COPY poetry.lock* ./
COPY uv.lock* ./

# Copy application files (appuser already exists in base image)
# COPY --chown=appuser:appuser main_video_parallel.py ./
COPY --chown=appuser:appuser main.py ./
# COPY --chown=appuser:appuser test_hello.py ./

# Copy the complete sn_gamestate package with all subdirectories
COPY --chown=appuser:appuser sn_gamestate/ ./sn_gamestate/

# Copy plugins
COPY --chown=appuser:appuser plugins/calibration/ ./plugins/calibration/

# Copy pretrained models (if they weren't included in base image)
COPY --chown=appuser:appuser pretrained_models/yolo/yolov11_A100_640_batch1_fp16_ultra.engine ./pretrained_models/yolo/
COPY --chown=appuser:appuser pretrained_models/reid/prtreid-fixed-opset17-simplified.onnx ./pretrained_models/reid/
COPY --chown=appuser:appuser pretrained_models/reid/prtreid-onnx-opset20-simplified.onnx ./pretrained_models/reid/
COPY --chown=appuser:appuser pretrained_models/gta_link/sports_model.pth.tar-60 ./pretrained_models/gta_link/
COPY --chown=appuser:appuser pretrained_models/gta_link/feature_extractor_osnet_x1_0.onnx ./pretrained_models/gta_link/
COPY --chown=appuser:appuser pretrained_models/calibration/ ./pretrained_models/calibration/

# Install local packages (need to reinstall since we copied new code)
USER root
RUN pip install --no-cache-dir albumentations==1.4.3 boto3
RUN pip install --no-cache-dir --no-deps -e ./plugins/calibration
RUN pip install --no-cache-dir --no-deps -e .

# Copy the script and make it executable
COPY --chown=appuser:appuser run_parallel.sh /app/
RUN chmod +x /app/run_parallel.sh

# Create minimal directory structure and cleanup
RUN mkdir -p /app/data/input /app/data/output /app/logs /app/tmp && \
    chown -R appuser:appuser /app

COPY --chown=appuser:appuser test_videos/* /app/data/input

# Create startup script with GPU info
# RUN echo '#!/bin/bash\n\
# echo "Football Player Tracker (GPU-Optimized)"\n\
# echo "Python: $(python --version)"\n\
# echo "PyTorch: $(python -c \"import torch; print(torch.__version__)\" 2>/dev/null || echo \"Not available\")"\n\
# echo "CUDA Available: $(python -c \"import torch; print(torch.cuda.is_available())\" 2>/dev/null || echo \"False\")"\n\
# echo "GPU Count: $(python -c \"import torch; print(torch.cuda.device_count())\" 2>/dev/null || echo \"0\")"\n\
# if [ \"$CUDA_VISIBLE_DEVICES\" ]; then echo "CUDA Devices: $CUDA_VISIBLE_DEVICES"; fi\n\
# exec "$@"' > /app/entrypoint.sh && chmod +x /app/entrypoint.sh

# Switch to non-root user
USER appuser

# Set environment variables
ENV PYTHONPATH=/app
ENV HYDRA_FULL_ERROR=1
ENV CUDA_VISIBLE_DEVICES=0
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

# Default command - you can override this when running the container
# Examples:
# docker run kvelertak/player_tracker:app python main.py
# docker run kvelertak/player_tracker:app python main_video_parallel.py
# docker run kvelertak/player_tracker:app python test_hello.py
# docker run -it kvelertak/player_tracker:app /bin/bash
# CMD ["python", "test_hello.py"]
# CMD ["python", "main.py", "--config-path", "pkg://sn_gamestate.configs", "--config-name", "soccernet_test"]
CMD ["/app/run_parallel.sh", "/app/data/input/7_06_15fps.mp4", "/app/data/output", "soccernet_test", "600", "1", "2"]
