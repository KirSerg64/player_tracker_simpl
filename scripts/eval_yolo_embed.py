"""
Evaluation script: YOLO detection + embed feature extraction + K-means clustering.

Usage (from repo root, torchreid env):
    python scripts/eval_yolo_embed.py video_path='C:/path/to/video.mp4'

Stripped from video_processor.py — only detector + embed eval logic retained.

Embed strategy:
  1. Phase 1 — detect all players on every frame (normal YOLO predict, no embed).
  2. Phase 2 — crop each accepted detection and run model.predict(crop, embed=[layer])
               to obtain a per-crop spatial feature tensor; avg-pool → fixed-size vector.
  3. Phase 3 — L2-normalise, estimate K from median per-frame detection count, KMeans.
  4. Phase 4 — save 4 plots to <output_dir>/embed_eval/:
               silhouette.png   (silhouette score vs K)
               scatter.png      (UMAP/PCA 2D coloured by cluster and by frame)
               timeline.png     (cluster × frame-bin heatmap)
               crop_grid.png    (sample detection crops per cluster)
"""

import os
import sys
from pathlib import Path

import cv2
from hydra.utils import instantiate
import numpy as np
from pika import frame
import torch
import hydra
import logging
from tqdm import tqdm

from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf
from ultralytics import YOLO
import supervision as sv
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from tracklab.utils.coordinates import sanitize_bbox_ltrb
from tracker.detector import yolo_features
from tracker.utils.pipeline_base import MessageType, PipelineMessage
from tracker.detector.yolo_features import YOLOFeatureExtractor

os.environ["HYDRA_FULL_ERROR"] = "1"

log = logging.getLogger(__name__)

# Single backbone layer to embed (15 = P3 neck output in YOLOv11)
EMBED_LAYER = 15


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_hydra_output_dir() -> str:
    try:
        return HydraConfig.get().runtime.output_dir
    except Exception:
        return os.getcwd()


def init_device() -> str:
    import os as _os
    cpu_count = _os.cpu_count() or 1
    torch.set_num_threads(min(4, cpu_count))
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def extract_embed_vector(model: YOLO, crop: np.ndarray, device: str) -> np.ndarray:
    """Run model.predict on a single crop with embed, avg-pool spatial dim → 1-D vector."""
    results = model.predict(crop, embed=[EMBED_LAYER], verbose=False, device=device)
    t = results[0]  # Tensor [B, C, spatial] or [C, spatial]
    if t.ndim == 3:
        t = t[0]   # [C, spatial]
    return t.mean(dim=-1).cpu().numpy()  # [C]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

@hydra.main(version_base=None, config_path="pkg://tracker.configs", config_name="main")
def main(cfg):
    device = init_device()
    output_dir = get_hydra_output_dir()

    video_path = cfg.video_path
    if not video_path or not os.path.exists(video_path):
        log.error(f"Video not found: {video_path!r}. Pass video_path='...' on the command line.")
        return 1

    # Use ONNX model (CPU-compatible)
    model_path = str(Path(cfg.project_dir) / "pretrained_models/yolo/yolo_best.pt")
    log.info(f"Loading model: {model_path}")
    model = YOLO(model_path, task="detect")

    detector_cfg = cfg.detector.cfg
    class_map = {cls: i for i, cls in enumerate(detector_cfg.classes)}
    classes_to_detect = {class_map[c] for c in ["player", "goalkeeper", "referee", "person"]}
    min_conf: float = detector_cfg.min_confidence

    # YOLO backbone feature extractor
    cfg = OmegaConf.create({
        "scale_idx": 2,
        "embed_layers": [EMBED_LAYER],
    })
    yolo_features = YOLOFeatureExtractor(
        yolo_model=model, 
        cfg=cfg,
    )

    # --- Open video ---
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        log.error(f"Cannot open video: {video_path}")
        return 1
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    vid_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    vid_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    log.info(f"Video: {vid_w}x{vid_h} @ {fps:.1f} FPS, {total_frames} frames")

    # -----------------------------------------------------------------------
    # Phase 1 — detection pass
    # -----------------------------------------------------------------------
    shapes = [(vid_h, vid_w)]  # model.predict expects list of images; we have just one
    all_detections = []   # list of (frame_id, crop_bgr, xyxy)
    embeddings = []
    frame_det_counts = []
    frame_id = 0

    pbar = tqdm(total=total_frames, desc="Phase 1 — detecting", ncols=100)
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results_by_image = model.predict(frame, device=device, agnostic_nms=True)
        results_by_image = [sv.Detections.from_ultralytics(res) for res in results_by_image]

        detections = []
        embed_features = []

        for results, shape in zip(results_by_image, shapes):
            for xyxy, _, conf, class_id, _, _ in results:
                # check for `player` and 'goalkeeper' class
                if conf >= min_conf and class_id in classes_to_detect:
                    detections.append(
                        np.concatenate([sanitize_bbox_ltrb(xyxy, (shape[1], shape[0])), [conf, class_id]], axis=0)
                    )

        embed_results = yolo_features.extract_roi_features(detections, shapes[0]) if detections else []
        embeddings.extend(embed_results)
        count = 0
        for det  in detections:
            xyxy, conf, class_id = det[:4], det[4], det[5]
            if conf >= min_conf and class_id in classes_to_detect:
                x1, y1, x2, y2 = int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(vid_w, x2), min(vid_h, y2)
                if x2 > x1 and y2 > y1:
                    crop = frame[y1:y2, x1:x2].copy()
                    all_detections.append((frame_id, crop, xyxy))
                    count += 1

        frame_det_counts.append(count)
        frame_id += 1
        pbar.update(1)

    cap.release()
    pbar.close()

    n_det = len(all_detections)
    log.info(f"Phase 1 done — {n_det} accepted detections across {frame_id} frames")
    if n_det == 0:
        log.error("No detections found. Check model path and min_confidence.")
        return 1

    # -----------------------------------------------------------------------
    # Phase 2 — embed extraction per crop
    # -----------------------------------------------------------------------
    # embeddings = []
    # for _, crop, _ in tqdm(all_detections, desc="Phase 2 — embedding", ncols=100):
    #     vec = extract_embed_vector(model, crop, device)
    #     embeddings.append(vec)

    embeddings = np.array(embeddings, dtype=np.float32)   # [N, C]
    log.info(f"Embedding matrix: {embeddings.shape}")

    # L2-normalise (makes KMeans behave like cosine similarity)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    emb_norm = embeddings / np.maximum(norms, 1e-8)

    # -----------------------------------------------------------------------
    # Phase 3 — estimate K and cluster
    # -----------------------------------------------------------------------
    nonzero_counts = [c for c in frame_det_counts if c > 0]
    k_estimate = int(np.median(nonzero_counts)) if nonzero_counts else 22
    k = max(2, min(k_estimate, n_det - 1))
    log.info(f"Estimated K={k}  (median per-frame detection count={k_estimate})")

    kmeans = KMeans(n_clusters=k, n_init=10, random_state=42)
    labels = kmeans.fit_predict(emb_norm)
    log.info(f"Cluster sizes: { {i: int((labels==i).sum()) for i in range(k)} }")

    # -----------------------------------------------------------------------
    # Phase 4 — visualise
    # -----------------------------------------------------------------------
    out_dir = Path(output_dir) / "embed_eval"
    out_dir.mkdir(parents=True, exist_ok=True)

    frame_ids = np.array([d[0] for d in all_detections])
    crops = [d[1] for d in all_detections]

    _plot_silhouette(emb_norm, labels, k, out_dir)
    _plot_scatter(emb_norm, labels, frame_ids, k, out_dir)
    _plot_timeline(labels, frame_ids, k, out_dir)
    _plot_crop_grid(crops, labels, k, out_dir)

    log.info(f"All plots saved to: {out_dir}")
    return 0


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def _plot_silhouette(emb: np.ndarray, labels: np.ndarray, chosen_k: int, out_dir: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(emb)
    k_max = min(chosen_k * 2, n // 2)
    k_range = range(2, k_max + 1)
    scores = []
    for kk in tqdm(k_range, desc="Silhouette sweep", ncols=80, leave=False):
        lbl = KMeans(n_clusters=kk, n_init=5, random_state=42).fit_predict(emb)
        scores.append(silhouette_score(emb, lbl))

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(list(k_range), scores, marker="o", linewidth=1.5)
    ax.axvline(chosen_k, color="red", linestyle="--", label=f"Chosen K={chosen_k}")
    ax.set_xlabel("K"); ax.set_ylabel("Silhouette Score")
    ax.set_title("Silhouette Score vs Number of Clusters")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "silhouette.png", dpi=150)
    plt.close(fig)
    log.info("Saved silhouette.png")


def _plot_scatter(emb: np.ndarray, labels: np.ndarray, frame_ids: np.ndarray,
                  k: int, out_dir: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA

    # Dimensionality reduction
    try:
        import umap as umap_lib
        reducer = umap_lib.UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1)
        proj = reducer.fit_transform(emb)
        method = "UMAP"
    except ImportError:
        pca = PCA(n_components=2, random_state=42)
        proj = pca.fit_transform(emb)
        method = "PCA"
    log.info(f"2D projection via {method}")

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    sc0 = axes[0].scatter(proj[:, 0], proj[:, 1], c=labels, cmap="tab20",
                          s=8, alpha=0.7, linewidths=0)
    plt.colorbar(sc0, ax=axes[0], label="Cluster")
    axes[0].set_title(f"{method} — coloured by K-means cluster")
    axes[0].set_xlabel("Dim 1"); axes[0].set_ylabel("Dim 2")

    sc1 = axes[1].scatter(proj[:, 0], proj[:, 1], c=frame_ids, cmap="viridis",
                          s=8, alpha=0.7, linewidths=0)
    plt.colorbar(sc1, ax=axes[1], label="Frame index")
    axes[1].set_title(f"{method} — coloured by temporal position")
    axes[1].set_xlabel("Dim 1"); axes[1].set_ylabel("Dim 2")

    fig.suptitle("YOLO embed features — 2D projection", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_dir / "scatter.png", dpi=150)
    plt.close(fig)
    log.info("Saved scatter.png")


def _plot_timeline(labels: np.ndarray, frame_ids: np.ndarray, k: int, out_dir: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n_bins = min(300, int(frame_ids.max()) + 1)
    bin_edges = np.linspace(0, frame_ids.max() + 1, n_bins + 1)
    heatmap = np.zeros((k, n_bins), dtype=np.float32)

    for fi, li in zip(frame_ids, labels):
        b = min(int(np.searchsorted(bin_edges, fi, side="right")) - 1, n_bins - 1)
        heatmap[li, b] += 1

    # Normalise each cluster row by its max for readability
    row_max = heatmap.max(axis=1, keepdims=True)
    row_max[row_max == 0] = 1
    heatmap /= row_max

    fig, ax = plt.subplots(figsize=(18, max(4, k * 0.35)))
    im = ax.imshow(heatmap, aspect="auto", cmap="hot", origin="upper",
                   vmin=0, vmax=1, interpolation="nearest")
    ax.set_xlabel("Frame bin (time →)")
    ax.set_ylabel("Cluster ID")
    ax.set_title("Cluster occupancy over time  (per-row normalised)")
    plt.colorbar(im, ax=ax, label="Relative presence")
    fig.tight_layout()
    fig.savefig(out_dir / "timeline.png", dpi=150)
    plt.close(fig)
    log.info("Saved timeline.png")


def _plot_crop_grid(crops: list, labels: np.ndarray, k: int, out_dir: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n_cols = 10
    thumb_w, thumb_h = 45, 90

    crops_by_cluster: dict[int, list] = {i: [] for i in range(k)}
    for crop, label in zip(crops, labels):
        crops_by_cluster[label].append(crop)

    fig, axes = plt.subplots(k, n_cols, figsize=(n_cols * 1.1, k * 2.2))
    if k == 1:
        axes = axes[np.newaxis, :]

    for ci in range(k):
        row_crops = crops_by_cluster[ci]
        for si in range(n_cols):
            ax = axes[ci, si]
            ax.axis("off")
            if si < len(row_crops):
                crop = row_crops[si]
                if crop is not None and crop.size > 0:
                    thumb = cv2.resize(crop, (thumb_w, thumb_h))
                    ax.imshow(cv2.cvtColor(thumb, cv2.COLOR_BGR2RGB))
            if si == 0:
                ax.set_ylabel(f"C{ci}", fontsize=7, rotation=0, labelpad=22, va="center")

    fig.suptitle("Sample detection crops per cluster  (first 10 per cluster)", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_dir / "crop_grid.png", dpi=150)
    plt.close(fig)
    log.info("Saved crop_grid.png")


if __name__ == "__main__":
    main()
