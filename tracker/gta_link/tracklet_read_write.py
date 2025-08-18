from typing import List, Dict
import pickle

import numpy as np

from tracker.gta_link.utils.Tracklet import Tracklet
from tracker.utils.pipeline_base import PipelineMessage
from tracklab.utils.coordinates import ltrb_to_ltwh


class TrackletReadWrite:
    """
    Class for reading and writing tracklet data.
    """
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.tracklets: Dict[int, Tracklet] = {}

    def add_tracklet(self, input: PipelineMessage):
        frame_id = input.metadata.get("frame_id", -1)
        tracklets = input.data.get("tracklets", np.array([]))
        features = input.data.get("features", np.array([]))
        self._update_tracklets(frame_id, tracklets, features)

    def _update_tracklets(self, frame_id, tracklets: np.ndarray, features: np.ndarray):
        """Add tracklets to the internal dictionary."""
        #  tracklet: M X (x, y, x, y, id, conf, cls, ind)        
        for i, tracklet in enumerate(tracklets):
            l, t, w, h = ltrb_to_ltwh(tracklet[:4])
            track_id = int(tracklet[4])
            bbox_conf = tracklet[5]
            # Update tracklet with detection info
            if track_id not in self.tracklets:
                self.tracklets[track_id] = Tracklet(track_id, frame_id, bbox_conf, [l, t, w, h])
            else:
                self.tracklets[track_id].append_det(frame_id, bbox_conf, [l, t, w, h])
            self.tracklets[track_id].append_feat(np.squeeze(features[i, :])) 

    def get_tracklets(self) -> Dict[int, Tracklet]:
        """Get all tracklets."""
        return self.tracklets

    def read_tracklets(self) -> Dict[int, Tracklet]:
        """
        Read tracklets from a file.
        """
        with open(self.file_path, 'rb') as f:
            tracklets = pickle.load(f)
        return tracklets

    def save_tracklets(self):
        """
        Write tracklets to a file.
        """
        with open(self.file_path, 'wb') as f:
            pickle.dump(self.tracklets, f)
