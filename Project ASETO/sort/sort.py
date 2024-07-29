import numpy as np
from .track import Track
from scipy.optimize import linear_sum_assignment

class Sort:
    def __init__(self, max_age=3, min_hits=1, iou_threshold=0.3):  # Penyesuaian parameter
        self.max_age = max_age
        self.min_hits = min_hits
        self.trackers = []
        self.frame_count = 0
        self.track_id = 0

    def update(self, dets=np.empty((0, 6))):
        self.frame_count += 1

        # Predict new locations of existing trackers
        for tracker in self.trackers:
            tracker.predict()
            tracker.no_losses += 1

        if len(dets) > 0:
            # Associate detections to trackers
            matched, unmatched_dets, unmatched_tracks = self.associate_detections_to_trackers(dets)

            # Update matched trackers with assigned detections
            for t, trk in enumerate(self.trackers):
                if t not in unmatched_tracks:
                    d = matched[np.where(matched[:, 1] == t)[0], 0]
                    trk.update(dets[d, :][0])

            # Create and initialize new trackers for unmatched detections
            for i in unmatched_dets:
                self.track_id += 1
                trk = Track(dets[i], self.track_id)
                self.trackers.append(trk)
        else:
            unmatched_dets = range(len(dets))

        # Remove dead tracklets
        self.trackers = [t for t in self.trackers if t.no_losses <= self.max_age]

        return np.array([[t.bbox[0], t.bbox[1], t.bbox[2], t.bbox[3], t.track_id] for t in self.trackers])

    def associate_detections_to_trackers(self, detections, iou_threshold=0.3):
        if len(self.trackers) == 0:
            return np.empty((0, 2), dtype=int), np.arange(len(detections)), np.empty((0, 5), dtype=int)

        iou_matrix = np.zeros((len(detections), len(self.trackers)), dtype=np.float32)

        for d, det in enumerate(detections):
            for t, trk in enumerate(self.trackers):
                iou_matrix[d, t] = self.iou(det, trk.bbox)

        matched_indices = linear_sum_assignment(-iou_matrix)
        matched_indices = np.array(matched_indices).T

        unmatched_detections = []
        for d, det in enumerate(detections):
            if d not in matched_indices[:, 0]:
                unmatched_detections.append(d)
        unmatched_trackers = []
        for t, trk in enumerate(self.trackers):
            if t not in matched_indices[:, 1]:
                unmatched_trackers.append(t)

        matches = []
        for m in matched_indices:
            if iou_matrix[m[0], m[1]] < iou_threshold:
                unmatched_detections.append(m[0])
                unmatched_trackers.append(m[1])
            else:
                matches.append(m.reshape(1, 2))
        if len(matches) == 0:
            matches = np.empty((0, 2), dtype=int)
        else:
            matches = np.concatenate(matches, axis=0)

        return matches, np.array(unmatched_detections), np.array(unmatched_trackers)

    def iou(self, bb_test, bb_gt):
        xx1 = np.maximum(bb_test[0], bb_gt[0])
        yy1 = np.maximum(bb_test[1], bb_gt[1])
        xx2 = np.minimum(bb_test[2], bb_gt[2])
        yy2 = np.minimum(bb_test[3], bb_gt[3])
        w = np.maximum(0., xx2 - xx1)
        h = np.maximum(0., yy2 - yy1)
        wh = w * h
        o = wh / ((bb_test[2] - bb_test[0]) * (bb_test[3] - bb_test[1])
                  + (bb_gt[2] - bb_gt[0]) * (bb_gt[3] - bb_gt[1]) - wh)
        return o

# Untuk pengujian
if __name__ == "__main__":
    sort_tracker = Sort(max_age=3, min_hits=1, iou_threshold=0.3)
    detections = np.array([
        [100, 100, 200, 200, 0.9],
        [150, 150, 250, 250, 0.8]
    ])
    result = sort_tracker.update(detections)
    print("Tracking result:", result)
