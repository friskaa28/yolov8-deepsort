import numpy as np
from .kalman_filter import KalmanFilter

class Track:
    def __init__(self, bbox, track_id):
        self.bbox = bbox
        self.track_id = track_id
        self.kf = KalmanFilter()
        self.kf.state[:2] = np.array([[bbox[0]], [bbox[1]]])
        self.kf.motion_covariance = np.diag([1.0, 1.0, 0.1, 0.1])  # Penyesuaian motion covariance untuk lebih stabil
        self.kf.measurement_covariance = np.diag([0.5, 0.5])  # Penyesuaian measurement covariance
        self.hits = 1
        self.no_losses = 0

    def predict(self):
        self.kf.predict()
        self.bbox[:2] = self.kf.state[:2].flatten()

    def update(self, bbox):
        self.kf.update([bbox[0], bbox[1]])
        self.bbox = bbox
        self.hits += 1
        self.no_losses = 0

# Untuk pengujian
if __name__ == "__main__":
    track = Track([100, 100, 200, 200], 1)
    print("Initial state:", track.kf.state.flatten())
    track.predict()
    print("Predicted state:", track.kf.state.flatten())
    track.update([150, 150, 250, 250])
    print("Updated state:", track.kf.state.flatten())
