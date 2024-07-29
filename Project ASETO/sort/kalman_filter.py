import numpy as np

class KalmanFilter:
    def __init__(self):
        self.state = np.zeros((4, 1))  # State: [x, y, vx, vy]
        self.P = np.eye(4)  # State covariance matrix
        self.F = np.eye(4)  # State transition matrix
        self.F[0, 2] = 1
        self.F[1, 3] = 1
        self.Q = np.eye(4) * 0.01  # Process noise covariance matrix
        self.H = np.eye(2, 4)  # Measurement matrix
        self.R = np.eye(2) * 0.5  # Measurement noise covariance matrix

        # Penyesuaian parameter motion_covariance dan measurement_covariance
        self.motion_covariance = np.diag([1.0, 1.0, 0.1, 0.1])  # Penyesuaian motion covariance
        self.measurement_covariance = np.diag([0.5, 0.5])  # Penyesuaian measurement covariance

    def predict(self):
        self.state = np.dot(self.F, self.state)
        self.P = np.dot(np.dot(self.F, self.P), self.F.T) + self.Q * self.motion_covariance  # Penyesuaian dengan motion covariance
        return self.state

    def update(self, measurement):
        z = np.array([[measurement[0]], [measurement[1]]])
        y = z - np.dot(self.H, self.state)
        S = np.dot(np.dot(self.H, self.P), self.H.T) + self.R * self.measurement_covariance  # Penyesuaian dengan measurement covariance
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))
        self.state = self.state + np.dot(K, y)
        self.P = self.P - np.dot(np.dot(K, self.H), self.P)

# Untuk pengujian
if __name__ == "__main__":
    kf = KalmanFilter()
    measurements = [[1, 2], [2, 3], [3, 4]]
    for measurement in measurements:
        print("Predicted state:", kf.predict().flatten())
        kf.update(measurement)
        print("Updated state:", kf.state.flatten())
