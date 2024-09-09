import cv2
import mediapipe as mp
import numpy as np
import os
from scipy.spatial.distance import euclidean
import time
from PyQt5 import QtCore, QtGui, QtWidgets

# setting up the mediapipe
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils

# dictionary of pose options
pose_options = {
    'tree': 'yoga_poses/train/tree',
    'chair': 'yoga_poses/train/chair',
    'cobra': 'yoga_poses/train/cobra',
    'dog': 'yoga_poses/train/dog',
    'shoulder_stand': 'yoga_poses/train/shoulder_stand',
    'triangle': 'yoga_poses/train/triangle',
    'warrior': 'yoga_poses/train/warrior',
    'no_pose': 'yoga_poses/train/no_pose'
}

# processes an image to extract the key points and saves their coordinates in a list
def extract_keypoints(image):
    results = pose.process(image)
    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark
        keypoints = [(landmark.x, landmark.y) for landmark in landmarks]
        return keypoints
    return None

# loads images , extracts keypoints, and stores them
def load_reference_keypoints(folder_path):
    reference_keypoints = []
    for filename in os.listdir(folder_path):
        img_path = os.path.join(folder_path, filename)
        image = cv2.imread(img_path)
        if image is not None:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            keypoints = extract_keypoints(image_rgb)
            if keypoints:
                reference_keypoints.append(keypoints)
    return reference_keypoints

# stores reference keypoints for each pose
reference_keypoints_dict = {pose: load_reference_keypoints(path) for pose, path in pose_options.items()}

# calculates the similarity between the user's keypoints and reference keypoints
def calculate_similarity(user_keypoints, reference_keypoints):
    similarities = []
    for ref in reference_keypoints:
        if len(ref) == len(user_keypoints):
            dist = np.mean([euclidean(ref[i], user_keypoints[i]) for i in range(len(ref))])
            similarities.append(dist)
    return min(similarities) if similarities else float('inf')

# captures video from the webcam
# processes each frame to detect the user's pose
# compare it to reference keypoints
# provides feedback on the pose accuracy
def start_pose_detection(selected_pose):
    cap = cv2.VideoCapture(0)
    pose_start_time = None
    pose_held_time = 0
    similarity_threshold = 0.1

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb_frame)

        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        
        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            user_keypoints = [(landmark.x, landmark.y) for landmark in landmarks]

            selected_keypoints = reference_keypoints_dict[selected_pose]
            similarity = calculate_similarity(user_keypoints, selected_keypoints)
            correct_pose = similarity < similarity_threshold

            if correct_pose:
                if pose_start_time is None:
                    pose_start_time = time.time()
                pose_held_time = time.time() - pose_start_time
                cv2.putText(frame, f"Correct {selected_pose.capitalize()}! Holding for {pose_held_time:.2f} seconds", 
                            (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            else:
                pose_start_time = None
                pose_held_time = 0
                cv2.putText(frame, f"Incorrect {selected_pose.capitalize()}. Adjust your posture.", 
                            (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

        cv2.imshow('Pose Detection', frame)
        
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

class YogaPoseApp(QtWidgets.QWidget): #  GUI
    def __init__(self):
        super().__init__()

        self.initUI()

    def initUI(self):
        self.setWindowTitle('Yoga Pose Detection')
        self.setGeometry(100, 100, 600, 400)

        layout = QtWidgets.QVBoxLayout()

        label = QtWidgets.QLabel("Select a Pose to Practice:", self)
        label.setFont(QtGui.QFont("Helvetica", 12))
        layout.addWidget(label)

        self.pose_combo = QtWidgets.QComboBox(self)
        self.pose_combo.addItems(list(pose_options.keys()))
        self.pose_combo.setFont(QtGui.QFont("Helvetica", 12))
        self.pose_combo.currentTextChanged.connect(self.update_pose_image)
        layout.addWidget(self.pose_combo)

        self.pose_image_label = QtWidgets.QLabel(self)
        self.pose_image_label.setAlignment(QtCore.Qt.AlignCenter)
        layout.addWidget(self.pose_image_label)

        start_button = QtWidgets.QPushButton("Start Pose Detection", self)
        start_button.setFont(QtGui.QFont("Helvetica", 12))
        start_button.clicked.connect(self.start_detection)
        layout.addWidget(start_button)

        self.setLayout(layout)
        self.update_pose_image()

    def update_pose_image(self):
        selected_pose = self.pose_combo.currentText()
        img_path = f'example_poses/{selected_pose}.png'
        if os.path.exists(img_path):
            pixmap = QtGui.QPixmap(img_path)
            self.pose_image_label.setPixmap(pixmap.scaled(200, 200, QtCore.Qt.KeepAspectRatio))
        else:
            self.pose_image_label.setText("No image available")

    def start_detection(self):
        selected_pose = self.pose_combo.currentText()
        start_pose_detection(selected_pose)

# initializes and runs Qt
if __name__ == '__main__':
    import sys
    app = QtWidgets.QApplication(sys.argv)
    ex = YogaPoseApp()
    ex.show()
    sys.exit(app.exec_())