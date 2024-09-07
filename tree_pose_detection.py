import time
import cv2
import mediapipe as mp
import numpy as np
import os
from scipy.spatial.distance import euclidean

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils

# Start capturing video from the webcam
cap = cv2.VideoCapture(0)

# Function to extract keypoints from an image
def extract_keypoints(image):
	results = pose.process(image)
	if results.pose_landmarks:
		landmarks = results.pose_landmarks.landmark
		keypoints = [(landmark.x, landmark.y) for landmark in landmarks]
		return keypoints
	return None

# Load reference keypoints from the images in yoga_poses/train/tree
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

# Load the reference keypoints
reference_keypoints = load_reference_keypoints('yoga_poses/train/tree')

# Function to calculate similarity between two sets of keypoints
def calculate_similarity(user_keypoints, reference_keypoints):
	similarities = []
	for ref in reference_keypoints:
		if len(ref) == len(user_keypoints):
			dist = np.mean([euclidean(ref[i], user_keypoints[i]) for i in range(len(ref))])
			similarities.append(dist)
	return min(similarities) if similarities else float('inf')

# Timer variables
pose_start_time = None
pose_held_time = 0
similarity_threshold = 0.1  # Adjust this threshold based on your tests

while cap.isOpened():
	ret, frame = cap.read()
	if not ret:
		break
	
	# Flip the frame horizontally (unmirrored)
	frame = cv2.flip(frame, 1)
	rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
	results = pose.process(rgb_frame)

	# Draw landmarks and connections
	mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
	
	# Extract landmarks if they exist
	if results.pose_landmarks:
		landmarks = results.pose_landmarks.landmark
		user_keypoints = [(landmark.x, landmark.y) for landmark in landmarks]

		# Calculate similarity between user keypoints and reference keypoints
		similarity = calculate_similarity(user_keypoints, reference_keypoints)
		correct_pose = similarity < similarity_threshold

		# Display feedback
		if correct_pose:
			if pose_start_time is None:
				pose_start_time = time.time()
			pose_held_time = time.time() - pose_start_time
			cv2.putText(frame, f"Correct Pose! Holding for {pose_held_time:.2f} seconds", (10, 50), 
						cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
		else:
			pose_start_time = None
			pose_held_time = 0
			cv2.putText(frame, "Incorrect Pose. Adjust your posture.", (10, 50), 
						cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

	# Show the video frame
	cv2.imshow('Tree Pose Detection', frame)
	
	# Break loop with 'q' key
	if cv2.waitKey(10) & 0xFF == ord('q'):
		break

cap.release()
cv2.destroyAllWindows()