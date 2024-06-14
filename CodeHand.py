import cv2
import mediapipe as mp
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import pickle
import os

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

# Initialize variables
X = []
y = []
gesture_names = {}
data_save_path = 'C:/Users/kshet/Desktop/Coding/Projects/GestureHand' #add a path in where the collected data and trained model will stored

# Ensure the data save directory exists
if not os.path.exists(data_save_path):
    os.makedirs(data_save_path)

# Define a function to extract hand landmarks as features
def extract_features(hand_landmarks):
    features = []
    for landmark in hand_landmarks.landmark:
        features.append(landmark.x)
        features.append(landmark.y)
    return features

# Define a function to collect data for training
def collect_data():
    cap = cv2.VideoCapture(0)

    while True:
        current_gesture = input("Enter the gesture label (or type 'done' to finish): ")
        if current_gesture.lower() == 'done':
            break

        gesture_id = len(gesture_names)
        gesture_names[gesture_id] = current_gesture
        print(f"Collecting data for gesture: {current_gesture}")

        while cap.isOpened():
            success, image = cap.read()
            if not success:
                break

            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            result = hands.process(image_rgb)

            if result.multi_hand_landmarks:
                for hand_landmarks in result.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                    features = extract_features(hand_landmarks)
                    X.append(features)
                    y.append(gesture_id)

            cv2.putText(image, f'Gesture: {current_gesture}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow('Collecting Data', image)

            key = cv2.waitKey(1)
            if key == ord('q'):
                break
            elif key == ord('n'):
                break

    cap.release()
    cv2.destroyAllWindows()

    # Save the collected data
    with open(os.path.join(data_save_path, 'gesture_data.pkl'), 'wb') as f:
        pickle.dump((X, y, gesture_names), f)

    print("Data collection complete and saved.")

# Train the KNN model
def train_model():
    global model

    # Load the collected data
    with open(os.path.join(data_save_path, 'gesture_data.pkl'), 'rb') as f:
        X, y, gesture_names = pickle.load(f)

    model = KNeighborsClassifier(n_neighbors=3)
    model.fit(X, y)
    with open(os.path.join(data_save_path, 'gesture_model.pkl'), 'wb') as f:
        pickle.dump((model, gesture_names), f)
    print("Model trained and saved.")

# Load the trained model
def load_model():
    global model, gesture_names
    with open(os.path.join(data_save_path, 'gesture_model.pkl'), 'rb') as f:
        model, gesture_names = pickle.load(f)
    print("Model loaded.")

# Predict gesture
def predict_gesture(image):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    result = hands.process(image_rgb)
    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            features = extract_features(hand_landmarks)
            features = np.array(features).reshape(1, -1)
            prediction = model.predict(features)
            confidence = model.predict_proba(features).max()
            gesture = gesture_names[prediction[0]]
            cv2.putText(image, f'{gesture} ({confidence:.2f})', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow('Gesture Recognition', image)

# Main function
if __name__ == "__main__":
    choice = input("Enter 'c' to collect data, 't' to train model, 'r' to run recognition: ")

    if choice == 'c':
        collect_data()
    elif choice == 't':
        train_model()
    elif choice == 'r':
        load_model()
        cap = cv2.VideoCapture(0)
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                break
            predict_gesture(image)
            if cv2.waitKey(1) & 0xFF == 27:
                break
        cap.release()
        cv2.destroyAllWindows()
