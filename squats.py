import cv2
import mediapipe as mp
import numpy as np

# Initialize Mediapipe modules
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose


# Function to calculate angle between three points
def calculate_angle(a, b, c):
    a = np.array(a)  # First point
    b = np.array(b)  # Midpoint
    c = np.array(c)  # Last point

    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)

    if angle > 180.0:
        angle = 360 - angle

    return angle


# Function to track squats
def track_squats(landmarks, counter, stage):
    # Extract relevant joints
    hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
           landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
    knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
            landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
    ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
             landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]

    # Calculate angle
    angle = calculate_angle(hip, knee, ankle)

    # Squat logic
    if angle > 160:
        stage = "up"
    if angle < 70 and stage == "up":
        stage = "down"
        counter += 1

    return angle, counter, stage


# Function to track push-ups
def track_pushups(landmarks, counter, stage):
    # Extract relevant joints
    shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
    elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
             landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
    wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
             landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]

    # Calculate angle
    angle = calculate_angle(shoulder, elbow, wrist)

    # Push-up logic
    if angle > 160:
        stage = "up"
    if angle < 90 and stage == "up":
        stage = "down"
        counter += 1

    return angle, counter, stage


# Main script
def main():
    cap = cv2.VideoCapture(0)
    counter = 0  # Repetition counter
    stage = None  # Current stage ("up" or "down")
    exercise = input("Enter the exercise (squats/pushups): ").lower()

    # Setup Mediapipe instance
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            ret, frame = cap.read()

            # Recolor image to RGB
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False

            # Make detection
            results = pose.process(image)

            # Recolor back to BGR
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            # Extract landmarks
            try:
                landmarks = results.pose_landmarks.landmark

                # Track the selected exercise
                if exercise == "squats":
                    angle, counter, stage = track_squats(landmarks, counter, stage)
                elif exercise == "pushups":
                    angle, counter, stage = track_pushups(landmarks, counter, stage)
                else:
                    print("Invalid exercise selected!")
                    break

                # Display angle
                cv2.putText(image, f'Angle: {int(angle)}',
                            (50, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

                # Display repetitions
                cv2.putText(image, f'Reps: {counter}',
                            (50, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

                # Display stage
                cv2.putText(image, f'Stage: {stage}',
                            (50, 150),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

            except:
                pass

            # Render detections
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                      mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                                      mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2))

            # Display the frame
            cv2.imshow('Mediapipe Feed', image)

            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
