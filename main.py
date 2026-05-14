import cv2
import mediapipe as mp
import numpy as np

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)

cap = cv2.VideoCapture(0)
canvas = None
last_x, last_y = 0, 0
color = (0, 215, 255)
thickness = 5

def count_fingers(hand_landmarks):
    landmarks = hand_landmarks.landmark
    finger_tips  = [8, 12, 16, 20]
    finger_bases = [6, 10, 14, 18]
    fingers_up = 0
    if landmarks[4].x < landmarks[3].x:
        fingers_up += 1
    for tip, base in zip(finger_tips, finger_bases):
        if landmarks[tip].y < landmarks[base].y:
            fingers_up += 1
    return fingers_up == 5

def draw_fancy_hand(frame, hand_landmarks, w, h):
    for connection in mp_hands.HAND_CONNECTIONS:
        x1 = int(hand_landmarks.landmark[connection[0]].x * w)
        y1 = int(hand_landmarks.landmark[connection[0]].y * h)
        x2 = int(hand_landmarks.landmark[connection[1]].x * w)
        y2 = int(hand_landmarks.landmark[connection[1]].y * h)
        cv2.line(frame, (x1, y1), (x2, y2), (255, 220, 0), 2)
        cv2.line(frame, (x1, y1), (x2, y2), (255, 255, 100), 1)
    for landmark in hand_landmarks.landmark:
        x = int(landmark.x * w)
        y = int(landmark.y * h)
        cv2.circle(frame, (x, y), 4, (255, 255, 255), -1)
        cv2.circle(frame, (x, y), 7, (255, 200, 0), 1)

while True:
    success, frame = cap.read()
    if not success:
        break
    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    if canvas is None:
        canvas = np.zeros_like(frame)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)
    drawing_paused = False
    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]
        draw_fancy_hand(frame, hand_landmarks, w, h)
        if count_fingers(hand_landmarks):
            drawing_paused = True
            last_x, last_y = 0, 0
        else:
            index_finger = hand_landmarks.landmark[8]
            x, y = int(index_finger.x * w), int(index_finger.y * h)
            cv2.circle(frame, (x, y), 10, color, -1)
            if last_x != 0 and last_y != 0:
                cv2.line(canvas, (last_x, last_y), (x, y), color, thickness)
            last_x, last_y = x, y
    else:
        last_x, last_y = 0, 0
    combined = cv2.addWeighted(frame, 0.7, canvas, 0.3, 0)
    status = ("", (0, 100, 255)) if drawing_paused else ("", (0, 215, 255))
    cv2.putText(combined, status[0], (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, status[1], 2)
    cv2.putText(combined, "C = clear | Q = quit", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.imshow("Drawing with Hand", combined)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('c'):
        canvas = np.zeros_like(frame)

cap.release()
cv2.destroyAllWindows()