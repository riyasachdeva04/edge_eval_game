import cv2
import numpy as np
import mediapipe as mp


mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.7)

mp_drawing = mp.solutions.drawing_utils

def count_fingers(hand_landmarks):
    fingers = []

    if hand_landmarks[mp_hands.HandLandmark.THUMB_TIP].y < hand_landmarks[mp_hands.HandLandmark.THUMB_IP].y:
        fingers.append(1)

    if hand_landmarks[mp_hands.HandLandmark.INDEX_FINGER_TIP].y < hand_landmarks[mp_hands.HandLandmark.INDEX_FINGER_DIP].y:
        fingers.append(1)

    if hand_landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y < hand_landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_DIP].y:
        fingers.append(1)

    if hand_landmarks[mp_hands.HandLandmark.RING_FINGER_TIP].y < hand_landmarks[mp_hands.HandLandmark.RING_FINGER_DIP].y:
        fingers.append(1)

    if hand_landmarks[mp_hands.HandLandmark.PINKY_TIP].y < hand_landmarks[mp_hands.HandLandmark.PINKY_DIP].y:
        fingers.append(1)

    return len(fingers)

def apply_color_filter(frame, num_fingers):
    filtered_frame = frame.copy()
    applied_filter = ""

    if num_fingers == 1:

        kernel = np.ones((5, 5), np.uint8)
        filtered_frame = cv2.dilate(filtered_frame, kernel)
        applied_filter = "Dilation Filter"
    elif num_fingers == 2:

        kernel = np.ones((5, 5), np.uint8)
        filtered_frame = cv2.erode(filtered_frame, kernel)
        applied_filter = "Erosion Filter"
    elif num_fingers == 3:

        filtered_frame = cv2.blur(filtered_frame, (15, 15), 0)
        applied_filter = "Mean Filter"
    elif num_fingers == 4:

        filtered_frame = cv2.medianBlur(filtered_frame, 29)
        applied_filter = "Median Filter"
    elif num_fingers == 5:

        filtered_frame = cv2.GaussianBlur(filtered_frame, (5, 5), 0)
        applied_filter = "Gaussian Filter"
    
    return filtered_frame, applied_filter

def run_game1():
    temp = False
    vid = cv2.VideoCapture(0)
    if not vid.isOpened():
        print("Error: Could not open video.")
        return None 

    selected_fingers = 0 
    level = None 
    button_position = (50, 400, 200, 100) 

    level_map = {
        1: "Level One",
        2: "Level Two",
        3: "Level Three",
        4: "Level Four",
        5: "Level Five"
    }

    while True:
        ret, frame = vid.read()
        if not ret:
            print("Error: Could not read frame.")
            break

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                selected_fingers = count_fingers(hand_landmarks.landmark)

                level_text = level_map.get(selected_fingers, "")  

                font_scale = 7  
                thickness = 15  
                color = (0, 255, 255) 

                if level_text:  
                    text_size = cv2.getTextSize(level_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0]
                    text_x = (frame.shape[1] - text_size[0]) // 2
                    text_y = (frame.shape[0] + text_size[1]) // 2 
                    cv2.putText(frame, level_text, (text_x, text_y), 
                                cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness, cv2.LINE_AA)

                cv2.rectangle(frame, (button_position[0], button_position[1]), 
                              (button_position[0] + button_position[2], button_position[1] + button_position[3]), 
                              (255, 0, 0), -1) 
                cv2.putText(frame, "OK", (button_position[0] + 40, button_position[1] + 60), 
                            cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3) 

                if selected_fingers >= 2: 
                    for hand_landmarks in results.multi_hand_landmarks:
                        index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                        if (button_position[0] < index_finger_tip.x * frame.shape[1] < button_position[0] + button_position[2] and
                            button_position[1] < index_finger_tip.y * frame.shape[0] < button_position[1] + button_position[3]):
                            level = selected_fingers
                            print(f"Level selected: {level}") 
                            temp = True
                            return level
                filtered_frame, applied_text = apply_color_filter(frame, selected_fingers)
                cv2.putText(frame, applied_text, (150, 150), 
                                cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness, cv2.LINE_AA)

                frame = cv2.addWeighted(frame, 0.5, filtered_frame, 0.5, 0) 

        cv2.imshow("Finger Count", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    vid.release()
    cv2.destroyAllWindows()

    return level

