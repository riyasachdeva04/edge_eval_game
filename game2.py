import cv2
import numpy as np
import time
import mediapipe as mp

def generate_grid(n):
    if n == 1:
        return np.array([[1]])
    elif n == 2:
        return np.array([[1, 2],
                         [4, 3]])
    elif n == 3:
        return np.array([[1, 6, 7],
                         [2, 5, 8],
                         [3, 4, 9]])
    elif n == 4:
        return np.array([[1,  2,  3,  4],
                         [8,  7,  6,  5],
                         [9, 10, 11, 12],
                         [16, 15, 14, 13]])
    elif n == 5:
        return np.array([[ 1,  2,  3,  4,  5],
                         [ 16,  17,  18,  19, 6],
                         [15, 24, 25, 20, 7],
                         [14, 23, 22, 21, 8],
                         [13, 12, 11, 10, 9]])
    else:
        raise ValueError("Only n values from 1 to 5 are supported.")

def calculate_grid_parameters(frame, n, margin_percentage=0.1):
    height, width, _ = frame.shape
    vertical_margin = int(height * margin_percentage)
    available_height = height - 2 * vertical_margin
    grid_size = available_height // n
    horizontal_offset = (width - (grid_size * n)) // 2
    vertical_offset = vertical_margin
    return grid_size, horizontal_offset, vertical_offset

def display_grid(frame, grid, guessed_cells=set(), grid_size=None, horizontal_offset=0, vertical_offset=0, show_numbers=True):
    n = len(grid)
    for i in range(n):
        for j in range(n):
            top_left = (horizontal_offset + j * grid_size, vertical_offset + i * grid_size)
            bottom_right = (horizontal_offset + (j + 1) * grid_size, vertical_offset + (i + 1) * grid_size)

            cv2.rectangle(frame, top_left, bottom_right, (255, 255, 255), 2)

            if show_numbers or (i, j) in guessed_cells:
                number = str(grid[i, j])
                font_scale = 2
                font_thickness = 4
                text_size = cv2.getTextSize(number, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)[0]
                text_x = top_left[0] + (grid_size - text_size[0]) // 2
                text_y = top_left[1] + (grid_size + text_size[1]) // 2
                cv2.putText(frame, number, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), font_thickness)

    return frame

def show_countdown(frame, seconds_left):
    countdown_text = f"Memorize: {seconds_left} seconds"
    cv2.putText(frame, countdown_text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)

def display_hearts(frame, hearts):
    heart_icon = cv2.imread('heart_icon.webp') 
    heart_icon = cv2.resize(heart_icon, (40, 40)) 
    height, width, _ = frame.shape
    for i in range(hearts):
        x_offset = width - (i + 1) * 50
        y_offset = 20
        frame[y_offset:y_offset + heart_icon.shape[0], x_offset:x_offset + heart_icon.shape[1]] = heart_icon
    
    return frame

def detect_hand_and_point(frame, hands):
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    finger_position = None

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            index_tip = hand_landmarks.landmark[8]  

            h, w, _ = frame.shape
            index_x, index_y = int(index_tip.x * w), int(index_tip.y * h)

            cv2.circle(frame, (index_x, index_y), 10, (0, 255, 0), -1) 

            finger_position = (index_x, index_y)

    return finger_position


def guess_grid(frame, grid, correct_sequence, current_guess_idx, finger_position, guessed_cells, grid_size, horizontal_offset, vertical_offset):
    n = len(grid)

    if finger_position:
        fx, fy = finger_position

        for i in range(n):
            for j in range(n):
                top_left = (horizontal_offset + j * grid_size, vertical_offset + i * grid_size)
                bottom_right = (horizontal_offset + (j + 1) * grid_size, vertical_offset + (i + 1) * grid_size)

                if (i, j) not in guessed_cells:
                    if top_left[0] < fx < bottom_right[0] and top_left[1] < fy < bottom_right[1]:
                        guessed_cell = grid[i][j]

                        if guessed_cell == correct_sequence.flatten()[current_guess_idx]:
                            guessed_cells.add((i, j)) 
                            current_guess_idx += 1 
                            return True, current_guess_idx 
                        else:
                            return False, current_guess_idx 
    return True, current_guess_idx 


def run_game(selected_level):
    video_feed = cv2.VideoCapture(0)
    hearts = 3
    level = 1
    if selected_level is not None: level = selected_level

    scaling_factor = 1.5  


    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(min_detection_confidence=0.7)

    while hearts > 0:
        n = level + 1
        grid = generate_grid(n)
        correct_sequence = np.arange(1, n * n + 1).reshape(n, n)
        current_guess_idx = 0
        guessed_cells = set() 

        start_time = time.time()
        while True:
            ret, frame = video_feed.read()
            if not ret:
                break
            
            frame = cv2.flip(frame, 1)
            frame = cv2.resize(frame, None, fx=scaling_factor, fy=scaling_factor)

            grid_size, horizontal_offset, vertical_offset = calculate_grid_parameters(frame, n)
            total_time = 4
            elapsed_time = time.time() - start_time
            time_left = max(0, total_time - int(elapsed_time))
            cv2.putText(frame, "Time: " + str(elapsed_time), (150, 150), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 2)

            frame = display_grid(frame, grid, show_numbers=True, grid_size=grid_size, horizontal_offset=horizontal_offset, vertical_offset=vertical_offset)
            frame = display_hearts(frame, hearts)

            height, width, _ = frame.shape
            level_text = f"Level {level}"
            text_size = cv2.getTextSize(level_text, cv2.FONT_HERSHEY_SIMPLEX, 1.5, 2)[0]
            text_x = (width - text_size[0]) // 2
            cv2.putText(frame, level_text, (text_x, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 2)

            show_countdown(frame, time_left)

            cv2.imshow('Grid Master', frame)

            if elapsed_time >= total_time:
                break

            key = cv2.waitKey(1)
            if key == ord('q'):
                hearts = 0
                break

        while current_guess_idx < n * n:
            
            ret, frame = video_feed.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            frame = cv2.resize(frame, None, fx=scaling_factor, fy=scaling_factor)

            grid_size, horizontal_offset, vertical_offset = calculate_grid_parameters(frame, n)
            frame = display_grid(frame, grid, guessed_cells, grid_size=grid_size, horizontal_offset=horizontal_offset, vertical_offset=vertical_offset, show_numbers=False)

            finger_position = detect_hand_and_point(frame, hands)


            guessed_correctly, current_guess_idx = guess_grid(frame, grid, correct_sequence, current_guess_idx, finger_position, guessed_cells, grid_size, horizontal_offset, vertical_offset)
            
            if not guessed_correctly and finger_position:
                hearts -= 1  
                break 

            cv2.imshow('Grid Master', frame)
            key = cv2.waitKey(1)
            if key == ord('q'):
                hearts = 0
                break

        if len(guessed_cells) >= n * n: 
            level += 1 
        else:
            if hearts == 0: 

                video_feed.release()
                cv2.destroyAllWindows()
                return

    

