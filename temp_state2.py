import cv2
import mediapipe as mp
import numpy as np

flower_band = cv2.imread('horns.png', cv2.IMREAD_UNCHANGED)
pookie_image = cv2.imread('pookie.png', cv2.IMREAD_UNCHANGED)

pookie_height, pookie_width = pookie_image.shape[:2]
pookie_image = cv2.resize(pookie_image, (pookie_width * 3, pookie_height * 3), interpolation=cv2.INTER_LINEAR)

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, 
                                   max_num_faces=1, 
                                   min_detection_confidence=0.5)

def apply_kirsch(gray):
    kernel_kirsch = np.array([[5, 5, 5], [-3, 0, -3], [-3, -3, -3]], dtype=np.float32)
    return cv2.filter2D(gray, cv2.CV_64F, kernel_kirsch)

def run_game():
    cap = cv2.VideoCapture(0)

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 3
    font_thickness = 8
    text_color = (0, 0, 255)  

    current_filter = 'Laplacian'

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if current_filter == 'Canny':
            edges = cv2.Canny(gray_frame, 100, 200)
        elif current_filter == 'Sobel':
            sobelx = cv2.Sobel(gray_frame, cv2.CV_64F, 1, 0, ksize=5)
            sobely = cv2.Sobel(gray_frame, cv2.CV_64F, 0, 1, ksize=5)
            edges = cv2.magnitude(sobelx, sobely)
        # elif current_filter == 'Prewitt':
        #     kernelx = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]], dtype=np.float32)
        #     kernely = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]], dtype=np.float32)
        #     prewittx = cv2.filter2D(gray_frame, -1, kernelx)
        #     prewitty = cv2.filter2D(gray_frame, -1, kernely)
        #     edges = cv2.magnitude(prewittx, prewitty)
        elif current_filter == 'Laplacian':
            edges = cv2.Laplacian(gray_frame, cv2.CV_64F)
        elif current_filter == 'Kirsch':
            edges = apply_kirsch(gray_frame)

        edges_colored = cv2.cvtColor(np.uint8(edges), cv2.COLOR_GRAY2BGR)

        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        faces = face_cascade.detectMultiScale(frame, scaleFactor=1.1, minNeighbors=5)

        for (x, y, w, h) in faces:
            face_roi = image_rgb[y:y + h, x:x + w]
            results = face_mesh.process(face_roi)

            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    for landmark in face_landmarks.landmark:
                        h, w, _ = face_roi.shape
                        cx, cy = int(landmark.x * w), int(landmark.y * h)

                    band_width = w
                    band_height = int(band_width * (flower_band.shape[0] / flower_band.shape[1]))
                    band_x = x
                    band_y = y - band_height // 2
                    resized_band = cv2.resize(flower_band, (band_width, band_height))

                    for i in range(band_height):
                        for j in range(band_width):
                            if resized_band[i, j][3] != 0:  
                                frame[band_y + i, band_x + j] = resized_band[i, j][:3]

            frame_width = frame.shape[1]
            text = "GAME OVER!"
            text_size = cv2.getTextSize(text, font, font_scale, font_thickness)[0]

            text_x = frame_width - text_size[0] - 20
            text_y = 100
            cv2.putText(frame, text, (text_x, text_y), font, font_scale, text_color, font_thickness)
            cv2.putText(frame, current_filter, (150, 150), font, font_scale, text_color, font_thickness)

        bottom_right_x = frame.shape[1] - pookie_image.shape[1] - 20
        bottom_right_y = frame.shape[0] - pookie_image.shape[0] - 20
        for i in range(pookie_image.shape[0]):
            for j in range(pookie_image.shape[1]):
                if pookie_image[i, j][3] != 0: 
                    frame[bottom_right_y + i, bottom_right_x + j] = pookie_image[i, j][:3]

        blended_frame = cv2.addWeighted(frame, 0.6, edges_colored, 0.4, 0)

        cv2.imshow('Flower Band Snap Filter with Edge Filters', blended_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('c'):
            current_filter = 'Canny'
        elif key == ord('s'):
            current_filter = 'Sobel'
        # elif key == ord('p'):
        #     current_filter = 'Prewitt'
        elif key == ord('l'):
            current_filter = 'Laplacian'
        elif key == ord('k'):
            current_filter = 'Kirsch'
        elif key == ord('n'):
            current_filter = 'None'

    cap.release()
    cv2.destroyAllWindows()