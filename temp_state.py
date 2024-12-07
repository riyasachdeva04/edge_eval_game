import cv2
import mediapipe as mp

flower_band = cv2.imread('flower_band.png', cv2.IMREAD_UNCHANGED)

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, 
                                   max_num_faces=1, 
                                   min_detection_confidence=0.5)

def run_game():
    cap = cv2.VideoCapture(0)

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 3
    font_thickness = 8
    text_color = (255, 255, 255) 

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

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
            text = "LETS PLAYYYY!"
            text_size = cv2.getTextSize(text, font, font_scale, font_thickness)[0]

            text_x = frame_width - text_size[0] - 20  
            text_y = 100  

            cv2.putText(frame, text, (text_x, text_y), font, font_scale, text_color, font_thickness)

        cv2.imshow('Flower Band Snap Filter', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
