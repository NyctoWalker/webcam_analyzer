import cv2
import mediapipe as mp
import time
from scipy.spatial import distance as dist

cam = cv2.VideoCapture(0)
font = cv2.FONT_HERSHEY_DUPLEX
stat_text_color = (0, 200, 0)
metrics_text_color = (0, 150, 255)

# DEBUG
draw_landmarks = True
show_metrics = True

# face mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)


# EAR for blinking
def calculate_EAR(eye):
    v1 = dist.euclidean(eye[1], eye[5])
    v2 = dist.euclidean(eye[2], eye[4])
    h = dist.euclidean(eye[0], eye[3])
    return (v1 + v2) / (2.0 * h) if h else 0.0

# eye landmarks
L_EYE = [33, 160, 158, 133, 153, 144]
R_EYE = [362, 385, 387, 263, 373, 380]
L_IRIS, R_IRIS = 468, 473

# blink state
BLINK_THRESH = 0.21
BLINK_FRAMES = 2
blink_counter = 0
blink_count = 0

# smile state
SMILE_THRESH = 1
SMILE_FRAMES = 5
NON_SMILE_FRAMES = 5
smile_frame_counter = 0
non_smile_counter = 0
is_smiling = False
smile_start_time = None
smile_count = 0
total_smile_time = 0.0

while True:
    ret, frame = cam.read()
    if not ret:
        break

    frame = cv2.resize(frame, (640, 480))
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    smile_metric = 0.0
    avg_EAR = 0.0

    if results.multi_face_landmarks:
        for fl in results.multi_face_landmarks:
            h, w = frame.shape[:2]

            def pt(i):
                return (fl.landmark[i].x * w, fl.landmark[i].y * h)

            # blink
            left_eye = [pt(i) for i in L_EYE]
            right_eye = [pt(i) for i in R_EYE]
            avg_EAR = (calculate_EAR(left_eye) + calculate_EAR(right_eye)) / 2.0

            if avg_EAR < BLINK_THRESH:
                blink_counter += 1
            else:
                if blink_counter >= BLINK_FRAMES:
                    blink_count += 1
                blink_counter = 0

            # smile
            l_corner = pt(61)
            r_corner = pt(291)
            upper_lip = pt(13)
            lower_lip = pt(14)
            l_iris = pt(L_IRIS)
            r_iris = pt(R_IRIS)

            pupil_dist = dist.euclidean(l_iris, r_iris)
            mouth_width = dist.euclidean(l_corner, r_corner)

            # width
            width_ratio = mouth_width / pupil_dist if pupil_dist else 0.0

            # corner elevation
            center_y = (upper_lip[1] + lower_lip[1]) / 2.0
            elev_l = center_y - l_corner[1]
            elev_r = center_y - r_corner[1]
            avg_elevation = (elev_l + elev_r) / 2.0
            elevation_ratio = avg_elevation / pupil_dist if pupil_dist else 0.0

            # "magic" combined metric
            smile_metric = width_ratio + (elevation_ratio * 2.0)

            # DEBUG
            if draw_landmarks:
                for idx in L_EYE + R_EYE + [61, 291, 13, 14]:
                    px = int(fl.landmark[idx].x * w)
                    py = int(fl.landmark[idx].y * h)
                    cv2.circle(frame, (px, py), 2, (0, 255, 0), -1)
                for idx in (L_IRIS, R_IRIS):
                    px = int(fl.landmark[idx].x * w)
                    py = int(fl.landmark[idx].y * h)
                    cv2.circle(frame, (px, py), 2, (255, 200, 0), -1)

    # smile state machine
    if smile_metric > SMILE_THRESH:
        smile_frame_counter += 1
        non_smile_counter = 0
    else:
        smile_frame_counter = 0
        non_smile_counter += 1

    if not is_smiling and smile_frame_counter >= SMILE_FRAMES:
        is_smiling = True
        smile_count += 1
        smile_start_time = time.time()

    if is_smiling and non_smile_counter >= NON_SMILE_FRAMES:
        is_smiling = False
        if smile_start_time is not None:
            total_smile_time += time.time() - smile_start_time
            smile_start_time = None

    # smile timer
    live_smile_time = total_smile_time
    if is_smiling and smile_start_time is not None:
        live_smile_time += time.time() - smile_start_time

    # text
    cv2.putText(frame, f'Blink count: {blink_count}', (30, 30), font, 1, stat_text_color, 2)
    cv2.putText(frame, f'Smile count: {smile_count}', (30, 60), font, 1, stat_text_color, 2)
    cv2.putText(frame, f'Smile time: {live_smile_time:.2f} s', (30, 90),  font, 1, stat_text_color, 2)

    if show_metrics and results.multi_face_landmarks:
        cv2.putText(frame, f'EAR: {avg_EAR:.2f}', (30, 120), font, 0.75, metrics_text_color, 2)
        cv2.putText(frame, f'Smile custom metric: {smile_metric:.2f}', (30, 140), font, 0.75, metrics_text_color, 2)

    cv2.imshow("Camera", frame)
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()
