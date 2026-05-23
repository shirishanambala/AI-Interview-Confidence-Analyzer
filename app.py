import cv2
import mediapipe as mp
import time
import sounddevice as sd
import numpy as np

# -------------------------
# MEDIAPIPE SETUP
# -------------------------
mp_face_mesh = mp.solutions.face_mesh

face_mesh = mp_face_mesh.FaceMesh(
    refine_landmarks=True,
    max_num_faces=1
)

# -------------------------
# CAMERA SETUP
# -------------------------
url = "http://192.168.1.9:4747/video"

cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)

# Reduce lag
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

if not cap.isOpened():
    print("❌ Camera error")
    exit()

# -------------------------
# VARIABLES
# -------------------------
frame_count = 0

emotion = "neutral"
direction = "No Face"

focused_time = 0
not_looking_frames = 0

confidence_list = []

voice = 0

start_time = time.time()

# -------------------------
# MIC FUNCTION
# -------------------------
def get_voice_level():

    duration = 0.3

    recording = sd.rec(
        int(duration * 44100),
        samplerate=44100,
        channels=1
    )

    sd.wait()

    volume = np.linalg.norm(recording)

    return volume

# -------------------------
# MAIN LOOP
# -------------------------
while True:

    ret, frame = cap.read()

    if not ret:
        print("❌ Failed to read frame")
        break

    # Mirror effect
    frame = cv2.flip(frame, 1)

    h, w, _ = frame.shape

    # Convert to RGB
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # FaceMesh processing
    results = face_mesh.process(rgb)

    frame_count += 1

    # Default values
    emotion = "neutral"
    direction = "No Face"
    confidence = 0
    warning = ""

    # -------------------------
    # FACE DETECTION
    # -------------------------
    if results.multi_face_landmarks:

        for face_landmarks in results.multi_face_landmarks:

            # -------------------------
            # FACE BOX
            # -------------------------
            x_min = int(min([lm.x for lm in face_landmarks.landmark]) * w)
            y_min = int(min([lm.y for lm in face_landmarks.landmark]) * h)

            x_max = int(max([lm.x for lm in face_landmarks.landmark]) * w)
            y_max = int(max([lm.y for lm in face_landmarks.landmark]) * h)

            cv2.rectangle(
                frame,
                (x_min, y_min),
                (x_max, y_max),
                (255, 0, 0),
                2
            )

            # -------------------------
            # EYE TRACKING
            # -------------------------
            left_eye = face_landmarks.landmark[468]
            right_eye = face_landmarks.landmark[473]

            lx, ly = int(left_eye.x * w), int(left_eye.y * h)
            rx, ry = int(right_eye.x * w), int(right_eye.y * h)

            eye_x = (lx + rx) / 2
            eye_y = (ly + ry) / 2

            center_x = w / 2
            center_y = h / 2

            dx = eye_x - center_x
            dy = eye_y - center_y

            # -------------------------
            # DIRECTION DETECTION
            # -------------------------
            threshold = 60

            if abs(dx) < threshold and abs(dy) < threshold:
                direction = "Center"

            elif abs(dx) > abs(dy):

                if dx < 0:
                    direction = "Left"
                else:
                    direction = "Right"

            else:

                if dy < 0:
                    direction = "Up"
                else:
                    direction = "Down"

            # -------------------------
            # ATTENTION TRACKING
            # -------------------------
            if direction == "Center":

                focused_time += 1
                not_looking_frames = 0

            else:
                not_looking_frames += 1

            if not_looking_frames > 50:
                warning = "⚠️ Look at camera!"

            # -------------------------
            # VOICE ANALYSIS
            # -------------------------
            # Run only every 20 frames
            if frame_count % 20 == 0:
                voice = get_voice_level()

            # Simple confidence emotion
            if voice > 10:
                emotion = "confident"
            else:
                emotion = "neutral"

            # -------------------------
            # CONFIDENCE SCORE
            # -------------------------
            confidence = 50

            # Eye contact
            if direction == "Center":
                confidence += 20
            else:
                confidence -= 10

            # Emotion
            if emotion == "confident":
                confidence += 20

            elif emotion == "neutral":
                confidence += 10

            else:
                confidence -= 10

            # Voice strength
            if voice > 10:
                confidence += 10
            else:
                confidence -= 5

            # Limit 0-100
            confidence = max(0, min(confidence, 100))

            confidence_list.append(confidence)

    # -------------------------
    # LIVE FOCUS %
    # -------------------------
    focus_percent_live = (
        (focused_time / frame_count) * 100
        if frame_count
        else 0
    )

    # -------------------------
    # DISPLAY TEXT
    # -------------------------
    cv2.putText(
        frame,
        f"{direction} | {emotion}",
        (30, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    cv2.putText(
        frame,
        f"Confidence: {confidence}%",
        (30, 100),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 255, 0),
        2
    )

    cv2.putText(
        frame,
        f"Focus: {focus_percent_live:.1f}%",
        (30, 150),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 255),
        2
    )

    if warning:

        cv2.putText(
            frame,
            warning,
            (30, 200),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2
        )

    # -------------------------
    # SHOW WINDOW
    # -------------------------
    cv2.imshow("AI Interview Analyzer", frame)

    # Exit key
    key = cv2.waitKey(1)

    if key & 0xFF in [27, ord('q')]:
        print("Interview ended.")
        break

# -------------------------
# FINAL REPORT
# -------------------------
total_time = time.time() - start_time

focus_percent = (
    (focused_time / frame_count) * 100
    if frame_count
    else 0
)

avg_confidence = (
    sum(confidence_list) / len(confidence_list)
    if confidence_list
    else 0
)

print("\n📊 FINAL REPORT")
print(f"Total Time: {round(total_time, 2)} sec")
print(f"Focus: {round(focus_percent, 2)} %")
print(f"Avg Confidence: {round(avg_confidence, 2)} %")

# -------------------------
# CLEANUP
# -------------------------
cap.release()

cv2.destroyAllWindows()