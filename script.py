import cv2
import mediapipe as mp

# Gunakan kamera laptop
cap = cv2.VideoCapture(0)

# Validasi kamera berhasil dibuka
if not cap.isOpened():
    print("❌ Kamera tidak bisa dibuka.")
    exit()

# Inisialisasi MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False,
                       max_num_hands=2,
                       min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Nama dan ID jari
finger_names = ["Jempol", "Telunjuk", "Tengah", "Manis", "Kelingking"]
tip_ids = [4, 8, 12, 16, 20]

def finger_status(landmarks, handedness):
    fingers = []

    # Jempol: arah X tergantung kiri atau kanan
    if handedness == "Right":
        fingers.append(landmarks[4].x < landmarks[3].x)
    else:
        fingers.append(landmarks[4].x > landmarks[3].x)

    # 4 jari lainnya
    for tip_id in tip_ids[1:]:
        fingers.append(landmarks[tip_id].y < landmarks[tip_id - 2].y)

    return fingers

while True:
    success, img = cap.read()
    if not success:
        print("❌ Gagal membaca frame.")
        break

    img = cv2.flip(img, 1)
    img = cv2.resize(img, (800, 600))  # Ukuran tidak terlalu besar
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)
    height, width, _ = img.shape

    total_open_fingers = 0
    total_hands = len(results.multi_hand_landmarks) if results.multi_hand_landmarks else 0

    if results.multi_hand_landmarks and results.multi_handedness:
        for hand_landmarks, hand_handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
            mp_draw.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            lm = hand_landmarks.landmark
            handedness_label = hand_handedness.classification[0].label
            cx = int(lm[9].x * width)
            cy = int(lm[9].y * height)

            fingers = finger_status(lm, handedness_label)
            total_open_fingers += sum(fingers)

            for i, is_open in enumerate(fingers):
                if is_open:
                    tip_id = tip_ids[i]
                    fx = int(lm[tip_id].x * width)
                    fy = int(lm[tip_id].y * height)

                    # Laser hijau dan efek lingkaran
                    color = (0, 255, 0)
                    cv2.line(img, (cx, cy), (fx, fy), color, 3)
                    cv2.circle(img, (fx, fy), 10, color, cv2.FILLED)
                    cv2.circle(img, (fx, fy), 20, color, 2)

                    cv2.putText(img, f"{handedness_label} {finger_names[i]}", 
                                (fx - 40, fy - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                                0.5, color, 2)

    # Tampilkan jumlah tangan & jari
    cv2.putText(img, f"Tangan Terdeteksi: {total_hands}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    cv2.putText(img, f"Jari Terbuka: {total_open_fingers}", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

    cv2.imshow("Hand Laser Visualizer", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
