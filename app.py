
import streamlit as st
import tensorflow as tf
import numpy as np
import joblib
import cv2
import mediapipe as mp
import tempfile

# -------------------------------
# CONFIG
# -------------------------------
SEQUENCE_LENGTH = 60  
mp_hands = mp.solutions.hands

# -------------------------------
# UI
# -------------------------------
st.title("✋ Hearing assistive technology system")
st.write("กรุณาอัปโหลดวิดีโอภาษามือ แล้วระบบจะทำนายเป็นข้อความให้โดยอัตโนมัติ")

# -------------------------------
# โหลดโมเดลและ Label Encoder
# -------------------------------
@st.cache_resource
def load_model_and_labels():
    model = tf.keras.models.load_model("signlang_lstm.h5")
    encoder = joblib.load("label_encoder.pkl")
    labels = list(encoder.classes_)
    return model, labels

model, labels = load_model_and_labels()

# -------------------------------
# อัปโหลดไฟล์วิดีโอ
# -------------------------------
uploaded_file = st.file_uploader("📂 เลือกวิดีโอ", type=["mp4", "avi", "mov"])

if uploaded_file is not None:
    # แสดงวิดีโอที่อัปโหลด
    st.video(uploaded_file)

    # เก็บไฟล์ชั่วคราว
    tfile = tempfile.NamedTemporaryFile(delete=False) 
    tfile.write(uploaded_file.read())

    # อ่านวิดีโอด้วย OpenCV
    cap = cv2.VideoCapture(tfile.name)
    sequence = []

    with mp_hands.Hands(
        max_num_hands=2,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.5
    ) as hands:

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # เตรียมภาพสำหรับ Mediapipe
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = hands.process(image)

            landmarks_all = []
            if result.multi_hand_landmarks:
                for hand_landmarks in result.multi_hand_landmarks:
                    for lm in hand_landmarks.landmark:
                        landmarks_all.extend([lm.x, lm.y, lm.z])

            # เก็บเฉพาะถ้ามี 21 จุดครบ (2 มือ = 126 ค่า)
            if len(landmarks_all) == 126:
                sequence.append(landmarks_all)

        cap.release()


    # -------------------------------
    # ทำการทำนาย (รองรับทุกความยาว)
    # -------------------------------
    if len(sequence) > 0:
        seq_array = np.array(sequence)

        # ถ้าน้อยกว่า 60 → pad ด้วยศูนย์
        if len(seq_array) < SEQUENCE_LENGTH:
            pad_length = SEQUENCE_LENGTH - len(seq_array)
            pad_array = np.zeros((pad_length, 126))  # 126 = keypoints ต่อเฟรม
            seq_array = np.vstack([seq_array, pad_array])

        # ถ้ามากกว่า 60 → เอา 60 เฟรมสุดท้าย
        else:
            seq_array = seq_array[-SEQUENCE_LENGTH:]

        # reshape ให้เข้ากับโมเดล
        seq_input = np.expand_dims(seq_array, axis=0)

        prediction = model.predict(seq_input, verbose=0)[0]
        max_index = np.argmax(prediction)
        predicted_label = labels[max_index]
        confidence = prediction[max_index]

        st.success(f"✅ ผลลัพธ์: **{predicted_label}**")
        st.write(f"📊 ความมั่นใจ: {confidence:.2f}")
    else:
        st.warning("⚠️ ไม่เจอ keypoints จากมือในวิดีโอ")
