import streamlit as st
from streamlit_webrtc import webrtc_streamer
import cv2
import mediapipe as mp
import numpy as np

st.title("像個偉(偽)人一樣 📸 (全新基礎相機)")
st.write("👉 先測試新專案的鏡頭與拍照功能是否正常！")

# 初始化拍照歷史紀錄 (保留最多 8 張)
if "history" not in st.session_state:
    st.session_state.history = []

# 宣告 MediaPipe 模組
mp_face_mesh = mp.solutions.face_mesh
mp_hands = mp.solutions.hands

class VideoProcessor:
    def __init__(self):
        # 💡 【高效率關鍵一】：MediaPipe 偵測器只在開機時建立一次，避免重複宣告崩潰
        self.face_mesh = mp_face_mesh.FaceMesh(
            max_num_faces=1, 
            refine_landmarks=True, 
            min_detection_confidence=0.5
        )
        self.hands = mp_hands.Hands(
            max_num_hands=1, 
            min_detection_confidence=0.5
        )
        
        # 💡 【高效率關鍵二】：這輩子只讀取這 5 個偉人的貼圖一次，絕不重複讀檔
        self.buddha_light = cv2.imread("assets/buddha_light.png", cv2.IMREAD_UNCHANGED)
        self.einstein_hair = cv2.imread("assets/einstein_hair.png", cv2.IMREAD_UNCHANGED)
        self.einstein_tongue = cv2.imread("assets/einstein_tongue.png", cv2.IMREAD_UNCHANGED)
        self.kongzi_beard = cv2.imread("assets/kongzi_beard.png", cv2.IMREAD_UNCHANGED)
        self.kongzi_cap = cv2.imread("assets/kongzi_cap.png", cv2.IMREAD_UNCHANGED)
        self.kongzi_sleeves = cv2.imread("assets/kongzi_sleeves.png", cv2.IMREAD_UNCHANGED)
        self.polar_bear = cv2.imread("assets/polar_bear.png", cv2.IMREAD_UNCHANGED)
        self.qin_cap = cv2.imread("assets/qin_cap.png", cv2.IMREAD_UNCHANGED)
        self.tomato = cv2.imread("assets/tomato.png", cv2.IMREAD_UNCHANGED)
        
        # 用於拍照存檔的變數
        self.latest_orig = None
        self.latest_filter = None

    def recv(self, frame):
        # 讀取當前影像
        img = frame.to_ndarray(format="bgr24")
        self.latest_orig = img.copy()
        
        # 這裡目前是純相機測試，所以濾鏡影像直接等於原圖
        # 等一下我們就在這裡一個一個偉人加回去！
        
        self.latest_filter = img.copy()
        return frame.from_ndarray(img, format="bgr24")

# --- 網頁畫面佈局與 WebRTC 鏡頭 ---
ctx = webrtc_streamer(
    key="new-filter-project-camera",
    video_processor_factory=VideoProcessor,
    media_stream_constraints={
        "video": True, 
        "audio": False
    }
)

# 📸 拍照按鈕
if st.button("📸 Capture (拍照)", use_container_width=True):
    if ctx.video_processor and ctx.video_processor.latest_orig is not None:
        orig_rgb = cv2.cvtColor(ctx.video_processor.latest_orig, cv2.COLOR_BGR2RGB)
        filter_rgb = cv2.cvtColor(ctx.video_processor.latest_filter, cv2.COLOR_BGR2RGB)
        
        # 把最新拍的照片塞進歷史紀錄第一筆
        st.session_state.history.insert(0, (orig_rgb, filter_rgb))
        
        # 上限 8 張，超過自動刪除最舊的
        if len(st.session_state.history) > 8:
            st.session_state.history.pop()
            
        st.success("拍照成功！已加到下方紀錄中。")
    else:
        st.warning("請先點擊 START 開啟鏡頭再拍照喔！")

st.markdown("---")

# 🖼️ 顯示剛拍好的大照片與一排 4 張的歷史紀錄
if st.session_state.history:
    st.subheader("🖼️ 剛剛拍到的影像")
    current_orig, current_filter = st.session_state.history[0]
    
    col_orig, col_filt = st.columns(2)
    with col_orig:
        st.image(current_orig, caption="拍到的原影像", use_container_width=True)
    with col_filt:
        st.image(current_filter, caption="加上濾鏡後的影相", use_container_width=True)

    st.markdown("---")

    st.subheader("📜 歷史拍照紀錄 (最多儲存 8 張)")
    cols = st.columns(4)
    for idx, (orig, filt) in enumerate(st.session_state.history):
        col_idx = idx % 4
        with cols[col_idx]:
            # 這裡完全去除了文字標籤，保持排版精美乾淨！
            st.image(filt, use_container_width=True)

