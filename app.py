import streamlit as st
from PIL import Image, ImageFilter


st.cache_data.clear()
st.cache_resource.clear()

#import streamlit as st
from streamlit_webrtc import webrtc_streamer
import cv2
import mediapipe as mp
import numpy as np

st.title("像個偉(偽)人一樣 📸 (相機復活版)")
st.write("👉 先確認鏡頭開不開得起來！")

if "history" not in st.session_state:
    st.session_state.history = []

class VideoProcessor:
    def __init__(self):
        # 開機時只讀取一次貼圖，超級省記憶體
        self.buddha_light = cv2.imread("assets/buddha_light.png", cv2.IMREAD_UNCHANGED)
        self.einstein_hair = cv2.imread("assets/einstein_hair.png", cv2.IMREAD_UNCHANGED)
        self.einstein_tongue = cv2.imread("assets/einstein_tongue.png", cv2.IMREAD_UNCHANGED)
        self.kongzi_beard = cv2.imread("assets/kongzi_beard.png", cv2.IMREAD_UNCHANGED)
        self.kongzi_cap = cv2.imread("assets/kongzi_cap.png", cv2.IMREAD_UNCHANGED)
        self.kongzi_sleeves = cv2.imread("assets/kongzi_sleeves.png", cv2.IMREAD_UNCHANGED)
        self.polar_bear = cv2.imread("assets/polar_bear.png", cv2.IMREAD_UNCHANGED)
        self.qin_cap = cv2.imread("assets/qin_cap.png", cv2.IMREAD_UNCHANGED)
        self.tomato = cv2.imread("assets/tomato.png", cv2.IMREAD_UNCHANGED)
        
        self.latest_orig = None
        self.latest_filter = None

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        self.latest_orig = img.copy()
        self.latest_filter = img.copy()
        return frame.from_ndarray(img, format="bgr24")

ctx = webrtc_streamer(
    key="fixed-camera",
    video_processor_factory=VideoProcessor,
    media_stream_constraints={"video": True, "audio": False}
)

if st.button("📸 Capture (拍照)", use_container_width=True):
    if ctx.video_processor and ctx.video_processor.latest_orig is not None:
        orig_rgb = cv2.cvtColor(ctx.video_processor.latest_orig, cv2.COLOR_BGR2RGB)
        filter_rgb = cv2.cvtColor(ctx.video_processor.latest_filter, cv2.COLOR_BGR2RGB)
        st.session_state.history.insert(0, (orig_rgb, filter_rgb))
        if len(st.session_state.history) > 8:
            st.session_state.history.pop()
        st.success("拍照成功！")
    else:
        st.warning("請先點擊 START 開啟鏡頭再拍照喔！")

st.markdown("---")

if st.session_state.history:
    st.subheader("🖼️ 剛剛拍到的影像")
    current_orig, current_filter = st.session_state.history[0]
    col_orig, col_filt = st.columns(2)
    with col_orig:
        st.image(current_orig, caption="原影像", use_container_width=True)
    with col_filt:
        st.image(current_filter, caption="濾鏡影像", use_container_width=True)

    st.markdown("---")
    st.subheader("📜 歷史拍照紀錄 (最多儲存 8 張)")
    cols = st.columns(4)
    for idx, (orig, filt) in enumerate(st.session_state.history):
        col_idx = idx % 4
        with cols[col_idx]:
            st.image(filt, use_container_width=True)
