import streamlit as st
from streamlit_webrtc import webrtc_streamer
import cv2
import mediapipe as mp
import numpy as np

st.title("像個偉(偽)人一樣 📸part2")
st.write("👉 請對鏡頭【比個耶 ✌️】來召喚至聖先師！")

if "history" not in st.session_state:
    st.session_state.history = []

@st.cache_data
def load_resources():
    # 載入 assets 資料夾中的孔子素材
    k_cap = cv2.imread("assets/kongzi_cap.png", cv2.IMREAD_UNCHANGED)
    k_beard = cv2.imread("assets/kongzi_beard.png", cv2.IMREAD_UNCHANGED)
    k_sleeves = cv2.imread("assets/kongzi_sleeves.png", cv2.IMREAD_UNCHANGED)
    return k_cap, k_beard, k_sleeves

kongzi_cap, kongzi_beard, kongzi_sleeves = load_resources()

def overlay_image(background, overlay, x, y, size=None):
    if overlay is None: return background
    bg_h, bg_w = background.shape[:2]
    if size is not None:
        overlay = cv2.resize(overlay, size, interpolation=cv2.INTER_AREA)
    h, w = overlay.shape[:2]
    if x >= bg_w or y >= bg_h or x + w <= 0 or y + h <= 0: return background
    x1, y1 = max(0, x), max(0, y)
    x2, y2 = min(bg_w, x + w), min(bg_h, y + h)
    overlay_x1, overlay_y1 = x1 - x, y1 - y
    overlay_x2, overlay_y2 = overlay_x1 + (x2 - x1), overlay_y1 + (y2 - y1)
    crop_overlay = overlay[overlay_y1:overlay_y2, overlay_x1:overlay_x2]
    crop_bg = background[y1:y2, x1:x2]
    
    if crop_overlay.shape[2] == 4:
        alpha = crop_overlay[:, :, 3] / 255.0
        alpha = np.expand_dims(alpha, axis=2)
        composite = crop_overlay[:, :, :3] * alpha + crop_bg * (1 - alpha)
        background[y1:y2, x1:x2] = composite.astype(np.uint8)
    else:
        background[y1:y2, x1:x2] = crop_overlay[:, :, :3]
    return background

mp_face_mesh = mp.solutions.face_mesh
mp_hands = mp.solutions.hands

class VideoProcessor:
    def __init__(self):
        self.face_mesh = mp_face_mesh.FaceMesh(
            max_num_faces=1, 
            refine_landmarks=True, 
            min_detection_confidence=0.5
        )
        self.hands = mp_hands.Hands(
            max_num_hands=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.latest_orig = None
        self.latest_filter = None
        self.is_kongzi_active = False

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        self.latest_orig = img.copy()
        
        h, w, _ = img.shape
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        face_results = self.face_mesh.process(rgb_img)
        hand_results = self.hands.process(rgb_img)
        
        # 🎯 關鍵：每格影格一開始就將孔子狀態歸零，只要手放下效果就會立刻消失！
        self.is_kongzi_active = False 
        
        # ─── 手勢辨識：比耶 ✌️ ───
        if hand_results.multi_hand_landmarks:
            hand_landmarks = hand_results.multi_hand_landmarks[0].landmark
            
            # 判斷每隻手指伸直或收起 (Y軸朝下，指尖 Y < 關節 Y 代表伸直)
            index_is_straight = hand_landmarks[8].y < hand_landmarks[6].y   # 食指伸直
            middle_is_straight = hand_landmarks[12].y < hand_landmarks[10].y # 中指伸直
            ring_is_closed = hand_landmarks[16].y > hand_landmarks[14].y     # 無名指收起
            pinky_is_closed = hand_landmarks[20].y > hand_landmarks[18].y     # 小指收起
            
            # 🎯 ✌️手勢演算法：只有食指和中指伸直，無名指和小指收起
            if index_is_straight and middle_is_straight and ring_is_closed and pinky_is_closed:
                self.is_kongzi_active = True

        # ─── 人臉特徵點與貼紙疊加 ───
        if face_results.multi_face_landmarks:
            face_landmarks = face_results.multi_face_landmarks[0].landmark
            
            forehead = face_landmarks[10]
            chin = face_landmarks[152]
            
            # 如果觸發孔子狀態，就 P 上帽子、鬍鬚、袖子
            if self.is_kongzi_active:
                left_face = face_landmarks[234]
                right_face = face_landmarks[454]
                face_width = abs(right_face.x - left_face.x) * w
                
                # A. 孔子帽子
                if kongzi_cap is not None:
                    cap_w = int(face_width * 2.2)
                    cap_scale = kongzi_cap.shape[0] / kongzi_cap.shape[1]
                    cap_h = int(cap_w * cap_scale)
                    cap_x = int(forehead.x * w - cap_w / 1.85)
                    cap_y = int(forehead.y * h - cap_h * 0.45)
                    img = overlay_image(img, kongzi_cap, cap_x, cap_y, size=(cap_w, cap_h))
                    
                # B. 孔子鬍鬚
                if kongzi_beard is not None:
                    beard_w = int(face_width * 1.2)
                    beard_scale = kongzi_beard.shape[0] / kongzi_beard.shape[1]
                    beard_h = int(beard_w * beard_scale)
                    beard_x = int(chin.x * w - beard_w / 2)
                    beard_y = int(chin.y * h - beard_h * 0.45)
                    img = overlay_image(img, kongzi_beard, beard_x, beard_y, size=(beard_w, beard_h))
                    
                # C. 古裝袖子在畫面最下方正中間
                if kongzi_sleeves is not None:
                    s_w = int(w * 0.8)
                    s_scale = kongzi_sleeves.shape[0] / kongzi_sleeves.shape[1]
                    s_h = int(s_w * s_scale)
                    s_x = int(w / 2 - s_w / 2)
                    s_y = h - s_h
                    img = overlay_image(img, kongzi_sleeves, s_x, s_y, size=(s_w, s_h))

        # 🎯 這裡已經拿掉 cv2.putText，畫面上絕對不會再出現黃色除錯英文字
        self.latest_filter = img.copy()
        return frame.from_ndarray(img, format="bgr24")

# --- 網頁畫面佈局 ---
ctx = webrtc_streamer(
    key="camera-step-by-step",
    video_processor_factory=VideoProcessor,
    media_stream_constraints={"video": True, "audio": False}
)

if st.button("📸 Capture (拍照)", use_container_width=True):
    if ctx.video_processor and ctx.video_processor.latest_orig is not None:
        orig_img = ctx.video_processor.latest_orig.copy()
        filter_img = ctx.video_processor.latest_filter.copy()
        
        orig_rgb = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)
        filter_rgb = cv2.cvtColor(filter_img, cv2.COLOR_BGR2RGB)
        
        st.session_state.history.insert(0, (orig_rgb, filter_rgb))
        if len(st.session_state.history) > 8:
            st.session_state.history.pop()
        st.success("拍照成功！已加到下方紀錄中。")
    else:
        st.warning("請先點擊 START 開啟鏡頭再拍照喔！")

st.markdown("---")

if st.session_state.history:
    st.subheader("🖼️ 剛剛拍到的影像")
    current_orig, current_filter = st.session_state.history[0]
    col_orig, col_filt = st.columns(2)
    with col_orig:
        st.image(current_orig, caption="拍到的原影像", use_container_width=True)
    with col_filt:
        st.image(current_filter, caption="濾鏡影像", use_container_width=True)

    st.markdown("---")
    st.subheader("📜 歷史拍照紀錄 (最多儲存 8 張)")
    
    cols = st.columns(4)
    for idx, (orig, filt) in enumerate(st.session_state.history):
        col_idx = idx % 4
        with cols[col_idx]:
            st.image(filt, use_container_width=True)
