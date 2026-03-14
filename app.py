import streamlit as st
import cv2
import numpy as np
from PIL import Image

st.set_page_config(page_title="Advanced Cartoonifier", layout="wide")
st.title("🎨 Advanced Image Cartoonification App")
st.write("Upload an image and convert it into different cartoon styles.")

# --- Upload Image ---
uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

# --- Style Selection ---
style = st.selectbox(
    "Choose Cartoon Style",
    ["Classic Cartoon", "Pencil Sketch", "Watercolor"]
)

# --- Sliders ---
edge_strength = st.slider("Edge Strength", 3, 15, 9, step=2)
blur_strength = st.slider("Blur Strength", 3, 15, 5, step=2)

# --- Edge Preview Checkbox ---
show_edges = st.checkbox("Show Edge Detection")

if uploaded_file is not None:

    # --- Convert uploaded image to OpenCV format ---
    image = Image.open(uploaded_file)
    img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    # --- Grayscale and Blurring ---
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.medianBlur(gray, blur_strength)

    # --- Edge Detection ---
    edges = cv2.adaptiveThreshold(
        blur,
        255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY,
        edge_strength,
        9
    )
    edges = edges.astype(np.uint8)  # ensure correct mask type
    edges = cv2.dilate(edges, np.ones((1, 1), np.uint8), iterations=1)  # optional enhancement

    # --- Cartoon Styles ---
    if style == "Classic Cartoon":
        color = cv2.bilateralFilter(img, 9, 250, 250)
        cartoon = cv2.bitwise_and(color, color, mask=edges)

    elif style == "Pencil Sketch":
        cartoon = cv2.Canny(gray, 100, 200)

    else:  # Watercolor
        color = cv2.bilateralFilter(img, 15, 300, 300)
        cartoon = cv2.bitwise_and(color, color, mask=edges)

    # --- Display Images Side by Side ---
    col1, col2 = st.columns(2)

    display_width = 400  # Set a fixed width for both images

    with col1:
        st.subheader("Original Image")
        st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), width=display_width)

    with col2:
        st.subheader("Cartoon Image")
        if style == "Pencil Sketch":
            display_img = cv2.cvtColor(cartoon, cv2.COLOR_GRAY2RGB)
        else:
            display_img = cv2.cvtColor(cartoon, cv2.COLOR_BGR2RGB)
        st.image(display_img, width=display_width)

    # --- Show Edges if Checked ---
    if show_edges:
        st.subheader("Edge Detection")
        st.image(edges, width=display_width, clamp=True)

    # --- Download Cartoon Image ---
    _, buffer = cv2.imencode(".png", display_img)
    st.download_button(
        label="Download Cartoon Image",
        data=buffer.tobytes(),
        file_name="cartoon_image.png",
        mime="image/png"
    )

    # --- Image Information ---
    st.subheader("Image Information")
    st.write("Height:", img.shape[0])
    st.write("Width:", img.shape[1])
    st.write("Channels:", img.shape[2])