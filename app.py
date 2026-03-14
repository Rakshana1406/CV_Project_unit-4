import streamlit as st
import cv2
import numpy as np
from PIL import Image

st.title("🎨 Advanced Image Cartoonification App")

st.write("Upload an image and convert it into different cartoon styles.")

# Upload image
uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

# Style selection
style = st.selectbox(
    "Choose Cartoon Style",
    ["Classic Cartoon", "Pencil Sketch", "Watercolor"]
)

# Sliders (step=2 ensures only odd values)
edge_strength = st.slider("Edge Strength", 3, 15, 9, step=2)
blur_strength = st.slider("Blur Strength", 3, 15, 5, step=2)

# Edge preview
show_edges = st.checkbox("Show Edge Detection")

if uploaded_file is not None:

    # Convert uploaded image to array
    image = Image.open(uploaded_file)
    img = np.array(image)

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply median blur
    blur = cv2.medianBlur(gray, blur_strength)

    # Edge detection
    edges = cv2.adaptiveThreshold(
        blur,
        255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY,
        edge_strength,
        9
    )

    # Cartoon Styles
    if style == "Classic Cartoon":

        color = cv2.bilateralFilter(img, 9, 250, 250)
        cartoon = cv2.bitwise_and(color, color, mask=edges)

    elif style == "Pencil Sketch":

        cartoon = cv2.Canny(gray, 100, 200)

    else:  # Watercolor

        color = cv2.bilateralFilter(img, 15, 300, 300)
        cartoon = cv2.bitwise_and(color, color, mask=edges)

    # Display side-by-side images
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Original Image")
        st.image(img, width="stretch")

    with col2:
        st.subheader("Cartoon Image")
        st.image(cartoon, width="stretch")

    # Show edge detection if selected
    if show_edges:
        st.subheader("Edge Detection")
        st.image(edges, width="stretch")

    # Download button
    st.download_button(
        label="Download Cartoon Image",
        data=cv2.imencode(".png", cartoon)[1].tobytes(),
        file_name="cartoon_image.png",
        mime="image/png"
    )

    # Image info
    st.subheader("Image Information")
    st.write("Height:", img.shape[0])
    st.write("Width:", img.shape[1])
    st.write("Channels:", img.shape[2])