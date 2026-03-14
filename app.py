import streamlit as st
import cv2
import numpy as np
from PIL import Image

st.title("🖼️ Background Remover & Blur App")
st.write("Upload an image, remove its background, or apply artistic blur effects.")

# Upload image
uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

# Options
blur_strength = st.slider("Blur Strength", 5, 35, 15, step=2)
show_mask = st.checkbox("Show Background Mask")

if uploaded_file is not None:

    # Convert uploaded image to array
    image = Image.open(uploaded_file)
    img = np.array(image)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    # Resize for faster processing
    img_small = cv2.resize(img, (0,0), fx=0.5, fy=0.5)

    # Background removal using simple color segmentation
    # Convert to HSV for better color separation
    hsv = cv2.cvtColor(img_small, cv2.COLOR_BGR2HSV)
    
    # Assuming background is mostly light (like white wall)
    lower_bg = np.array([0,0,180])
    upper_bg = np.array([180,50,255])
    mask = cv2.inRange(hsv, lower_bg, upper_bg)
    
    # Refine mask
    mask = cv2.resize(mask, (img.shape[1], img.shape[0]))
    mask_inv = cv2.bitwise_not(mask)

    # Extract foreground
    fg = cv2.bitwise_and(img, img, mask=mask_inv)

    # Blur the background
    bg = cv2.bitwise_and(img, img, mask=mask)
    bg_blur = cv2.GaussianBlur(bg, (blur_strength, blur_strength), 0)

    # Combine foreground with blurred background
    final_img = cv2.add(fg, bg_blur)

    # Display side-by-side
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Original Image")
        st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), width=300)
    with col2:
        st.subheader("Processed Image")
        st.image(cv2.cvtColor(final_img, cv2.COLOR_BGR2RGB), width=300)

    # Show mask if selected
    if show_mask:
        st.subheader("Background Mask")
        st.image(mask, width=300)

    # Download button
    st.download_button(
        label="Download Processed Image",
        data=cv2.imencode(".png", final_img)[1].tobytes(),
        file_name="background_removed.png",
        mime="image/png"
    )

    # Image info
    st.subheader("Image Information")
    st.write("Height:", img.shape[0])
    st.write("Width:", img.shape[1])
    st.write("Channels:", img.shape[2])