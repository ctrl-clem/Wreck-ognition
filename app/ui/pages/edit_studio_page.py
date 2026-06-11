import streamlit as st
from PIL import Image
import io
from app.ui.utils.visualization_functions import add_realistic_smoke
import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF

st.title("Adding noise Studio")

st.write("Upload picture:")
uploaded_file = st.file_uploader("Choose a picture", type=["png", "jpg", "jpeg"], label_visibility="collapsed")

if uploaded_file is not None:
    st.divider()

    original_image = Image.open(uploaded_file).convert("RGB")

    img_col, slider_col = st.columns([0.8, 1.0])

    with slider_col:
        st.markdown("### Noise Controls")

        smoke_opacity = st.slider("Smoke Opacity", min_value=0.0, max_value=1.0, value=0.0, step=0.05)

        st.markdown("### Color Jitter")
        st.caption("Adjust the random variance range for each property.")
        jitter_brightness = st.slider("Brightness Variance", min_value=0.0, max_value=1.0, value=0.0, step=0.05)
        jitter_contrast = st.slider("Contrast Variance", min_value=0.0, max_value=1.0, value=0.0, step=0.05)
        jitter_saturation = st.slider("Saturation Variance", min_value=0.0, max_value=1.0, value=0.0, step=0.05)

    with img_col:
        torch.manual_seed(42)
        img_tensor = TF.to_tensor(original_image)

        if smoke_opacity > 0:
            img_tensor = add_realistic_smoke(img_tensor, max_opacity=smoke_opacity)

        if jitter_brightness > 0 or jitter_contrast > 0 or jitter_saturation > 0:
            jitter = transforms.ColorJitter(
                brightness=jitter_brightness,
                contrast=jitter_contrast,
                saturation=jitter_saturation,
                hue=0.0
            )
            img_tensor = jitter(img_tensor)

        img_tensor = torch.clamp(img_tensor, 0.0, 1.0)
        final_image = TF.to_pil_image(img_tensor)

        st.image(final_image, caption="Modified Picture", use_container_width=True)

        buf = io.BytesIO()
        final_image.save(buf, format="PNG")
        img_bytes = buf.getvalue()

        st.download_button(
            label="Download Picture",
            data=img_bytes,
            file_name="noisy_picture.png",
            mime="image/png",
            use_container_width=True
        )