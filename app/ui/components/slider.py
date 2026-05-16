import base64
import io
import streamlit.components.v1 as components
from PIL import Image
import numpy as np

def _image_to_base64(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def mask_slider(
        post_image: str,
        mask_overlay: Image.Image,
        height: int = 512,
        key: str = "mask_slider"
) -> None:
    post_image = Image.open(post_image).convert("RGB")
    post_b64 = _image_to_base64(post_image)
    mask_b64 = _image_to_base64(mask_overlay)

    html = f"""
    <div style="position:relative; width:100%; height:{height}px;
                overflow:hidden; cursor:col-resize; user-select:none;
                touch-action: none;"
         id="{key}_container">

      <img src="data:image/png;base64,{post_b64}"
           draggable="false"
           style="position:absolute; top:0; left:0;
                  width:100%; height:100%; object-fit:contain;
                  pointer-events:none;"/>

      <img id="{key}_mask"
           src="data:image/png;base64,{mask_b64}"
           draggable="false"
           style="position:absolute; top:0; left:0;
                  width:100%; height:100%; object-fit:contain;
                  opacity:0.80; clip-path: inset(0 50% 0 0);
                  pointer-events:none;"/>

      <div id="{key}_handle"
           style="position:absolute; top:0; left:50%;
                  width:3px; height:100%;
                  background:white; box-shadow:0 0 6px rgba(0,0,0,0.6);
                  transform:translateX(-50%); pointer-events:none;">
        <div style="position:absolute; top:50%; left:50%;
                    transform:translate(-50%,-50%);
                    background:white; border-radius:50%;
                    width:32px; height:32px;
                    box-shadow:0 0 6px rgba(0,0,0,0.4);
                    display:flex; align-items:center; justify-content:center;
                    font-size:14px;">⇔</div>
      </div>
    </div>

    <script>
      (function() {{
        const container = document.getElementById("{key}_container");
        const mask      = document.getElementById("{key}_mask");
        const handle    = document.getElementById("{key}_handle");

        function setPosition(clientX) {{
          const rect  = container.getBoundingClientRect();
          const pct   = Math.min(Math.max((clientX - rect.left) / rect.width, 0), 1);
          mask.style.clipPath = `inset(0 ${{(1 - pct) * 100}}% 0 0)`;
          handle.style.left   = (pct * 100) + "%";
        }}

        container.addEventListener("pointerdown", e => {{
          e.preventDefault();                       
          container.setPointerCapture(e.pointerId);
          setPosition(e.clientX);
        }});

        container.addEventListener("pointermove", e => {{
          if (container.hasPointerCapture(e.pointerId)) setPosition(e.clientX);
        }});

        container.addEventListener("pointerup",     e => container.releasePointerCapture(e.pointerId));
        container.addEventListener("pointercancel", e => container.releasePointerCapture(e.pointerId));
      }})();
    </script>
    """
    components.html(html, height=height + 10)