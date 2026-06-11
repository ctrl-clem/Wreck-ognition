import streamlit as st
import io

GRADIENTS = [
    "linear-gradient(135deg, #667EEA 0%, #764BA2 100%)",  # Purple
    "linear-gradient(135deg, #FF0844 0%, #FFB199 100%)",  # Red
    "linear-gradient(135deg, #4FACFE 0%, #00F2FE 100%)",  # Blue
    "linear-gradient(135deg, #43E97B 0%, #38F9D7 100%)",  # Green
    "linear-gradient(135deg, #0BA360 0%, #3CB0FD 100%)",  # Teal/Blue
    "linear-gradient(135deg, #FA709A 0%, #FEE140 100%)",  # Pink/Orange
]
def format_title(text: str) -> str:
    return text.replace("-", " ").title()


def update_selection(disaster_name: str):
    st.session_state.selected_disaster = disaster_name

def convert_image_to_bytes(img):
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def render_disaster_groups(grouped: dict[str, list[str]]) -> str | None:
    if 'selected_disaster' not in st.session_state:
        st.session_state.selected_disaster = None

    selected = st.session_state.selected_disaster

    st.write("# Disasters by type")

    cols = st.columns(3)

    max_disasters = max(len(d) for d in grouped.values()) if grouped else 0

    card_height = 120 + (max_disasters * 45)

    for i, (disaster_type, disasters) in enumerate(grouped.items()):
        gradient = GRADIENTS[i % len(GRADIENTS)]

        with cols[i % 3]:
            with st.container(border=True, height=card_height):
                pretty_type = format_title(disaster_type)

                st.markdown(
                    f"""
                    <div style="background: {gradient}; padding: 12px; border-radius: 6px; 
                                margin-bottom: 15px; color: white; text-shadow: 1px 1px 2px rgba(0,0,0,0.2);">
                        <div style="font-weight: 600; font-size: 16px;">{pretty_type}</div>
                        <div style="font-size: 12px; opacity: 0.9;">
                            {len(disasters)} Disaster{'s' if len(disasters) != 1 else ''}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                for disaster in disasters:
                    is_active = disaster == selected
                    pretty_disaster = format_title(disaster)
                    label = f"●\u00A0\u00A0{pretty_disaster}" if is_active else f"○\u00A0\u00A0{pretty_disaster}"

                    st.button(
                        label,
                        key=f"btn_{disaster}",
                        use_container_width=True,
                        type="primary" if is_active else "secondary",
                        on_click=update_selection,
                        args=(disaster,)
                    )

    return st.session_state.selected_disaster