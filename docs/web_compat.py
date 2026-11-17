import base64
import html

import streamlit as st


def download_button(label, data, file_name, mime=None, **kwargs):
    """Render a download link that works inside the browser-only Stlite runtime."""
    if isinstance(data, str):
        data = data.encode("utf-8")

    encoded = base64.b64encode(data).decode("ascii")
    href = f"data:{mime or 'application/octet-stream'};base64,{encoded}"
    safe_name = html.escape(file_name, quote=True)
    safe_label = html.escape(label)

    st.markdown(
        (
            f'<a href="{href}" download="{safe_name}" '
            'style="display:block;width:100%;box-sizing:border-box;padding:0.55rem 0.75rem;'
            'border:1px solid #d1d5db;border-radius:0.5rem;text-align:center;'
            'text-decoration:none;color:#172033;background:#ffffff;">'
            f"{safe_label}</a>"
        ),
        unsafe_allow_html=True,
    )
