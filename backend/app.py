import base64
import io

import requests
import streamlit as st

# URL of your FastAPI backend
BACKEND_URL = "http://localhost:8000"


def main():
    st.set_page_config(
        page_title="AI Creative Studio",
        page_icon="🎨",
        layout="wide",
    )

    st.title("🎨 AI Creative Studio — Hackathon Edition (H-003)")
    st.markdown(
        "Upload a brand logo + product image and generate multiple ad creatives.\n\n"
        "Images are generated via **Stable Diffusion (Hugging Face)** and captions via **Gemini LLM**, "
        "wrapped behind a clean FastAPI backend."
    )

    with st.sidebar:
        st.header("Configuration")
        brand_name = st.text_input("Brand Name", value="Demo Brand")
        product_desc = st.text_area(
            "Product Description",
            value="Premium cold brew coffee in a sleek bottle, energizing and refreshing.",
            height=80,
        )
        num_creatives = st.slider("Number of creatives", 4, 12, 6, 1)

        st.markdown("---")
        if st.button("Check Backend Health"):
            try:
                resp = requests.get(f"{BACKEND_URL}/health", timeout=10)
                st.write("Backend health:", resp.json())
            except Exception as e:
                st.error(f"Backend health check failed: {e}")

    col1, col2 = st.columns(2)
    with col1:
        logo_file = st.file_uploader(
            "Upload Brand Logo", type=["png", "jpg", "jpeg"], accept_multiple_files=False
        )
    with col2:
        product_file = st.file_uploader(
            "Upload Product Image (for context only)",
            type=["png", "jpg", "jpeg"],
            accept_multiple_files=False,
        )

    generate_btn = st.button("Generate Ad Creatives 🚀", type="primary")

    if generate_btn:
        if not logo_file or not product_file:
            st.error("Please upload both a brand logo and a product image.")
            return

        try:
            with st.spinner("Calling backend to generate creatives..."):
                # Prepare multipart form-data
                files = {
                    "logo": (logo_file.name, logo_file.getvalue(), logo_file.type),
                    "product": (
                        product_file.name,
                        product_file.getvalue(),
                        product_file.type,
                    ),
                }
                data = {
                    "brand_name": brand_name,
                    "product_desc": product_desc,
                    "num_creatives": str(num_creatives),
                }

                resp = requests.post(
                    f"{BACKEND_URL}/api/generate-creatives",
                    data=data,
                    files=files,
                    timeout=600,
                )
                resp.raise_for_status()
                payload = resp.json()

            st.success("Creatives generated successfully!")

            colors = payload.get("colors", [])
            images = payload.get("images", [])
            captions = payload.get("captions", [])
            zip_b64 = payload.get("zip_base64", "")

            # Show brand color swatches
            if colors:
                st.subheader("Detected Brand Colors")
                color_cols = st.columns(len(colors))
                for c, col in zip(colors, color_cols):
                    with col:
                        st.markdown(
                            f"""
                            <div style="background:{c};width:100%;height:40px;
                                        border-radius:8px;border:1px solid #111;"></div>
                            <p style="font-size:0.8rem;margin-top:4px;">{c}</p>
                            """,
                            unsafe_allow_html=True,
                        )

            # Show preview grid
            st.subheader("Preview Creatives")
            grid_cols = st.columns(3)
            for idx, img_data_url in enumerate(images):
                col = grid_cols[idx % 3]
                with col:
                    # img_data_url is 'data:image/png;base64,...'
                    if img_data_url.startswith("data:image"):
                        header, b64 = img_data_url.split(",", 1)
                    else:
                        b64 = img_data_url
                    img_bytes = base64.b64decode(b64)
                    st.image(img_bytes, use_column_width=True)
                    cap = captions[idx] if idx < len(captions) else ""
                    st.caption(f"**Caption:** {cap}")

            # Download ZIP
            if zip_b64:
                zip_bytes = base64.b64decode(zip_b64)
                st.download_button(
                    label="⬇️ Download ZIP (images + captions.csv)",
                    data=zip_bytes,
                    file_name="ai_creatives_bundle.zip",
                    mime="application/zip",
                )

        except Exception as e:
            st.error(f"Something went wrong: {e}")
            st.exception(e)


if __name__ == "__main__":
    main()
