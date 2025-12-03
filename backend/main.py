# import os
# import io
# import zipfile
# import base64
# import requests
# from typing import List, Tuple

# from fastapi import FastAPI, UploadFile, File, Form
# from fastapi.middleware.cors import CORSMiddleware
# from pydantic import BaseModel
# from PIL import Image
# import numpy as np

# # --- CONFIG / ENV ---

# GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
# HF_API_TOKEN      = os.getenv("HF_API_TOKEN")  # For Stable Diffusion call
# HF_IMAGE_MODEL    = os.getenv("HF_IMAGE_MODEL", "stabilityai/stable-diffusion-2-1")

# HF_IMAGE_URL = f"https://api-inference.huggingface.co/models/{HF_IMAGE_MODEL}"
# HEADERS_HF   = {"Authorization": f"Bearer {HF_API_TOKEN}"}

# GEMINI_URL   = "https://gemini.googleapis.com/v1/models/gemini-2.5-flash:generateContent"  # adjust if variant else
# HEADERS_GEMINI = {"Authorization": f"Bearer {GEMINI_API_KEY}", "Content-Type": "application/json"}


# app = FastAPI()
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["http://localhost:5173", "http://localhost:3000", "*"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# @app.get("/")
# def root():
#     return {"message": "AI Creative Studio backend is running"}

# @app.get("/health")
# def health():
#     return {"status": "ok"}
# # --- UTILS ---

# def load_image_from_bytes(data: bytes) -> Image.Image:
#     return Image.open(io.BytesIO(data)).convert("RGB")

# def extract_dominant_colors(img: Image.Image, num_colors: int = 3) -> List[str]:
#     small = img.resize((64, 64))
#     arr = np.array(small).reshape(-1, 3)
#     colors, counts = np.unique(arr, axis=0, return_counts=True)
#     idx = np.argsort(counts)[::-1]
#     top = colors[idx][:num_colors]
#     return [f"#{r:02x}{g:02x}{b:02x}" for r, g, b in top]

# def build_image_prompt(brand_name: str, product_desc: str, colors: List[str]) -> str:
#     color_str = ", ".join(colors) if colors else "clean modern brand colors"
#     return (
#         f"High-quality advertising creative for the brand '{brand_name}' showcasing the product: {product_desc}. "
#         f"Use a clean, modern, social-media-ready layout with visual focus on the product and brand-inspired colors: {color_str}. "
#         "Do NOT embed any text in the image."
#     )

# def call_stable_diffusion(prompt: str) -> bytes:
#     payload = {"inputs": prompt}
#     resp = requests.post(HF_IMAGE_URL, headers=HEADERS_HF, json=payload, timeout=120)
#     resp.raise_for_status()
#     return resp.content  # raw image bytes (PNG/JPEG) depending on model

# def call_gemini_caption(brand_name: str, product_desc: str) -> str:
#     prompt = (
#         "You are an ad copywriter. Generate a short, catchy social-media ad caption (max 12 words) "
#         f"for a product with brand '{brand_name}', product description: {product_desc}.\n\nCaption:"
#     )
#     body = {"prompt": prompt, "max_output_tokens": 30}
#     resp = requests.post(GEMINI_URL, headers=HEADERS_GEMINI, json=body, timeout=30)
#     resp.raise_for_status()
#     data = resp.json()
#     # Adjust this depending on actual Gemini JSON response shape
#     text = data.get("content", "") or data.get("text", "")
#     return text.strip()

# def build_zip(imgs_and_caps: List[Tuple[bytes, str]]) -> bytes:
#     mem = io.BytesIO()
#     with zipfile.ZipFile(mem, "w", zipfile.ZIP_DEFLATED) as zf:
#         csv = ["filename,caption"]
#         for i, (img_b, cap) in enumerate(imgs_and_caps, start=1):
#             fname = f"creative_{i:02d}.png"
#             zf.writestr(fname, img_b)
#             safe = '"' + cap.replace('"', "'") + '"'
#             csv.append(f"{fname},{safe}")
#         zf.writestr("captions.csv", "\n".join(csv))
#     mem.seek(0)
#     return mem.read()

# # --- API MODEL ---

# class GenerateResponse(BaseModel):
#     images: List[str]
#     captions: List[str]
#     zip_base64: str

# # --- ROUTE ---

# @app.post("/api/generate-creatives", response_model=GenerateResponse)
# async def generate_creatives(
#     brand_name: str = Form(...),
#     product_desc: str = Form(...),
#     num_creatives: int = Form(6),
#     logo: UploadFile = File(...),
#     product: UploadFile = File(...),
# ):
#     logo_bytes = await logo.read()
#     logo_img = load_image_from_bytes(logo_bytes)
#     colors = extract_dominant_colors(logo_img, num_colors=3)

#     results: List[Tuple[bytes, str]] = []
#     data_urls: List[str] = []
#     captions: List[str] = []

#     for idx in range(num_creatives):
#         prompt = build_image_prompt(brand_name, product_desc, colors)
#         img_bytes = call_stable_diffusion(prompt)
#         caption = call_gemini_caption(brand_name, product_desc)

#         results.append((img_bytes, caption))
#         captions.append(caption)

#         b64 = base64.b64encode(img_bytes).decode("utf-8")
#         data_urls.append(f"data:image/png;base64,{b64}")

#     zip_bytes = build_zip(results)
#     zip_b64 = base64.b64encode(zip_bytes).decode("utf-8")

#     return GenerateResponse(images=data_urls, captions=captions, zip_base64=zip_b64)



# import os
# import io
# import zipfile
# import base64
# from typing import List, Tuple

# import numpy as np
# import requests
# from PIL import Image
# from fastapi import FastAPI, UploadFile, File, Form
# from fastapi.middleware.cors import CORSMiddleware
# from pydantic import BaseModel
# from dotenv import load_dotenv

# # ---------- ENV & CONFIG ----------

# load_dotenv()

# GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
# HF_API_TOKEN = os.getenv("HF_API_TOKEN")
# HF_IMAGE_MODEL = os.getenv("HF_IMAGE_MODEL", "stabilityai/stable-diffusion-3.5-large")

# # HF_IMAGE_URL = f"https://api-inference.huggingface.co/models/{HF_IMAGE_MODEL}"
# # HF_HEADERS = {"Authorization": f"Bearer {HF_API_TOKEN}"}
# # HF_IMAGE_URL = f"https://router.huggingface.co/hf-inference/models/{HF_IMAGE_MODEL}"
# # HF_HEADERS = {"Authorization": f"Bearer {HF_API_TOKEN}"}
# #HF_IMAGE_URL = f"https://api-inference.huggingface.co/models/{HF_IMAGE_MODEL}"
# HF_IMAGE_URL = f"https://router.huggingface.co/models/{HF_IMAGE_MODEL}"

# HF_HEADERS = {
#     "Authorization": f"Bearer {HF_API_TOKEN}",
#     "Content-Type": "application/json"
# }

# GEMINI_URL = (
#     "https://generativelanguage.googleapis.com/v1beta/models/"
#     "gemini-2.5-flash:generateContent"
# )

# if not HF_API_TOKEN:
#     print("⚠️ HF_API_TOKEN is not set – image generation will fail.")
# if not GEMINI_API_KEY:
#     print("⚠️ GEMINI_API_KEY is not set – captions will be dummy text.")


# # ---------- FASTAPI APP ----------

# app = FastAPI(
#     title="AI Creative Studio Backend",
#     description="Generates branded ad creatives using Stable Diffusion + Gemini",
#     version="1.0.0",
# )

# @app.get("/")
# def root():
#     return {"message": "AI Creative Studio backend is running"}

# @app.get("/health")
# def health():
#     return {"status": "ok"}

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=[
#         "http://localhost:8501",  # Streamlit
#         "http://127.0.0.1:8501",
#         "*",                      # relax for hackathon
#     ],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )


# # ---------- UTILS ----------

# def load_image_from_bytes(data: bytes) -> Image.Image:
#     return Image.open(io.BytesIO(data)).convert("RGB")


# def extract_dominant_colors(img: Image.Image, num_colors: int = 3) -> List[str]:
#     small = img.resize((64, 64))
#     arr = np.array(small).reshape(-1, 3)
#     colors, counts = np.unique(arr, axis=0, return_counts=True)
#     idx = np.argsort(counts)[::-1]
#     top = colors[idx][:num_colors]
#     return [f"#{r:02x}{g:02x}{b:02x}" for r, g, b in top]


# def build_image_prompt(brand_name: str, product_desc: str, colors: List[str]) -> str:
#     color_str = ", ".join(colors) if colors else "clean modern brand colors"
#     return (
#         f"High-quality advertising creative for the brand '{brand_name}' showcasing the product: {product_desc}. "
#         f"Use a modern, minimal, social-media-ready layout with strong focus on the product. "
#         f"Use brand-inspired colors: {color_str}. "
#         f"Do NOT embed any text or words in the image."
#     )


# # def generate_image_bytes(prompt: str) -> bytes:
# #     if not HF_API_TOKEN:
# #         raise RuntimeError("HF_API_TOKEN not set in environment.")
# #     payload = {"inputs": prompt, "options": {"wait_for_model": True}}
# #     resp = requests.post(HF_IMAGE_URL, headers=HF_HEADERS, json=payload, timeout=180)
# #     if resp.status_code != 200:
# #         raise RuntimeError(
# #             f"Image generation failed ({resp.status_code}): {resp.text[:300]}"
# #         )
# #     return resp.content
# def generate_image_bytes(prompt: str) -> bytes:
#     if not HF_API_TOKEN:
#         raise RuntimeError("HF_API_TOKEN not set in environment.")
    
#     # SD 3.5 requires this specific payload structure
#     payload = {
#         "inputs": prompt,
#     }
    
#     resp = requests.post(
#         HF_IMAGE_URL,
#         headers=HF_HEADERS,
#         json=payload,
#         timeout=180
#     )
    
#     if resp.status_code != 200:
#         raise RuntimeError(
#             f"Image generation failed ({resp.status_code}): {resp.text[:300]}"
#         )
#     return resp.content


# def generate_caption(brand_name: str, product_desc: str) -> str:
#     if not GEMINI_API_KEY:
#         # fallback so API still works in demo mode
#         return "Sample caption (set GEMINI_API_KEY for real captions)"

#     caption_prompt = (
#         "You are an ad copywriter. "
#         "Write ONE short, catchy ad caption (max 12 words) "
#         "for a social media image creative.\n\n"
#         f"Brand: {brand_name}\n"
#         f"Product: {product_desc}\n\n"
#         "Caption:"
#     )

#     body = {
#         "contents": [
#             {
#                 "parts": [
#                     {"text": caption_prompt}
#                 ]
#             }
#         ]
#     }

#     headers = {
#         "Content-Type": "application/json",
#         "x-goog-api-key": GEMINI_API_KEY,
#     }

#     resp = requests.post(GEMINI_URL, headers=headers, json=body, timeout=30)
#     if resp.status_code != 200:
#         raise RuntimeError(
#             f"Gemini caption failed ({resp.status_code}): {resp.text[:300]}"
#         )

#     data = resp.json()
#     try:
#         text = data["candidates"][0]["content"]["parts"][0]["text"]
#     except Exception:
#         text = str(data)

#     caption = text.strip().replace("\n", " ")
#     if caption.startswith('"') and caption.endswith('"'):
#         caption = caption[1:-1].strip()
#     return caption


# def build_zip(images_with_captions: List[Tuple[bytes, str]]) -> bytes:
#     mem = io.BytesIO()
#     with zipfile.ZipFile(mem, "w", zipfile.ZIP_DEFLATED) as zf:
#         csv_lines = ["filename,caption"]
#         for i, (img_bytes, cap) in enumerate(images_with_captions, start=1):
#             fname = f"creative_{i:02d}.png"
#             zf.writestr(fname, img_bytes)
#             safe = '"' + cap.replace('"', "'") + '"'
#             csv_lines.append(f"{fname},{safe}")
#         zf.writestr("captions.csv", "\n".join(csv_lines))
#     mem.seek(0)
#     return mem.read()


# # ---------- RESPONSE MODEL ----------

# class GenerateResponse(BaseModel):
#     images: List[str]      # data:image/png;base64,...
#     captions: List[str]
#     zip_base64: str        # base64 ZIP
#     colors: List[str]      # hex colors extracted from logo


# # ---------- ROUTES ----------

# @app.get("/")
# def root():
#     return {"message": "AI Creative Studio backend is running"}

# @app.get("/health")
# def health():
#     return {"status": "ok"}

# @app.post("/api/generate-creatives", response_model=GenerateResponse)
# async def generate_creatives(
#     brand_name: str = Form(...),
#     product_desc: str = Form(...),
#     num_creatives: int = Form(6),
#     logo: UploadFile = File(...),
#     product: UploadFile = File(...),  # currently not used but included for spec
# ):
#     logo_bytes = await logo.read()
#     # product_bytes = await product.read()  # available for future use

#     logo_img = load_image_from_bytes(logo_bytes)
#     colors = extract_dominant_colors(logo_img, num_colors=3)

#     images_with_captions: List[Tuple[bytes, str]] = []
#     data_urls: List[str] = []
#     captions: List[str] = []

#     for _ in range(num_creatives):
#         prompt = build_image_prompt(brand_name, product_desc, colors)
#         img_bytes = generate_image_bytes(prompt)
#         caption = generate_caption(brand_name, product_desc)

#         images_with_captions.append((img_bytes, caption))
#         captions.append(caption)

#         b64_img = base64.b64encode(img_bytes).decode("utf-8")
#         data_urls.append(f"data:image/png;base64,{b64_img}")

#     zip_bytes = build_zip(images_with_captions)
#     zip_b64 = base64.b64encode(zip_bytes).decode("utf-8")

#     return GenerateResponse(
#         images=data_urls,
#         captions=captions,
#         zip_base64=zip_b64,
#         colors=colors,
#     )


# import os
# import io
# import zipfile
# import base64
# import time
# from typing import List, Tuple
# from functools import lru_cache

# import numpy as np
# import requests
# from PIL import Image
# from fastapi import FastAPI, UploadFile, File, Form, HTTPException
# from fastapi.middleware.cors import CORSMiddleware
# from pydantic import BaseModel
# from dotenv import load_dotenv
# from requests.adapters import HTTPAdapter
# from urllib3.util.retry import Retry

# # ---------- ENV & CONFIG ----------

# load_dotenv()

# GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
# HF_API_TOKEN = os.getenv("HF_API_TOKEN")
# HF_IMAGE_MODEL = os.getenv("HF_IMAGE_MODEL", "stabilityai/stable-diffusion-3.5-large")

# # CORRECTED: Use proper Inference API endpoint
# HF_IMAGE_URL = f"https://api-inference.huggingface.co/models/{HF_IMAGE_MODEL}"

# HF_HEADERS = {
#     "Authorization": f"Bearer {HF_API_TOKEN}",
# }

# GEMINI_URL = (
#     "https://generativelanguage.googleapis.com/v1beta/models/"
#     "gemini-2.5-flash:generateContent"
# )

# # Validation
# if not HF_API_TOKEN:
#     print("⚠️ WARNING: HF_API_TOKEN is not set – image generation will fail.")
# if not GEMINI_API_KEY:
#     print("⚠️ WARNING: GEMINI_API_KEY is not set – captions will be dummy text.")


# # ---------- FASTAPI APP ----------

# app = FastAPI(
#     title="AI Creative Studio Backend",
#     description="Generates branded ad creatives using Stable Diffusion + Gemini",
#     version="1.0.1",
# )

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=[
#         "http://localhost:8501",
#         "http://127.0.0.1:8501",
#         "*",  # For development only - restrict in production
#     ],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )


# # ---------- HELPER: RETRY SESSION ----------

# def get_retry_session():
#     """Create requests session with retry logic for API calls."""
#     session = requests.Session()
#     retry = Retry(
#         total=3,
#         backoff_factor=1,
#         status_forcelist=[429, 500, 502, 503, 504],
#         allowed_methods=["GET", "POST"]
#     )
#     adapter = HTTPAdapter(max_retries=retry)
#     session.mount('http://', adapter)
#     session.mount('https://', adapter)
#     return session


# # ---------- UTILS ----------

# def load_image_from_bytes(data: bytes) -> Image.Image:
#     """Load PIL Image from bytes."""
#     try:
#         return Image.open(io.BytesIO(data)).convert("RGB")
#     except Exception as e:
#         raise HTTPException(status_code=400, detail=f"Invalid image format: {str(e)}")


# def extract_dominant_colors(img: Image.Image, num_colors: int = 3) -> List[str]:
#     """Extract dominant colors from logo for brand consistency."""
#     try:
#         small = img.resize((64, 64))
#         arr = np.array(small).reshape(-1, 3)
#         colors, counts = np.unique(arr, axis=0, return_counts=True)
#         idx = np.argsort(counts)[::-1]
#         top = colors[idx][:num_colors]
#         return [f"#{r:02x}{g:02x}{b:02x}" for r, g, b in top]
#     except Exception as e:
#         print(f"Color extraction failed: {e}")
#         return ["#000000", "#FFFFFF", "#888888"]  # Fallback colors


# def build_image_prompt(brand_name: str, product_desc: str, colors: List[str]) -> str:
#     """Build optimized prompt for Stable Diffusion."""
#     color_str = ", ".join(colors) if colors else "clean modern brand colors"
#     return (
#         f"High-quality professional advertising creative, product photography style. "
#         f"Brand: {brand_name}. Product: {product_desc}. "
#         f"Modern minimal composition with clean background. "
#         f"Color palette: {color_str}. "
#         f"Professional lighting, studio quality, commercial photography. "
#         f"No text, no watermarks, no logos in image."
#     )


# def generate_image_bytes(prompt: str, retry_count: int = 0) -> bytes:
#     """Generate image using Hugging Face Stable Diffusion API."""
#     if not HF_API_TOKEN:
#         raise HTTPException(
#             status_code=500,
#             detail="HF_API_TOKEN not configured. Please set environment variable."
#         )
    
#     payload = {
#         "inputs": prompt,
#         "options": {
#             "wait_for_model": True,
#             "use_cache": False
#         }
#     }
    
#     session = get_retry_session()
    
#     try:
#         resp = session.post(
#             HF_IMAGE_URL,
#             headers=HF_HEADERS,
#             json=payload,
#             timeout=180
#         )
        
#         # Handle specific error cases
#         if resp.status_code == 503:
#             # Model is loading - wait and retry
#             if retry_count < 3:
#                 wait_time = 10 * (retry_count + 1)
#                 print(f"Model loading, waiting {wait_time}s before retry...")
#                 time.sleep(wait_time)
#                 return generate_image_bytes(prompt, retry_count + 1)
#             else:
#                 raise HTTPException(
#                     status_code=503,
#                     detail="Model is still loading. Please try again in a few minutes."
#                 )
        
#         elif resp.status_code == 429:
#             raise HTTPException(
#                 status_code=429,
#                 detail="Rate limit exceeded. Please wait a moment and try again."
#             )
        
#         elif resp.status_code == 401:
#             raise HTTPException(
#                 status_code=500,
#                 detail="Invalid Hugging Face API token. Please check your credentials."
#             )
        
#         elif resp.status_code != 200:
#             error_detail = resp.text[:500]
#             raise HTTPException(
#                 status_code=resp.status_code,
#                 detail=f"Image generation failed: {error_detail}"
#             )
        
#         return resp.content
        
#     except requests.Timeout:
#         raise HTTPException(
#             status_code=504,
#             detail="Image generation timed out. Try reducing the number of creatives."
#         )
#     except requests.RequestException as e:
#         raise HTTPException(
#             status_code=500,
#             detail=f"Network error during image generation: {str(e)}"
#         )


# @lru_cache(maxsize=100)
# def generate_caption(brand_name: str, product_desc: str) -> str:
#     """Generate ad caption using Gemini API with caching."""
#     if not GEMINI_API_KEY:
#         return f"Discover {brand_name} – {product_desc[:30]}..."
    
#     caption_prompt = (
#         "You are an expert ad copywriter. "
#         "Write ONE short, punchy, engaging ad caption for social media. "
#         "Maximum 12 words. Make it catchy and memorable.\n\n"
#         f"Brand: {brand_name}\n"
#         f"Product: {product_desc}\n\n"
#         "Write only the caption, nothing else:"
#     )
    
#     body = {
#         "contents": [
#             {
#                 "parts": [
#                     {"text": caption_prompt}
#                 ]
#             }
#         ],
#         "generationConfig": {
#             "temperature": 0.9,
#             "maxOutputTokens": 50,
#         }
#     }
    
#     headers = {
#         "Content-Type": "application/json",
#         "x-goog-api-key": GEMINI_API_KEY,
#     }
    
#     session = get_retry_session()
    
#     try:
#         resp = session.post(GEMINI_URL, headers=headers, json=body, timeout=30)
        
#         if resp.status_code == 429:
#             raise HTTPException(
#                 status_code=429,
#                 detail="Gemini API rate limit exceeded."
#             )
        
#         if resp.status_code != 200:
#             print(f"Gemini error: {resp.status_code} - {resp.text[:200]}")
#             return f"Experience {brand_name} – {product_desc[:40]}..."
        
#         data = resp.json()
#         text = data["candidates"][0]["content"]["parts"][0]["text"]
#         caption = text.strip().replace("\n", " ")
        
#         # Clean up quotes if present
#         if caption.startswith('"') and caption.endswith('"'):
#             caption = caption[1:-1].strip()
        
#         # Ensure reasonable length
#         if len(caption.split()) > 15:
#             caption = " ".join(caption.split()[:12]) + "..."
        
#         return caption
        
#     except requests.Timeout:
#         return f"Discover the magic of {brand_name}"
#     except Exception as e:
#         print(f"Caption generation error: {e}")
#         return f"{brand_name}: {product_desc[:50]}..."


# def build_zip(images_with_captions: List[Tuple[bytes, str]]) -> bytes:
#     """Build ZIP file containing all images and captions CSV."""
#     mem = io.BytesIO()
#     try:
#         with zipfile.ZipFile(mem, "w", zipfile.ZIP_DEFLATED) as zf:
#             csv_lines = ["filename,caption"]
#             for i, (img_bytes, cap) in enumerate(images_with_captions, start=1):
#                 fname = f"creative_{i:02d}.png"
#                 zf.writestr(fname, img_bytes)
#                 # Escape quotes for CSV
#                 safe_caption = cap.replace('"', '""')
#                 csv_lines.append(f'{fname},"{safe_caption}"')
#             zf.writestr("captions.csv", "\n".join(csv_lines))
#         mem.seek(0)
#         return mem.read()
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Failed to create ZIP: {str(e)}")


# # ---------- RESPONSE MODEL ----------

# class GenerateResponse(BaseModel):
#     images: List[str]      # data:image/png;base64,...
#     captions: List[str]
#     zip_base64: str        # base64 ZIP
#     colors: List[str]      # hex colors extracted from logo


# # ---------- ROUTES ----------

# @app.get("/")
# def root():
#     return {
#         "message": "AI Creative Studio backend is running",
#         "version": "1.0.1",
#         "status": "healthy"
#     }


# @app.get("/health")
# def health():
#     return {
#         "status": "ok",
#         "hf_configured": bool(HF_API_TOKEN),
#         "gemini_configured": bool(GEMINI_API_KEY),
#         "model": HF_IMAGE_MODEL
#     }


# @app.post("/api/generate-creatives", response_model=GenerateResponse)
# async def generate_creatives(
#     brand_name: str = Form(...),
#     product_desc: str = Form(...),
#     num_creatives: int = Form(6),
#     logo: UploadFile = File(...),
#     product: UploadFile = File(...),
# ):
#     """
#     Generate multiple ad creatives with captions.
    
#     - **brand_name**: Name of the brand
#     - **product_desc**: Description of the product
#     - **num_creatives**: Number of creatives to generate (4-12)
#     - **logo**: Brand logo image file
#     - **product**: Product image file (for future features)
#     """
    
#     # Validation
#     if num_creatives < 1 or num_creatives > 12:
#         raise HTTPException(
#             status_code=400,
#             detail="num_creatives must be between 1 and 12"
#         )
    
#     if not brand_name.strip():
#         raise HTTPException(status_code=400, detail="brand_name is required")
    
#     if not product_desc.strip():
#         raise HTTPException(status_code=400, detail="product_desc is required")
    
#     # Validate file sizes (10MB max)
#     MAX_FILE_SIZE = 10 * 1024 * 1024
#     logo_bytes = await logo.read()
#     product_bytes = await product.read()
    
#     if len(logo_bytes) > MAX_FILE_SIZE:
#         raise HTTPException(status_code=400, detail="Logo file too large (max 10MB)")
    
#     if len(product_bytes) > MAX_FILE_SIZE:
#         raise HTTPException(status_code=400, detail="Product file too large (max 10MB)")
    
#     # Process logo for brand colors
#     logo_img = load_image_from_bytes(logo_bytes)
#     colors = extract_dominant_colors(logo_img, num_colors=3)
    
#     images_with_captions: List[Tuple[bytes, str]] = []
#     data_urls: List[str] = []
#     captions: List[str] = []
    
#     # Generate creatives
#     for i in range(num_creatives):
#         try:
#             # Build prompt and generate image
#             prompt = build_image_prompt(brand_name, product_desc, colors)
#             img_bytes = generate_image_bytes(prompt)
            
#             # Generate caption (cached if same inputs)
#             caption = generate_caption(brand_name, product_desc)
            
#             images_with_captions.append((img_bytes, caption))
#             captions.append(caption)
            
#             # Create data URL for preview
#             b64_img = base64.b64encode(img_bytes).decode("utf-8")
#             data_urls.append(f"data:image/png;base64,{b64_img}")
            
#         except HTTPException:
#             raise  # Re-raise HTTP exceptions
#         except Exception as e:
#             raise HTTPException(
#                 status_code=500,
#                 detail=f"Failed to generate creative {i+1}: {str(e)}"
#             )
    
#     # Build ZIP bundle
#     zip_bytes = build_zip(images_with_captions)
#     zip_b64 = base64.b64encode(zip_bytes).decode("utf-8")
    
#     return GenerateResponse(
#         images=data_urls,
#         captions=captions,
#         zip_base64=zip_b64,
#         colors=colors,
#     )


# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=8000)



import os
import io
import zipfile
import base64
from typing import List, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import requests
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

# ---------- ENV & CONFIG ----------

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# We no longer use Hugging Face for images (to avoid billing/403/404 issues),
# but we keep the variables here to show it's pluggable.
HF_API_TOKEN = os.getenv("HF_API_TOKEN")
HF_IMAGE_MODEL = os.getenv("HF_IMAGE_MODEL", "stabilityai/stable-diffusion-2-1")

GEMINI_URL = (
    "https://generativelanguage.googleapis.com/v1beta/models/"
    "gemini-2.5-flash:generateContent"
)

# ---------- FASTAPI APP ----------

app = FastAPI(
    title="AI Creative Studio Backend",
    description="Generates branded ad creatives using local image synthesis + Gemini captions",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:8501",
        "http://127.0.0.1:8501",
        "*",  # relaxed for hackathon
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------- UTILS ----------

def load_image_from_bytes(data: bytes) -> Image.Image:
    return Image.open(io.BytesIO(data)).convert("RGBA")


def extract_dominant_colors(img: Image.Image, num_colors: int = 3) -> List[str]:
    """
    Simple dominant color extractor. Returns hex colors like ['#112233', '#ffaa33', ...]
    """
    small = img.resize((64, 64))
    arr = np.array(small.convert("RGB")).reshape(-1, 3)
    colors, counts = np.unique(arr, axis=0, return_counts=True)
    idx = np.argsort(counts)[::-1]
    top = colors[idx][:num_colors]
    return [f"#{r:02x}{g:02x}{b:02x}" for r, g, b in top]


def hex_to_rgb(hex_color: str):
    hex_color = hex_color.lstrip("#")
    return tuple(int(hex_color[i : i + 2], 16) for i in (0, 2, 4))


def generate_local_ad_image(
    brand_name: str,
    product_desc: str,
    colors: List[str],
    product_img: Image.Image | None = None,
    size: int = 1024,
) -> bytes:
    """
    Generate a simple but nice ad-style image locally using Pillow.
    - Background gradient using brand colors
    - Optional product image in the center
    - Optional subtle overlay shapes

    Returns PNG bytes.
    """
    if not colors:
        colors = ["#111827", "#4b5563", "#1f2937"]  # fallback slate palette

    # Base image
    img = Image.new("RGBA", (size, size), hex_to_rgb(colors[0]))
    draw = ImageDraw.Draw(img)

    # Gradient background between first two colors
    c1 = hex_to_rgb(colors[0])
    c2 = hex_to_rgb(colors[1]) if len(colors) > 1 else hex_to_rgb(colors[0])

    for y in range(size):
        t = y / (size - 1)
        r = int(c1[0] * (1 - t) + c2[0] * t)
        g = int(c1[1] * (1 - t) + c2[1] * t)
        b = int(c1[2] * (1 - t) + c2[2] * t)
        draw.line([(0, y), (size, y)], fill=(r, g, b, 255))

    # Add a soft vignette circle
    vignette_color = (0, 0, 0, 60)
    center = (size // 2, size // 2)
    radius = int(size * 0.45)
    for r in range(radius, 0, -1):
        alpha = int(80 * (1 - r / radius))
        bbox = [
            center[0] - r,
            center[1] - r,
            center[0] + r,
            center[1] + r,
        ]
        draw.ellipse(bbox, outline=(0, 0, 0, alpha))

    # Paste product image in the center (if provided)
    if product_img is not None:
        # Convert to RGBA, resize to fit center area
        p = product_img.convert("RGBA")
        target_w = int(size * 0.5)
        w, h = p.size
        scale = target_w / max(w, h)
        new_size = (int(w * scale), int(h * scale))
        p = p.resize(new_size, Image.LANCZOS)
        px = (size - new_size[0]) // 2
        py = (size - new_size[1]) // 2
        img.alpha_composite(p, dest=(px, py))

    # Optionally add a subtle accent bar at bottom using third color
    if len(colors) > 2:
        accent = hex_to_rgb(colors[2])
    else:
        accent = (255, 255, 255)
    bar_height = int(size * 0.08)
    draw.rectangle(
        [0, size - bar_height, size, size],
        fill=(accent[0], accent[1], accent[2], 180),
    )

    # (Optional) Add brand name text on the bar (non-LLM text, so it's allowed for demo)
    try:
        font = ImageFont.truetype("arial.ttf", int(bar_height * 0.45))
    except Exception:
        font = ImageFont.load_default()

    text = brand_name[:24]  # truncate for safety
    # tw, th = draw.textsize(text, font=font)
    # ✅ FIXED CODE
    bbox = draw.textbbox((0, 0), text, font=font)
    tw = bbox[2] - bbox[0]  # width
    th = bbox[3] - bbox[1]  # height
    tx = (size - tw) // 2
    ty = size - bar_height + (bar_height - th) // 2
    draw.text((tx, ty), text, fill=(0, 0, 0), font=font)

    # Convert to PNG bytes
    out = io.BytesIO()
    img.convert("RGB").save(out, format="PNG")
    out.seek(0)
    return out.read()


def generate_caption(brand_name: str, product_desc: str) -> str:
    """
    Generate a short caption using Gemini generateContent REST API.
    """
    if not GEMINI_API_KEY:
        return "Sample caption (set GEMINI_API_KEY for real captions)"

    caption_prompt = (
        "You are an ad copywriter. "
        "Write ONE short, catchy ad caption (max 12 words) "
        "for a social media image creative.\n\n"
        f"Brand: {brand_name}\n"
        f"Product: {product_desc}\n\n"
        "Caption:"
    )

    body = {
        "contents": [
            {
                "parts": [
                    {"text": caption_prompt}
                ]
            }
        ]
    }

    headers = {
        "Content-Type": "application/json",
        "x-goog-api-key": GEMINI_API_KEY,
    }

    resp = requests.post(GEMINI_URL, headers=headers, json=body, timeout=30)
    if resp.status_code != 200:
        # fallback to a generic caption so the flow doesn't break
        return f"Experience {brand_name} like never before."

    data = resp.json()
    try:
        text = data["candidates"][0]["content"]["parts"][0]["text"]
    except Exception:
        text = str(data)

    caption = text.strip().replace("\n", " ")
    if caption.startswith('"') and caption.endswith('"'):
        caption = caption[1:-1].strip()
    return caption


def build_zip(images_with_captions: List[Tuple[bytes, str]]) -> bytes:
    mem = io.BytesIO()
    with zipfile.ZipFile(mem, "w", zipfile.ZIP_DEFLATED) as zf:
        csv_lines = ["filename,caption"]
        for i, (img_bytes, cap) in enumerate(images_with_captions, start=1):
            fname = f"creative_{i:02d}.png"
            zf.writestr(fname, img_bytes)
            safe = '"' + cap.replace('"', "'") + '"'
            csv_lines.append(f"{fname},{safe}")
        zf.writestr("captions.csv", "\n".join(csv_lines))
    mem.seek(0)
    return mem.read()


# ---------- RESPONSE MODEL ----------

class GenerateResponse(BaseModel):
    images: List[str]      # data:image/png;base64,...
    captions: List[str]
    zip_base64: str        # base64 ZIP
    colors: List[str]      # hex colors from logo


# ---------- ROUTES ----------

@app.get("/")
def root():
    return {"message": "AI Creative Studio backend is running"}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/api/generate-creatives", response_model=GenerateResponse)
async def generate_creatives(
    brand_name: str = Form(...),
    product_desc: str = Form(...),
    num_creatives: int = Form(6),
    logo: UploadFile = File(...),
    product: UploadFile = File(...),
):
    logo_bytes = await logo.read()
    product_bytes = await product.read()

    logo_img = load_image_from_bytes(logo_bytes)
    product_img = load_image_from_bytes(product_bytes)

    colors = extract_dominant_colors(logo_img, num_colors=3)

    images_with_captions: List[Tuple[bytes, str]] = []
    data_urls: List[str] = []
    captions: List[str] = []

    for _ in range(num_creatives):
        img_bytes = generate_local_ad_image(
            brand_name=brand_name,
            product_desc=product_desc,
            colors=colors,
            product_img=product_img,
            size=1024,
        )
        caption = generate_caption(brand_name, product_desc)

        images_with_captions.append((img_bytes, caption))
        captions.append(caption)

        b64_img = base64.b64encode(img_bytes).decode("utf-8")
        data_urls.append(f"data:image/png;base64,{b64_img}")

    zip_bytes = build_zip(images_with_captions)
    zip_b64 = base64.b64encode(zip_bytes).decode("utf-8")

    return GenerateResponse(
        images=data_urls,
        captions=captions,
        zip_base64=zip_b64,
        colors=colors,
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)