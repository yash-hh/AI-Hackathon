# 🎨 AI Creative Studio  
### Automated Ad Creative Generation with Stable Diffusion + Gemini  
**Hackathon Submission — Problem Statement H-003**

## 🚀 Overview

AI Creative Studio is a multimodal system that automatically transforms a brand’s identity into professional ad creatives.

Upload a brand logo and product image, and the system generates:

- Multiple themed ad creatives  
- Brand-consistent color palettes  
- High-quality captions  
- Downloadable ZIP bundle  
- Ready for social media campaigns  

## 🤖 Core Features

| Capability | Description |
|---|---|
| Automated Creative Generation | Generates multiple marketing images per run |
| Brand Identity Awareness | Extracts dominant colors from logo for visual coherence |
| Caption Generation via Gemini | Produces short, catchy ad copy |
| Batch Export | ZIP file with images + captions.csv |
| Visual Preview UI | Preview creatives directly |
| API-first Architecture | Fully decoupled backend |

## 🏗️ Architecture

```
┌───────────────────┐      ┌───────────────────────────┐
│   Streamlit UI     │◄────►│       FastAPI Backend     │
│ (Frontend Client)  │      │  (REST API endpoints)     │
└───────▲───────────┘      └───────▲───────────────────┘
        │                           │
        │                           │
        │                   ┌───────┴─────────────────┐
        │                   │         Services         │
        │                   │ - Stable Diffusion API  │
        │                   │ - Gemini LLM API        │
        │                   └─────────────────────────┘
```

## 🧰 Technology Stack

- Streamlit (frontend)  
- FastAPI (backend)  
- Stable Diffusion (Hugging Face API)  
- Gemini (Google)  
- ZIP export + CSV metadata  

## ⚙️ Installation & Setup

### 1. Clone the repository

```
git clone https://github.com/yash-hh/AI-Hackathon.git
cd ai-creative-studio
```

### 2. Navigate to backend

```
cd backend
```

### 3. Create .env

```
.env
```

Fill in:

```
HF_API_TOKEN="your_huggingface_token_here"
GEMINI_API_KEY="your_gemini_api_key_here"
HF_IMAGE_MODEL="stabilityai/stable-diffusion-2-1"
```

### 4. Install dependencies

```
pip install -r requirements.txt
```

## 🖥️ Run the System

### Backend

```
uvicorn main:app --reload --port 8000
```

### UI

```
streamlit run app.py
```

UI opens at:

```
http://localhost:8501
```

## 🎯 How to Use

1. Enter brand name  
2. Enter product description  
3. Upload brand logo  
4. Upload product image  
5. Choose number of creatives  
6. Click “Generate Ad Creatives 🚀”  

## 📦 Output Format

```
ai_creatives_bundle.zip
├── creative_01.png
├── creative_02.png
└── captions.csv
```

## 📌 Limitations

- Depends on inference latency  
- Product image not used visually  

## 🧭 Future Work

- Local GPU support  
- Style templates  
- Aspect ratio presets  
- Background removal  

## 🏆 Why It Solves H-003

This solution delivers:

- Automated creative generation  
- Brand consistency  
- Captions  
- Bulk export  
- Clean architecture  

## 🙌 Acknowledgements

- Stable Diffusion (via Hugging Face)  
- Gemini (Google AI)  
- Streamlit & FastAPI  

## 📄 License

MIT

**AI Creative Studio — Turning brand identity into automated creativity.**
