# VITA — AI-Powered Yoga Assistant

VITA is a full-stack yoga practice application combining real-time pose detection, form feedback, and an AI-powered yoga chatbot.

## Features

- **Pose Detection & Scoring** — YOLO11m-Pose extracts 17 COCO keypoints; a trained Random Forest classifier identifies the pose (Downdog, Goddess, Plank, Tree, Warrior2) and per-pose form analyzers return actionable feedback.
- **VITA AI Chatbot** — Locally-hosted LLM via [Ollama](https://ollama.com/) (`qwen2.5:1.5b-instruct`) answers yoga questions using a curated knowledge base.
- **React / Vite Frontend** — Responsive UI with live webcam capture, chat interface, and pose session pages.

---

## Project Structure

```
VITA-main/
├── frontend/          # React + Vite UI
├── backend/
│   ├── server.js      # Node/Express chatbot API + ML proxy (port 5000)
│   ├── ml/
│   │   ├── api.py                        # FastAPI inference service (port 8001)
│   │   ├── yogamat_inference.py          # YOLO + RF inference pipeline
│   │   ├── yogamat_pose_classifier_rf.pkl # Trained RF classifier (committed)
│   │   ├── yolo11m-pose.pt               # YOLO weights — see below
│   │   └── requirements.txt
│   ├── app/           # Alternate FastAPI entry point
│   └── .env.example   # Environment variable template
├── data/
│   ├── yoga_knowledge.json   # Chatbot knowledge base
│   └── yoga.json
├── .gitattributes     # Git LFS rules for *.pt weights
└── .gitignore
```

---

## Setup

### Prerequisites

| Tool | Version |
|------|---------|
| Node.js | 18 + |
| Python | 3.10 + |
| Ollama | latest |
| git-lfs | latest |

---

### 1. Clone & pull LFS files

```bash
git clone https://github.com/<your-username>/VITA-main.git
cd VITA-main
git lfs pull
```

> `git lfs pull` downloads `yolo11m-pose.pt`. If Git LFS is not available,
> download the weights manually — see **Model Files** below.

---

### 2. Frontend

```bash
cd frontend
npm install
cp .env.example .env.local   # edit if your backend runs on a different port
npm run dev
```

---

### 3. Node / Express backend

```bash
cd backend
npm install
cp .env.example .env         # edit if needed
node server.js
```

Runs on **http://localhost:5000**.

---

### 4. Python ML service (FastAPI)

```bash
cd backend
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate

pip install -r ml/requirements.txt
uvicorn ml.api:app --port 8001 --reload
```

Runs on **http://127.0.0.1:8001**.

> The service loads `yolo11m-pose.pt` and `yogamat_pose_classifier_rf.pkl`
> from `backend/ml/` using paths relative to `yogamat_inference.py`.
> No absolute paths are used.

---

### 5. Ollama chatbot model

```bash
ollama pull qwen2.5:1.5b-instruct
ollama serve   # usually auto-started
```

---

## Model Files

| File | Size | How to obtain |
|------|------|---------------|
| `backend/ml/yogamat_pose_classifier_rf.pkl` | ~4.8 MB | **Committed** — pulled automatically with the repo. |
| `backend/ml/yolo11m-pose.pt` | ~40 MB | **Git LFS** — run `git lfs pull` after cloning. Alternatively, [download from Ultralytics](https://github.com/ultralytics/assets/releases/) or let Ultralytics auto-download on first inference run by removing the file (it will download to the local cache). |

### Manual YOLO download (no LFS)

```bash
pip install ultralytics
python -c "from ultralytics import YOLO; YOLO('yolo11m-pose.pt')"
# Then copy the downloaded file to backend/ml/yolo11m-pose.pt
```

---

## Environment Variables

Copy `.env.example` → `.env` in each service directory and adjust as needed.
**Never commit `.env` files.**

| Variable | Default | Used by |
|----------|---------|---------|
| `PORT` | `5000` | Node backend |
| `OLLAMA_URL` | `http://localhost:11434/api/chat` | Node backend |
| `OLLAMA_MODEL` | `qwen2.5:1.5b-instruct` | Node backend |
| `ML_SERVICE_URL` | `http://127.0.0.1:8001` | Node backend |
| `VITE_API_BASE_URL` | `http://localhost:5000` | React frontend |

---

## Supported Yoga Poses

Downdog · Goddess · Plank · Tree · Warrior2

---

## Notes

- The `localhost` URLs in `ChatBot.jsx` and `PoseSession.jsx` are development defaults. For production, set `VITE_API_BASE_URL` in your frontend environment and update the proxy target accordingly.
- The `backend/ml/yogamat_inference.py` file contains a duplicate definition block (merged from an earlier version). Both blocks are functionally equivalent; the second definition (line ~877) takes precedence at import time.
