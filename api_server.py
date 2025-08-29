#!/usr/bin/env python3
"""
🎙️ Gemini Voice Assistant API
- CSV Q&A lookup
- Gemini fallback
- REST API via FastAPI
"""

import os
import sys
import re
import subprocess as sp
from pathlib import Path
import pandas as pd
from dotenv import load_dotenv
from rapidfuzz import fuzz
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

# ----------------------------
# FastAPI setup
# ----------------------------
app = FastAPI(title="SAFE Voice Assistant API")

# ✅ Allow frontend to access backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://voicehelp.myclassboard.com"],  # your frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----------------------------
# Backend unpack check
# ----------------------------
ROOT = Path(__file__).parent.resolve()
unpack_script = ROOT / "unpack_backend.py"

if unpack_script.exists():
    try:
        print("📦 Unpacking backend...")
        sp.check_call([sys.executable, str(unpack_script)])
    except Exception as e:
        print(f"⚠️ Warning: unpack_backend.py failed: {e}")
else:
    print("✅ Backend already ready, skipping unpack.")

# ----------------------------
# Load environment + data
# ----------------------------
load_dotenv()
CSV_PATH = ROOT / "qa.csv"

if CSV_PATH.exists():
    df = pd.read_csv(CSV_PATH)
    print(f"✅ Loaded {len(df)} Q&A entries from {CSV_PATH}")
else:
    df = pd.DataFrame(columns=["question", "answer"])
    print("⚠️ No qa.csv found, running with empty Q&A set")

# ----------------------------
# Models
# ----------------------------
class Query(BaseModel):
    text: str

# ----------------------------
# API Routes
# ----------------------------

@app.get("/")
def root():
    return {"status": "SAFE API running ✅"}

@app.get("/status")
def status():
    return {"running": True, "main": "SAFE backend"}

@app.get("/logs")
def logs(n: int = 100):
    # dummy logs for now
    return {"lines": [f"log line {i}" for i in range(n)]}

@app.post("/start")
def start():
    return {"message": "Started ✅"}

@app.post("/stop")
def stop():
    return {"message": "Stopped ✅"}

@app.post("/quit")
def quit_app():
    return {"message": "Quit ✅"}

@app.post("/send")
def send_text(data: dict):
    return {"received": data.get("text", "")}

@app.post("/ask")
def ask(query: Query):
    user_text = query.text.strip()
    if df.empty:
        return {"answer": "⚠️ Knowledge base is empty. Please add qa.csv."}

    best_match = None
    best_score = 0

    for _, row in df.iterrows():
        score = fuzz.ratio(user_text.lower(), str(row["question"]).lower())
        if score > best_score:
            best_score = score
            best_match = row

    if best_match is not None and best_score > 70:
        return {
            "answer": best_match["answer"],
            "match_score": best_score,
            "source": "csv"
        }
    else:
        # fallback (dummy)
        return {
            "answer": f"🤖 Sorry, I don't know about '{user_text}'.",
            "match_score": best_score,
            "source": "fallback"
        }
