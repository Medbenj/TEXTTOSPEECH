modules = [
    ("numpy","import numpy as np"),
    ("kokoro","from kokoro import KPipeline"),
    ("google-genai","from google import genai"),
    ("dotenv","from dotenv import load_dotenv"),
    ("streamlit","import streamlit as st"),
    ("pypdf","from pypdf import PdfReader"),
    ("struct","import struct"),
    ("tempfile","import tempfile"),
    ("shutil","import shutil"),
]

ok = True
for name, stmt in modules:
    try:
        exec(stmt, {})
        print(f"OK: {name}")
    except Exception as e:
        ok = False
        print(f"FAIL: {name} -> {e.__class__.__name__}: {e}")

if not ok:
    raise SystemExit(1)
