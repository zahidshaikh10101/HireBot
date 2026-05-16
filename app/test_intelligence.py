"""
test_intelligence.py
--------------------
Quick sanity-check for the OpenRouter integration.
Run this BEFORE starting the FastAPI server.

Usage:
  cd E:/projects/HireBot
  python test_intelligence.py
"""

import os
import sys
import json
from dotenv import load_dotenv

load_dotenv()

# Add app/ to path so imports work
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "app"))

from intelligence.analyze import analyze_intelligence

SAMPLE_RESUME = """
Zahid Salim Shaikh
Mumbai, India | zahidshaikh10101@gmail.com | +91-8286092787

Data & BI professional with 4 years of experience building end-to-end analytics systems.
At IVY Entertainment, independently architected data pipelines across Spotify, YouTube,
and JioSaavn platforms, cutting data collection time by 80%.
Skilled in Python, SQL, Pandas, PySpark, and Power BI.
Currently completing MTech in Data Science & Engineering from BITS Pilani.

Experience:
Senior Executive Business Intelligence, IVY Entertainment - Apr 2024 to Present
- Built automated data pipelines using Python and REST APIs
- Developed GenAI-based pipeline for YouTube metadata generation

Software Engineer, LTIMindtree - June 2022 to Mar 2024
- Java Spring Boot backend services for insurance data processing

Skills:
Languages: Python, SQL
Data & Analytics: Pandas, NumPy, PySpark
ML & AI: Scikit-learn, NLP, GenAI APIs, Hugging Face
"""

if __name__ == "__main__":
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        print("❌ OPENROUTER_API_KEY not set in .env file")
        print("   Get your free key at: https://openrouter.ai/keys")
        sys.exit(1)

    print("🔍 Testing OpenRouter Intelligence Module...")
    print("─" * 50)

    try:
        result = analyze_intelligence(SAMPLE_RESUME)
        print("✅ Success!\n")
        print(json.dumps(result, indent=2))

    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)