import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Try to import from the src/ folder if we're running locally or on Vercel
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from sentiment_analysis.orchestration.analyzer import analyze_document
from sentiment_analysis.schemas import DocumentAnalysisResult
from fastapi.staticfiles import StaticFiles

app = FastAPI()

# For local testing, serve the dashboard static files.
# In production on Vercel, this is handled by Vercel's edge network based on vercel.json.
dashboard_path = os.path.join(os.path.dirname(__file__), "..", "dashboard")
if os.path.exists(dashboard_path):
    app.mount("/dashboard", StaticFiles(directory=dashboard_path, html=True), name="dashboard")

class AnalyzeRequest(BaseModel):
    text: str

@app.post("/api/analyze")
async def analyze_api(request: AnalyzeRequest):
    if not request.text or len(request.text.strip()) < 10:
        raise HTTPException(status_code=400, detail="Text is too short or empty.")
    
    # Check if anthropic API key is available (it is needed for LLM inference)
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise HTTPException(status_code=401, detail="ANTHROPIC_API_KEY environment variable is not set. Please set it in Vercel or locally.")

    try:
        # We disable cache for serverless environments since Redis won't be available
        result: DocumentAnalysisResult = await analyze_document(
            content=request.text,
            use_cache=False
        )
        # return the pydantic model directly; FastAPI will serialize it to JSON
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
