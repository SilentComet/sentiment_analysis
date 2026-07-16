# Sentiment Analysis Dashboard

An end-to-end sentiment analysis pipeline powered by local SLMs and Anthropic Claude for complex linguistic evaluation. The dashboard provides insights into document sentiment, trajectory, emotion profile, aspect-based sentiment (ABSA), and underlying narrative arcs.

## Features

- **Document Ingestion**: Fast processing and semantic chunking.
- **Hybrid Inference**: Routes simple chunks to local SLMs (HuggingFace) and complex chunks to Claude 3.5 Sonnet.
- **Post-Processing**: Cultural idiom decoding, entity extraction, and sentiment contradictions resolution.
- **UI Dashboard**: Dynamic, interactive UI built with native Web Components and Spring Physics.

## Tech Stack

- **Backend**: FastAPI, Pydantic, Python 3.13, Uvicorn
- **Frontend**: Vanilla HTML/JS/CSS, Canvas API
- **Deployment**: Vercel Serverless Functions

## Local Development

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Set up Environment Variables**:
   Create a `.env` file in the root directory (this file is ignored by Git).
   ```
   ANTHROPIC_API_KEY=sk-ant-api03...
   ```

3. **Run the API server**:
   ```bash
   uvicorn api.index:app --reload
   ```

4. **View Dashboard**:
   Open `http://localhost:8000/dashboard/` in your browser.

## Deployment

This repository is configured for zero-configuration deployment on **Vercel**.
Ensure that you add `ANTHROPIC_API_KEY` to your Vercel Project Settings under Environment Variables.
