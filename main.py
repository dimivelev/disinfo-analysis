# main.py
# To run this file:
# 1. Ensure all packages from requirements.txt are installed.
# 2. In your terminal, run the command: uvicorn main:app --reload

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import asyncio
import os
import json # Import the json library

# Import the crew and parsing function from your agent.py file
from agent import disinfo_crew, parse_html_content_without_llm

class HtmlPayload(BaseModel):
    html: str

app = FastAPI(
    title="Disinformation Analysis API",
    description="An API that accepts an HTML payload and analyzes its content for disinformation using a multi-agent CrewAI system.",
    version="1.0.1"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# This async generator will be responsible for running the crew and streaming updates.
async def run_crew_and_stream(inputs: dict):
    """
    An async generator that yields real-time updates from the CrewAI kickoff process.
    """
    try:
        # Step 1: Announce that the analysis is starting
        yield f"data: {json.dumps({'status': 'Starting analysis...', 'type': 'update'})}\n\n"
        await asyncio.sleep(0.1) 

        # Step 2: Run the CrewAI kickoff in a separate thread
        loop = asyncio.get_event_loop()
        
        yield f"data: {json.dumps({'status': 'Agents are now analyzing the content...', 'type': 'update'})}\n\n"
        
        _ = await loop.run_in_executor(None, disinfo_crew.kickoff, inputs)

        yield f"data: {json.dumps({'status': 'Analysis complete!', 'type': 'update'})}\n\n"
        await asyncio.sleep(0.1)

        # Step 3: Collect all individual task outputs
        all_results = {
            'Headline Analyzer': disinfo_crew.tasks[0].output.raw,
            'Emotional Manipulation Analyst': disinfo_crew.tasks[1].output.raw,
            'Bias Detection Analyst': disinfo_crew.tasks[2].output.raw,
            'Medical Detector': disinfo_crew.tasks[3].output.raw
        }

        # Step 4: Stream the final result as a dictionary of all outputs
        final_data = {"source": "chrome-extension", "analysis_result": all_results}
        yield f"data: {json.dumps({'result': final_data, 'type': 'result'})}\n\n"

    except Exception as e:
        import traceback
        # Also print the error to the server console for better logging
        print(f"ERROR IN STREAM: {traceback.format_exc()}")
        error_message = f"An unexpected error occurred: {str(e)}"
        yield f"data: {json.dumps({'error': error_message, 'type': 'error'})}\n\n"

@app.post("/analyze-html/",
          summary="Analyze HTML content for disinformation (Streaming)",
          response_description="A stream of analysis updates followed by the final result.")
async def analyze_html_streaming(payload: HtmlPayload):
    """
    This endpoint accepts a JSON payload with an 'html' key, parses the content,
    and streams the CrewAI analysis process and final result.
    """
    try:
        html_content_str = payload.html
        parsed_result = parse_html_content_without_llm(html_content_str)

        if not parsed_result or not parsed_result.get('body'):
            raise HTTPException(status_code=422, detail="Could not parse HTML or body was empty.")

        inputs = {
            'headline': parsed_result.get('headline', ''),
            'body': parsed_result.get('body', '')
        }

        # Return a StreamingResponse that uses the async generator
        return StreamingResponse(run_crew_and_stream(inputs), media_type="text/event-stream")

    except HTTPException as e:
        # This will catch parsing errors before the stream starts
        return JSONResponse(content={"error": e.detail}, status_code=e.status_code)
    except Exception as e:
        return JSONResponse(content={"error": f"An unexpected error occurred: {str(e)}"}, status_code=500)


@app.get("/",
         summary="Root Endpoint",
         description="A simple endpoint to confirm the API is running.")
async def read_root():
    return {"message": "Welcome to the Disinformation Analysis API. Visit /docs to see the API documentation."}

port = int(os.getenv("PORT", 8000))

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=port)