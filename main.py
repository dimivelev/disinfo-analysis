# main.py
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import asyncio
import os
import json

from agent import create_disinfo_crew, parse_html_content_without_llm

class HtmlPayload(BaseModel):
    html: str

app = FastAPI(
    title="Disinformation Analysis API",
    description="An API that accepts an HTML payload and analyzes its content for disinformation using a multi-agent CrewAI system.",
    version="1.0.3" # Incremented version
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

async def run_crew_and_stream(inputs: dict):
    crew = create_disinfo_crew()
    
    try:
        yield f"data: {json.dumps({'status': 'Starting analysis...', 'type': 'update'})}\n\n"
        await asyncio.sleep(0.1) 

        loop = asyncio.get_event_loop()
        
        yield f"data: {json.dumps({'status': 'Agents are now analyzing the content...', 'type': 'update'})}\n\n"
        
        _ = await loop.run_in_executor(None, crew.kickoff, inputs)

        yield f"data: {json.dumps({'status': 'Analysis complete! Collecting results...', 'type': 'update'})}\n\n"
        await asyncio.sleep(0.1)

        def get_safe_output(task):
            """Helper function to safely get raw output from a task."""
            if task and task.output and hasattr(task.output, 'raw'):
                return task.output.raw
            # Provide a fallback message if output is missing
            return "Error: This agent failed to produce a valid output."

        all_results = {
            'Headline Analyzer': get_safe_output(crew.tasks[0]),
            'Emotional Manipulation Analyst': get_safe_output(crew.tasks[1]),
            'Bias Detection Analyst': get_safe_output(crew.tasks[2]),
            'Medical Detector': get_safe_output(crew.tasks[3])
        }
      
        final_data = {"source": "chrome-extension", "analysis_result": all_results}
        yield f"data: {json.dumps({'result': final_data, 'type': 'result'})}\n\n"

    except Exception as e:
        import traceback
        print(f"ERROR IN STREAM: {traceback.format_exc()}")
        error_message = f"An unexpected error occurred: {str(e)}"
        yield f"data: {json.dumps({'error': error_message, 'type': 'error'})}\n\n"

@app.post("/analyze-html/",
          summary="Analyze HTML content for disinformation (Streaming)",
          response_description="A stream of analysis updates followed by the final result.")
async def analyze_html_streaming(payload: HtmlPayload):
    try:
        html_content_str = payload.html
        parsed_result = parse_html_content_without_llm(html_content_str)

        if not parsed_result or not parsed_result.get('body'):
            raise HTTPException(status_code=422, detail="Could not parse HTML or body was empty.")

        inputs = {
            'headline': parsed_result.get('headline', ''),
            'body': parsed_result.get('body', '')
        }

        return StreamingResponse(run_crew_and_stream(inputs), media_type="text/event-stream")

    except HTTPException as e:
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