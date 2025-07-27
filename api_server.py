#!/usr/bin/env python3
"""
FastAPI Server for Legal Document Compliance Analyzer

Simple REST API that loads Legal-BERT once and provides document analysis endpoint.
"""

import os
import json
import time
from datetime import datetime
from typing import Optional
from pathlib import Path
import tempfile

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import uvicorn

# Import the analysis functions from app.py
import app

app_fastapi = FastAPI(title="Legal Compliance Analytics API", version="1.0.0")

# Global variables to store loaded data
legal_clauses = None
clause_translations = None

class AnalysisRequest(BaseModel):
    document_type: str = "nda"
    enable_benchmark: bool = True

@app_fastapi.on_event("startup")
async def startup_event():
    """Load models on startup"""
    global legal_clauses, clause_translations
    
    print("Loading Legal-BERT model...")
    app.load_legal_bert()
    print("Legal-BERT loaded successfully")
    
    print("Loading spaCy model...")
    app.load_spacy_model()
    print("spaCy model loaded successfully")
    
    print("Loading legal clauses...")
    legal_clauses = app.load_legal_clauses()
    print("Legal clauses loaded successfully")
    
    print("Loading clause translations...")
    clause_translations = app.load_clause_translations()
    print("Clause translations loaded successfully")
    
    print("API server ready!")

@app_fastapi.post("/analyze")
async def analyze_document(
    file: UploadFile = File(...),
    document_type: str = "nda",
    enable_benchmark: bool = True
):
    """
    Analyze a legal document for compliance
    
    - **file**: PDF document to analyze
    - **document_type**: Type of document (nda, employment_contract, etc.)
    - **enable_benchmark**: Include performance metrics
    """
    
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")
    
    try:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
            content = await file.read()
            temp_file.write(content)
            temp_file_path = temp_file.name
        
        # Parse PDF
        app.start_timer('pdf_parsing')
        text = app.parse_pdf(temp_file_path)
        app.end_timer('pdf_parsing')
        
        if not text:
            raise HTTPException(status_code=400, detail="Could not extract text from PDF")
        
        # Analyze document (models are already loaded globally in app module)
        app.start_timer('api_analysis')
        
        analysis_result = app.analyze_legal_compliance_enhanced(
            text=text,
            document_type=document_type,
            clauses_data=legal_clauses,
            use_legal_bert=True
        )
        
        app.end_timer('api_analysis')
        
        # Add performance metrics if enabled
        if enable_benchmark:
            system_info = app.get_system_info()
            analysis_result["performance_metrics"] = {
                "system_info": system_info,
                "timing": app.performance_metrics['timing'],
                "analysis_duration": app.performance_metrics['timing'].get('api_analysis', {}).get('duration_seconds', 0)
            }
        
        # Clean up temporary file
        os.unlink(temp_file_path)
        
        return JSONResponse(content=analysis_result)
        
    except Exception as e:
        # Clean up temporary file if it exists
        if 'temp_file_path' in locals():
            try:
                os.unlink(temp_file_path)
            except:
                pass
        
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

@app_fastapi.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "models_loaded": app.legal_model is not None}

if __name__ == "__main__":
    uvicorn.run(app_fastapi, host="0.0.0.0", port=8000) 