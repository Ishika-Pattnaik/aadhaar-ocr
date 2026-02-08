"""
Aadhaar OCR API Server
FastAPI-based web service for extracting information from Aadhaar cards.
"""
import os
import re
import json
import logging
import cv2
import numpy as np
from datetime import datetime
from typing import Optional

from fastapi import FastAPI, UploadFile, File, HTTPException, Depends, Header, status, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure Logging with PII masking
class PIIMaskingFormatter(logging.Formatter):
    def format(self, record):
        original = super().format(record)
        # Mask Aadhaar numbers: XXXX XXXX 1234
        masked = re.sub(r'(\d{4})\s(\d{4})\s(\d{4})', r'XXXX XXXX \3', original)
        # Mask 12-digit sequences
        masked = re.sub(r'\b\d{12}\b', r'XXXX XXXX ' + original[-4:] if len(original) >= 4 else 'XXXX', masked)
        return masked

handler = logging.StreamHandler()
handler.setFormatter(PIIMaskingFormatter('%(asctime)s - %(levelname)s - %(message)s'))
logger = logging.getLogger(__name__ + ".server")
logger.setLevel(logging.INFO)
logger.addHandler(handler)

# Import OCR modules
from config import AADHAAR_CONFIDENCE_THRESHOLD, NAME_MATCH_THRESHOLD
from preprocessor import Preprocessor
from ocr_engine import OCREngine
from extractor import Extractor
from validator import Validator

# ============= CONFIGURATION =============
API_KEY = os.getenv("API_KEY", "your-secret-api-key-change-in-production")
API_KEY_HEADER = os.getenv("API_KEY_HEADER", "X-API-Key")
DEBUG_MODE = os.getenv("DEBUG_MODE", "false").lower() == "true"

# ============= PYDANTIC MODELS =============
class ExtractionResponse(BaseModel):
    success: bool
    message: str
    data: Optional[dict] = None
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())

class HealthResponse(BaseModel):
    status: str
    ocr_engine_ready: bool
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())

class AadhaarValidateRequest(BaseModel):
    aadhaar_number: str = Field(..., min_length=12, max_length=16, description="Aadhaar number to validate")

class ValidationResponse(BaseModel):
    success: bool
    valid: bool
    checksum_valid: bool
    formatted_number: str
    message: str

# ============= FASTAPI APP =============
app = FastAPI(
    title="Aadhaar OCR API",
    description="Extract and validate information from Aadhaar card images",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware for browser access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Allow all origins for API flexibility
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Initialize OCR components (lazy loading)
ocr_components = {}

def get_ocr_components():
    """Initialize and return OCR components (singleton pattern)."""
    global ocr_components
    if not ocr_components:
        logger.info("Initializing OCR components...")
        try:
            ocr_components['preprocessor'] = Preprocessor()
            ocr_components['ocr_engine'] = OCREngine()
            ocr_components['extractor'] = Extractor()
            ocr_components['validator'] = Validator()
            logger.info("OCR components initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize OCR components: {e}")
            raise
    return ocr_components

# ============= API KEY AUTHENTICATION =============
async def verify_api_key(x_api_key: str = Header(..., alias=API_KEY_HEADER)):
    """Verify the API key provided in the request header."""
    if not x_api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="API key is required"
        )
    if x_api_key != API_KEY:
        logger.warning(f"Invalid API key attempt: {x_api_key[:4]}...")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key"
        )
    return x_api_key

# ============= ENDPOINTS =============

@app.get("/", response_class=FileResponse)
async def root():
    """Root endpoint serving the Frontend UI."""
    return FileResponse("static/index.html")

@app.get("/api-info", response_model=dict)
async def api_info():
    """Old root endpoint with API information."""
    return {
        "name": "Aadhaar OCR API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint to verify the service is running."""
    try:
        components = get_ocr_components()
        ocr_ready = components['ocr_engine'] is not None and components['ocr_engine'].ocr is not None
        
        return HealthResponse(
            status="healthy" if ocr_ready else "degraded",
            ocr_engine_ready=ocr_ready
        )
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return HealthResponse(
            status="unhealthy",
            ocr_engine_ready=False
        )

@app.post("/extract", response_model=ExtractionResponse, dependencies=[Depends(verify_api_key)])
async def extract_aadhaar_info(
    file: UploadFile = File(..., description="Aadhaar card image file (JPG, PNG)"),
    user_name: Optional[str] = Query(None, alias="name", description="User provided name for fuzzy matching"),
    user_aadhaar: Optional[str] = Query(None, alias="aadhaar", description="User provided Aadhaar number for validation")
):
    """
    Extract information from an Aadhaar card image.
    
    - **file**: Image file (JPG, PNG)
    - **name**: (Optional) User's name for fuzzy matching
    - **aadhaar**: (Optional) User's Aadhaar number for exact match validation
    
    Returns extracted name, Aadhaar number, validation results, and confidence score.
    """
    # Validate file type
    allowed_types = ["image/jpeg", "image/png", "image/jpg"]
    if file.content_type not in allowed_types:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid file type. Allowed: {', '.join(allowed_types)}"
        )

    try:
        # Read and decode image
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Could not decode image file"
            )

        logger.info(f"Processing image: {file.filename}, size: {len(contents)} bytes")

        # Get OCR components
        components = get_ocr_components()
        preprocessor = components['preprocessor']
        ocr_engine = components['ocr_engine']
        extractor = components['extractor']
        validator = components['validator']

        # Save temp image for preprocessing
        temp_path = f"/tmp/aadhaar_{datetime.now().timestamp()}.jpg"
        cv2.imwrite(temp_path, image)

        # Preprocessing
        gray_img, bin_img = preprocessor.preprocess_pipeline(temp_path)
        
        # Ensure both images are 3-channel BGR for PaddleOCR
        # Note: Preprocessor already returns gray_img as 3-channel BGR.
        # We only need to check bin_img which might be 1-channel from adaptiveThreshold.
        if len(bin_img.shape) == 2:
            bin_img = cv2.cvtColor(bin_img, cv2.COLOR_GRAY2BGR)

        # OCR
        logger.info("Running OCR...")
        ocr_results = ocr_engine.extract_text(gray_img)
        
        if not ocr_results:
            logger.info("First pass empty. Trying binary image...")
            ocr_results = ocr_engine.extract_text(bin_img)

        # Extraction
        logger.info("Extracting attributes...")
        ext_aadhaar, aadhaar_conf = extractor.extract_aadhaar_number(ocr_results)
        ext_name, name_orc_conf = extractor.extract_name(ocr_results, user_name_hint=user_name)

        # Build result
        result = {
            "extracted_name": ext_name if ext_name else "",
            "extracted_aadhaar": ext_aadhaar if ext_aadhaar else "",
            "confidence_score": 0.0
        }

        # Validation
        validation_results = {
            "aadhaar_checksum_valid": False,
            "aadhaar_match": False,
            "name_match": False
        }

        if ext_aadhaar:
            clean_num = ext_aadhaar.replace(" ", "")
            validation_results["aadhaar_checksum_valid"] = validator.validate_verhoeff(clean_num)
            
            if user_aadhaar:
                validation_results["aadhaar_match"] = validator.exact_match_aadhaar(clean_num, user_aadhaar)
        
        # Calculate Name Confidence
        # If user verified (matched), use match score.
        # If found but not verified (no user input), use OCR confidence.
        final_name_conf = 0.0
        
        if ext_name:
            if user_name:
                match, score = validator.fuzzy_match_name(ext_name, user_name, threshold=NAME_MATCH_THRESHOLD)
                validation_results["name_match"] = match
                final_name_conf = score / 100.0
            else:
                # No user input to verify against, trust the OCR confidence
                final_name_conf = name_orc_conf

        # Calculate Final Confidence
        # If both name and aadhaar found -> average
        # If only one found -> take that one
        if ext_name and ext_aadhaar:
            final_conf = (aadhaar_conf + final_name_conf) / 2.0
        elif ext_name:
            final_conf = final_name_conf
        elif ext_aadhaar:
            final_conf = aadhaar_conf
        else:
            final_conf = 0.0
        
        result["confidence_score"] = round(final_conf, 2)

        # Add validation results
        result.update(validation_results)

        # Cleanup
        if os.path.exists(temp_path):
            os.remove(temp_path)

        logger.info(f"Extraction completed. Name: {ext_name[:20] if ext_name else 'N/A'}, "
                   f"Aadhaar: {ext_aadhaar[:9] + 'XXXX' if ext_aadhaar else 'N/A'}, "
                   f"Confidence: {result['confidence_score']}")

        return ExtractionResponse(
            success=True,
            message="Extraction completed successfully",
            data=result
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Extraction error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return ExtractionResponse(
            success=False,
            message=f"Extraction failed: {str(e)}"
        )

@app.post("/validate-aadhaar", response_model=ValidationResponse, dependencies=[Depends(verify_api_key)])
async def validate_aadhaar_number(request: AadhaarValidateRequest):
    """
    Validate an Aadhaar number using Verhoeff checksum algorithm.
    
    - **aadhaar_number**: Aadhaar number to validate (can be formatted with spaces)
    
    Returns validation result and formatted number.
    """
    try:
        # Clean the input
        clean_number = re.sub(r'\D', '', request.aadhaar_number)
        
        if len(clean_number) != 12:
            return ValidationResponse(
                success=True,
                valid=False,
                checksum_valid=False,
                formatted_number="",
                message="Aadhaar number must be exactly 12 digits"
            )

        # Format for display
        formatted = f"{clean_number[:4]} {clean_number[4:8]} {clean_number[8:]}"

        # Validate
        components = get_ocr_components()
        validator = components['validator']
        is_valid = validator.validate_verhoeff(clean_number)

        return ValidationResponse(
            success=True,
            valid=is_valid,
            checksum_valid=is_valid,
            formatted_number=formatted,
            message="Aadhaar number is valid" if is_valid else "Aadhaar number has invalid checksum"
        )

    except Exception as e:
        logger.error(f"Validation error: {e}")
        return ValidationResponse(
            success=False,
            valid=False,
            checksum_valid=False,
            formatted_number="",
            message=f"Validation failed: {str(e)}"
        )

# ============= MAIN ENTRY POINT =============
if __name__ == "__main__":
    import uvicorn
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", 7860))  # Default 7860 for HF Spaces
    
    print(f"""
╔══════════════════════════════════════════════════════════╗
║              Aadhaar OCR API Server                       ║
╠══════════════════════════════════════════════════════════╣
║  Server running at: http://{host}:{port}                      ║
║  API Documentation: http://{host}:{port}/docs                ║
║  Health Check: http://{host}:{port}/health                   ║
╚══════════════════════════════════════════════════════════╝
    """)
    
    uvicorn.run(app, host=host, port=port)

