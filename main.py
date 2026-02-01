import argparse
import json
import os
import sys
import logging
import cv2

# Configure Logging (Masking PII)
class PIIMaskingFormatter(logging.Formatter):
    def format(self, record):
        original = super().format(record)
        # Regex to mask XXXX XXXX 1234
        masked = re.sub(r'(\d{4})\s(\d{4})\s(\d{4})', r'XXXX XXXX \3', original)
        masked = re.sub(r'\b\d{12}\b', r'XXXX XXXX ' + original[-4:], masked)
        return masked

import re # Need to re-import re here for the formatter class binding

handler = logging.StreamHandler()
handler.setFormatter(PIIMaskingFormatter('%(asctime)s - %(levelname)s - %(message)s'))
logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(handler)


from config import AADHAAR_CONFIDENCE_THRESHOLD, NAME_MATCH_THRESHOLD
from preprocessor import Preprocessor
from ocr_engine import OCREngine
from extractor import Extractor
from validator import Validator

def main():
    parser = argparse.ArgumentParser(description="Aadhaar Extraction Engine")
    parser.add_argument("--image", required=True, help="Path to Aadhaar card image")
    parser.add_argument("--name", required=True, help="User provided Name")
    parser.add_argument("--aadhaar", required=True, help="User provided Aadhaar Number")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    
    args = parser.parse_args()

    if args.debug:
        logger.setLevel(logging.DEBUG)

    # 1. Init Modules
    preprocessor = Preprocessor()
    ocr_engine = OCREngine()
    extractor = Extractor()
    validator = Validator()

    result = {
        "extracted_name": "",
        "extracted_aadhaar": "",
        "aadhaar_checksum_valid": False,
        "name_match": False,
        "aadhaar_match": False,
        "confidence_score": 0.0
    }

    try:
        if not os.path.exists(args.image):
            raise FileNotFoundError(f"Image not found: {args.image}")

        logger.info(f"Processing image: {args.image}")

        # 2. Preprocessing
        # We get both gray (good for details) and binary (good for contrast)
        gray_img, bin_img = preprocessor.preprocess_pipeline(args.image)
        
        # Ensure both images are 3-channel BGR for PaddleOCR
        if len(gray_img.shape) == 2:
            gray_img = cv2.cvtColor(gray_img, cv2.COLOR_GRAY2BGR)
        if len(bin_img.shape) == 2:
            bin_img = cv2.cvtColor(bin_img, cv2.COLOR_GRAY2BGR)
        
        # 3. OCR
        # We try OCR on the gray image first as it often retains more context
        logger.info("Running OCR...")
        ocr_results = ocr_engine.extract_text(gray_img)
        
        # Debug: Log OCR results
        logger.debug(f"OCR Results count: {len(ocr_results)}")
        for i, item in enumerate(ocr_results[:10]):
            logger.debug(f"  [{i}] text='{item.get('text', '')}' conf={item.get('conf', 0)}")
        
        if not ocr_results:
            logger.info("First pass empty. Trying binary image...")
            ocr_results = ocr_engine.extract_text(bin_img)
            logger.debug(f"Binary OCR Results count: {len(ocr_results)}")
            for i, item in enumerate(ocr_results[:10]):
                logger.debug(f"  [{i}] text='{item.get('text', '')}' conf={item.get('conf', 0)}")

        # 4. Extraction
        logger.info("Extracting attributes...")
        ext_aadhaar, aadhaar_conf = extractor.extract_aadhaar_number(ocr_results)
        ext_name, name_orc_conf = extractor.extract_name(ocr_results, user_name_hint=args.name)

        result["extracted_aadhaar"] = ext_aadhaar if ext_aadhaar else ""
        result["extracted_name"] = ext_name

        # 5. Validation & Scoring
        if ext_aadhaar:
            clean_num = ext_aadhaar.replace(" ", "")
            result["aadhaar_checksum_valid"] = validator.validate_verhoeff(clean_num)
            result["aadhaar_match"] = validator.exact_match_aadhaar(clean_num, args.aadhaar)
        
        # Calculate Name Confidence
        final_name_conf = 0.0
        if ext_name:
            if args.name:
                match, score = validator.fuzzy_match_name(ext_name, args.name, threshold=NAME_MATCH_THRESHOLD)
                result["name_match"] = match
                final_name_conf = score / 100.0
            else:
                final_name_conf = name_orc_conf

        # Final Confidence
        if ext_name and ext_aadhaar:
            final_conf = (aadhaar_conf + final_name_conf) / 2.0
        elif ext_name:
            final_conf = final_name_conf
        elif ext_aadhaar:
            final_conf = aadhaar_conf
        else:
            final_conf = 0.0

        result["confidence_score"] = round(final_conf, 2)

    except Exception as e:
        logger.error(f"Pipeline processing error: {e}")
    
    # Output JSON
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    main()
