import logging
try:
    from paddleocr import PaddleOCR
    PADDLE_AVAILABLE = True
except ImportError:
    PADDLE_AVAILABLE = False

from config import OCR_LANG

logger = logging.getLogger(__name__)

class OCREngine:
    def __init__(self):
        if PADDLE_AVAILABLE:
            # Initialize once to load model into memory
            # use_angle_cls=True helps with rotated text
            self.ocr = PaddleOCR(use_angle_cls=True, lang=OCR_LANG)
        else:
            logger.warning("PaddleOCR not found. Please install it.")
            self.ocr = None

    def extract_text(self, image):
        """
        Runs OCR on the image.
        Returns a list of dicts: {'text': str, 'conf': float, 'box': list}
        """
        if not self.ocr:
            return []

        try:
            # PaddleOCR.ocr() returns a list of results (one per image).
            # Each result is a list of [box, (text, score)].
            # Note: cls parameter was removed in PaddleOCR 3.x - use_angle_cls in init handles this
            result = self.ocr.ocr(image)
            
            parsed_result = []
            
            # Parse results
            # PaddleOCR 3.x result structure: 
            # result = [ [ [box_points, text, score], ... ] ]
            # Or: result = [ [ [box, (text, score)], ... ] ] (older format)
            # Note: result can be None or empty list if no text found.
            
            if result and len(result) > 0 and result[0]:
                first_item = result[0]
                
                # Check for dictionary format (New PaddleOCR/PaddleX structure)
                if isinstance(first_item, dict):
                    texts = first_item.get('rec_texts', [])
                    scores = first_item.get('rec_scores', [])
                    boxes = first_item.get('dt_polys', [])
                    
                    # If dt_polys is missing, try rec_boxes or Rec_polys
                    if not boxes:
                        boxes = first_item.get('rec_boxes', [])
                    if not boxes:
                        boxes = first_item.get('rec_polys', [])
                        
                    # Zip them together
                    count = min(len(texts), len(scores))
                    if isinstance(boxes, list) and len(boxes) >= count:
                         pass
                    else:
                         # specialized handling if boxes mismatch or are missing
                         boxes = [[]] * count
                    
                    for i in range(count):
                        parsed_result.append({
                            'text': texts[i],
                            'conf': float(scores[i]),
                            'box': boxes[i]
                        })
                else:
                    # Standard List format
                    for line in first_item:
                         # Handle both old and new PaddleOCR list formats
                        if len(line) >= 3:
                            # New format: [box_points, text, score]
                            box = line[0]
                            text = line[1]
                            conf = line[2]
                        elif len(line) == 2:
                            # Old format: [box, (text, score)]
                            box = line[0]
                            text_conf = line[1]
                            if isinstance(text_conf, tuple) and len(text_conf) >= 2:
                                text = text_conf[0]
                                conf = text_conf[1]
                            else:
                                text = str(text_conf)
                                conf = 0.0
                        else:
                            continue
                        
                        parsed_result.append({
                            'text': text,
                            'conf': conf,
                            'box': box
                        })
            
            return parsed_result

        except Exception as e:
            logger.error(f"OCR Failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return []
