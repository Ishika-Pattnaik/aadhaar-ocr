import re
import logging
from config import AADHAAR_REGEX_PATTERNS, OCR_CORRECTIONS, NAME_ANCHORS_TOP, NON_NAME_WORDS
from validator import Validator

logger = logging.getLogger(__name__)

class Extractor:
    def __init__(self):
        self.validator = Validator()

    def clean_ocr_text(self, text):
        """
        Cleans OCR text by fixing common confusions.
        e.g. '1234 567B' -> '1234 5678'
        """
        res = ""
        for char in text:
            if char in OCR_CORRECTIONS:
                res += OCR_CORRECTIONS[char]
            else:
                res += char
        return res.strip()



    def is_valid_name(self, text):
        """
        Validates if extracted text could be a person's name.
        Returns True if it passes name criteria, False otherwise.
        """
        if not text or not text.strip():
            return False
        
        text_lower = text.lower().strip()
        words = text_lower.split()
        
        # Must have at least 2 words to be a name (first name + last name)
        if len(words) < 2:
            logger.debug(f"Rejected name '{text}': Less than 2 words")
            return False
        
        # Filter out text containing non-name words (using word boundary matching)
        for non_name in NON_NAME_WORDS:
            # Use regex with word boundaries to avoid substring matches
            # e.g., "m" should only match " m ", not "kumar"
            pattern = r'\b' + re.escape(non_name) + r'\b'
            if re.search(pattern, text_lower):
                logger.debug(f"Rejected name '{text}': Contains non-name word '{non_name}'")
                return False
        
        # Check if text starts with digits (unlikely for names)
        if text.strip()[0].isdigit():
            logger.debug(f"Rejected name '{text}': Starts with digit")
            return False
        
        # Check if text is too long (likely not a name, more likely a sentence/address)
        if len(words) > 6:
            logger.debug(f"Rejected name '{text}': Too many words ({len(words)})")
            return False
        
        # Reject if all words are uppercase and very long (likely a header/label)
        if text.isupper() and len(text) > 20:
            logger.debug(f"Rejected name '{text}': All caps and too long")
            return False
        
        # Reject if text contains special characters (except hyphens/apostrophes in names)
        # Allow spaces, hyphens, and apostrophes
        if re.search(r'[^\w\s\-\']', text):
            # Check if it contains only allowed special chars
            clean_text = re.sub(r'[\w\s\-\']', '', text)
            if clean_text.strip():
                logger.debug(f"Rejected name '{text}': Contains invalid special characters")
                return False
        
        return True

    def extract_aadhaar_number(self, ocr_results):
        """
        Scans all OCR lines for a valid 12-digit Aadhaar number.
        Returns: (number_string, confidence_score)
        """
        best_candidate = None
        max_conf = 0.0
        best_is_valid = False

        # First, collect all digit sequences from OCR results
        all_digits = []
        for item in ocr_results:
            text = item['text']
            conf = item['conf']
            # Extract all digit sequences
            digit_seqs = re.findall(r'\d+', text)
            for seq in digit_seqs:
                all_digits.append((seq, conf))

        # Try to find 12-digit combination from collected sequences
        # Strategy 1: Look for direct match using Configured Patterns & Cleaning
        for item in ocr_results:
            text = item['text']
            conf = item['conf']
            
            # Apply robust cleaning (fixes B->8, O->0 etc)
            cleaned_text = self.clean_ocr_text(text)
            
            # Check against configured Regex Patterns (e.g. "1234 5678 9012")
            for pattern in AADHAAR_REGEX_PATTERNS:
                matches = re.findall(pattern, cleaned_text)
                for match in matches:
                     # Remove spaces for validation
                    clean_num = match.replace(" ", "")
                    if len(clean_num) == 12:
                         is_valid = self.validator.validate_verhoeff(clean_num)
                         if conf > max_conf:
                            max_conf = conf
                            best_candidate = clean_num
                            best_is_valid = is_valid

            # Fallback: Check for any 12-digit sequence in the cleaned line
            # This handles cases not strictly matching the regex (e.g. weird spacing)
            clean_digits_only = cleaned_text.replace(" ", "")
            matches = re.findall(r'\b\d{12}\b', clean_digits_only)
            for match in matches:
                 is_valid = self.validator.validate_verhoeff(match)
                 # Only update if we haven't found a stronger match or if this has higher confidence
                 if conf > max_conf: 
                        max_conf = conf
                        best_candidate = match
                        best_is_valid = is_valid

        # Strategy 2: Combine 4-digit + 8-digit sequences (with cleaning)
        for i, (seq1, conf1) in enumerate(all_digits):
            seq1_clean = self.clean_ocr_text(seq1) # Clean individual chunks too
            if len(seq1_clean) == 4 and seq1_clean.isdigit():
                for j, (seq2, conf2) in enumerate(all_digits):
                    if i != j:
                        seq2_clean = self.clean_ocr_text(seq2)
                        if len(seq2_clean) >= 8 and seq2_clean[:8].isdigit():
                            # Try combining: 4 digits + next 8 digits from seq2
                            combined = seq1_clean + seq2_clean[:8]
                            if len(combined) == 12:
                                is_valid = self.validator.validate_verhoeff(combined)
                                avg_conf = (conf1 + conf2) / 2
                                if avg_conf > max_conf:
                                    max_conf = avg_conf
                                    best_candidate = combined
                                    best_is_valid = is_valid

        # Strategy 3: Look for 6-digit + 6-digit patterns (with cleaning)
        for i, (seq1, conf1) in enumerate(all_digits):
            seq1_clean = self.clean_ocr_text(seq1)
            if len(seq1_clean) == 6 and seq1_clean.isdigit():
                for j, (seq2, conf2) in enumerate(all_digits):
                    if i != j:
                        seq2_clean = self.clean_ocr_text(seq2)
                        if len(seq2_clean) == 6 and seq2_clean.isdigit():
                            combined = seq1_clean + seq2_clean
                            is_valid = self.validator.validate_verhoeff(combined)
                            avg_conf = (conf1 + conf2) / 2
                            if avg_conf > max_conf:
                                max_conf = avg_conf
                                best_candidate = combined
                                best_is_valid = is_valid

        if best_candidate:
            formatted = f"{best_candidate[:4]} {best_candidate[4:8]} {best_candidate[8:]}"
            return formatted, max_conf
        
        return None, 0.0

    def extract_name(self, ocr_results, user_name_hint=None):
        """
        Extracts name based on anchors or fuzzy matching with user hint.
        Validates that extracted text is a valid person's name using is_valid_name().
        Returns: (name_string, confidence_score)
        """
        # Strategy 1: If user provided a name, look for it (fuzzy match)
        if user_name_hint:
            best_match = None
            best_score = 0
            best_conf = 0.0
            
            for item in ocr_results:
                text = item['text']
                conf = item.get('conf', 0.0)
                # First check if it's a valid name format
                if not self.is_valid_name(text):
                    continue
                passed, score = self.validator.fuzzy_match_name(text, user_name_hint)
                if passed and score > best_score:
                    best_score = score
                    best_match = text
                    best_conf = conf
            
            if best_match:
                return best_match, best_conf
        
        # Strategy 2: Heuristic / Spatial
        # Find "Government of India" / "Govt of India"
        # The line physically BELOW it is often the Name.
        
        govt_index = -1
        for i, item in enumerate(ocr_results):
            txt = item['text'].lower()
            if "govt" in txt or "government" in txt:
                govt_index = i
                break
        
        if govt_index != -1 and govt_index + 1 < len(ocr_results):
            # Candidate is the next line
            item = ocr_results[govt_index + 1]
            candidate = item['text']
            conf = item.get('conf', 0.0)
            
            # Use is_valid_name() for comprehensive validation
            # This replaces the old ad-hoc checks for dob/male/female
            if self.is_valid_name(candidate):
                logger.debug(f"Valid name found below government: '{candidate}'")
                return candidate, conf
            else:
                # Try looking at more lines below if immediate next line failed validation
                for offset in range(2, 5):  # Check next 3 lines
                    if govt_index + offset < len(ocr_results):
                        item = ocr_results[govt_index + offset]
                        next_candidate = item['text']
                        conf = item.get('conf', 0.0)
                        if self.is_valid_name(next_candidate):
                            logger.debug(f"Valid name found {offset} lines below government: '{next_candidate}'")
                            return next_candidate, conf
        
        # Strategy 3: Look for text above DOB/Gender anchors that looks like a name
        for i, item in enumerate(ocr_results):
            txt = item['text'].lower()
            if any(anchor in txt for anchor in ["dob", "year of birth", "gender", "male", "female"]):
                # Check the line above
                if i > 0:
                    item = ocr_results[i - 1]
                    candidate = item['text']
                    conf = item.get('conf', 0.0)
                    if self.is_valid_name(candidate):
                        logger.debug(f"Valid name found above DOB/gender: '{candidate}'")
                        return candidate, conf
        
        # Strategy 4: Scan all OCR results for any valid name-like text
        # This is a fallback for documents where anchor-based extraction failed
        for item in ocr_results:
            text = item['text']
            conf = item.get('conf', 0.0)
            if self.is_valid_name(text):
                # Additional check: ensure it has at least one capital letter (proper noun indicator)
                has_capital = any(c.isupper() for c in text)
                if has_capital:
                    logger.debug(f"Valid name found via scan: '{text}'")
                    return text, conf
        
        return "", 0.0
