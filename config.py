import re

# Configuration Constants

# OCR Settings
OCR_LANG = 'en' # PaddleOCR lang code. Using 'en' often works better than 'devanagari' for mixing unless specifically trained.
# For Tesseract, we might want 'eng+hin'

# Preprocessing Thresholds
MIN_DPI = 300
GAUSSIAN_KERNEL = (3, 3)
ADAPTIVE_BLOCK_SIZE = 11
ADAPTIVE_C = 2

# Aadhaar Validation
AADHAAR_REGEX_PATTERNS = [
    r'\b\d{4}\s\d{4}\s\d{4}\b',  # 1234 5678 9012
    r'\b\d{12}\b'                # 123456789012
]

# Common OCR Confusions Map
OCR_CORRECTIONS = {
    'O': '0', 'D': '0', 'Q': '0', 'o': '0',
    'I': '1', 'l': '1', '|': '1', 'i': '1', 'L': '1',
    'S': '5', 's': '5',
    'B': '8',
    'Z': '2',
    'A': '4'
}

# Name Anchors
NAME_ANCHORS_TOP = ["Government of India", "Govt of India", "Male", "Female", "DOB", "Year of Birth", "Address"]
# Note: Usually Name is *below* Govt of India, or *above* DOB/Gender.

# Non-Name Words - Words that should never be considered as a person's name
# This prevents the model from extracting document labels/headers as names
NON_NAME_WORDS = [
    # Government/Document labels
    "government", "govt", "government of india", "govt of india",
    "uidai", "unique identification authority of india",
    "aadhaar", "aadhar", "adhar",
    "address", "dob", "date of birth", "year of birth",
    "gender", "male", "female", "m", "f",
    # Common document text
    "card", "number", "no", "hindi", "english",
    "name", "father's", "father", "mother's", "mother",
    "house", "street", "road", "post", "village",
    "district", "state", "pin", "pincode", "postcode",
    # OCR artifacts/common false positives
    "photo", "signature", "sign", "handwriting",
    "digitally", "signed", "certificate",
    "temporary", "permanent", "residential",
    # Common single words that aren't names
    "the", "and", "for", "with", "from",
    # Common technical terms
    "sector", "block", "sub", "ward",
    "area", "locality", "landline", "mobile",
    # Administrative terms
    "office", "department", "ministry", "civil",
    "verification", "verified", "authentic",
    # Additional document terms
    "issue", "date", "valid", "official",
    "identification", "identity", "resident",
    "citizen", "citizenship", "nationality"
]

# Verification Thresholds
NAME_MATCH_THRESHOLD = 80  # Fuzzy match score out of 100
AADHAAR_CONFIDENCE_THRESHOLD = 60 # Minimum OCR confidence to consider valid candidates
