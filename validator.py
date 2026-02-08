from rapidfuzz import fuzz

class Validator:
    def __init__(self):
        # Verhoeff Algorithm Tables
        self.d = [
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
            [1, 2, 3, 4, 0, 6, 7, 8, 9, 5],
            [2, 3, 4, 0, 1, 7, 8, 9, 5, 6],
            [3, 4, 0, 1, 2, 8, 9, 5, 6, 7],
            [4, 0, 1, 2, 3, 9, 5, 6, 7, 8],
            [5, 9, 8, 7, 6, 0, 4, 3, 2, 1],
            [6, 5, 9, 8, 7, 1, 0, 4, 3, 2],
            [7, 6, 5, 9, 8, 2, 1, 0, 4, 3],
            [8, 7, 6, 5, 9, 3, 2, 1, 0, 4],
            [9, 8, 7, 6, 5, 4, 3, 2, 1, 0]
        ]
        self.p = [
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
            [1, 5, 7, 6, 2, 8, 3, 0, 9, 4],
            [5, 8, 0, 3, 7, 9, 6, 1, 4, 2],
            [8, 9, 1, 6, 0, 4, 3, 5, 2, 7],
            [9, 4, 5, 3, 1, 2, 6, 8, 7, 0],
            [4, 2, 8, 6, 5, 7, 3, 9, 0, 1],
            [2, 7, 9, 3, 8, 0, 6, 4, 1, 5],
            [7, 0, 4, 6, 9, 1, 3, 2, 5, 8]
        ]
        self.inv = [0, 4, 3, 2, 1, 5, 6, 7, 8, 9]

    def validate_verhoeff(self, number_str):
        """
        Validates the Verhoeff checksum of a number string.
        Aadhaar uses Verhoeff algorithm.
        """
        if not number_str.isdigit():
            return False
        
        # Aadhaar is 12 digits
        if len(number_str) != 12:
            return False

        c = 0
        reversed_num = map(int, reversed(number_str))
        for i, n in enumerate(reversed_num):
            c = self.d[c][self.p[i % 8][n]]
        
        return c == 0

    def generate_verhoeff_checksum(self, number_str):
        """
        Generates the Verhoeff checksum digit for a given number.
        Uses self.inv table which was previously unused.
        """
        if not number_str.isdigit():
            raise ValueError("Input must be digits")
        
        c = 0
        reversed_num = map(int, reversed(number_str))
        for i, n in enumerate(reversed_num):
            c = self.d[c][self.p[(i + 1) % 8][n]]
            
        return self.inv[c]

    def fuzzy_match_name(self, extracted, target, threshold=80):
        """
        Compares extracted name with target name using fuzzy matching.
        Uses Token Sort Ratio to handle name reordering (Sujit Kumar vs Kumar Sujit).
        """
        if not extracted or not target:
            return False, 0.0

        score = fuzz.token_sort_ratio(extracted.lower(), target.lower())
        return score >= threshold, score

    def exact_match_aadhaar(self, extracted, target):
        """
        Compares processed aadhaar number (digits only) with target.
        """
        clean_ext = "".join(filter(str.isdigit, extracted))
        clean_target = "".join(filter(str.isdigit, target))
        return clean_ext == clean_target
