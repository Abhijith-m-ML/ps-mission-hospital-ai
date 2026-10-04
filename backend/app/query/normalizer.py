"""Query Normalizer for Hospital AI Assistant.

Normalizes whitespace, punctuation, common typos, chat abbreviations, and clinical
specialty misspellings while safely preserving medical titles, degrees, and doctor names.
"""
import re
from typing import Dict, List, Pattern, Tuple


class Normalizer:
    """Preprocesses and normalizes raw patient inquiries."""

    # Common chat and informal word contractions / misspellings
    COMMON_TYPO_MAP: Dict[str, str] = {
        "wat": "what",
        "wht": "what",
        "wats": "what is",
        "whats": "what is",
        "r": "are",
        "u": "you",
        "ur": "your",
        "urs": "yours",
        "pls": "please",
        "plz": "please",
        "bcoz": "because",
        "coz": "because",
        "abt": "about",
        "hsptl": "hospital",
        "hosp": "hospital",
        "docter": "doctor",
        "docters": "doctors",
        "doctrs": "doctors",
        "doctr": "doctor",
        "doc": "doctor",
        "docs": "doctors",
        "timin": "timing",
        "timings": "timings",
        "timngs": "timings",
        "shud": "should",
        "cud": "could",
        "wud": "would",
        "availble": "available",
        "availabel": "available",
        "avialable": "available",
        "avail": "available",
        "apointment": "appointment",
        "appoinment": "appointment",
        "appnt": "appointment",
        "scheldue": "schedule",
        "schdule": "schedule",
        "emrgency": "emergency",
        "emergancy": "emergency",
    }

    # Clinical and department misspellings / short names mapped to clean canonical names
    DEPARTMENT_TYPO_MAP: Dict[str, str] = {
        "cardiolgy": "Cardiology",
        "cardio": "Cardiology",
        "cardilogy": "Cardiology",
        "cardiologst": "Cardiology",
        "neurolgy": "Neurology",
        "neuro": "Neurology",
        "neurolgist": "Neurology",
        "paediatrics": "Paediatrics & Neonatology",
        "pediatrics": "Paediatrics & Neonatology",
        "paediatric": "Paediatrics & Neonatology",
        "pediatric": "Paediatrics & Neonatology",
        "pedia": "Paediatrics & Neonatology",
        "orthopaedics": "Orthopaedics & Trauma Care",
        "orthopedics": "Orthopaedics & Trauma Care",
        "orthopaedic": "Orthopaedics & Trauma Care",
        "orthopedic": "Orthopaedics & Trauma Care",
        "ortho": "Orthopaedics & Trauma Care",
        "gynaecology": "Obstetrics & Gynaecology",
        "gynecology": "Obstetrics & Gynaecology",
        "gynaec": "Obstetrics & Gynaecology",
        "gynec": "Obstetrics & Gynaecology",
        "obgyn": "Obstetrics & Gynaecology",
        "dermatolgy": "Dermatology",
        "derma": "Dermatology",
        "nephrolgy": "Nephrology, Toxicology & Dialysis",
        "nephro": "Nephrology, Toxicology & Dialysis",
        "opthalmology": "Ophthalmology",
        "opthal": "Ophthalmology",
        "psychiatry": "Psychiatry & Behavioral Sciences",
        "psych": "Psychiatry & Behavioral Sciences",
        "pulmonolgy": "Pulmonology",
        "pulmo": "Pulmonology",
    }

    # Common lay terms for clinical specialties and medical conditions
    LAY_MEDICAL_TERMS: Dict[str, str] = {
        "heart doctor": "Cardiology",
        "kidney doctor": "Nephrology",
        "skin doctor": "Dermatology",
        "brain doctor": "Neurology",
        "bone doctor": "Orthopaedics",
        "children doctor": "Paediatrics",
        "child doctor": "Paediatrics",
        "kids doctor": "Paediatrics",
        "eye doctor": "Ophthalmology",
        "bp": "blood pressure",
    }

    # Medical titles and honorifics that must NOT be aggressively lowercased or stripped
    PROTECTED_HONORIFICS = [
        "Dr.", "Dr", "Sr.", "Sr", "Prof.", "Prof", "Fr.", "Fr",
        "MBBS", "MD", "MS", "DM", "DNB", "MCh", "FRCS", "MRCP",
        "FICC", "FESC", "DA", "DCH", "DGO", "ICU", "NICU", "PICU",
        "OPD", "OP", "ECG", "EEG", "TMT", "ENT", "MRI", "CT"
    ]

    def __init__(self):
        # Precompile regex replacements with word boundaries
        self._compiled_typos: List[Tuple[Pattern, str]] = [
            (re.compile(r"\b" + re.escape(k) + r"\b", re.IGNORECASE), v)
            for k, v in self.COMMON_TYPO_MAP.items()
        ]
        self._compiled_lay_terms: List[Tuple[Pattern, str]] = [
            (re.compile(r"\b" + re.escape(k) + r"\b", re.IGNORECASE), v)
            for k, v in self.LAY_MEDICAL_TERMS.items()
        ]
        self._compiled_dept_typos: List[Tuple[Pattern, str]] = [
            (re.compile(r"\b" + re.escape(k) + r"\b", re.IGNORECASE), v)
            for k, v in self.DEPARTMENT_TYPO_MAP.items()
        ]

    def normalize(self, text: str) -> str:
        """Runs the complete normalization pipeline."""
        if not text:
            return ""

        # 1. Clean unicode whitespace and collapse multiple spaces
        cleaned = re.sub(r"[\r\n\t]+", " ", text)
        cleaned = re.sub(r"\s+", " ", cleaned).strip()

        # 2. Normalize punctuation: curly quotes to straight quotes, repeated question marks
        cleaned = cleaned.replace("“", '"').replace("”", '"').replace("’", "'").replace("‘", "'")
        cleaned = re.sub(r"\s+([?!.,;:])", r"\1", cleaned)
        cleaned = re.sub(r"\?{2,}", "?", cleaned)
        cleaned = re.sub(r"!{2,}", "!", cleaned)
        cleaned = re.sub(r"\s+([?!.,;:])", r"\1", cleaned)

        # 3. Handle common chat contractions and typos
        for pattern, replacement in self._compiled_typos:
            cleaned = pattern.sub(replacement, cleaned)

        # 4. Handle lay medical terms (e.g. bone doctor -> Orthopaedics)
        for pattern, replacement in self._compiled_lay_terms:
            cleaned = pattern.sub(replacement, cleaned)

        # 5. Handle department typos (e.g. cardiolgy -> Cardiology)
        for pattern, replacement in self._compiled_dept_typos:
            cleaned = pattern.sub(replacement, cleaned)

        # 5. Fix capitalization at beginning of query if lowercase
        if cleaned and cleaned[0].islower():
            cleaned = cleaned[0].upper() + cleaned[1:]

        # 6. Ensure question mark if clearly a question and ends with no punctuation
        q_words = ("what", "who", "whom", "where", "when", "why", "how", "which", "is", "are", "can", "do", "does")
        if (
            cleaned.lower().startswith(q_words)
            and not cleaned.endswith(("?", ".", "!"))
        ):
            cleaned += "?"

        return cleaned
