"""Entity extraction module for Hospital AI Assistant.

Extracts symptoms, body parts, clinical departments, doctors, facilities,
hospital name, and query language without hallucinating medical diagnoses.
"""
import re
from typing import Any, Dict, List, Optional, Tuple

from app.core.logging_config import logger
from app.retrieval.entity_store import HospitalEntityStore


class EntityExtractor:
    """Extracts clinical and conversational entities from user inquiries."""

    # Common body parts in English, Malayalam script, and Manglish
    BODY_PARTS: Dict[str, List[str]] = {
        "leg": ["leg", "legs", "കാൽ", "കാലിൽ", "കാലിനു", "കാലിന്", "കാലിലൊരു", "കാലുകൾ", "kaal", "kaalil", "kaalinu", "kalil", "kaalu"],
        "calf": ["calf", "calves", "കണംകാൽ", "കണങ്കാൽ", "kanamkaal"],
        "knee": ["knee", "knees", "മുട്ട്", "മുട്ടിൽ", "മുട്ടുകൾ", "മുട്ടിന്", "muttu", "muttil"],
        "foot": ["foot", "feet", "പാദം", "പാദത്തിൽ", "paadam"],
        "ankle": ["ankle", "ankles", "കണങ്കാൽ"],
        "chest": ["chest", "നെഞ്ച്", "നെഞ്ചിൽ", "നെഞ്ചിനു", "നെഞ്ചിന്", "nenju", "nenjil", "nenjinu"],
        "heart": ["heart", "ഹൃദയം", "ഹൃദയത്തിൽ", "hridayam"],
        "head": ["head", "തല", "തലയിൽ", "തലയ്ക്ക്", "തലയ്ക്കു", "thala", "thalayil", "thalakku"],
        "brain": ["brain", "തലച്ചോറ്", "thalamunda"],
        "stomach": ["stomach", "abdomen", "belly", "വയർ", "വയറ്റിൽ", "വയറിനു", "വയറിന്", "വയറു", "vayar", "vayattil", "vayarinu"],
        "eye": ["eye", "eyes", "കണ്ണ്", "കണ്ണിൽ", "കണ്ണുകൾ", "കണ്ണിന്", "kannu", "kannil", "kanninu"],
        "ear": ["ear", "ears", "ചെവി", "ചെവിയിൽ", "ചെവിക്ക്", "chevi", "cheviyil", "cheviku"],
        "throat": ["throat", "തൊണ്ട", "തൊണ്ടയിൽ", "തൊണ്ടയ്ക്ക്", "thonda", "thondayil", "thondakku"],
        "back": ["back", "spine", "നടു", "നടുവിൽ", "നടുവിന്", "മുതുക്", "nadu", "naduvil", "naduvedana"],
        "skin": ["skin", "ചർമ്മം", "തൊലി", "tholi", "charmam"],
        "tooth": ["tooth", "teeth", "പല്ല്", "പല്ലിൽ", "pallu", "pallil"],
        "kidney": ["kidney", "kidneys", "വൃക്ക", "വൃക്കയിൽ", "കിഡ്നി", "vrigga"],
        "bone": ["bone", "bones", "എല്ല്", "എല്ലിൽ", "ellu", "ellil"],
        "joint": ["joint", "joints", "സന്ധി", "സന്ധികൾ", "sandhi"],
    }

    # Pain & Hurt trigger tokens in English, Malayalam script, and Manglish
    PAIN_TRIGGERS: List[str] = [
        "pain", "hurting", "hurt", "hurts", "ache", "aching", "sore", "soreness",
        "വേദന", "വേദനയുണ്ട്", "വേദനയാണ്", "വേദനിക്കുന്നു", "വേദനയ്ക്ക്", "വേദനക്ക്",
        "vedana", "vedhana", "vedanayaanu", "vedanayanu", "vedanikkunnu",
    ]

    # Specific symptom phrases mapped directly
    COMMON_SYMPTOMS: Dict[str, List[str]] = {
        "leg pain": [
            "leg pain", "pain in leg", "pain in my leg", "legs pain",
            "കാലിൽ വേദന", "കാലിനു വേദന", "കാലിന് വേദന", "കാൽ വേദന", "കാലിലൊരു വേദന",
            "കാലിൽ വേദനയുണ്ട്", "കാലിനു വേദനയാണ്", "കാലിന് വേദനയാണ്", "കാലിലൊരു വേദനയുണ്ട്",
            "kaalil pain", "kaalinu pain", "kaal pain", "kaalil vedana", "kaalinu vedana",
            "enik kaalil pain", "enikku kaalil pain",
        ],
        "calf pain": [
            "calf pain", "pain in calf", "pain in my calf", "calves pain",
            "കണംകാൽ വേദന", "കണങ്കാൽ വേദന", "calf pain und",
        ],
        "chest pain": [
            "chest pain", "pain in chest", "pain in my chest",
            "നെഞ്ചുവേദന", "നെഞ്ചിൽ വേദന", "നെഞ്ചുവേദനയുണ്ട്", "നെഞ്ചുവേദനയാണ്",
            "nenju vedana", "nenjil pain", "nenju pain",
        ],
        "headache": [
            "headache", "head ache", "head pain", "pain in head", "migraine",
            "തലവേദന", "തലയിൽ വേദന", "തലവേദനയുണ്ട്", "തലവേദനയാണ്",
            "thala vedana", "thalavedana", "thala pain", "thalavali", "thala vali", "thalavaly",
        ],
        "stomach pain": [
            "stomach pain", "stomach ache", "belly pain", "abdominal pain", "tummy ache",
            "വയറുവേദന", "വയറ്റിൽ വേദന", "വയറുവേദനയുണ്ട്", "വയറുവേദനയാണ്", "വയറിനു വേദന", "വയറിന് വേദന",
            "vayar vedana", "vayar pain", "vayattil pain",
        ],
        "knee pain": [
            "knee pain", "pain in knee", "knees pain",
            "മുട്ടുവേദന", "മുട്ടിൽ വേദന", "മുട്ടുവേദനയുണ്ട്", "മുട്ടുവേദനയാണ്",
            "muttu vedana", "muttil pain",
        ],
        "back pain": [
            "back pain", "backache", "lower back pain", "spine pain",
            "നടുവേദന", "നടുവിന് വേദന", "നടുവേദനയുണ്ട്", "നടുവേദനയാണ്",
            "nadu vedana", "nadu pain",
        ],
        "eye pain": [
            "eye pain", "pain in eye", "eye irritation",
            "കണ്ണുവേദന", "കണ്ണിൽ വേദന", "കണ്ണുവേദനയുണ്ട്",
            "kannu vedana", "kannil pain",
        ],
        "ear pain": [
            "ear pain", "ear ache", "pain in ear",
            "ചെവിവേദന", "ചെവിയിൽ വേദന", "ചെവിവേദനയുണ്ട്",
            "chevi vedana", "chevi pain",
        ],
        "fever": ["fever", "high temperature", "പനി", "പനിയുണ്ട്", "pani", "fever und"],
        "cough": ["cough", "cold", "ചുമ", "ചുമയുണ്ട്", "chuma", "chuma und"],
    }

    # Department affinity mapping from symptoms/body parts
    # (Hospital department routing without diagnosing medical condition)
    BODY_PART_TO_DEPARTMENT: Dict[str, str] = {
        "leg": "Orthopaedics & Trauma Care",
        "calf": "Orthopaedics & Trauma Care",
        "knee": "Orthopaedics & Trauma Care",
        "foot": "Orthopaedics & Trauma Care",
        "ankle": "Orthopaedics & Trauma Care",
        "bone": "Orthopaedics & Trauma Care",
        "joint": "Orthopaedics & Trauma Care",
        "back": "Orthopaedics & Trauma Care",
        "chest": "Cardiology",
        "heart": "Cardiology",
        "head": "Neurology",
        "brain": "Neurology",
        "skin": "Dermatology",
        "eye": "Ophthalmology",
        "ear": "ENT (Ear, Nose & Throat)",
        "throat": "ENT (Ear, Nose & Throat)",
        "tooth": "Maxillofacial & Dental Services",
        "kidney": "Nephrology, Toxicology & Dialysis",
        "stomach": "General Medicine",
    }

    def __init__(self, entity_store: Optional[HospitalEntityStore] = None):
        self.entity_store = entity_store or HospitalEntityStore()

    def detect_language(self, text: str) -> str:
        """Identifies language: Malayalam, Hindi, Manglish, or English."""
        if not text:
            return "English"

        # Check for Malayalam Unicode range (\u0D00 - \u0D7F)
        if re.search(r"[\u0d00-\u0d7f]", text):
            return "Malayalam"

        # Check for Devanagari Unicode range (\u0900 - \u097F)
        if re.search(r"[\u0900-\u097f]", text):
            return "Hindi"

        # Check for Manglish phonetic markers in latin text
        text_lower = text.lower()
        manglish_markers = [
            r"\benik\b", r"\benikku\b", r"\benikk\b", r"\bente\b",
            r"\bkaalil\b", r"\bkaalinu\b", r"\bkaal\b",
            r"\bpain und\b", r"\bpain aanu\b", r"\bpain anu\b", r"\bpain aane\b",
            r"\bund\b", r"\baanu\b", r"\baane\b",
            r"\bdoctor aar\b", r"\baar doctor\b", r"\baarodu\b", r"\baare\b",
            r"\bkaanikkanam\b", r"\bkaananam\b", r"\bkanikkanam\b",
            r"\beth department\b", r"\bethu department\b",
            r"\bvedana\b", r"\bvedhana\b",
        ]
        if any(re.search(pat, text_lower) for pat in manglish_markers):
            return "Manglish"

        return "English"

    def extract_symptom_and_body_part(self, text: str) -> Tuple[Optional[str], Optional[str]]:
        """Extracts symptom and body part without diagnosing or inventing diseases."""
        if not text:
            return None, None

        text_lower = text.lower().strip()

        # 1. Match predefined exact symptom phrases
        for sym_name, patterns in self.COMMON_SYMPTOMS.items():
            for pat in patterns:
                if pat.lower() in text_lower or re.search(r"\b" + re.escape(pat.lower()) + r"\b", text_lower):
                    # Identify matching body part
                    body_part = None
                    for bp_name in self.BODY_PARTS:
                        if bp_name in sym_name:
                            body_part = bp_name
                            break
                    return sym_name, body_part

        # 2. Extract body part
        matched_bp: Optional[str] = None
        for bp_name, aliases in self.BODY_PARTS.items():
            for alias in aliases:
                # Check Malayalam / unicode or word boundary for latin
                if re.search(r"[\u0d00-\u0d7f]", alias):
                    if alias in text_lower:
                        matched_bp = bp_name
                        break
                else:
                    if re.search(r"\b" + re.escape(alias) + r"\b", text_lower):
                        matched_bp = bp_name
                        break
            if matched_bp:
                break

        # 3. Check for pain / hurt / discomfort attached to body part
        has_pain = False
        for trigger in self.PAIN_TRIGGERS:
            if re.search(r"[\u0d00-\u0d7f]", trigger):
                if trigger in text_lower:
                    has_pain = True
                    break
            else:
                if re.search(r"\b" + re.escape(trigger) + r"\b", text_lower):
                    has_pain = True
                    break

        if matched_bp and has_pain:
            symptom = f"{matched_bp} pain"
            return symptom, matched_bp
        elif matched_bp:
            return None, matched_bp
        elif has_pain:
            return "pain", None

        return None, None

    def infer_department_from_symptom(self, symptom: Optional[str], body_part: Optional[str]) -> Optional[str]:
        """Maps symptom/body_part to appropriate hospital department without diagnosing."""
        if body_part and body_part in self.BODY_PART_TO_DEPARTMENT:
            return self.BODY_PART_TO_DEPARTMENT[body_part]
        if symptom:
            for bp, dept in self.BODY_PART_TO_DEPARTMENT.items():
                if bp in symptom.lower():
                    return dept
        return None

    def extract_all(self, text: str) -> Dict[str, Any]:
        """Extracts complete dictionary of entities: symptom, body_part, doctor, department, hospital, language."""
        clean_text = (text or "").strip()
        lang = self.detect_language(clean_text)
        symptom, body_part = self.extract_symptom_and_body_part(clean_text)

        # Department can be explicit or inferred from symptom/body_part
        inferred_dept = self.infer_department_from_symptom(symptom, body_part)

        return {
            "symptom": symptom,
            "body_part": body_part,
            "hospital": "P.S. Mission Hospital",
            "language": lang,
            "inferred_department": inferred_dept,
        }
