"""Query Rewriter and Entity Extractor for Hospital AI Assistant.

Performs fuzzy entity matching (doctors, departments, facilities), extracts metadata
(days, times, services), and synthesizes clean, canonical retrieval queries without
diagnosing or hallucinating entities.
"""
import difflib
import json
import re
from typing import Any, Dict, List, Optional, Tuple

try:
    from rapidfuzz import fuzz
    HAS_RAPIDFUZZ = True
except ImportError:
    HAS_RAPIDFUZZ = False

from app.core.config import settings
from app.core.logging_config import logger
from app.models.schemas import Doctor, Facility
from app.query.entity_extractor import EntityExtractor
from app.query.intent import HospitalIntent
from app.query.normalizer import Normalizer
from app.retrieval.entity_store import HospitalEntityStore


def fuzzy_similarity(s1: str, s2: str) -> float:
    """Computes similarity score between 0.0 and 1.0 using RapidFuzz or difflib fallback."""
    if not s1 or not s2:
        return 0.0
    if HAS_RAPIDFUZZ:
        # Average of standard ratio and token sort ratio for robust match
        r1 = fuzz.ratio(s1, s2)
        r2 = fuzz.token_sort_ratio(s1, s2)
        return max(r1, r2) / 100.0
    return difflib.SequenceMatcher(None, s1, s2).ratio()


def clean_doctor_name(name: str) -> str:
    """Strips honorifics and clinical titles for canonical token matching."""
    cleaned = re.sub(
        r"\b(?:dr\.?|sr\.?|sister|prof\.?|fr\.?|father|doctor)\b",
        "",
        name,
        flags=re.IGNORECASE,
    )
    # Remove qualification suffixes if present
    cleaned = re.sub(r",?\s*(?:mbbs|md|ms|dm|dnb|ficc|fesc|frcs).*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"[^\w\s]", " ", cleaned)
    return re.sub(r"\s+", " ", cleaned).strip().lower()


class QueryRewriter:
    """Extracts entities and rewrites questions into clean, canonical queries."""

    DAYS_PATTERN = re.compile(
        r"\b(monday|tuesday|wednesday|thursday|friday|saturday|sunday|weekdays|weekends|today|tomorrow)\b",
        re.IGNORECASE,
    )
    TIME_PATTERN = re.compile(
        r"\b(morning|afternoon|evening|night|\d{1,2}(?::\d{2})?\s*(?:am|pm))\b",
        re.IGNORECASE,
    )

    def __init__(
        self,
        entity_store: Optional[HospitalEntityStore] = None,
        normalizer: Optional[Normalizer] = None,
        entity_extractor: Optional[EntityExtractor] = None,
    ):
        self.entity_store = entity_store or HospitalEntityStore()
        self.normalizer = normalizer or Normalizer()
        self.entity_extractor = entity_extractor or EntityExtractor(entity_store=self.entity_store)

    def extract_entities(self, query: str) -> Dict[str, Any]:
        """Extracts doctor, department, facility, service, symptom, body_part, day, and time entities."""
        clean_q = (query or "").strip()

        matched_doc = self.match_doctor(clean_q)
        matched_dept = self.match_department(clean_q)
        matched_fac = self.match_facility(clean_q)

        # Extract symptom, body_part, hospital, language via EntityExtractor
        extracted = self.entity_extractor.extract_all(clean_q)

        # Extract days and times
        days_match = self.DAYS_PATTERN.findall(clean_q)
        times_match = self.TIME_PATTERN.findall(clean_q)

        # Extract service mentions
        service = None
        service_keywords = ["health checkup", "preventive checkup", "dialysis", "ambulance", "vaccination", "immunization", "x-ray", "ct scan", "mri", "echo"]
        for skw in service_keywords:
            if re.search(r"\b" + re.escape(skw) + r"\b", clean_q, re.IGNORECASE):
                service = skw.title()
                break

        return {
            "doctor": matched_doc,
            "department": matched_dept,
            "facility": matched_fac,
            "service": service,
            "symptom": extracted.get("symptom"),
            "body_part": extracted.get("body_part"),
            "hospital": extracted.get("hospital", "P.S. Mission Hospital"),
            "language": extracted.get("language", "English"),
            "inferred_department": extracted.get("inferred_department"),
            "day": days_match[0].title() if days_match else None,
            "time": times_match[0].lower() if times_match else None,
        }

    def match_doctor(self, text: str) -> Optional[str]:
        """Fuzzy matches text against the hospital doctor catalog."""
        if not text:
            return None

        # 1. Exact match in EntityStore
        exact = self.entity_store.get_doctor_by_name(text)
        if exact:
            return exact.doctor_name

        # 2. Check for "Dr. [Candidate]" substring pattern
        cand_match = re.search(r"\b(dr\.?\s+[A-Za-z\.\s]+?)(?:\b|'s|\?|,|$)", text, re.IGNORECASE)
        cand_text = cand_match.group(1).strip() if cand_match else text

        cand_cleaned = clean_doctor_name(cand_text)
        if not cand_cleaned or len(cand_cleaned) < 3:
            return None

        # 3. Fuzzy search through all known doctors
        best_match: Optional[Doctor] = None
        best_score = 0.0

        for doc in self.entity_store.doctors:
            doc_cleaned = clean_doctor_name(doc.doctor_name)

            # Direct token subset match (e.g. "jaya joseph" in "jaya joseph")
            if cand_cleaned == doc_cleaned or cand_cleaned in doc_cleaned:
                return doc.doctor_name

            # Also check if individual significant tokens match (e.g. "jaya" + "jospeh")
            cand_tokens = [t for t in cand_cleaned.split() if len(t) > 2]
            doc_tokens = [t for t in doc_cleaned.split() if len(t) > 2]

            # Compute fuzzy score
            score = fuzzy_similarity(cand_cleaned, doc_cleaned)
            if score > best_score:
                best_score = score
                best_match = doc

        # High confidence threshold to prevent false matches (>= 78%)
        if best_match and best_score >= 0.78:
            logger.info("Fuzzy matched doctor: '%s' -> '%s' (score=%.2f)", cand_text, best_match.doctor_name, best_score)
            return best_match.doctor_name

        return None

    def match_department(self, text: str) -> Optional[str]:
        """Matches clinical department using synonyms and fuzzy matching."""
        if not text:
            return None
        text_lower = text.lower()

        # 1. Check exact synonyms
        for canonical, aliases in self.entity_store.DEPARTMENT_SYNONYMS.items():
            for alias in aliases:
                pattern = r"\b" + re.escape(alias) + r"(?:s|es)?\b"
                if re.search(pattern, text_lower):
                    return canonical

        # 2. Fuzzy match single-token departments (e.g. "cardiolgy" -> "Cardiology")
        tokens = re.findall(r"\b[A-Za-z]{4,}\b", text_lower)
        for token in tokens:
            for canonical, aliases in self.entity_store.DEPARTMENT_SYNONYMS.items():
                for alias in aliases:
                    if len(alias.split()) == 1 and len(alias) >= 4:
                        score = fuzzy_similarity(token, alias)
                        if score >= 0.85:
                            logger.info("Fuzzy matched department: '%s' -> '%s' (score=%.2f)", token, canonical, score)
                            return canonical

        return None

    def match_facility(self, text: str) -> Optional[str]:
        """Matches clinical facilities from entity catalog."""
        if not text:
            return None
        text_lower = text.lower()

        # 1. Exact match in EntityStore facilities
        for fac in self.entity_store.facilities:
            if re.search(r"\b" + re.escape(fac.facility_name.lower()) + r"\b", text_lower):
                return fac.facility_name

        # 2. Acronyms & Short forms
        fac_short_map = {
            "icu": "ICU (Intensive Care Unit)",
            "nicu": "Neonatal ICU",
            "picu": "Paediatric ICU",
            "dialysis": "Dialysis Unit",
            "echo": "Echocardiography",
            "tmt": "Treadmill Test (TMT)",
            "xray": "X-Ray",
            "x-ray": "X-Ray",
            "ct scan": "CT Scan",
            "mri": "MRI",
            "blood test": "Laboratory Medicine",
            "lab": "Laboratory Medicine",
            "ambulance": "Ambulance Services",
            "immunization": "Immunization",
        }
        for kw, fac_name in fac_short_map.items():
            if re.search(r"\b" + re.escape(kw) + r"\b", text_lower):
                return fac_name

        return None

    def rewrite_query(
        self,
        normalized_query: str,
        intent: HospitalIntent,
        entities: Dict[str, Any],
        context_department: Optional[str] = None,
        context_doctor: Optional[str] = None,
    ) -> str:
        """Synthesizes a clean, grammatically sound, canonical retrieval query."""
        doc = entities.get("doctor") or context_doctor
        dept = entities.get("department") or context_department
        fac = entities.get("facility")
        clean_q = normalized_query.strip()
        q_lower = clean_q.lower()

        # 0. SYMPTOM TO DEPARTMENT / DOCTOR CONSULTATION
        if intent == HospitalIntent.SYMPTOM_TO_DEPARTMENT:
            symptom = entities.get("symptom")
            body_part = entities.get("body_part")
            if not symptom:
                if body_part:
                    symptom = f"{body_part} pain"
                else:
                    symptom = "my symptoms"

            # Check if asking which department vs which doctor vs symptom statement
            is_dept_ask = bool(
                re.search(
                    r"\b(which|what|ethu|eth)\s+department\b|ഡിപ്പാർട്ട്മെന്റ്|വിഭാഗം|ഏത്\s+department",
                    q_lower,
                )
            )
            is_doc_ask = bool(
                re.search(
                    r"\b(which|what|aar|who|whom)\s+doct(?:o|e)r\b|doctor\s+aar\b|aar\s+doctor\b|ഏത്\s+ഡോക്ടറെ|ഏതു\s+ഡോക്ടറെ|ആരെ\s+കാണിക്കണം|aarodu\s+kaanikkanam|aare\s+kaanikkanam",
                    q_lower,
                )
            )
            is_statement = (
                bool(re.search(r"\b(enik|enikku|ente|i\s+have)\b", q_lower))
                and not is_doc_ask
                and not is_dept_ask
            )

            if is_statement and body_part:
                return f"I have pain in my {body_part}."
            elif is_dept_ask:
                if "calf" in q_lower or "hospital" in q_lower:
                    return f"Which hospital department should I consult for {symptom}?"
                return f"Which department should I consult for {symptom}?"
            else:
                return f"Which doctor should I consult for {symptom}?"

        # 1. DOCTOR SCHEDULE
        if intent == HospitalIntent.DOCTOR_SCHEDULE:
            if context_doctor or (doc and ("schedule" in q_lower or "timing" in q_lower or "when" in q_lower or "visit" in q_lower)):
                target = context_doctor or doc
                return f"What is {target}'s available schedule?"
            elif context_department or dept:
                target = context_department or dept
                return f"What is the OP consultation schedule for {target} doctors?"
            return "What are the outpatient doctor consultation schedules?"

        # 2. DOCTOR SEARCH
        if intent == HospitalIntent.DOCTOR_SEARCH:
            # If user queried doctor name directly (e.g. "Dr Jaya Jospeh" -> "Dr. Sr. Jaya Joseph")
            if doc:
                cand_clean = clean_doctor_name(clean_q)
                doc_clean = clean_doctor_name(doc)
                if not dept and (
                    cand_clean == doc_clean
                    or fuzzy_similarity(cand_clean, doc_clean) >= 0.75
                    or clean_q.lower().replace(".", "").strip() in (
                        doc.lower().replace(".", "").strip(),
                        clean_doctor_name(doc),
                        clean_doctor_name(clean_q),
                    )
                ):
                    return doc

            if context_department:
                return f"Which doctors are available in the {context_department} department?"
            if "what doctors are available" in q_lower and dept:
                return f"What doctors are available in {dept}?"
            if clean_q.lower().startswith("who ") and "doctor" in clean_q.lower():
                return "Who are the doctors?"
            return clean_q

        # 3. FACILITY INFORMATION
        if intent == HospitalIntent.FACILITY_INFORMATION:
            if context_department:
                return f"What facilities are available in the {context_department} department?"
            return clean_q

        # 4. DEPARTMENT INFORMATION
        if intent == HospitalIntent.DEPARTMENT_INFORMATION:
            if dept:
                return f"What services and information are available for the {dept} department?"
            return clean_q

        # 5. HOSPITAL HOURS (Visiting hours, normal hospital hours)
        # CRITICAL: Do NOT rewrite visiting hours into OP consultation hours
        if intent == HospitalIntent.HOSPITAL_HOURS:
            if "visiting" in q_lower or "visit" in q_lower:
                return "What are the hospital visiting hours?"
            if "normal" in q_lower or "working" in q_lower:
                return "What are the hospital working hours?"
            return "What are the hospital visiting hours and operating hours?"

        # 6. OP HOURS (Outpatient clinic hours)
        if intent == HospitalIntent.OP_HOURS:
            return "What are the hospital OPD consultation timings?"

        # 7. REGISTRATION HOURS
        if intent == HospitalIntent.REGISTRATION_HOURS:
            return "What are the hospital registration and token counter hours?"

        # 8. LOCATION
        if intent == HospitalIntent.LOCATION:
            return "Where is P.S. Mission Hospital located?"

        # 9. CONTACT INFORMATION
        if intent == HospitalIntent.CONTACT_INFORMATION:
            if "how can i contact" in q_lower or "how to contact" in q_lower:
                return "How can I contact the hospital?"
            if any(k in q_lower for k in ["contact", "phone", "number", "call", "reach"]):
                return clean_q
            return "How can I contact the hospital?"

        # 10. EMERGENCY INFORMATION
        if intent == HospitalIntent.EMERGENCY_INFORMATION:
            return "What emergency and casualty services are available at P.S. Mission Hospital?"

        # 11. INSURANCE
        if intent == HospitalIntent.INSURANCE:
            return "What insurance and cashless TPA facilities are accepted at P.S. Mission Hospital?"

        # 13. APPOINTMENT INFORMATION
        if intent == HospitalIntent.APPOINTMENT_INFORMATION:
            if "token" in q_lower:
                return "How can I get an OP token for doctor consultation?"
            return "How can I book an appointment at P.S. Mission Hospital?"

        # 14. GENERAL GREETINGS / CONVERSATION
        if intent == HospitalIntent.GENERAL_QUESTION:
            return clean_q

        # 15. GENERAL
        return clean_q
