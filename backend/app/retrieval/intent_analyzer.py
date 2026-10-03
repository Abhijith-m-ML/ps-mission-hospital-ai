import re
from dataclasses import dataclass
from enum import Enum
from typing import Optional

from app.core.logging_config import logger
from app.retrieval.entity_store import HospitalEntityStore


class QueryIntent(str, Enum):
    DOCTOR_SEARCH = "doctor_search"
    FACILITY_SEARCH = "facility_search"
    SCHEDULE_SEARCH = "schedule_search"
    DEPARTMENT_SEARCH = "department_search"
    GENERAL_RAG = "general_rag"


@dataclass
class ParsedIntent:
    intent: QueryIntent
    department: Optional[str] = None
    doctor_name: Optional[str] = None
    raw_query: str = ""

    def to_dict(self):
        return {
            "intent": self.intent.value,
            "department": self.department,
            "doctor_name": self.doctor_name,
            "raw_query": self.raw_query,
        }


class IntentAnalyzer:
    """Analyzes user questions to determine query intent (doctor search, facility search,
    schedule search, department search, or general RAG) and extracts referenced entities.
    """

    DOCTOR_KEYWORDS = [
        "doctor", "doctors", "dr", "dr.", "physician", "physicians",
        "consultant", "consultants", "specialist", "specialists",
        "pediatrician", "paediatrician", "cardiologist", "cardiologists",
        "gynecologist", "gynaecologist", "surgeon", "surgeons",
        "neurologist", "dermatologist", "urologist", "nephrologist",
        "radiologist", "psychiatrist", "orthopedician",
        "consult", "consultation", "who to see", "whom to see", "which doctor",
    ]

    FACILITY_KEYWORDS = [
        "facility", "facilities", "service", "services", "amenity", "amenities",
        "infrastructure", "equipment", "diagnostic", "diagnostics",
        "nicu", "picu", "icu", "lab", "laboratory", "test", "tests", "scan",
        "dialysis", "ecg", "echo", "phototherapy", "immunization",
        "vaccination", "radiology", "therapy",
    ]

    SCHEDULE_KEYWORDS = [
        "when can i visit", "when is dr", "op timing", "op timings",
        "op schedule", "visiting hours", "visit timing", "consultation time",
        "consultation timing", "consultation hours", "schedule of dr",
        "schedule for dr", "timing for dr", "appointment with dr",
        "available timing", "availability of dr", "when to meet",
        "when to visit", "working hours", "timing", "timings", "days",
    ]

    def __init__(self, entity_store: Optional[HospitalEntityStore] = None):
        self.entity_store = entity_store or HospitalEntityStore()

    def analyze(self, query: str) -> ParsedIntent:
        """Determines the specific hospital intent and identifies department or doctor."""
        clean_q = (query or "").strip()
        q_lower = clean_q.lower()

        # 1. Check for Doctor Reference or Schedule Intent
        # Look for explicit doctor mention, e.g. "Dr. Jaya", "Dr. Annie", "Dr. Sudheer"
        doc_match = self._extract_doctor_name(clean_q)

        # 2. Check for Schedule Search Intent
        is_schedule = any(k in q_lower for k in self.SCHEDULE_KEYWORDS)
        if doc_match and (is_schedule or "when" in q_lower or "visit" in q_lower or "time" in q_lower or "available" in q_lower):
            logger.info("Intent classified as SCHEDULE_SEARCH for doctor: %s", doc_match)
            return ParsedIntent(
                intent=QueryIntent.SCHEDULE_SEARCH,
                doctor_name=doc_match,
                raw_query=clean_q,
            )

        # 3. Detect Clinical Department
        detected_dept = self._detect_department(clean_q)

        # 4. Check for Doctor Search Intent
        has_doctor_keyword = any(
            re.search(r"\b" + re.escape(kw) + r"\b", q_lower)
            for kw in self.DOCTOR_KEYWORDS
        )

        if has_doctor_keyword:
            logger.info("Intent classified as DOCTOR_SEARCH for department: %s", detected_dept)
            return ParsedIntent(
                intent=QueryIntent.DOCTOR_SEARCH,
                department=detected_dept,
                doctor_name=doc_match,
                raw_query=clean_q,
            )

        # 5. Check for Facility Search Intent
        has_facility_keyword = any(
            re.search(r"\b" + re.escape(kw) + r"\b", q_lower)
            for kw in self.FACILITY_KEYWORDS
        )

        if has_facility_keyword:
            logger.info("Intent classified as FACILITY_SEARCH for department: %s", detected_dept)
            return ParsedIntent(
                intent=QueryIntent.FACILITY_SEARCH,
                department=detected_dept,
                raw_query=clean_q,
            )

        # 6. If only department is detected and asking "tell me about"
        if detected_dept and any(k in q_lower for k in ["about", "overview", "info", "tell me", "what is"]):
            logger.info("Intent classified as DEPARTMENT_SEARCH for: %s", detected_dept)
            return ParsedIntent(
                intent=QueryIntent.DEPARTMENT_SEARCH,
                department=detected_dept,
                raw_query=clean_q,
            )

        # 7. Fallback to General RAG
        logger.info("Intent classified as GENERAL_RAG for query: %s", clean_q[:50])
        return ParsedIntent(
            intent=QueryIntent.GENERAL_RAG,
            department=detected_dept,
            doctor_name=doc_match,
            raw_query=clean_q,
        )

    def _extract_doctor_name(self, text: str) -> Optional[str]:
        """Identifies any known doctor mentioned in the query."""
        # 1. Check against known doctors in entity store
        doc = self.entity_store.get_doctor_by_name(text)
        if doc:
            return doc.doctor_name

        # 2. Check for regex pattern "Dr. [Name]"
        match = re.search(r"\b(dr\.?\s+[A-Za-z\.\s]+?)(?:\b|'s|\?|,|$)", text, re.IGNORECASE)
        if match:
            cand = match.group(1).strip()
            # Verify minimum length
            if len(cand) > 4:
                # Check entity store again with candidate
                matched = self.entity_store.get_doctor_by_name(cand)
                if matched:
                    return matched.doctor_name
                return cand

        return None

    def _detect_department(self, text: str) -> Optional[str]:
        """Maps query tokens to standard hospital department name."""
        clean_text = text.lower()
        for canonical, aliases in self.entity_store.DEPARTMENT_SYNONYMS.items():
            for alias in aliases:
                pattern = r"\b" + re.escape(alias) + r"(?:s|es)?\b"
                if re.search(pattern, clean_text):
                    return canonical
        return None

