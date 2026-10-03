import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from app.core.logging_config import logger
from app.models.schemas import ChatMessage
from app.retrieval.entity_store import HospitalEntityStore


@dataclass
class ResolutionResult:
    """Outcome of conversational query contextualization."""
    original_query: str
    resolved_query: str
    intent: str
    department: Optional[str] = None
    target_doctor: Optional[str] = None
    used_conversation_context: bool = False

    @property
    def is_context_dependent(self) -> bool:
        return self.used_conversation_context

    @property
    def target_department(self) -> Optional[str]:
        return self.department

    def to_dict(self) -> Dict[str, Any]:
        return {
            "original_query": self.original_query,
            "resolved_query": self.resolved_query,
            "intent": self.intent,
            "department": self.department,
            "used_conversation_context": self.used_conversation_context,
        }


class QueryResolver:
    """Resolves context-dependent, follow-up, or elliptical queries against
    temporary in-memory conversation history.
    
    Examples:
    - Previous: "Paediatrics & Neonatology" | Current: "doctor name please"
      -> "Which doctors are available in the Paediatrics & Neonatology department?"
    - Previous: "Cardiology" | Current: "what are the doctor are available"
      -> "Which doctors are available in the Cardiology department?"
    - Previous: "Cardiology" | Current: "doctors"
      -> "Which doctors are available in the Cardiology department?"
    - Previous: "Cardiology" | Current: "What facilities are available?"
      -> "What facilities are available in the Cardiology department?"
    - Previous: "Neurology" | Current: "what are their timings?"
      -> "What is the OP consultation schedule for Neurology doctors?"
    - Previous: "Cardiology" | Current: "Where is the hospital located?"
      -> "Where is the hospital located?" (independent - no rewriting)
    """

    DOCTOR_KEYWORDS = [
        "doctor", "doctors", "dr", "dr.", "physician", "physicians",
        "consultant", "consultants", "specialist", "specialists",
        "cardiologist", "pediatrician", "paediatrician", "neurologist",
        "orthopedician", "gynecologist", "gynaecologist", "surgeon",
        "urologist", "dermatologist", "nephrologist",
        # Malayalam doctor keywords
        "ഡോക്ടർ", "ഡോക്ടർമാർ", "ഡോക്ടർമാരുണ്ട്", "ഡോക്ടറെ", "ഡോക്ടർമാർ ഉണ്ടോ", "ഡോക്ടർ ആരാണ്",
        # Hindi doctor keywords
        "डॉक्टर", "डॉक्टरों", "चिकित्सक", "डॉक्टर कौन हैं", "कौन से डॉक्टर",
    ]

    FACILITY_KEYWORDS = [
        "facility", "facilities", "service", "services", "amenity", "amenities",
        "infrastructure", "equipment", "diagnostic", "diagnostics",
        "nicu", "picu", "icu", "lab", "laboratory", "test", "tests", "scan",
        # Malayalam facility keywords
        "സൗകര്യം", "സൗകര്യങ്ങൾ", "സേവനങ്ങൾ", "സേവനം",
        # Hindi facility keywords
        "सुविधा", "सुविधाएं", "सेवाएं", "सेवा",
    ]

    SCHEDULE_KEYWORDS = [
        "timing", "timings", "schedule", "visiting hours", "visiting time",
        "op timing", "op timings", "op schedule", "consultation time",
        "consultation schedule", "consultation timing", "when can i visit",
        "when is", "when are they available", "what are their timings",
        "what are his timings", "what are her timings", "when to come",
        "when to visit", "available timing",
        # Malayalam schedule keywords
        "സമയം", "സമയങ്ങൾ", "സന്ദർശന സമയം", "ഒപി സമയം", "ഒ.പി സമയം",
        # Hindi schedule keywords
        "समय", "मिलने का समय", "ओपी समय",
    ]

    INDEPENDENT_HOSPITAL_PATTERNS = [
        r"\bwhere\s+is\s+(?:the\s+)?hospital(?:\s+located)?\b",
        r"\bhospital\s+location\b",
        r"\bwhere\s+are\s+you\s+located\b",
        r"\bwhere\s+is\s+it\s+located\b",
        r"\bwhere\s+is\s+(?:p\.?s\.?\s*mission|ps\s*mission)\b",
        r"\baddress\s+of\s+(?:the\s+)?hospital\b",
        r"\bhow\s+to\s+reach(?:\s+the\s+hospital)?\b",
        r"\bdirections\s+to\s+(?:the\s+)?hospital\b",
        r"\b(?:contact|phone|emergency|ambulance|telephone)\s+number\b",
        r"\bhospital\s+history\b",
        r"\bwho\s+founded\b",
        r"\bwhen\s+was\s+(?:the\s+)?hospital\s+established\b",
        r"\bvisiting\s+hours\s+for\s+(?:the\s+)?hospital\b",
        r"\bhospital\s+visiting\s+hours\b",
    ]

    def __init__(self, entity_store: Optional[HospitalEntityStore] = None):
        self.entity_store = entity_store or HospitalEntityStore()

    def resolve(
        self,
        query: str,
        history: Optional[List[ChatMessage]] = None,
    ) -> ResolutionResult:
        """Determines if the question depends on previous dialogue context and
        rewrites it into a standalone query for retrieval.
        """
        clean_q = (query or "").strip()
        if not clean_q:
            return ResolutionResult(
                original_query=clean_q,
                resolved_query=clean_q,
                intent="general_rag",
                department=None,
                target_doctor=None,
                used_conversation_context=False,
            )

        q_lower = clean_q.lower().strip()
        q_norm = re.sub(r"[^\w\s\?]", " ", q_lower).strip()

        # ----------------------------------------------------------------------
        # 1. Independent General Hospital Query Check
        # (e.g. "Where is the hospital located?" -> NEVER forced into Cardiology)
        # ----------------------------------------------------------------------
        if any(re.search(pat, q_lower) for pat in self.INDEPENDENT_HOSPITAL_PATTERNS):
            logger.info("Query '%s' classified as independent hospital inquiry. Preserving intact.", clean_q)
            return ResolutionResult(
                original_query=clean_q,
                resolved_query=clean_q,
                intent="general_rag",
                department=None,
                target_doctor=None,
                used_conversation_context=False,
            )

        # ----------------------------------------------------------------------
        # 2. Check if current query ALREADY contains an explicit department or doctor
        # If so, it is self-contained and sets its own context
        # ----------------------------------------------------------------------
        explicit_dept = self._find_department_in_text(clean_q)
        explicit_doc = self._find_doctor_in_text(clean_q)

        if explicit_dept or explicit_doc:
            inferred_intent = self._infer_intent_from_query(clean_q, has_dept=bool(explicit_dept), has_doc=bool(explicit_doc))
            logger.debug(
                "Query '%s' contains explicit entity (dept='%s', doc='%s', intent='%s'). Keeping intact.",
                clean_q,
                explicit_dept,
                explicit_doc,
                inferred_intent,
            )
            return ResolutionResult(
                original_query=clean_q,
                resolved_query=clean_q,
                intent=inferred_intent,
                department=explicit_dept,
                target_doctor=explicit_doc,
                used_conversation_context=False,
            )

        # ----------------------------------------------------------------------
        # 3. Check for History Context
        # ----------------------------------------------------------------------
        if not history:
            # Fresh session or browser refreshed - no context can be assumed
            inferred_intent = self._infer_intent_from_query(clean_q, has_dept=False, has_doc=False)
            logger.debug("No conversation history available. Query '%s' remains independent (intent=%s).", clean_q, inferred_intent)
            return ResolutionResult(
                original_query=clean_q,
                resolved_query=clean_q,
                intent=inferred_intent,
                department=None,
                target_doctor=None,
                used_conversation_context=False,
            )

        # ----------------------------------------------------------------------
        # 4. Extract active entities from recent conversation history
        # ----------------------------------------------------------------------
        last_dept, last_doc = self._extract_recent_entities(history)
        logger.debug("Extracted active history context: last_dept='%s', last_doc='%s'", last_dept, last_doc)

        # If no active department or doctor was found in history, query cannot be context-resolved
        if not last_dept and not last_doc:
            inferred_intent = self._infer_intent_from_query(clean_q, has_dept=False, has_doc=False)
            return ResolutionResult(
                original_query=clean_q,
                resolved_query=clean_q,
                intent=inferred_intent,
                department=None,
                target_doctor=None,
                used_conversation_context=False,
            )

        # ----------------------------------------------------------------------
        # 5. Evaluate Follow-up Intent Categories
        # ----------------------------------------------------------------------

        # CASE A: Doctor Follow-up (e.g. "what are the doctor are available", "doctors", "who to see", "doctor name please")
        is_doctor_query = self._is_doctor_followup(clean_q, q_norm)
        if is_doctor_query and last_dept:
            resolved = f"Which doctors are available in the {last_dept} department?"
            logger.info("Resolved doctor query: '%s' -> '%s' (dept=%s)", clean_q, resolved, last_dept)
            return ResolutionResult(
                original_query=clean_q,
                resolved_query=resolved,
                intent="doctor_search",
                department=last_dept,
                target_doctor=None,
                used_conversation_context=True,
            )

        # CASE B: Schedule / Timings Follow-up (e.g. "what are their timings?", "when can I visit?", "visiting hours")
        is_schedule_query = self._is_schedule_followup(clean_q, q_norm)
        if is_schedule_query:
            if last_doc:
                resolved = f"What is {last_doc}'s available schedule?"
                logger.info("Resolved schedule query with doctor: '%s' -> '%s'", clean_q, resolved)
                return ResolutionResult(
                    original_query=clean_q,
                    resolved_query=resolved,
                    intent="schedule_search",
                    department=last_dept,
                    target_doctor=last_doc,
                    used_conversation_context=True,
                )
            elif last_dept:
                resolved = f"What is the OP consultation schedule for {last_dept} doctors?"
                logger.info("Resolved schedule query with department: '%s' -> '%s'", clean_q, resolved)
                return ResolutionResult(
                    original_query=clean_q,
                    resolved_query=resolved,
                    intent="schedule_search",
                    department=last_dept,
                    target_doctor=None,
                    used_conversation_context=True,
                )

        # CASE C: Facility / Services Follow-up (e.g. "What facilities are available?", "services please")
        is_facility_query = self._is_facility_followup(clean_q, q_norm)
        if is_facility_query and last_dept:
            resolved = f"What facilities are available in the {last_dept} department?"
            logger.info("Resolved facility query: '%s' -> '%s' (dept=%s)", clean_q, resolved, last_dept)
            return ResolutionResult(
                original_query=clean_q,
                resolved_query=resolved,
                intent="facility_search",
                department=last_dept,
                target_doctor=None,
                used_conversation_context=True,
            )

        # CASE D: Pronoun Replacement (he, she, they, them, their, there, this department)
        if any(re.search(r"\b" + p + r"\b", q_lower) for p in ["he", "she", "him", "her"]):
            if last_doc:
                resolved = re.sub(r"\b(he|she|him|her)\b", last_doc, clean_q, flags=re.IGNORECASE)
                logger.info("Resolved pronoun with doctor: '%s' -> '%s'", clean_q, resolved)
                return ResolutionResult(
                    original_query=clean_q,
                    resolved_query=resolved,
                    intent="doctor_search",
                    department=last_dept,
                    target_doctor=last_doc,
                    used_conversation_context=True,
                )

        if any(re.search(r"\b" + p + r"\b", q_lower) for p in ["they", "them", "there", "their", "this department"]):
            if last_dept:
                resolved = re.sub(r"\b(they|them|there|their|this department)\b", last_dept, clean_q, flags=re.IGNORECASE)
                logger.info("Resolved pronoun with department: '%s' -> '%s'", clean_q, resolved)
                return ResolutionResult(
                    original_query=clean_q,
                    resolved_query=resolved,
                    intent="department_search",
                    department=last_dept,
                    target_doctor=last_doc,
                    used_conversation_context=True,
                )

        # ----------------------------------------------------------------------
        # 6. Fallback: Independent query that does not require rewriting
        # ----------------------------------------------------------------------
        inferred_intent = self._infer_intent_from_query(clean_q, has_dept=False, has_doc=False)
        logger.debug("Query '%s' does not match follow-up patterns. Remaining intact.", clean_q)
        return ResolutionResult(
            original_query=clean_q,
            resolved_query=clean_q,
            intent=inferred_intent,
            department=None,
            target_doctor=None,
            used_conversation_context=False,
        )

    def _is_doctor_followup(self, query: str, q_norm: str) -> bool:
        """Determines if the query is asking about doctors without naming one."""
        q_lower = query.lower()

        # Check explicit keywords
        for kw in self.DOCTOR_KEYWORDS:
            if not kw.isascii():
                if kw in q_lower:
                    return True
            elif re.search(r"\b" + re.escape(kw) + r"\b", q_lower):
                return True

        # Check common question phrasing patterns
        doctor_phrases = [
            r"who\s+(?:is|are)\s+(?:the\s+)?available",
            r"who\s+(?:can|should)\s+i\s+see",
            r"who\s+(?:can|should)\s+i\s+consult",
            r"whom\s+(?:can|should)\s+i\s+consult",
            r"who\s+treats",
            r"who\s+is\s+in\s+charge",
            r"who\s+is\s+available",
            r"is\s+anyone\s+available",
        ]
        return any(re.search(p, q_lower) for p in doctor_phrases)

    def _is_facility_followup(self, query: str, q_norm: str) -> bool:
        """Determines if the query is asking about facilities or services."""
        q_lower = query.lower()
        for kw in self.FACILITY_KEYWORDS:
            if not kw.isascii():
                if kw in q_lower:
                    return True
            elif re.search(r"\b" + re.escape(kw) + r"\b", q_lower):
                return True
        return False

    def _is_schedule_followup(self, query: str, q_norm: str) -> bool:
        """Determines if the query is asking about timing or schedules."""
        q_lower = query.lower()
        for kw in self.SCHEDULE_KEYWORDS:
            if not kw.isascii():
                if kw in q_lower:
                    return True
            elif re.search(r"\b" + re.escape(kw) + r"\b", q_lower):
                return True
        return False

    def _infer_intent_from_query(self, query: str, has_dept: bool, has_doc: bool) -> str:
        """Infers intent category for standalone or resolved queries."""
        q_norm = query.lower()
        if self._is_schedule_followup(query, q_norm) and (has_doc or "dr" in q_norm):
            return "schedule_search"
        if self._is_doctor_followup(query, q_norm):
            return "doctor_search"
        if self._is_facility_followup(query, q_norm):
            return "facility_search"
        if has_dept and any(k in q_norm for k in ["about", "tell me", "overview", "what is"]):
            return "department_search"
        return "general_rag"

    def _extract_recent_entities(
        self,
        history: List[ChatMessage],
    ) -> Tuple[Optional[str], Optional[str]]:
        """Scans recent conversation history to identify the active department and doctor.
        Prioritizes user messages first because they express the patient's explicit topic,
        then checks assistant messages if needed.
        """
        last_dept: Optional[str] = None
        last_doc: Optional[str] = None

        recent_history = history[-6:] if len(history) > 6 else history

        # Pass 1: Scan user turns in reverse chronological order
        user_turns = [msg for msg in reversed(recent_history) if msg.role == "user"]
        for msg in user_turns:
            content = msg.content or ""
            if not last_dept:
                dept = self._find_department_in_text(content)
                if dept:
                    last_dept = dept
            if not last_doc:
                doc = self._find_doctor_in_text(content)
                if doc:
                    last_doc = doc
            if last_dept and last_doc:
                return last_dept, last_doc

        # Pass 2: If department still not found, scan assistant turns
        if not last_dept or not last_doc:
            assistant_turns = [msg for msg in reversed(recent_history) if msg.role == "assistant"]
            for msg in assistant_turns:
                content = msg.content or ""
                if not last_dept:
                    dept = self._find_department_in_text(content)
                    if dept:
                        last_dept = dept
                if not last_doc:
                    doc = self._find_doctor_in_text(content)
                    if doc:
                        last_doc = doc
                if last_dept and last_doc:
                    break

        return last_dept, last_doc

    def _find_department_in_text(self, text: str) -> Optional[str]:
        """Finds any hospital department mentioned in the text using strict priority:
        1. Exact canonical department names (e.g. 'Cardiology', 'Neurology')
        2. Long multi-token department aliases (e.g. 'chest pain', 'heart attack')
        3. Single-word distinct aliases with word boundaries (e.g. 'cardio', 'heart')
        """
        if not text:
            return None
        clean_text = text.lower()

        # Step 1: Check canonical department names first
        for canonical in self.entity_store.DEPARTMENT_SYNONYMS.keys():
            canon_lower = canonical.lower()
            pattern = r"\b" + re.escape(canon_lower) + r"\b"
            if re.search(pattern, clean_text):
                return canonical

        # Step 2: Check multi-word aliases (e.g. "chest pain", "heart attack", "child fever")
        multi_word_matches: List[Tuple[int, str]] = []
        for canonical, aliases in self.entity_store.DEPARTMENT_SYNONYMS.items():
            for alias in aliases:
                if " " in alias:
                    if not alias.isascii() and alias in clean_text:
                        multi_word_matches.append((len(alias), canonical))
                    else:
                        pattern = r"\b" + re.escape(alias) + r"\b"
                        if re.search(pattern, clean_text):
                            multi_word_matches.append((len(alias), canonical))

        if multi_word_matches:
            # Return longest matching multi-word alias
            multi_word_matches.sort(key=lambda x: x[0], reverse=True)
            return multi_word_matches[0][1]

        # Step 3: Check single-word aliases (excluding common ambiguous words in general text)
        ambiguous_words = {"child", "children", "baby", "care", "infant", "bone", "test", "tests", "scan"}
        for canonical, aliases in self.entity_store.DEPARTMENT_SYNONYMS.items():
            for alias in aliases:
                if " " not in alias and alias not in ambiguous_words:
                    # Non-ASCII Malayalam or Devanagari script substring check
                    if not alias.isascii():
                        if alias in clean_text:
                            return canonical
                    else:
                        pattern = r"\b" + re.escape(alias) + r"(?:s|es)?\b"
                        if re.search(pattern, clean_text):
                            return canonical

        # Step 4: Fallback check for remaining aliases with strict word boundary
        for canonical, aliases in self.entity_store.DEPARTMENT_SYNONYMS.items():
            for alias in aliases:
                if not alias.isascii():
                    if alias in clean_text:
                        return canonical
                else:
                    pattern = r"\b" + re.escape(alias) + r"(?:s|es)?\b"
                    if re.search(pattern, clean_text):
                        return canonical

        return None

    def _find_doctor_in_text(self, text: str) -> Optional[str]:
        """Finds any hospital doctor mentioned in the text."""
        if not text:
            return None

        # 1. Match from entity store
        doc = self.entity_store.get_doctor_by_name(text)
        if doc:
            return doc.doctor_name

        # 2. Check regex for "Dr. [Name]"
        match = re.search(r"\b(dr\.?\s+[A-Za-z\.\s]+?)(?:\b|'s|\?|,|$)", text, re.IGNORECASE)
        if match:
            cand = match.group(1).strip()
            if len(cand) > 4:
                matched = self.entity_store.get_doctor_by_name(cand)
                if matched:
                    return matched.doctor_name
                return cand

        return None
