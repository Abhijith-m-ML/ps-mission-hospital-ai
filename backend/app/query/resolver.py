"""Unified Query Resolver coordinating normalization, entity fuzzy matching,
conversation context resolution, intent classification, and query rewriting.
"""
from dataclasses import dataclass, field
import json
import re
from typing import Any, Dict, List, Optional

from app.core.logging_config import logger
from app.models.schemas import ChatMessage
from app.query.intent import HospitalIntent, IntentClassifier
from app.query.normalizer import Normalizer
from app.query.query_rewriter import QueryRewriter
from app.query.semantic_analyzer import SemanticQueryAnalyzer
from app.retrieval.entity_store import HospitalEntityStore


@dataclass
class QueryUnderstandingResult:
    """Outcome of Query Understanding and Rewriting."""
    original_query: str
    resolved_query: str
    intent: str
    confidence: float
    entities: Dict[str, Any] = field(default_factory=dict)
    conversation_context_used: bool = False
    department: Optional[str] = None
    target_doctor: Optional[str] = None
    doctor: Optional[str] = None
    language: Optional[str] = None

    @property
    def raw_query(self) -> str:
        return self.original_query

    @property
    def raw_transcript(self) -> str:
        return self.original_query

    @property
    def normalized_query(self) -> str:
        return self.resolved_query

    @property
    def is_context_dependent(self) -> bool:
        return self.conversation_context_used

    @property
    def used_conversation_context(self) -> bool:
        return self.conversation_context_used

    @property
    def target_department(self) -> Optional[str]:
        return self.department

    def to_dict(self) -> Dict[str, Any]:
        return {
            "raw_query": self.original_query,
            "raw_transcript": self.original_query,
            "original_query": self.original_query,
            "normalized_query": self.resolved_query,
            "resolved_query": self.resolved_query,
            "intent": self.intent,
            "entities": self.entities,
            "confidence": self.confidence,
            "department": self.department,
            "doctor": self.doctor or self.target_doctor,
            "conversation_context_used": self.conversation_context_used,
            "language": self.language or self.entities.get("language", "English"),
        }


class QueryResolver:
    """End-to-end Query Understanding & Resolution Engine."""

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
        r"\bnormal\s+hospital\s+hours\b",
    ]

    def __init__(
        self,
        entity_store: Optional[HospitalEntityStore] = None,
        normalizer: Optional[Normalizer] = None,
        intent_classifier: Optional[IntentClassifier] = None,
        query_rewriter: Optional[QueryRewriter] = None,
        semantic_analyzer: Optional[SemanticQueryAnalyzer] = None,
    ):
        self.entity_store = entity_store or HospitalEntityStore()
        self.normalizer = normalizer or Normalizer()
        self.intent_classifier = intent_classifier or IntentClassifier()
        self.query_rewriter = query_rewriter or QueryRewriter(
            entity_store=self.entity_store,
            normalizer=self.normalizer,
        )
        self.semantic_analyzer = semantic_analyzer or SemanticQueryAnalyzer()

    def resolve(
        self,
        query: str,
        history: Optional[List[ChatMessage]] = None,
    ) -> QueryUnderstandingResult:
        """Processes raw user question into fully understood and rewritten query."""
        raw_query = (query or "").strip()
        if not raw_query:
            return QueryUnderstandingResult(
                original_query=raw_query,
                resolved_query=raw_query,
                intent=HospitalIntent.GENERAL_HOSPITAL_INFORMATION.value,
                confidence=1.0,
                entities={},
                conversation_context_used=False,
                language="English",
            )

        # Step 1: Normalization (whitespace, punctuation, common chat typos, medical aliases)
        norm_query = self.normalizer.normalize(raw_query)

        # Step 2: Entity extraction & fuzzy matching on current turn
        # Extract from both raw_query and norm_query to preserve native script details
        entities = self.query_rewriter.extract_entities(raw_query)
        if not entities.get("doctor"):
            entities["doctor"] = self.query_rewriter.match_doctor(norm_query)
        if not entities.get("department"):
            entities["department"] = self.query_rewriter.match_department(norm_query)
        if not entities.get("facility"):
            entities["facility"] = self.query_rewriter.match_facility(norm_query)
        if not entities.get("symptom"):
            norm_extracted = self.query_rewriter.entity_extractor.extract_all(norm_query)
            if norm_extracted.get("symptom"):
                entities["symptom"] = norm_extracted.get("symptom")
            if not entities.get("body_part") and norm_extracted.get("body_part"):
                entities["body_part"] = norm_extracted.get("body_part")
            if not entities.get("inferred_department") and norm_extracted.get("inferred_department"):
                entities["inferred_department"] = norm_extracted.get("inferred_department")

        doc = entities.get("doctor")
        dept = entities.get("department")
        fac = entities.get("facility")
        has_symptom = bool(entities.get("symptom") or entities.get("body_part"))

        # Step 3: Base Intent Classification on current turn
        base_intent, base_conf = self.intent_classifier.classify(
            norm_query,
            has_doctor=bool(doc),
            has_department=bool(dept),
            has_facility=bool(fac),
            has_symptom=has_symptom,
        )
        if base_intent == HospitalIntent.GENERAL_HOSPITAL_INFORMATION:
            raw_intent, raw_conf = self.intent_classifier.classify(
                raw_query,
                has_doctor=bool(doc),
                has_department=bool(dept),
                has_facility=bool(fac),
                has_symptom=has_symptom,
            )
            if raw_intent != HospitalIntent.GENERAL_HOSPITAL_INFORMATION:
                base_intent = raw_intent
                base_conf = raw_conf

        # Step 3B: Semantic Query Understanding via hosted LLM
        # For non-English inputs, Manglish, colloquial symptoms, or ambiguous intents,
        # consult the SemanticQueryAnalyzer to obtain high-precision semantic parsing.
        semantic_normalized = None
        needs_semantic_analysis = (
            (base_intent == HospitalIntent.GENERAL_HOSPITAL_INFORMATION and (
                any(ord(c) > 127 for c in raw_query)
                or entities.get("language") in ("Malayalam", "Hindi", "Manglish")
            ))
            or (has_symptom and not entities.get("symptom"))
            or base_conf < 0.85
        )
        if self.semantic_analyzer and needs_semantic_analysis:
            try:
                sem_res = self.semantic_analyzer.analyze(raw_query, history)
                if sem_res:
                    if sem_res.entities.get("symptom") and not entities.get("symptom"):
                        entities["symptom"] = sem_res.entities["symptom"]
                        has_symptom = True
                    if sem_res.entities.get("body_part") and not entities.get("body_part"):
                        entities["body_part"] = sem_res.entities["body_part"]
                    if sem_res.inferred_department:
                        entities["inferred_department"] = sem_res.inferred_department
                    if sem_res.entities.get("doctor") and not doc:
                        doc = sem_res.entities["doctor"]
                    if sem_res.entities.get("department") and not dept:
                        dept = sem_res.entities["department"]

                    for h_intent in HospitalIntent:
                        if h_intent.value == sem_res.intent:
                            base_intent = h_intent
                            base_conf = max(base_conf, sem_res.confidence)
                            break

                    if sem_res.normalized_query:
                        semantic_normalized = sem_res.normalized_query
            except Exception as sem_err:
                logger.debug("Semantic analysis fallback to local rules: %s", sem_err)

        # General hospital intents that must NEVER inherit department/doctor context
        GENERAL_HOSPITAL_INTENTS = {
            HospitalIntent.CONTACT_INFORMATION,
            HospitalIntent.HOSPITAL_HOURS,
            HospitalIntent.LOCATION,
            HospitalIntent.EMERGENCY_INFORMATION,
            HospitalIntent.GENERAL_QUESTION,
            HospitalIntent.GENERAL_HOSPITAL_INFORMATION,
        }

        # Step 4: Independent query check (e.g. general questions or symptoms don't inherit old dept)
        is_independent = (
            (base_intent in GENERAL_HOSPITAL_INTENTS and not dept)
            or (base_intent == HospitalIntent.SYMPTOM_TO_DEPARTMENT)
            or any(
                re.search(pat, norm_query.lower())
                for pat in self.INDEPENDENT_HOSPITAL_PATTERNS
            )
        )

        used_context = False
        context_dept = None
        context_doc = None

        if not is_independent and not dept and not doc:
            # Step 5: Context resolution from recent history for genuine follow-ups only
            last_dept, last_doc = self._extract_recent_entities(history or [])
            if last_dept or last_doc:
                q_lower = norm_query.lower()

                # Check if asking about doctors in follow-up
                is_doc_followup = self._is_doctor_followup(q_lower)
                # Check if asking about schedule in follow-up
                is_sched_followup = self._is_schedule_followup(q_lower)
                # Check if asking about facilities in follow-up
                is_fac_followup = self._is_facility_followup(q_lower)
                # Pronoun check (doctor / department personal pronouns only)
                has_pronoun = bool(re.search(r"\b(he|she|they|them|their|his|her|this department)\b", q_lower))

                if is_doc_followup and last_dept:
                    dept = last_dept
                    context_dept = last_dept
                    used_context = True
                elif is_sched_followup:
                    if last_doc:
                        doc = last_doc
                        context_doc = last_doc
                        used_context = True
                    elif last_dept:
                        dept = last_dept
                        context_dept = last_dept
                        used_context = True
                elif is_fac_followup and last_dept:
                    dept = last_dept
                    context_dept = last_dept
                    used_context = True
                elif has_pronoun:
                    if any(w in q_lower for w in ["he", "she", "his", "her"]) and last_doc:
                        doc = last_doc
                        context_doc = last_doc
                        used_context = True
                    elif last_dept:
                        dept = last_dept
                        context_dept = last_dept
                        used_context = True

        # Update entities dictionary
        entities["doctor"] = doc
        entities["department"] = dept
        entities["facility"] = fac

        # Step 6: Final Intent classification
        intent_enum, confidence = self.intent_classifier.classify(
            norm_query,
            has_doctor=bool(doc),
            has_department=bool(dept),
            has_facility=bool(fac),
            has_symptom=has_symptom,
        )
        if intent_enum == HospitalIntent.GENERAL_HOSPITAL_INFORMATION and base_intent != HospitalIntent.GENERAL_HOSPITAL_INFORMATION:
            intent_enum = base_intent
            confidence = base_conf

        if intent_enum == HospitalIntent.SYMPTOM_TO_DEPARTMENT and not dept:
            dept = entities.get("inferred_department")

        # Step 7: Query rewriting (pass raw_query as well so Malayalam/Manglish triggers are preserved)
        if semantic_normalized:
            resolved_q = semantic_normalized
        else:
            target_norm = raw_query if (entities.get("language") in ("Malayalam", "Manglish") or any(ord(c) > 127 for c in raw_query)) else norm_query
            resolved_q = self.query_rewriter.rewrite_query(
                normalized_query=target_norm,
                intent=intent_enum,
                entities=entities,
                context_department=context_dept,
                context_doctor=context_doc,
            )

        return QueryUnderstandingResult(
            original_query=raw_query,
            resolved_query=resolved_q,
            intent=intent_enum.value,
            confidence=confidence,
            entities=entities,
            conversation_context_used=used_context,
            department=dept,
            target_doctor=doc,
            doctor=doc,
            language=entities.get("language", "English"),
        )

    def _extract_recent_entities(
        self,
        history: List[ChatMessage],
    ) -> tuple[Optional[str], Optional[str]]:
        """Scans recent conversation turns in reverse order to find active department/doctor."""
        active_dept = None
        active_doc = None

        for msg in reversed(history[-6:]):
            text = getattr(msg, "content", "") if hasattr(msg, "content") else str(msg.get("content", ""))

            # 1. Doctor check
            if not active_doc:
                active_doc = self.query_rewriter.match_doctor(text)

            # 2. Department check
            if not active_dept:
                active_dept = self.query_rewriter.match_department(text)

            if active_dept and active_doc:
                break

        # If doctor was found without explicit department, associate their department
        if active_doc and not active_dept:
            doc_obj = self.entity_store.get_doctor_by_name(active_doc)
            if doc_obj and doc_obj.department:
                active_dept = doc_obj.department

        return active_dept, active_doc

    def _is_doctor_followup(self, q_lower: str) -> bool:
        doctor_tokens = [
            "doctor", "doctors", "dr", "physician", "physicians", "specialist", "consultant",
            "who are the", "who is the", "who is available", "who to see", "whom to see",
            "who to consult", "doctor name please", "doctors please",
        ]
        return any(re.search(r"\b" + re.escape(tok) + r"\b", q_lower) for tok in doctor_tokens)

    def _is_schedule_followup(self, q_lower: str) -> bool:
        sched_tokens = [
            "timing", "timings", "schedule", "when can i visit", "when is",
            "what are their timings", "what are his timings", "what are her timings",
            "when to come", "when to visit", "available timing",
        ]
        return any(tok in q_lower for tok in sched_tokens)

    def _is_facility_followup(self, q_lower: str) -> bool:
        facility_tokens = [
            "facility", "facilities", "services", "equipment", "amenities",
            "what about facilities", "facilities please",
        ]
        return any(re.search(r"\b" + re.escape(tok) + r"\b", q_lower) for tok in facility_tokens)
