"""Semantic Query Analyzer for Hospital AI Assistant using hosted Gemini API.

Translates natural language questions in Malayalam, English, Hindi, Manglish, or colloquial
phrasing into structured semantic meaning (normalized_query, intent, language, entities,
inferred_department, confidence) WITHOUT diagnosing or hallucinating clinical conditions.
"""
from dataclasses import dataclass, field
import json
import os
import re
import time
from typing import Any, Dict, List, Optional

import httpx

from app.core.config import settings
from app.core.logging_config import logger
from app.models.schemas import ChatMessage


SEMANTIC_ANALYZER_INSTRUCTION = """You are an expert clinical query understanding and normalization engine for P.S. Mission Hospital.
Your task is to analyze user inquiries (which may be in Malayalam, English, Hindi, Manglish, or colloquial mixtures) and return a structured JSON analysis.

CRITICAL RULES:
1. DO NOT answer the user's question. DO NOT provide medical advice. Only return structured JSON.
2. CRITICAL SAFETY RULE: NEVER diagnose the user with medical diseases, conditions, or syndromes (e.g. NEVER output "arthritis", "fracture", "heart attack", "migraine", "nerve disorder", "appendicitis" unless the user explicitly used those exact words).
   - If user mentions pain in a body part (e.g. leg, knee, calf, head, stomach, chest, back), extract the symptom as simply "{body_part} pain" (e.g. "leg pain", "headache", "stomach pain", "chest pain", "knee pain").
3. Support colloquial and natural phrasing in Malayalam, English, Hindi, and Manglish.
   - For example:
     - "എന്റെ കാലിലൊരു വേദനയുണ്ട്, ഞാൻ ഏത് ഡോക്ടറെയാണ് കാണിക്കേണ്ടത്?"
     - "കാലിനു pain ആണ്, ആരെ കാണിക്കണം?"
     - "enik kaalil pain aanu, etha doctor?"
     - "leg pain und, which doctor?"
     - "my leg is hurting, which doctor should I see?"
     - "which doctor for pain in my leg?"
     - "എനിക്ക് കാലിൽ വേദനയുണ്ട്"
     ALL resolve to:
     normalized_query: "I have leg pain. Which doctor should I consult?"
     intent: "symptom_to_department"
     entities: {"symptom": "leg pain", "body_part": "leg"}
     inferred_department: "Orthopaedics & Trauma Care"
4. Normalize the query into a clean, canonical English retrieval question representing the user's genuine semantic intent.
   - "എനിക്ക് തലവേദനയുണ്ട് ഏത് ഡോക്ടറെ കാണണം?" -> "I have a headache. Which doctor should I consult?"
   - "എനിക്ക് വയറുവേദനയാണ് ആരെ കാണണം?" -> "I have stomach pain. Which doctor should I consult?"
   - "What are the hospital visiting hours?" -> "What are the hospital visiting hours?"
   - "Where is the hospital?" -> "Where is P.S. Mission Hospital located?"
   - "How can I contact the hospital?" -> "How can I contact the hospital?"
5. Supported hospital intents:
   - "symptom_to_department" (user describes symptoms, pain, injury or asks which doctor/department to consult)
   - "doctor_search" (user searching for doctor names, specialists)
   - "doctor_schedule" (when doctor is available, OPD hours of a doctor)
   - "department_information" (services and information about a hospital department)
   - "facility_information" (diagnostic facilities: lab, MRI, CT scan, ICU, dialysis, ambulance)
   - "hospital_hours" (visiting hours for relatives/visitors)
   - "op_hours" (outpatient clinic consultation hours)
   - "registration_hours" (counter and token desk timings)
   - "emergency_information" (casualty, trauma, 24/7 emergency care)
   - "contact_information" (phone numbers, reception, email)
   - "location" (hospital address, route, how to reach)
   - "insurance" (cashless insurance, TPA schemes)
   - "services" (preventive health checkups, packages)
   - "appointment_information" (how to book appointment or get token)
   - "general_hospital_information" (hospital history, founder, general info)
   - "general_question" (greetings, hi, hello)
6. Map symptoms to appropriate P.S. Mission Hospital departments (inferred_department):
   - Leg pain, knee pain, bone pain, joint pain, back pain, fracture, sprain -> "Orthopaedics & Trauma Care"
   - Chest pain, heart discomfort -> "Cardiology"
   - Headache, dizziness, seizures -> "Neurology"
   - Stomach pain, abdomen, indigestion, fever, cough -> "General Medicine"
   - Eye pain, vision -> "Ophthalmology"
   - Ear pain, throat pain, nose -> "ENT (Ear, Nose & Throat)"
   - Skin rash, itching -> "Dermatology"
   - Toothache, dental -> "Maxillofacial & Dental Services"
   - Child/infant illness -> "Paediatrics & Neonatology"
   - Pregnancy, female health -> "Obstetrics & Gynaecology"
   - Kidney, dialysis -> "Nephrology, Toxicology & Dialysis"
   - Urinary -> "Urology"
   If no symptom is present or confidence is low, set inferred_department to null.
7. Context and Follow-up:
   - set is_followup to true ONLY if the current message is a genuine, explicit follow-up dependent on the previous conversation turn (e.g. "Who are the doctors?", "When is he available?", "What about visiting hours for this department?").
   - If the user asks a new symptom ("I have leg pain"), an independent general question ("Where is the hospital?", "What are the hospital visiting hours?"), is_followup MUST be false, and previous department/doctor must NOT be inherited.
8. Output JSON schema:
{
  "normalized_query": "string",
  "intent": "string",
  "language": "string",
  "entities": {
    "symptom": "string or null",
    "body_part": "string or null",
    "doctor": "string or null",
    "department": "string or null",
    "facility": "string or null"
  },
  "inferred_department": "string or null",
  "confidence": number,
  "is_followup": boolean
}
"""


@dataclass
class SemanticAnalysisResult:
    """Structured semantic representation of user question."""
    normalized_query: str
    intent: str
    language: str
    entities: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 0.95
    inferred_department: Optional[str] = None
    is_followup: bool = False


class SemanticQueryAnalyzer:
    """Gemini-powered semantic analyzer for multilingual hospital queries."""

    def __init__(self, api_key: Optional[str] = None):
        raw_key = api_key or settings.GEMINI_API_KEY or settings.OPENAI_API_KEY
        self.api_key = raw_key.strip() if raw_key else None
        self._cache: Dict[str, SemanticAnalysisResult] = {}
        self.candidate_models = [
            "gemini-flash-lite-latest",
            "gemini-3.1-flash-lite",
            "gemini-flash-latest",
        ]

    def analyze(
        self,
        raw_query: str,
        history: Optional[List[ChatMessage]] = None,
        timeout: float = 5.0,
    ) -> Optional[SemanticAnalysisResult]:
        """Performs semantic understanding on raw user query using Gemini Flash Lite.
        Returns None if offline or timed out, allowing graceful fallback to local rules.
        """
        if not self.api_key:
            return None

        clean_q = (raw_query or "").strip()
        if not clean_q:
            return None

        if clean_q in self._cache:
            return self._cache[clean_q]

        # Build compact history summary for contextual follow-up awareness
        history_summary = []
        if history:
            for turn in history[-4:]:
                role = getattr(turn, "role", "user")
                content = getattr(turn, "content", "")
                if content:
                    history_summary.append(f"{role.capitalize()}: {content[:120]}")

        prompt_text = f"User query: {clean_q}\nRecent history: {json.dumps(history_summary)}"

        payload = {
            "system_instruction": {
                "parts": [{"text": SEMANTIC_ANALYZER_INSTRUCTION}]
            },
            "contents": [
                {
                    "parts": [{"text": prompt_text}]
                }
            ],
            "generationConfig": {
                "response_mime_type": "application/json",
                "temperature": 0.0,
                "maxOutputTokens": 400,
            },
        }

        for model in self.candidate_models:
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={self.api_key}"
            start_time = time.time()
            try:
                with httpx.Client(timeout=timeout) as client:
                    resp = client.post(url, json=payload)
                    duration = time.time() - start_time
                    if resp.status_code == 200:
                        data = resp.json()
                        candidates = data.get("candidates", [])
                        if not candidates:
                            continue
                        parts = candidates[0].get("content", {}).get("parts", [])
                        if not parts:
                            continue
                        text = parts[0].get("text", "").strip()
                        parsed = json.loads(text)

                        logger.info("SemanticQueryAnalyzer succeeded in %.3fs using %s.", duration, model)
                        res = SemanticAnalysisResult(
                            normalized_query=parsed.get("normalized_query", clean_q),
                            intent=parsed.get("intent", "general_hospital_information"),
                            language=parsed.get("language", "English"),
                            entities=parsed.get("entities", {}) or {},
                            confidence=float(parsed.get("confidence", 0.95)),
                            inferred_department=parsed.get("inferred_department"),
                            is_followup=bool(parsed.get("is_followup", False)),
                        )
                        self._cache[clean_q] = res
                        return res
                    elif resp.status_code in (429, 503):
                        logger.warning("SemanticQueryAnalyzer model %s busy (HTTP %d). Trying next...", model, resp.status_code)
                        continue
                    else:
                        logger.warning("SemanticQueryAnalyzer error status %d: %s", resp.status_code, resp.text[:150])
                        continue
            except httpx.TimeoutException:
                logger.warning("SemanticQueryAnalyzer timed out on model %s after %.1fs. Falling back to local rules.", model, timeout)
                return None
            except Exception as err:
                logger.warning("SemanticQueryAnalyzer model %s failed: %s", model, err)
                continue

        return None
