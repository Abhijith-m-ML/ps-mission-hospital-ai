"""Grounded prompt templates and formatting utilities for the Hospital AI Assistant."""
from typing import Any, List, Optional

HOSPITAL_SYSTEM_INSTRUCTION = """You are the official AI assistant for P.S. Mission Hospital, Maradu, Cochin.
Your duty is to provide clear, compassionate, and strictly accurate information to patients, families, and visitors.

STRICT GROUNDING RULES & MEDICAL SAFETY:
1. Use the provided hospital context as your SOLE source of truth for all hospital-specific facts.
2. The recent conversation history is provided ONLY for conversational context.
3. Do NOT invent, assume, or guess any hospital departments, doctors, services, hours, or policies.
4. If the retrieved context does not contain the information needed to answer the patient's question, clearly state:
   "I couldn't find that information in the available hospital information."
5. Preserve all phone numbers, email addresses, departments, timings, emergency numbers, and directions EXACTLY as written in the context.
6. Never fabricate URLs, sources, or citations. URLs and source pages are managed strictly by the hospital backend.
7. MEDICAL SAFETY:
   - Do NOT provide medical diagnoses or definitive clinical treatment recommendations.
   - When a user describes symptoms (e.g., severe headaches, chest pain, fever, joint pain), provide hospital-information-based guidance by identifying the relevant hospital department and consulting doctors available at P.S. Mission Hospital.
   - Do NOT claim that a specific doctor is medically guaranteed or definitively the only correct doctor based only on symptoms.
   - Always include an appropriate emergency/medical disclaimer advising clinical consultation or emergency desk visit if symptoms are severe, acute, or worsening.
8. RESPONSE FORMAT:
   You MUST return a JSON object with EXACTLY two fields:
   {
     "answer": "<Your compassionate, factual answer in clear plain text>",
     "source_ids": ["<id_1>", "<id_2>"]
   }
   - In "source_ids", list ONLY the exact IDs (from [SOURCE ID: ...]) of the records that directly support your answer.
   - Do NOT include IDs of doctors, departments, or facilities that were not used or mentioned in your answer.
   - If no specific sources apply or no information was found, return: "source_ids": []
   - Ensure the JSON is completely valid and properly escaped.
"""


def build_grounded_conversational_prompt(
    question: str,
    context_text: str,
    history: Optional[List[Any]] = None,
    resolved_query: Optional[str] = None,
    language: str = "en-IN",
) -> str:
    """Builds a grounded prompt combining retrieved hospital facts, recent session history,
    and the patient inquiry, instructing the model to return a structured JSON response with citations.
    
    Args:
        question: User inquiry or follow-up prompt.
        context_text: Grounded context blocks from retrieval layer.
        history: Recent messages in the current temporary session.
        resolved_query: Standalone contextualized query used for retrieval.
        language: Language code ('en-IN', 'ml-IN', or 'hi-IN').
        
    Returns:
        str: Formatted user message string.
    """
    clean_q = (question or "").strip()
    clean_ctx = (context_text or "").strip()

    history_section = ""
    if history:
        history_lines = []
        for msg in history:
            role = getattr(msg, "role", "user") if hasattr(msg, "role") else (msg.get("role", "user") if isinstance(msg, dict) else "user")
            content = getattr(msg, "content", "") if hasattr(msg, "content") else (msg.get("content", "") if isinstance(msg, dict) else "")
            speaker = "Patient" if role == "user" else "Assistant"
            if content:
                history_lines.append(f"{speaker}: {content}")
        if history_lines:
            history_section = (
                "=== RECENT CONVERSATION HISTORY (TEMPORARY SESSION CONTEXT ONLY) ===\n"
                + "\n".join(history_lines)
                + "\n=== END OF CONVERSATION HISTORY ===\n\n"
            )

    resolved_line = (
        f"\n(Context-Resolved Question Intent: {resolved_query})"
        if (resolved_query and resolved_query != clean_q)
        else ""
    )

    # Tailored language instructions
    if language == "ml-IN":
        lang_instruction = (
            "4. Language Requirement: Answer the question in clear, natural, and polite Malayalam (മലയാളത്തിൽ മറുപടി നൽകുക).\n"
            "   - CRITICAL: DO NOT mistranslate doctor names, qualifications (MBBS, MD, MS, etc.), consultation hours, phone numbers, or URLs.\n"
            "   - Keep doctor names readable in English or standard Malayalam transliteration (e.g. ഡോ. ആനി ഷീല / Dr. Sr. Annie Sheela).\n"
            "   - If information was not found, return: \"answer\": \"ലഭ്യമായ ആശുപത്രി വിവരങ്ങളിൽ ആ വിവരം കണ്ടെത്താൻ കഴിഞ്ഞില്ല.\""
        )
    elif language == "hi-IN":
        lang_instruction = (
            "4. Language Requirement: Answer the question in clear, natural, and polite Hindi (हिन्दी में उत्तर दें).\n"
            "   - CRITICAL: DO NOT mistranslate doctor names, qualifications (MBBS, MD, MS, etc.), consultation hours, phone numbers, or URLs.\n"
            "   - Keep doctor names readable in English or standard Hindi transliteration (e.g. डॉ. एनी शीला / Dr. Sr. Annie Sheela).\n"
            "   - If information was not found, return: \"answer\": \"उपलब्ध अस्पताल की जानकारी में यह जानकारी नहीं मिली।\""
        )
    else:
        lang_instruction = (
            "4. Language Requirement: Answer the question in clear, compassionate, and professional English."
        )

    return f"""Please answer the patient's question based strictly on the verified hospital context provided below:

{history_section}=== VERIFIED HOSPITAL CONTEXT ===
{clean_ctx}
=== END OF CONTEXT ===

Patient Question: {clean_q}{resolved_line}


INSTRUCTIONS:
1. Grounding: All hospital facts, doctor names, OP schedules, and facilities MUST come EXCLUSIVELY from the VERIFIED HOSPITAL CONTEXT above.
2. Medical Safety: Do not make medical diagnoses. Guide the patient on which department/consultants at P.S. Mission Hospital evaluate these conditions, and provide an appropriate medical/emergency disclaimer. Do not claim a doctor is medically guaranteed based solely on symptoms.
3. Output Format: Return a VALID JSON object with this exact structure:
{{
  "answer": "Grounded answer text here...",
  "source_ids": ["id_1", "id_2"]
}}
- In "source_ids", include ONLY the exact chunk IDs (from [SOURCE ID: ...]) of records that directly contributed to your answer.
- Do NOT include IDs of unrelated doctors or departments.
- If information was not found, return:
{{
  "answer": "I couldn't find that information in the available hospital information.",
  "source_ids": []
}}
{lang_instruction}"""



def build_grounded_user_prompt(question: str, context_text: str) -> str:
    """Legacy backward-compatible wrapper combining factual context with question."""
    return build_grounded_conversational_prompt(question=question, context_text=context_text)

