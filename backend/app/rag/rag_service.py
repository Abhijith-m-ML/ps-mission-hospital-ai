import json
import os
import re
import time
from typing import Any, Dict, List, Optional, Set, Tuple

from app.core.config import settings
from app.core.logging_config import logger
from app.models.schemas import ChatMessage, ChatResponse, RetrievalChunkItem, SourceItem
from app.rag.llm import BaseLLMProvider, get_llm_provider
from app.rag.prompts import (
    HOSPITAL_SYSTEM_INSTRUCTION,
    build_grounded_conversational_prompt,
)
from app.retrieval.query_resolver import QueryResolver
from app.retrieval.structured_retriever import StructuredRetriever
from app.voice.language import resolve_language


NO_CONTEXT_DEFAULT_REPLY = (
    "I couldn't find that information in the available hospital information."
)


def get_no_context_reply(language: str) -> str:
    """Returns polite no-context fallback in the appropriate language."""
    if language == "ml-IN":
        return "ലഭ്യമായ ആശുപത്രി വിവരങ്ങളിൽ ആ വിവരം കണ്ടെത്താൻ കഴിഞ്ഞില്ല."
    elif language == "hi-IN":
        return "उपलब्ध अस्पताल की जानकारी में यह जानकारी नहीं मिली।"
    return NO_CONTEXT_DEFAULT_REPLY


def format_doctor_display_name(name: Optional[str]) -> Optional[str]:
    """Cleans and standardizes doctor names for display."""
    if not name:
        return None
    cleaned = name.strip()
    cleaned = re.sub(r"\bJHON\b", "John", cleaned, flags=re.IGNORECASE)
    if cleaned.isupper() or cleaned.islower():
        words = cleaned.split()
        capitalized = []
        for w in words:
            if w.lower() in ("dr.", "dr", "sr.", "sr"):
                capitalized.append(w.capitalize())
            elif len(w) == 1:
                capitalized.append(w.upper())
            else:
                capitalized.append(w.capitalize())
        cleaned = " ".join(capitalized)
    return cleaned


def get_source_priority(item: SourceItem) -> int:
    """Returns priority rank (1 is most specific, 4 is least specific).
    Priority:
    1. Individual doctor page
    2. Department page
    3. Relevant hospital page / facility
    4. Generic homepage / contact page
    """
    ctype = (item.content_type or "").lower()
    url = (item.url or "").lower().rstrip("/")

    # 1. Individual doctor page
    if ctype == "doctor" or item.doctor_name:
        return 1

    # 4. Generic homepage or contact
    if url in (
        "https://www.psmissionhospital.org",
        "http://www.psmissionhospital.org",
        "https://psmissionhospital.org",
        "http://psmissionhospital.org",
        "https://www.psmissionhospital.org/contact",
        "https://www.psmissionhospital.org/contact-us",
        "",
    ):
        return 4

    # 2. Department page
    if ctype == "department" or "/category/department/" in url or "department" in (item.title or "").lower():
        return 2

    # 3. Relevant hospital page / facility / services
    return 3


def deduplicate_sources(sources: List[SourceItem]) -> List[SourceItem]:
    """Deduplicates sources prioritizing the most specific source.
    Rules:
    - Deduplicate identical URLs.
    - Prefer most specific source: Doctor (1) > Department (2) > Hospital Page (3) > Homepage (4).
    - If multiple distinct doctors share a URL (e.g. department page), preserve each distinct doctor.
    - If a doctor and a department share the same URL, prefer the doctor page.
    """
    doctor_sources: List[SourceItem] = []
    seen_doctor_names: Set[str] = set()
    other_sources_by_url: Dict[str, Tuple[int, SourceItem]] = {}
    doctor_urls: Set[str] = set()

    # Sort candidates by priority (1 to 4)
    sorted_sources = sorted(sources, key=get_source_priority)

    for item in sorted_sources:
        prio = get_source_priority(item)
        norm_url = (item.url or "").strip().rstrip("/").lower()

        if prio == 1 and item.doctor_name:
            doc_key = item.doctor_name.lower().strip()
            if doc_key not in seen_doctor_names:
                seen_doctor_names.add(doc_key)
                doctor_sources.append(item)
                if norm_url:
                    doctor_urls.add(norm_url)
        else:
            # If a doctor already claimed this exact URL, do not show redundant generic department/homepage
            if norm_url and norm_url in doctor_urls:
                continue
            if not norm_url:
                continue
            if norm_url not in other_sources_by_url:
                other_sources_by_url[norm_url] = (prio, item)
            else:
                existing_prio, _ = other_sources_by_url[norm_url]
                if prio < existing_prio:
                    other_sources_by_url[norm_url] = (prio, item)

    final_sources = list(doctor_sources)
    for _, item in other_sources_by_url.values():
        final_sources.append(item)

    final_sources.sort(key=get_source_priority)
    return final_sources


def parse_llm_response(
    raw_output: str,
    retrieved_chunks: List[RetrievalChunkItem],
) -> Tuple[str, List[str]]:
    """Extracts answer text and cited source IDs from hosted LLM output.
    Supports JSON output format {"answer": "...", "source_ids": [...]},
    with robust fallbacks for markdown code fences and entity mentions.
    """
    text = (raw_output or "").strip()
    clean_text = text

    # Strip markdown code blocks (e.g. ```json ... ```)
    if clean_text.startswith("```"):
        clean_text = re.sub(r"^```(?:json)?\s*", "", clean_text, flags=re.IGNORECASE)
        clean_text = re.sub(r"\s*```$", "", clean_text)
        clean_text = clean_text.strip()

    answer: Optional[str] = None
    source_ids: List[str] = []

    # 1. Attempt JSON Parsing
    try:
        data = json.loads(clean_text)
        if isinstance(data, dict):
            answer = str(data.get("answer", "")).strip()
            raw_sids = data.get("source_ids", [])
            if isinstance(raw_sids, list):
                source_ids = [str(sid).strip() for sid in raw_sids if sid]
    except Exception:
        # Check for JSON object embedded within response text
        json_match = re.search(r"\{[\s\S]*\"answer\"[\s\S]*\}", clean_text)
        if json_match:
            try:
                data = json.loads(json_match.group(0))
                if isinstance(data, dict):
                    answer = str(data.get("answer", "")).strip()
                    raw_sids = data.get("source_ids", [])
                    if isinstance(raw_sids, list):
                        source_ids = [str(sid).strip() for sid in raw_sids if sid]
            except Exception:
                pass

    # 2. Fallback: If answer is not parsed from JSON, treat raw text as answer
    if not answer:
        answer = text

    # 3. Check for "No context found" responses
    if NO_CONTEXT_DEFAULT_REPLY.lower() in answer.lower():
        return answer, []

    # 4. Fallback: If no source_ids were extracted, match chunks via entity mentions in answer
    if not source_ids and retrieved_chunks:
        answer_lower = answer.lower()
        matched_ids: List[str] = []

        # Check doctor mentions
        for chunk in retrieved_chunks:
            meta = chunk.metadata or {}
            doc_name = meta.get("doctor_name")
            if doc_name:
                doc_lower = doc_name.lower()
                clean_doc = re.sub(r"^(?:dr\.?|sr\.?)\s*", "", doc_lower).strip()
                tokens = [t for t in clean_doc.split() if len(t) > 2]
                if doc_lower in answer_lower or (tokens and any(t in answer_lower for t in tokens)):
                    matched_ids.append(chunk.chunk_id)

        # If doctors matched, use them
        if matched_ids:
            source_ids = matched_ids
        else:
            # Check facility mentions
            for chunk in retrieved_chunks:
                meta = chunk.metadata or {}
                fac_name = meta.get("facility_name")
                if fac_name and fac_name.lower() in answer_lower:
                    matched_ids.append(chunk.chunk_id)

            if matched_ids:
                source_ids = matched_ids
            else:
                # Check department mentions
                for chunk in retrieved_chunks:
                    sec_lower = (chunk.section or "").lower()
                    if sec_lower and sec_lower != "general" and sec_lower in answer_lower:
                        matched_ids.append(chunk.chunk_id)
                if matched_ids:
                    source_ids = matched_ids
                elif len(retrieved_chunks) <= 2:
                    # General query fallback
                    source_ids = [c.chunk_id for c in retrieved_chunks]

    return answer, source_ids


class RAGService:
    """Orchestrates end-to-end Retrieval-Augmented Generation for hospital inquiries.
    
    Pipeline:
    User Message + History -> QueryResolver -> StructuredRetriever -> Grounded Prompt -> Hosted LLM -> Answer + Sources
    """

    def __init__(
        self,
        retriever: Optional[Any] = None,
        llm_provider: Optional[BaseLLMProvider] = None,
        query_resolver: Optional[QueryResolver] = None,
    ):
        self.retriever = retriever or StructuredRetriever()
        self.llm_provider = llm_provider or get_llm_provider()
        self.query_resolver = query_resolver or QueryResolver(
            entity_store=getattr(self.retriever, "entity_store", None)
        )

    def answer_question(
        self,
        message: str,
        history: Optional[List[ChatMessage]] = None,
        session_id: Optional[str] = None,
        language: Optional[str] = "en-IN",
        top_k: Optional[int] = None,
        min_score: Optional[float] = None,
        namespace: Optional[str] = None,
        debug: Optional[bool] = False,
    ) -> ChatResponse:
        """Processes patient question, resolves conversational follow-ups, retrieves factual
        context, queries hosted LLM, and returns grounded answer with source citations.
        
        Args:
            message: User's question or message.
            history: Recent conversation history in the current temporary session.
            session_id: Client-generated temporary session UUID.
            language: Requested language code ('en-IN', 'ml-IN', 'hi-IN', or 'auto').
            top_k: Number of chunks to retrieve.
            min_score: Minimum similarity score cutoff.
            namespace: Hospital namespace partition.
            debug: Optional flag to return detailed development diagnostics.
            
        Returns:
            ChatResponse: Pydantic model with answer string, sources, session_id, resolved_query,
                          raw_transcript, and optional debug_pipeline trace.
        """
        start_time = time.time()
        clean_msg = (message or "").strip()
        if not clean_msg:
            raise ValueError("Message cannot be empty.")

        # Resolve target response language (English, Malayalam, Hindi)
        target_lang = resolve_language(language, clean_msg)

        # 1. Enforce Memory Window Limit (Configurable)
        history_limit = settings.MAX_HISTORY_MESSAGES
        recent_history = (history or [])[-history_limit:] if history else []

        # 2. History-Aware Conversational Query Resolution
        resolution = self.query_resolver.resolve(clean_msg, recent_history)
        resolved_query = resolution.resolved_query

        # 3. Retrieve Candidate Chunks & Assemble Grounded Context
        # Enforce IMPORTANT RETRIEVAL RULE:
        # Only apply department filters when the intent actually requires a department.
        effective_department = resolution.department
        intents_without_dept = {
            "contact_information",
            "hospital_hours",
            "location",
            "emergency_information",
            "services",
            "general_hospital_information",
            "insurance",
            "general_question",
        }
        dept_in_query = bool(resolution.entities.get("department")) if hasattr(resolution, "entities") else False
        if str(resolution.intent).lower() in intents_without_dept and not dept_in_query:
            effective_department = None

        entities_dict = getattr(resolution, "entities", {})
        if str(resolution.intent).lower() == "symptom_to_department" and not effective_department:
            effective_department = entities_dict.get("inferred_department")

        retrieval_filters = {
            "department": effective_department,
            "doctor": resolution.target_doctor,
            "intent": resolution.intent,
        }
        retrieval_response = self.retriever.retrieve(
            query=resolved_query,
            department=effective_department,
            doctor_name=resolution.target_doctor,
            intent=resolution.intent,
            top_k=top_k,
            min_score=min_score,
            namespace=namespace,
        )

        # 4. Compute Diagnostic Counts (No secrets logged)
        structured_result_count = len([
            r for r in (retrieval_response.results or [])
            if r.source != "website"
        ])
        semantic_result_count = len([
            r for r in (retrieval_response.results or [])
            if r.source == "website"
        ])
        final_context_count = len(retrieval_response.results or [])

        # Check debug mode
        is_debug = bool(
            debug
            or settings.DEBUG_QUERY_PIPELINE
            or os.getenv("DEBUG_QUERY_PIPELINE", "false").lower() in ("true", "1")
        )

        pinecone_results_preview = [
            {"id": r.chunk_id, "score": round(r.score, 4), "section": r.section, "source": r.source}
            for r in (retrieval_response.results or [])
            if r.source == "website"
        ]

        # Development query diagnostics in required format
        logger.info("==================================================")
        logger.info("DEVELOPMENT QUERY DIAGNOSTICS:")
        logger.info("RAW USER INPUT:\n%s", clean_msg)
        logger.info("DETECTED LANGUAGE:\n%s", getattr(resolution, "language", target_lang))
        logger.info("NORMALIZED QUERY:\n%s", resolved_query)
        logger.info("INTENT:\n%s", resolution.intent)
        logger.info("ENTITIES:\n%s", json.dumps(entities_dict))
        logger.info("RETRIEVAL QUERY:\n%s", resolved_query)
        logger.info("FILTERS:\n%s", json.dumps(retrieval_filters))
        logger.info("PINECONE RESULTS:\n%s", json.dumps(pinecone_results_preview))
        logger.info("FINAL CONTEXT:\n%s", retrieval_response.context.context_text[:500] if retrieval_response.context else "None")
        logger.info("raw_query: '%s'", clean_msg)
        logger.info("conversation_history_used: %s", resolution.used_conversation_context)
        logger.info("department: %s", effective_department)
        logger.info("doctor: %s", resolution.target_doctor)
        logger.info("facility: %s", entities_dict.get("facility"))
        logger.info("Pinecone result count: %d", semantic_result_count)
        logger.info("structured result count: %d", structured_result_count)
        logger.info("final context count: %d", final_context_count)
        logger.info("session_id: %s", session_id or "anonymous")
        logger.info("target_language: %s (requested: %s)", target_lang, language)
        logger.info("==================================================")

        debug_pipeline_data = None
        if is_debug:
            retrieved_docs = [
                {
                    "chunk_id": r.chunk_id,
                    "score": round(r.score, 4),
                    "section": r.section,
                    "title": r.title,
                    "source": r.source,
                    "url": r.url,
                    "content": r.content[:200] if r.content else "",
                }
                for r in (retrieval_response.results or [])
            ]
            debug_pipeline_data = {
                "raw_transcript": clean_msg,
                "language": getattr(resolution, "language", target_lang),
                "normalized_query": resolved_query,
                "intent": str(resolution.intent),
                "entities": entities_dict,
                "retrieval_query": resolved_query,
                "filters": retrieval_filters,
                "retrieved_documents": retrieved_docs,
                "final_context": retrieval_response.context.context_text if retrieval_response.context else "",
            }

        # 5. Check No-Context Behavior
        no_ctx_reply = get_no_context_reply(target_lang)
        if not retrieval_response.has_context or not retrieval_response.results:
            logger.info("No relevant hospital context found for: '%s'. Bypassing LLM call.", resolved_query[:80])
            return ChatResponse(
                answer=no_ctx_reply,
                sources=[],
                source_ids=[],
                session_id=session_id,
                resolved_query=resolved_query,
                raw_transcript=clean_msg,
                debug_pipeline=debug_pipeline_data,
            )

        context_text = retrieval_response.context.context_text.strip()
        if not context_text:
            logger.info("Context text was empty after assembly. Returning controlled fallback.")
            return ChatResponse(
                answer=no_ctx_reply,
                sources=[],
                source_ids=[],
                session_id=session_id,
                resolved_query=resolved_query,
                raw_transcript=clean_msg,
                debug_pipeline=debug_pipeline_data,
            )

        # 6. Build Grounded Prompt with History and Grounded Context
        user_prompt = build_grounded_conversational_prompt(
            question=clean_msg,
            context_text=context_text,
            history=recent_history,
            resolved_query=resolved_query,
            language=target_lang,
        )

        # 7. Generate Answer via Hosted LLM Provider
        logger.info("Calling hosted LLM provider to generate grounded conversational answer...")
        raw_llm_output = self.llm_provider.generate(
            system_instruction=HOSPITAL_SYSTEM_INSTRUCTION,
            user_message=user_prompt,
        )

        # 8. Parse LLM Answer and Source IDs
        answer_text, cited_source_ids = parse_llm_response(
            raw_output=raw_llm_output,
            retrieved_chunks=retrieval_response.results,
        )

        # 9. Map Selected Source IDs to Original Trusted Metadata
        chunks_by_id: Dict[str, RetrievalChunkItem] = {
            chunk.chunk_id: chunk for chunk in retrieval_response.results
        }

        raw_selected_sources: List[SourceItem] = []
        for s_id in cited_source_ids:
            chunk = chunks_by_id.get(s_id)
            if chunk:
                meta = chunk.metadata or {}
                doc_name = format_doctor_display_name(meta.get("doctor_name"))
                dept = meta.get("department", chunk.section)
                raw_selected_sources.append(
                    SourceItem(
                        chunk_id=chunk.chunk_id,
                        document_id=chunk.document_id or f"doc_{chunk.chunk_id}",
                        title=chunk.title or f"{dept} Department",
                        url=chunk.url,
                        section=chunk.section,
                        content_type=str(meta.get("content_type", "webpage")),
                        doctor_name=doc_name,
                        department=dept,
                        image_url=meta.get("image_url"),
                    )
                )

        # Fallback if cited IDs did not match candidate IDs
        if not raw_selected_sources and retrieval_response.results and NO_CONTEXT_DEFAULT_REPLY.lower() not in answer_text.lower():
            seen_cids = set()
            for chunk in retrieval_response.results:
                if chunk.chunk_id not in seen_cids:
                    seen_cids.add(chunk.chunk_id)
                    meta = chunk.metadata or {}
                    doc_name = format_doctor_display_name(meta.get("doctor_name"))
                    dept = meta.get("department", chunk.section)
                    raw_selected_sources.append(
                        SourceItem(
                            chunk_id=chunk.chunk_id,
                            document_id=chunk.document_id or f"doc_{chunk.chunk_id}",
                            title=chunk.title or f"{dept} Department",
                            url=chunk.url,
                            section=chunk.section,
                            content_type=str(meta.get("content_type", "webpage")),
                            doctor_name=doc_name,
                            department=dept,
                            image_url=meta.get("image_url"),
                        )
                    )

        # 10. Deduplicate and Sort by Specificity Priority
        final_sources = deduplicate_sources(raw_selected_sources)
        final_source_ids = [s.chunk_id for s in final_sources]

        duration = time.time() - start_time
        logger.info(
            "RAG response generated in %.3fs with %d cited sources [session_id=%s].",
            duration,
            len(final_sources),
            session_id or "anonymous",
        )

        return ChatResponse(
            answer=answer_text,
            sources=final_sources,
            source_ids=final_source_ids,
            session_id=session_id,
            resolved_query=resolved_query,
            raw_transcript=clean_msg,
            debug_pipeline=debug_pipeline_data,
        )

