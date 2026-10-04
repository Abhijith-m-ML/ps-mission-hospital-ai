import re
import time
from typing import Any, Dict, List, Optional

from app.core.logging_config import logger
from app.models.schemas import (
    ContextResult,
    Department,
    Doctor,
    Facility,
    RetrievalChunkItem,
    RetrievalDebugInfo,
    RetrievalPreviewResponse,
)
from app.retrieval.entity_store import HospitalEntityStore
from app.retrieval.intent_analyzer import IntentAnalyzer, QueryIntent
from app.retrieval.reranker import LightweightReranker
from app.retrieval.retriever import Retriever


class StructuredRetriever:
    """Unified retrieval abstraction coordinating structured hospital entity lookups
    (Department, Doctor, Facility, Schedule) with semantic vector search in Pinecone.
    
    Architecture:
    User Query -> Intent & Entity Analyzer
      ├─ doctor_search    -> EntityStore (doctors by department / doctor name)
      ├─ facility_search  -> EntityStore (facilities by department)
      ├─ schedule_search  -> EntityStore (doctor OP timing schedule)
      ├─ department_search-> EntityStore (department overview)
      └─ general_rag      -> VectorStore / Retriever (semantic text chunks)
    """

    def __init__(
        self,
        entity_store: Optional[HospitalEntityStore] = None,
        intent_analyzer: Optional[IntentAnalyzer] = None,
        semantic_retriever: Optional[Retriever] = None,
        reranker: Optional[LightweightReranker] = None,
    ):
        self.entity_store = entity_store or HospitalEntityStore()
        self.intent_analyzer = intent_analyzer or IntentAnalyzer(self.entity_store)
        self.semantic_retriever = semantic_retriever or Retriever()
        self.reranker = reranker or LightweightReranker()

    def retrieve_doctors(
        self,
        department: Optional[str] = None,
        doctor_name: Optional[str] = None,
    ) -> List[Doctor]:
        """Retrieves structured doctor records."""
        if doctor_name:
            doc = self.entity_store.get_doctor_by_name(doctor_name)
            return [doc] if doc else []
        if department:
            return self.entity_store.get_doctors_by_department(department)
        return self.entity_store.doctors

    def retrieve_facilities(self, department: Optional[str] = None) -> List[Facility]:
        """Retrieves structured clinical facility records."""
        if department:
            return self.entity_store.get_facilities_by_department(department)
        return self.entity_store.facilities

    def retrieve_department(self, department: str) -> Optional[Department]:
        """Retrieves structured department record."""
        return self.entity_store.get_department(department)

    def retrieve_semantic(self, query: str, **kwargs) -> RetrievalPreviewResponse:
        """Executes Pinecone vector semantic search."""
        return self.semantic_retriever.retrieve(query=query, **kwargs)

    def retrieve(
        self,
        query: str,
        department: Optional[str] = None,
        doctor_name: Optional[str] = None,
        intent: Optional[str] = None,
        top_k: Optional[int] = None,
        min_score: Optional[float] = None,
        namespace: Optional[str] = None,
        **kwargs,
    ) -> RetrievalPreviewResponse:
        """Primary retrieval entry point. Determines query intent, routes to either
        structured entity lookup or semantic Pinecone search, and formats the output
        as a standard RetrievalPreviewResponse for downstream LLM synthesis.
        """
        start_time = time.time()
        clean_query = (query or "").strip()
        if not clean_query:
            raise ValueError("Query string cannot be empty or blank.")

        # 1. Analyze Query Intent & Extract Entities
        parsed_intent = self.intent_analyzer.analyze(clean_query)

        # Apply explicit entity and intent overrides if provided from resolution
        if intent:
            intent_lower = str(intent).lower()
            if intent_lower in ("doctor_search",):
                parsed_intent.intent = QueryIntent.DOCTOR_SEARCH
            elif intent_lower in ("doctor_schedule", "schedule_search"):
                parsed_intent.intent = QueryIntent.SCHEDULE_SEARCH
            elif intent_lower in ("facility_information", "facility_search"):
                parsed_intent.intent = QueryIntent.FACILITY_SEARCH
            elif intent_lower in ("department_information", "department_search"):
                parsed_intent.intent = QueryIntent.DEPARTMENT_SEARCH
            elif intent_lower in ("symptom_to_department",):
                # When department is identified for symptom, search doctors in that department
                parsed_intent.intent = QueryIntent.DOCTOR_SEARCH
            else:
                try:
                    parsed_intent.intent = QueryIntent(intent)
                except ValueError:
                    parsed_intent.intent = QueryIntent.GENERAL_RAG

        # Enforce IMPORTANT RETRIEVAL RULE:
        # Only apply department filters when the intent actually requires a department.
        NO_DEPARTMENT_INTENTS = {
            "contact_information",
            "hospital_hours",
            "location",
            "emergency_information",
            "services",
            "general_hospital_information",
            "insurance",
            "general_question",
        }
        raw_intent_str = str(intent).lower() if intent else ""
        if raw_intent_str in NO_DEPARTMENT_INTENTS:
            parsed_intent.department = None
        elif department:
            parsed_intent.department = department

        if doctor_name:
            parsed_intent.doctor_name = doctor_name

        logger.info(
            "StructuredRetriever: intent=%s, dept='%s', doctor='%s'",
            parsed_intent.intent.value,
            parsed_intent.department,
            parsed_intent.doctor_name,
        )

        # 2. Route According to Intent
        # -------------------------------------------------------------
        # Case A: Doctor Search
        # -------------------------------------------------------------
        if parsed_intent.intent == QueryIntent.DOCTOR_SEARCH:
            doctors = self.retrieve_doctors(
                department=parsed_intent.department,
                doctor_name=parsed_intent.doctor_name,
            )
            if doctors:
                return self._build_doctor_retrieval_response(
                    query=clean_query,
                    doctors=doctors,
                    department=parsed_intent.department or (doctors[0].department if doctors else "Hospital"),
                    start_time=start_time,
                )

        # -------------------------------------------------------------
        # Case B: Schedule Search
        # -------------------------------------------------------------
        elif parsed_intent.intent == QueryIntent.SCHEDULE_SEARCH:
            if parsed_intent.doctor_name:
                doc = self.entity_store.get_doctor_by_name(parsed_intent.doctor_name)
                if doc:
                    return self._build_schedule_retrieval_response(
                        query=clean_query,
                        doctor=doc,
                        start_time=start_time,
                    )
            elif parsed_intent.department:
                # Department schedule query: fetch doctors for this department
                doctors = self.retrieve_doctors(department=parsed_intent.department)
                if doctors:
                    return self._build_doctor_retrieval_response(
                        query=clean_query,
                        doctors=doctors,
                        department=parsed_intent.department,
                        start_time=start_time,
                    )

        # -------------------------------------------------------------
        # Case C: Facility Search
        # -------------------------------------------------------------
        elif parsed_intent.intent == QueryIntent.FACILITY_SEARCH and parsed_intent.department:
            facilities = self.retrieve_facilities(department=parsed_intent.department)
            if facilities:
                return self._build_facility_retrieval_response(
                    query=clean_query,
                    facilities=facilities,
                    department=parsed_intent.department,
                    start_time=start_time,
                )

        # -------------------------------------------------------------
        # Case D: Department Search
        # -------------------------------------------------------------
        elif parsed_intent.intent == QueryIntent.DEPARTMENT_SEARCH and parsed_intent.department:
            dept = self.retrieve_department(parsed_intent.department)
            if dept:
                return self._build_department_retrieval_response(
                    query=clean_query,
                    department=dept,
                    start_time=start_time,
                )

        # -------------------------------------------------------------
        # Case E: Semantic Vector Search (Fallback for general RAG)
        # -------------------------------------------------------------
        logger.info("Executing standard semantic text retrieval for query: '%s'", clean_query[:60])
        semantic_res = self.semantic_retriever.retrieve(
            query=clean_query,
            top_k=top_k,
            min_score=min_score,
            namespace=namespace,
            **kwargs,
        )

        # Apply reranking to semantic candidate chunks
        if semantic_res.results and self.reranker:
            rerank_entities = {}
            if parsed_intent.intent in (QueryIntent.DOCTOR_SEARCH, QueryIntent.SCHEDULE_SEARCH):
                if parsed_intent.doctor_name:
                    rerank_entities["doctor"] = parsed_intent.doctor_name
                if parsed_intent.department:
                    rerank_entities["department"] = parsed_intent.department
            elif parsed_intent.intent in (QueryIntent.DEPARTMENT_SEARCH, QueryIntent.FACILITY_SEARCH):
                if parsed_intent.department:
                    rerank_entities["department"] = parsed_intent.department

            reranked_chunks = self.reranker.rerank(
                query=clean_query,
                chunks=semantic_res.results,
                entities=rerank_entities,
            )
            semantic_res.results = reranked_chunks
            context_res = self.semantic_retriever.context_builder.build_context(reranked_chunks)
            semantic_res.context = context_res

        # Fallback to structured department info ONLY for department-specific intents
        if (
            not semantic_res.has_context
            and parsed_intent.department
            and parsed_intent.intent in (QueryIntent.DEPARTMENT_SEARCH, QueryIntent.FACILITY_SEARCH, QueryIntent.DOCTOR_SEARCH)
        ):
            dept = self.retrieve_department(parsed_intent.department)
            if dept and dept.description:
                logger.info("Falling back to structured department info for '%s'", parsed_intent.department)
                return self._build_department_retrieval_response(
                    query=clean_query,
                    department=dept,
                    start_time=start_time,
                )

        return semantic_res

    def _build_doctor_retrieval_response(
        self,
        query: str,
        doctors: List[Doctor],
        department: str,
        start_time: float,
    ) -> RetrievalPreviewResponse:
        """Formats verified doctor records into RetrievalPreviewResponse."""
        items: List[RetrievalChunkItem] = []
        context_blocks: List[str] = [
            f"=== VERIFIED HOSPITAL DOCTOR RECORDS ===",
            f"Department: {department}",
            f"Total Doctors Found: {len(doctors)}\n",
        ]

        for idx, doc in enumerate(doctors, start=1):
            chunk_id = f"doctor_{self._slugify(doc.doctor_name)}"
            doc_lines = [
                f"--- DOCTOR RECORD {idx} [SOURCE ID: {chunk_id}] ---",
                f"Doctor Name: {doc.doctor_name}",
                f"Department: {doc.department or department}",
                f"Qualification: {doc.qualification if doc.qualification else 'Specialist Consultant'}",
                f"OP Consultation Schedule: {doc.schedule_text if doc.schedule_text else 'Hospital outpatient schedule'}",
                f"Profile Image URL: {doc.image_url if doc.image_url else 'N/A'}",
                f"Source URL: {doc.source_url if doc.source_url else 'https://www.psmissionhospital.org/doctors/'}",
            ]
            block_text = "\n".join(doc_lines)
            context_blocks.append(block_text)

            metadata = {
                "content_type": "doctor",
                "doctor_name": doc.doctor_name,
                "qualification": doc.qualification,
                "department": doc.department or department,
                "schedule_text": doc.schedule_text,
                "image_url": doc.image_url,
                "source_url": doc.source_url,
                "source_title": doc.source_title or f"Consultants - {doc.doctor_name}",
            }

            items.append(
                RetrievalChunkItem(
                    rank=idx,
                    chunk_id=chunk_id,
                    document_id=f"doc_{chunk_id}",
                    score=0.99,
                    content=block_text,
                    section=doc.department or department,
                    url=doc.source_url or "https://www.psmissionhospital.org/doctors/",
                    title=f"Doctor Profile: {doc.doctor_name}",
                    source="structured_entity",
                    metadata=metadata,
                )
            )

        full_context = "\n\n".join(context_blocks)
        duration = time.time() - start_time

        return RetrievalPreviewResponse(
            query=query,
            has_context=True,
            results=items,
            context=ContextResult(
                has_context=True,
                context_text=full_context,
                total_chunks=len(items),
                total_characters=len(full_context),
            ),
            debug=RetrievalDebugInfo(
                pinecone_top_k=len(items),
                candidates_retrieved=len(items),
                removed_by_threshold=0,
                removed_duplicates=0,
                final_chunks_count=len(items),
                context_character_count=len(full_context),
            ),
            retrieval_duration_seconds=round(duration, 3),
        )

    def _build_schedule_retrieval_response(
        self,
        query: str,
        doctor: Doctor,
        start_time: float,
    ) -> RetrievalPreviewResponse:
        """Formats doctor OP schedule into RetrievalPreviewResponse."""
        chunk_id = f"schedule_{self._slugify(doctor.doctor_name)}"
        content = (
            f"=== VERIFIED DOCTOR OP SCHEDULE [SOURCE ID: {chunk_id}] ===\n"
            f"Doctor Name: {doctor.doctor_name}\n"
            f"Department: {doctor.department}\n"
            f"Qualification: {doctor.qualification if doctor.qualification else 'Specialist Consultant'}\n"
            f"OP Consultation Timings: {doctor.schedule_text if doctor.schedule_text else 'Please check with hospital reception for OP schedule'}\n"
            f"Profile Photo: {doctor.image_url if doctor.image_url else 'N/A'}\n"
            f"Source URL: {doctor.source_url}"
        )

        metadata = {
            "content_type": "doctor",
            "doctor_name": doctor.doctor_name,
            "qualification": doctor.qualification,
            "department": doctor.department,
            "schedule_text": doctor.schedule_text,
            "image_url": doctor.image_url,
            "source_url": doctor.source_url,
        }

        item = RetrievalChunkItem(
            rank=1,
            chunk_id=chunk_id,
            document_id=f"doc_{chunk_id}",
            score=0.99,
            content=content,
            section=doctor.department or "Consultants",
            url=doctor.source_url or "https://www.psmissionhospital.org/doctors/",
            title=f"Schedule: {doctor.doctor_name}",
            source="structured_entity",
            metadata=metadata,
        )

        duration = time.time() - start_time
        return RetrievalPreviewResponse(
            query=query,
            has_context=True,
            results=[item],
            context=ContextResult(
                has_context=True,
                context_text=content,
                total_chunks=1,
                total_characters=len(content),
            ),
            debug=RetrievalDebugInfo(
                pinecone_top_k=1,
                candidates_retrieved=1,
                removed_by_threshold=0,
                removed_duplicates=0,
                final_chunks_count=1,
                context_character_count=len(content),
            ),
            retrieval_duration_seconds=round(duration, 3),
        )

    def _build_facility_retrieval_response(
        self,
        query: str,
        facilities: List[Facility],
        department: str,
        start_time: float,
    ) -> RetrievalPreviewResponse:
        """Formats department facilities into RetrievalPreviewResponse."""
        items: List[RetrievalChunkItem] = []
        lines = [
            f"=== VERIFIED DEPARTMENT FACILITIES & SERVICES ===",
            f"Department: {department}",
            f"Total Facilities / Services Found: {len(facilities)}\n",
        ]

        for idx, fac in enumerate(facilities, start=1):
            chunk_id = f"facility_{self._slugify(department)}_{idx}"
            f_text = f"• [SOURCE ID: {chunk_id}] {fac.facility_name}"
            if fac.description and "clinical" not in fac.description.lower():
                f_text += f": {fac.description}"
            lines.append(f_text)

            metadata = {
                "content_type": "facility",
                "facility_name": fac.facility_name,
                "description": fac.description,
                "department": fac.department or department,
                "source_url": fac.source_url,
            }
            items.append(
                RetrievalChunkItem(
                    rank=idx,
                    chunk_id=chunk_id,
                    document_id=f"doc_{chunk_id}",
                    score=0.95,
                    content=f"Facility: {fac.facility_name}\nDepartment: {department}",
                    section=department,
                    url=fac.source_url or "",
                    title=f"Facilities: {department}",
                    source="structured_entity",
                    metadata=metadata,
                )
            )

        full_context = "\n".join(lines)
        duration = time.time() - start_time

        return RetrievalPreviewResponse(
            query=query,
            has_context=True,
            results=items,
            context=ContextResult(
                has_context=True,
                context_text=full_context,
                total_chunks=len(items),
                total_characters=len(full_context),
            ),
            debug=RetrievalDebugInfo(
                pinecone_top_k=len(items),
                candidates_retrieved=len(items),
                removed_by_threshold=0,
                removed_duplicates=0,
                final_chunks_count=len(items),
                context_character_count=len(full_context),
            ),
            retrieval_duration_seconds=round(duration, 3),
        )

    def _build_department_retrieval_response(
        self,
        query: str,
        department: Department,
        start_time: float,
    ) -> RetrievalPreviewResponse:
        """Formats department entity into RetrievalPreviewResponse."""
        chunk_id = f"dept_{self._slugify(department.department_name)}"
        content = (
            f"=== VERIFIED DEPARTMENT OVERVIEW [SOURCE ID: {chunk_id}] ===\n"
            f"Department: {department.department_name}\n"
            f"Description: {department.description if department.description else 'Department at P.S. Mission Hospital'}\n"
            f"Source URL: {department.source_url}"
        )

        metadata = {
            "content_type": "department",
            "department": department.department_name,
            "description": department.description,
            "source_url": department.source_url,
        }

        item = RetrievalChunkItem(
            rank=1,
            chunk_id=chunk_id,
            document_id=f"doc_{chunk_id}",
            score=0.95,
            content=content,
            section=department.department_name,
            url=department.source_url or "",
            title=f"Department: {department.department_name}",
            source="structured_entity",
            metadata=metadata,
        )

        duration = time.time() - start_time
        return RetrievalPreviewResponse(
            query=query,
            has_context=True,
            results=[item],
            context=ContextResult(
                has_context=True,
                context_text=content,
                total_chunks=1,
                total_characters=len(content),
            ),
            debug=RetrievalDebugInfo(
                pinecone_top_k=1,
                candidates_retrieved=1,
                removed_by_threshold=0,
                removed_duplicates=0,
                final_chunks_count=1,
                context_character_count=len(content),
            ),
            retrieval_duration_seconds=round(duration, 3),
        )

    @staticmethod
    def _slugify(text: str) -> str:
        """Converts name to deterministic alphanumeric slug."""
        clean = re.sub(r"[^\w\s-]", "", (text or "").lower()).strip()
        return re.sub(r"[-\s]+", "_", clean)
