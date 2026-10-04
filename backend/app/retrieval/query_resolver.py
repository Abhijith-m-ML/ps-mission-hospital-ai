"""Conversational query resolution layer for the hospital retrieval engine.
Delegates to the modular Query Understanding engine in app.query.
"""
from typing import Any, Dict, List, Optional
from app.models.schemas import ChatMessage
from app.query.resolver import QueryResolver as ModularQueryResolver, QueryUnderstandingResult
from app.retrieval.entity_store import HospitalEntityStore

# Aliasing for backward compatibility
ResolutionResult = QueryUnderstandingResult


class QueryResolver(ModularQueryResolver):
    """Backward-compatible wrapper around the app.query.QueryResolver engine."""

    def __init__(self, entity_store: Optional[HospitalEntityStore] = None):
        super().__init__(entity_store=entity_store)
