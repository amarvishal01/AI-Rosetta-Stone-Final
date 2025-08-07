"""
Symbolic Knowledge Base Module

Handles ingestion and formalization of legal text into machine-readable ontologies.
"""

from .builder import SymbolicKnowledgeBase, LegalTextProcessor
from .ontology import RegulatoryOntology, EUAIActOntology
from .query import KnowledgeQueryEngine

__all__ = [
    "SymbolicKnowledgeBase",
    "LegalTextProcessor", 
    "RegulatoryOntology",
    "EUAIActOntology",
    "KnowledgeQueryEngine"
]