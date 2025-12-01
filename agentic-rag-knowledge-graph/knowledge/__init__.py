"""
Knowledge module for domain-specific ontologies.

This module provides static ontology structures that can be used for:
- Query expansion and concept matching
- Knowledge-enhanced retrieval
- Future Graphiti/Neo4j graph seeding

Usage:
    from knowledge import AstrologyOntology
    
    ontology = AstrologyOntology()
    matches = ontology.match_keywords("Venus und Beziehungen")
"""

from .ontology_utils import (
    AstrologyOntology,
    Entity,
    Relationship,
    load_ontology,
)

__all__ = [
    'AstrologyOntology',
    'Entity',
    'Relationship',
    'load_ontology',
]

