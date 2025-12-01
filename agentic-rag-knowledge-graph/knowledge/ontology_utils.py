"""
Utility functions for working with the astrology ontology.

This module provides functions for:
- Loading the ontology from JSON
- Expanding concepts to find related terms
- Matching query terms to ontology concepts
- Getting concept details and relationships

Usage:
    from knowledge.ontology_utils import AstrologyOntology
    
    ontology = AstrologyOntology()
    
    # Find related concepts
    related = ontology.expand_concept("venus")
    # Returns: ['relationships', 'creativity', 'taurus', 'libra', ...]
    
    # Match keywords in a query
    matches = ontology.match_keywords("Wie beeinflusst Venus meine Beziehungen?")
    # Returns: [{'id': 'venus', 'name': 'Venus', ...}, {'id': 'relationships', ...}]
"""

import json
import re
import logging
from pathlib import Path
from typing import Dict, List, Optional, Set, Any, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# Default path to ontology file (relative to this module)
DEFAULT_ONTOLOGY_PATH = Path(__file__).parent / "astrology_ontology.json"


@dataclass
class Entity:
    """Represents an entity from the ontology."""
    id: str
    name: str
    type: str
    description: str
    keywords: List[str]
    attributes: Dict[str, Any]


@dataclass
class Relationship:
    """Represents a relationship between entities."""
    source: str
    target: str
    type: str
    description: str


class AstrologyOntology:
    """
    Main class for working with the astrology ontology.
    
    Provides methods for loading, querying, and expanding concepts
    from the astrology knowledge structure.
    """
    
    def __init__(self, ontology_path: Optional[Path] = None):
        """
        Initialize the ontology.
        
        Args:
            ontology_path: Path to the ontology JSON file. 
                          Defaults to astrology_ontology.json in the same directory.
        """
        self.path = ontology_path or DEFAULT_ONTOLOGY_PATH
        self._data: Dict = {}
        self._entities: Dict[str, Entity] = {}
        self._relationships: List[Relationship] = []
        self._keyword_index: Dict[str, List[str]] = {}  # keyword -> entity_ids
        self._cross_references: Dict[str, List[str]] = {}
        
        self._load()
    
    def _load(self) -> None:
        """Load and index the ontology from JSON."""
        try:
            with open(self.path, 'r', encoding='utf-8') as f:
                self._data = json.load(f)
            
            # Index entities
            for entity_data in self._data.get('entities', []):
                entity = Entity(
                    id=entity_data['id'],
                    name=entity_data['name'],
                    type=entity_data['type'],
                    description=entity_data['description'],
                    keywords=entity_data.get('keywords', []),
                    attributes=entity_data.get('attributes', {})
                )
                self._entities[entity.id] = entity
                
                # Build keyword index
                for keyword in entity.keywords:
                    keyword_lower = keyword.lower()
                    if keyword_lower not in self._keyword_index:
                        self._keyword_index[keyword_lower] = []
                    self._keyword_index[keyword_lower].append(entity.id)
                
                # Also index the name
                name_lower = entity.name.lower()
                if name_lower not in self._keyword_index:
                    self._keyword_index[name_lower] = []
                if entity.id not in self._keyword_index[name_lower]:
                    self._keyword_index[name_lower].append(entity.id)
            
            # Index relationships
            for rel_data in self._data.get('relationships', []):
                self._relationships.append(Relationship(
                    source=rel_data['source'],
                    target=rel_data['target'],
                    type=rel_data['type'],
                    description=rel_data.get('description', '')
                ))
            
            # Load cross-references
            self._cross_references = self._data.get('cross_references', {})
            
            logger.info(f"Loaded ontology with {len(self._entities)} entities and {len(self._relationships)} relationships")
            
        except FileNotFoundError:
            logger.warning(f"Ontology file not found: {self.path}")
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse ontology JSON: {e}")
    
    def get_entity(self, entity_id: str) -> Optional[Entity]:
        """
        Get an entity by ID.
        
        Args:
            entity_id: The entity ID (e.g., 'venus', 'aries')
            
        Returns:
            Entity object or None if not found
        """
        return self._entities.get(entity_id)
    
    def get_entities_by_type(self, entity_type: str) -> List[Entity]:
        """
        Get all entities of a specific type.
        
        Args:
            entity_type: Type like 'planet', 'sign', 'house', 'aspect', etc.
            
        Returns:
            List of matching entities
        """
        return [e for e in self._entities.values() if e.type == entity_type]
    
    def expand_concept(self, concept_id: str, max_depth: int = 2) -> List[str]:
        """
        Expand a concept to find all related concepts.
        
        Uses cross-references and relationships to find connected concepts.
        
        Args:
            concept_id: The starting concept ID
            max_depth: Maximum traversal depth (default 2)
            
        Returns:
            List of related concept IDs
        """
        related: Set[str] = set()
        visited: Set[str] = set()
        to_explore: List[Tuple[str, int]] = [(concept_id, 0)]
        
        while to_explore:
            current_id, depth = to_explore.pop(0)
            
            if current_id in visited or depth > max_depth:
                continue
            
            visited.add(current_id)
            
            if depth > 0:  # Don't include the original concept
                related.add(current_id)
            
            if depth < max_depth:
                # Add cross-referenced concepts
                if current_id in self._cross_references:
                    for ref in self._cross_references[current_id]:
                        if ref not in visited and ref in self._entities:
                            to_explore.append((ref, depth + 1))
                
                # Add related concepts from relationships
                for rel in self._relationships:
                    if rel.source == current_id and rel.target not in visited:
                        to_explore.append((rel.target, depth + 1))
                    elif rel.target == current_id and rel.source not in visited:
                        to_explore.append((rel.source, depth + 1))
        
        return list(related)
    
    def match_keywords(self, text: str) -> List[Entity]:
        """
        Find all ontology concepts mentioned in a text.
        
        Performs case-insensitive keyword matching.
        
        Args:
            text: The text to search (e.g., a user query)
            
        Returns:
            List of matching Entity objects, sorted by relevance
        """
        text_lower = text.lower()
        matches: Dict[str, int] = {}  # entity_id -> match_count
        
        for keyword, entity_ids in self._keyword_index.items():
            # Use word boundary matching for accuracy
            pattern = r'\b' + re.escape(keyword) + r'\b'
            count = len(re.findall(pattern, text_lower))
            
            if count > 0:
                for entity_id in entity_ids:
                    matches[entity_id] = matches.get(entity_id, 0) + count
        
        # Sort by match count and return entities
        sorted_matches = sorted(matches.items(), key=lambda x: x[1], reverse=True)
        return [self._entities[entity_id] for entity_id, _ in sorted_matches if entity_id in self._entities]
    
    def get_relationships_for(self, entity_id: str) -> List[Relationship]:
        """
        Get all relationships involving an entity.
        
        Args:
            entity_id: The entity ID
            
        Returns:
            List of relationships where the entity is source or target
        """
        return [
            rel for rel in self._relationships
            if rel.source == entity_id or rel.target == entity_id
        ]
    
    def get_related_entities(self, entity_id: str, relationship_type: Optional[str] = None) -> List[Entity]:
        """
        Get entities directly related to a given entity.
        
        Args:
            entity_id: The starting entity ID
            relationship_type: Optional filter for relationship type (e.g., 'RULES', 'HAS_ELEMENT')
            
        Returns:
            List of related entities
        """
        related_ids: Set[str] = set()
        
        for rel in self._relationships:
            if relationship_type and rel.type != relationship_type:
                continue
                
            if rel.source == entity_id:
                related_ids.add(rel.target)
            elif rel.target == entity_id:
                related_ids.add(rel.source)
        
        return [self._entities[eid] for eid in related_ids if eid in self._entities]
    
    def expand_query(self, query: str) -> Dict[str, Any]:
        """
        Analyze a query and expand it with related concepts.
        
        This is useful for enhancing RAG retrieval by including
        related terms that might not be explicitly mentioned.
        
        Args:
            query: The user's query text
            
        Returns:
            Dictionary with:
                - matched_concepts: Direct keyword matches
                - expanded_concepts: Related concepts
                - all_keywords: All keywords to use for search
        """
        # Find direct matches
        matched = self.match_keywords(query)
        matched_ids = [e.id for e in matched]
        
        # Expand to find related concepts
        expanded_ids: Set[str] = set()
        for entity_id in matched_ids:
            for related_id in self.expand_concept(entity_id, max_depth=1):
                if related_id not in matched_ids:
                    expanded_ids.add(related_id)
        
        expanded = [self._entities[eid] for eid in expanded_ids if eid in self._entities]
        
        # Collect all keywords for search
        all_keywords: Set[str] = set()
        for entity in matched + expanded:
            all_keywords.update(entity.keywords)
            all_keywords.add(entity.name.lower())
        
        return {
            'matched_concepts': matched,
            'expanded_concepts': expanded,
            'all_keywords': list(all_keywords),
            'matched_ids': matched_ids,
            'expanded_ids': list(expanded_ids)
        }
    
    def get_signs_by_element(self, element_id: str) -> List[Entity]:
        """
        Get all zodiac signs with a specific element.
        
        Args:
            element_id: Element ID ('fire', 'earth', 'air', 'water')
            
        Returns:
            List of sign entities
        """
        signs = []
        for rel in self._relationships:
            if rel.type == 'HAS_ELEMENT' and rel.target == element_id:
                if rel.source in self._entities:
                    signs.append(self._entities[rel.source])
        return signs
    
    def get_signs_by_modality(self, modality_id: str) -> List[Entity]:
        """
        Get all zodiac signs with a specific modality.
        
        Args:
            modality_id: Modality ID ('cardinal', 'fixed', 'mutable')
            
        Returns:
            List of sign entities
        """
        signs = []
        for rel in self._relationships:
            if rel.type == 'HAS_MODALITY' and rel.target == modality_id:
                if rel.source in self._entities:
                    signs.append(self._entities[rel.source])
        return signs
    
    def get_planet_rulerships(self, planet_id: str) -> List[Entity]:
        """
        Get signs ruled by a planet.
        
        Args:
            planet_id: Planet ID (e.g., 'venus', 'mars')
            
        Returns:
            List of sign entities ruled by this planet
        """
        signs = []
        for rel in self._relationships:
            if rel.type in ('RULES', 'TRADITIONAL_RULES') and rel.source == planet_id:
                if rel.target in self._entities:
                    signs.append(self._entities[rel.target])
        return signs
    
    def to_graphiti_format(self) -> Dict[str, List[Dict]]:
        """
        Convert ontology to a format suitable for seeding Graphiti/Neo4j.
        
        Returns:
            Dictionary with 'nodes' and 'edges' lists ready for graph import
        """
        nodes = []
        for entity in self._entities.values():
            nodes.append({
                'name': entity.name,
                'entity_type': entity.type,
                'description': entity.description,
                'properties': {
                    'id': entity.id,
                    'keywords': entity.keywords,
                    **entity.attributes
                }
            })
        
        edges = []
        for rel in self._relationships:
            edges.append({
                'source': self._entities.get(rel.source, Entity(rel.source, rel.source, '', '', [], {})).name,
                'target': self._entities.get(rel.target, Entity(rel.target, rel.target, '', '', [], {})).name,
                'relationship_type': rel.type,
                'description': rel.description
            })
        
        return {
            'nodes': nodes,
            'edges': edges
        }
    
    @property
    def metadata(self) -> Dict:
        """Get ontology metadata."""
        return self._data.get('metadata', {})
    
    @property
    def entity_count(self) -> int:
        """Get total number of entities."""
        return len(self._entities)
    
    @property
    def relationship_count(self) -> int:
        """Get total number of relationships."""
        return len(self._relationships)


# Convenience function for quick loading
def load_ontology(path: Optional[Path] = None) -> AstrologyOntology:
    """
    Load the astrology ontology.
    
    Args:
        path: Optional custom path to ontology JSON
        
    Returns:
        AstrologyOntology instance
    """
    return AstrologyOntology(path)


# Example usage and testing
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Load ontology
    ontology = AstrologyOntology()
    
    print(f"\n=== Astrology Ontology ===")
    print(f"Entities: {ontology.entity_count}")
    print(f"Relationships: {ontology.relationship_count}")
    
    # Test concept expansion
    print(f"\n=== Expanding 'venus' ===")
    expanded = ontology.expand_concept('venus')
    print(f"Related concepts: {expanded[:10]}...")
    
    # Test keyword matching
    print(f"\n=== Matching query ===")
    query = "Wie beeinflusst der Vollmond meine Beziehungen?"
    matches = ontology.match_keywords(query)
    print(f"Query: {query}")
    print(f"Matches: {[m.name for m in matches]}")
    
    # Test query expansion
    print(f"\n=== Query Expansion ===")
    expansion = ontology.expand_query(query)
    print(f"Matched: {[e.name for e in expansion['matched_concepts']]}")
    print(f"Expanded: {[e.name for e in expansion['expanded_concepts'][:5]]}...")
    print(f"Keywords for search: {expansion['all_keywords'][:10]}...")
    
    # Test getting fire signs
    print(f"\n=== Fire Signs ===")
    fire_signs = ontology.get_signs_by_element('fire')
    print(f"Fire signs: {[s.name for s in fire_signs]}")
    
    # Test planet rulerships
    print(f"\n=== Venus Rulerships ===")
    venus_rules = ontology.get_planet_rulerships('venus')
    print(f"Venus rules: {[s.name for s in venus_rules]}")

