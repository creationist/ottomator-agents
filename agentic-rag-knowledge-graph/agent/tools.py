"""
Tools for the Pydantic AI agent.
"""

import os
import logging
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import asyncio

from pydantic import BaseModel, Field
from dotenv import load_dotenv

from .db_utils import (
    vector_search,
    hybrid_search,
    get_document,
    list_documents,
    get_document_chunks
)
from .graph_utils import (
    search_knowledge_graph,
    get_entity_relationships,
    graph_client
)
from .models import ChunkResult, GraphSearchResult, DocumentMetadata
from .providers import get_embedding_client, get_embedding_model

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)

# Initialize embedding client with flexible provider
embedding_client = get_embedding_client()
EMBEDDING_MODEL = get_embedding_model()

# Initialize astrology ontology for query expansion (optional)
try:
    from knowledge import AstrologyOntology
    _ontology = AstrologyOntology()
    logger.info(f"Astrology ontology loaded: {_ontology.entity_count} entities")
except Exception as e:
    logger.warning(f"Astrology ontology not available: {e}")
    _ontology = None


def expand_query_with_ontology(query: str, max_keywords: int = 10) -> Tuple[str, Dict[str, Any]]:
    """
    Expand a query with related astrology concepts if applicable.
    
    Only expands if the query contains recognized astrology terms.
    Returns the original query unchanged if no ontology or no matches.
    
    Args:
        query: The original search query
        max_keywords: Maximum number of keywords to add
        
    Returns:
        Tuple of (expanded_query, expansion_info)
    """
    if _ontology is None:
        return query, {"expanded": False, "reason": "ontology_not_loaded"}
    
    try:
        expansion = _ontology.expand_query(query)
        
        # Only expand if we found astrology-related concepts
        if not expansion['matched_concepts']:
            return query, {"expanded": False, "reason": "no_astrology_terms"}
        
        # Get unique keywords to add (excluding words already in query)
        query_lower = query.lower()
        new_keywords = [
            kw for kw in expansion['all_keywords']
            if kw.lower() not in query_lower
        ][:max_keywords]
        
        if new_keywords:
            expanded_query = query + " " + " ".join(new_keywords)
            return expanded_query, {
                "expanded": True,
                "matched_concepts": [e.name for e in expansion['matched_concepts']],
                "expanded_concepts": [e.name for e in expansion['expanded_concepts'][:5]],
                "added_keywords": new_keywords
            }
        
        return query, {"expanded": False, "reason": "no_new_keywords"}
        
    except Exception as e:
        logger.warning(f"Query expansion failed: {e}")
        return query, {"expanded": False, "reason": f"error: {e}"}


async def generate_embedding(text: str) -> List[float]:
    """
    Generate embedding for text using OpenAI.
    
    Args:
        text: Text to embed
    
    Returns:
        Embedding vector
    """
    try:
        response = await embedding_client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=text
        )
        return response.data[0].embedding
    except Exception as e:
        logger.error(f"Failed to generate embedding: {e}")
        raise


# Tool Input Models
class VectorSearchInput(BaseModel):
    """Input for vector search tool."""
    query: str = Field(..., description="Search query")
    limit: int = Field(default=10, description="Maximum number of results")


class GraphSearchInput(BaseModel):
    """Input for graph search tool."""
    query: str = Field(..., description="Search query")


class HybridSearchInput(BaseModel):
    """Input for hybrid search tool."""
    query: str = Field(..., description="Search query")
    limit: int = Field(default=10, description="Maximum number of results")
    text_weight: float = Field(default=0.3, description="Weight for text similarity (0-1)")


class DocumentInput(BaseModel):
    """Input for document retrieval."""
    document_id: str = Field(..., description="Document ID to retrieve")


class DocumentListInput(BaseModel):
    """Input for listing documents."""
    limit: int = Field(default=20, description="Maximum number of documents")
    offset: int = Field(default=0, description="Number of documents to skip")


class EntityRelationshipInput(BaseModel):
    """Input for entity relationship query."""
    entity_name: str = Field(..., description="Name of the entity")
    depth: int = Field(default=2, description="Maximum traversal depth")


class EntityTimelineInput(BaseModel):
    """Input for entity timeline query."""
    entity_name: str = Field(..., description="Name of the entity")
    start_date: Optional[str] = Field(None, description="Start date (ISO format)")
    end_date: Optional[str] = Field(None, description="End date (ISO format)")


# Tool Implementation Functions
async def vector_search_tool(input_data: VectorSearchInput) -> List[ChunkResult]:
    """
    Perform vector similarity search with optional ontology-based query expansion.
    
    If the query contains recognized astrology terms, it will be automatically
    expanded with related concepts for better retrieval.
    
    Args:
        input_data: Search parameters
    
    Returns:
        List of matching chunks
    """
    try:
        # Expand query with astrology ontology if applicable
        expanded_query, expansion_info = expand_query_with_ontology(input_data.query)
        
        if expansion_info.get("expanded"):
            logger.info(f"Query expanded: '{input_data.query}' -> added {expansion_info.get('added_keywords', [])}")
        
        # Generate embedding for the (potentially expanded) query
        embedding = await generate_embedding(expanded_query)
        
        # Perform vector search
        results = await vector_search(
            embedding=embedding,
            limit=input_data.limit
        )

        # Convert to ChunkResult models
        return [
            ChunkResult(
                chunk_id=str(r["chunk_id"]),
                document_id=str(r["document_id"]),
                content=r["content"],
                score=r["similarity"],
                metadata=r["metadata"],
                document_title=r["document_title"],
                document_source=r["document_source"]
            )
            for r in results
        ]
        
    except Exception as e:
        logger.error(f"Vector search failed: {e}")
        return []


async def graph_search_tool(input_data: GraphSearchInput) -> List[GraphSearchResult]:
    """
    Search the knowledge graph.
    
    Args:
        input_data: Search parameters
    
    Returns:
        List of graph search results
    """
    try:
        results = await search_knowledge_graph(
            query=input_data.query
        )
        
        # Convert to GraphSearchResult models
        return [
            GraphSearchResult(
                fact=r["fact"],
                uuid=r["uuid"],
                valid_at=r.get("valid_at"),
                invalid_at=r.get("invalid_at"),
                source_node_uuid=r.get("source_node_uuid")
            )
            for r in results
        ]
        
    except Exception as e:
        logger.error(f"Graph search failed: {e}")
        return []


async def hybrid_search_tool(input_data: HybridSearchInput) -> List[ChunkResult]:
    """
    Perform hybrid search (vector + keyword) with optional ontology-based query expansion.
    
    If the query contains recognized astrology terms, it will be automatically
    expanded with related concepts for better retrieval.
    
    Args:
        input_data: Search parameters
    
    Returns:
        List of matching chunks
    """
    try:
        # Expand query with astrology ontology if applicable
        expanded_query, expansion_info = expand_query_with_ontology(input_data.query)
        
        if expansion_info.get("expanded"):
            logger.info(f"Query expanded: '{input_data.query}' -> added {expansion_info.get('added_keywords', [])}")
        
        # Generate embedding for the (potentially expanded) query
        embedding = await generate_embedding(expanded_query)
        
        # Perform hybrid search (use expanded query for both vector and text matching)
        results = await hybrid_search(
            embedding=embedding,
            query_text=expanded_query,
            limit=input_data.limit,
            text_weight=input_data.text_weight
        )
        
        # Convert to ChunkResult models
        return [
            ChunkResult(
                chunk_id=str(r["chunk_id"]),
                document_id=str(r["document_id"]),
                content=r["content"],
                score=r["combined_score"],
                metadata=r["metadata"],
                document_title=r["document_title"],
                document_source=r["document_source"]
            )
            for r in results
        ]
        
    except Exception as e:
        logger.error(f"Hybrid search failed: {e}")
        return []


async def get_document_tool(input_data: DocumentInput) -> Optional[Dict[str, Any]]:
    """
    Retrieve a complete document.
    
    Args:
        input_data: Document retrieval parameters
    
    Returns:
        Document data or None
    """
    try:
        document = await get_document(input_data.document_id)
        
        if document:
            # Also get all chunks for the document
            chunks = await get_document_chunks(input_data.document_id)
            document["chunks"] = chunks
        
        return document
        
    except Exception as e:
        logger.error(f"Document retrieval failed: {e}")
        return None


async def list_documents_tool(input_data: DocumentListInput) -> List[DocumentMetadata]:
    """
    List available documents.
    
    Args:
        input_data: Listing parameters
    
    Returns:
        List of document metadata
    """
    try:
        documents = await list_documents(
            limit=input_data.limit,
            offset=input_data.offset
        )
        
        # Convert to DocumentMetadata models
        return [
            DocumentMetadata(
                id=d["id"],
                title=d["title"],
                source=d["source"],
                metadata=d["metadata"],
                created_at=datetime.fromisoformat(d["created_at"]),
                updated_at=datetime.fromisoformat(d["updated_at"]),
                chunk_count=d.get("chunk_count")
            )
            for d in documents
        ]
        
    except Exception as e:
        logger.error(f"Document listing failed: {e}")
        return []


async def get_entity_relationships_tool(input_data: EntityRelationshipInput) -> Dict[str, Any]:
    """
    Get relationships for an entity.
    
    Args:
        input_data: Entity relationship parameters
    
    Returns:
        Entity relationships
    """
    try:
        return await get_entity_relationships(
            entity=input_data.entity_name,
            depth=input_data.depth
        )
        
    except Exception as e:
        logger.error(f"Entity relationship query failed: {e}")
        return {
            "central_entity": input_data.entity_name,
            "related_entities": [],
            "relationships": [],
            "depth": input_data.depth,
            "error": str(e)
        }


async def get_entity_timeline_tool(input_data: EntityTimelineInput) -> List[Dict[str, Any]]:
    """
    Get timeline of facts for an entity.
    
    Args:
        input_data: Timeline query parameters
    
    Returns:
        Timeline of facts
    """
    try:
        # Parse dates if provided
        start_date = None
        end_date = None
        
        if input_data.start_date:
            start_date = datetime.fromisoformat(input_data.start_date)
        if input_data.end_date:
            end_date = datetime.fromisoformat(input_data.end_date)
        
        # Get timeline from graph
        timeline = await graph_client.get_entity_timeline(
            entity_name=input_data.entity_name,
            start_date=start_date,
            end_date=end_date
        )
        
        return timeline
        
    except Exception as e:
        logger.error(f"Entity timeline query failed: {e}")
        return []


# Ontology lookup functions
class OntologyLookupInput(BaseModel):
    """Input for ontology concept lookup."""
    concept: str = Field(..., description="Concept name or ID to look up (e.g., 'Venus', 'scorpio', 'transformation')")


def lookup_ontology_concept(input_data: OntologyLookupInput) -> Dict[str, Any]:
    """
    Look up an astrological concept in the ontology.
    
    Args:
        input_data: Lookup parameters
        
    Returns:
        Concept details including description, related concepts, and relationships
    """
    if _ontology is None:
        return {"error": "Astrology ontology not available"}
    
    try:
        # Try to find the concept by ID or name
        concept_id = input_data.concept.lower().replace(" ", "_")
        entity = _ontology.get_entity(concept_id)
        
        # If not found by ID, try matching keywords
        if entity is None:
            matches = _ontology.match_keywords(input_data.concept)
            if matches:
                entity = matches[0]
        
        if entity is None:
            return {
                "found": False,
                "query": input_data.concept,
                "suggestion": "Try searching for: planets, signs, aspects, houses, or themes like 'transformation', 'healing', 'relationships'"
            }
        
        # Get related concepts
        related_ids = _ontology.expand_concept(entity.id, max_depth=1)
        related = [_ontology.get_entity(rid) for rid in related_ids if _ontology.get_entity(rid)]
        
        # Get relationships
        relationships = _ontology.get_relationships_for(entity.id)
        
        return {
            "found": True,
            "concept": {
                "id": entity.id,
                "name": entity.name,
                "type": entity.type,
                "description": entity.description,
                "keywords": entity.keywords
            },
            "related_concepts": [
                {"id": e.id, "name": e.name, "type": e.type}
                for e in related[:10]
            ],
            "relationships": [
                {
                    "type": rel.type,
                    "target": rel.target if rel.source == entity.id else rel.source,
                    "description": rel.description
                }
                for rel in relationships[:10]
            ]
        }
        
    except Exception as e:
        logger.error(f"Ontology lookup failed: {e}")
        return {"error": str(e)}


def get_ontology_stats() -> Dict[str, Any]:
    """
    Get statistics about the loaded ontology.
    
    Returns:
        Ontology metadata and statistics
    """
    if _ontology is None:
        return {"loaded": False, "error": "Astrology ontology not available"}
    
    return {
        "loaded": True,
        "metadata": _ontology.metadata,
        "entity_count": _ontology.entity_count,
        "relationship_count": _ontology.relationship_count,
        "entity_types": list(set(e.type for e in _ontology._entities.values()))
    }


# Combined search function for agent use
async def perform_comprehensive_search(
    query: str,
    use_vector: bool = True,
    use_graph: bool = True,
    limit: int = 10
) -> Dict[str, Any]:
    """
    Perform a comprehensive search using multiple methods.
    
    Args:
        query: Search query
        use_vector: Whether to use vector search
        use_graph: Whether to use graph search
        limit: Maximum results per search type (only applies to vector search)
    
    Returns:
        Combined search results
    """
    results = {
        "query": query,
        "vector_results": [],
        "graph_results": [],
        "total_results": 0
    }
    
    tasks = []
    
    if use_vector:
        tasks.append(vector_search_tool(VectorSearchInput(query=query, limit=limit)))
    
    if use_graph:
        tasks.append(graph_search_tool(GraphSearchInput(query=query)))
    
    if tasks:
        search_results = await asyncio.gather(*tasks, return_exceptions=True)
        
        if use_vector and not isinstance(search_results[0], Exception):
            results["vector_results"] = search_results[0]
        
        if use_graph:
            graph_idx = 1 if use_vector else 0
            if not isinstance(search_results[graph_idx], Exception):
                results["graph_results"] = search_results[graph_idx]
    
    results["total_results"] = len(results["vector_results"]) + len(results["graph_results"])
    
    return results


# =============================================================================
# Ontology Traversal Tools (Multi-hop graph queries)
# =============================================================================

# Import ontology query client
try:
    from .ontology_queries import get_ontology_query_client, OntologyQueryClient
    _ontology_queries_available = True
except ImportError:
    _ontology_queries_available = False
    logger.warning("Ontology queries module not available")


class ElementSearchInput(BaseModel):
    """Input for element-based search."""
    element: str = Field(..., description="Element to search for (fire, earth, air, water)")


class ModalitySearchInput(BaseModel):
    """Input for modality-based search."""
    modality: str = Field(..., description="Modality to search for (cardinal, fixed, mutable)")


class PlanetRulershipInput(BaseModel):
    """Input for planet rulership search."""
    planet: str = Field(..., description="Planet ID (sun, moon, mercury, venus, mars, jupiter, saturn, uranus, neptune, pluto)")


class ThemeSearchInput(BaseModel):
    """Input for theme-based search."""
    theme: str = Field(..., description="Theme ID (transformation, healing, relationships, creativity, spirituality, karma, vocation, family, communication, finances, health)")


class OntologyTraversalInput(BaseModel):
    """Input for multi-hop ontology traversal."""
    entity_id: str = Field(..., description="Starting entity ID")
    max_hops: int = Field(default=2, description="Maximum relationship hops (1-3)")
    limit: int = Field(default=20, description="Maximum results to return")


class CoOccurrenceInput(BaseModel):
    """Input for co-occurrence query."""
    entity_id: str = Field(..., description="Entity to find co-occurrences for")
    min_count: int = Field(default=1, description="Minimum co-occurrence count")
    limit: int = Field(default=20, description="Maximum results")


class EntityPairInput(BaseModel):
    """Input for entity pair chunk search."""
    entity1: str = Field(..., description="First entity ID")
    entity2: str = Field(..., description="Second entity ID")


async def search_chunks_by_element(input_data: ElementSearchInput) -> Dict[str, Any]:
    """
    Find chunks mentioning zodiac signs of a specific element.
    
    Uses multi-hop graph traversal: Chunk -> Sign -> Element
    
    Args:
        input_data: Element search parameters
        
    Returns:
        Chunks mentioning signs of the specified element
    """
    if not _ontology_queries_available:
        return {"error": "Ontology queries not available"}
    
    try:
        client = await get_ontology_query_client()
        results = await client.get_chunks_by_element(input_data.element)
        
        return {
            "element": input_data.element,
            "chunk_count": len(results),
            "chunks": results
        }
    except Exception as e:
        logger.error(f"Element search failed: {e}")
        return {"error": str(e)}


async def search_chunks_by_modality(input_data: ModalitySearchInput) -> Dict[str, Any]:
    """
    Find chunks mentioning zodiac signs of a specific modality.
    
    Uses multi-hop graph traversal: Chunk -> Sign -> Modality
    
    Args:
        input_data: Modality search parameters
        
    Returns:
        Chunks mentioning signs of the specified modality
    """
    if not _ontology_queries_available:
        return {"error": "Ontology queries not available"}
    
    try:
        client = await get_ontology_query_client()
        results = await client.get_chunks_by_modality(input_data.modality)
        
        return {
            "modality": input_data.modality,
            "chunk_count": len(results),
            "chunks": results
        }
    except Exception as e:
        logger.error(f"Modality search failed: {e}")
        return {"error": str(e)}


async def search_chunks_by_planet_rulership(input_data: PlanetRulershipInput) -> Dict[str, Any]:
    """
    Find chunks mentioning zodiac signs ruled by a specific planet.
    
    Uses multi-hop graph traversal: Chunk -> Sign <- Planet (via RULES relationship)
    
    Args:
        input_data: Planet rulership search parameters
        
    Returns:
        Chunks mentioning signs ruled by the specified planet
    """
    if not _ontology_queries_available:
        return {"error": "Ontology queries not available"}
    
    try:
        client = await get_ontology_query_client()
        results = await client.get_chunks_by_planet_rulership(input_data.planet)
        
        return {
            "planet": input_data.planet,
            "chunk_count": len(results),
            "chunks": results
        }
    except Exception as e:
        logger.error(f"Planet rulership search failed: {e}")
        return {"error": str(e)}


async def search_chunks_by_theme(input_data: ThemeSearchInput) -> Dict[str, Any]:
    """
    Find chunks related to a specific astrological theme.
    
    Uses multi-hop graph traversal through theme relationships.
    
    Args:
        input_data: Theme search parameters
        
    Returns:
        Chunks related to the specified theme
    """
    if not _ontology_queries_available:
        return {"error": "Ontology queries not available"}
    
    try:
        client = await get_ontology_query_client()
        results = await client.get_chunks_by_theme(input_data.theme)
        
        return {
            "theme": input_data.theme,
            "chunk_count": len(results),
            "chunks": results
        }
    except Exception as e:
        logger.error(f"Theme search failed: {e}")
        return {"error": str(e)}


async def traverse_ontology_for_chunks(input_data: OntologyTraversalInput) -> Dict[str, Any]:
    """
    Find chunks connected to an entity through ontology relationships.
    
    Performs variable-length path traversal (1-3 hops) through the ontology,
    then finds chunks mentioning any connected entity.
    
    Args:
        input_data: Traversal parameters
        
    Returns:
        Chunks connected through the ontology graph
    """
    if not _ontology_queries_available:
        return {"error": "Ontology queries not available"}
    
    try:
        client = await get_ontology_query_client()
        results = await client.get_related_chunks_via_ontology(
            entity_id=input_data.entity_id,
            max_hops=input_data.max_hops,
            limit=input_data.limit
        )
        
        return {
            "start_entity": input_data.entity_id,
            "max_hops": input_data.max_hops,
            "chunk_count": len(results),
            "chunks": results
        }
    except Exception as e:
        logger.error(f"Ontology traversal failed: {e}")
        return {"error": str(e)}


async def get_entity_neighborhood_tool(input_data: OntologyTraversalInput) -> Dict[str, Any]:
    """
    Get the ontology neighborhood of an entity.
    
    Returns all entities and relationships within N hops, useful for
    understanding the context around a concept.
    
    Args:
        input_data: Traversal parameters
        
    Returns:
        Entity neighborhood with connected entities and relationships
    """
    if not _ontology_queries_available:
        return {"error": "Ontology queries not available"}
    
    try:
        client = await get_ontology_query_client()
        results = await client.get_entity_neighborhood(
            entity_id=input_data.entity_id,
            max_hops=input_data.max_hops
        )
        
        return results
    except Exception as e:
        logger.error(f"Entity neighborhood query failed: {e}")
        return {"error": str(e)}


async def find_co_occurring_entities(input_data: CoOccurrenceInput) -> Dict[str, Any]:
    """
    Find entities that frequently co-occur with a given entity in chunks.
    
    This reveals which concepts are commonly discussed together in the
    knowledge base.
    
    Args:
        input_data: Co-occurrence query parameters
        
    Returns:
        List of co-occurring entities with counts
    """
    if not _ontology_queries_available:
        return {"error": "Ontology queries not available"}
    
    try:
        client = await get_ontology_query_client()
        results = await client.get_co_occurring_entities(
            entity_id=input_data.entity_id,
            min_co_occurrences=input_data.min_count,
            limit=input_data.limit
        )
        
        return {
            "entity": input_data.entity_id,
            "co_occurring_count": len(results),
            "co_occurring_entities": results
        }
    except Exception as e:
        logger.error(f"Co-occurrence query failed: {e}")
        return {"error": str(e)}


async def find_chunks_with_entity_pair(input_data: EntityPairInput) -> Dict[str, Any]:
    """
    Find chunks that mention both specified entities.
    
    Useful for finding content that discusses the relationship or
    interaction between two concepts.
    
    Args:
        input_data: Entity pair parameters
        
    Returns:
        Chunks mentioning both entities
    """
    if not _ontology_queries_available:
        return {"error": "Ontology queries not available"}
    
    try:
        client = await get_ontology_query_client()
        results = await client.get_chunks_with_entity_pair(
            entity1=input_data.entity1,
            entity2=input_data.entity2
        )
        
        return {
            "entity1": input_data.entity1,
            "entity2": input_data.entity2,
            "chunk_count": len(results),
            "chunks": results
        }
    except Exception as e:
        logger.error(f"Entity pair search failed: {e}")
        return {"error": str(e)}


async def get_graph_statistics() -> Dict[str, Any]:
    """
    Get comprehensive statistics about the knowledge graph.
    
    Returns entity counts, relationship counts, and chunk coverage.
    """
    if not _ontology_queries_available:
        return {"error": "Ontology queries not available"}
    
    try:
        client = await get_ontology_query_client()
        return await client.get_ontology_statistics()
    except Exception as e:
        logger.error(f"Statistics query failed: {e}")
        return {"error": str(e)}