"""
Main Pydantic AI agent for agentic RAG with knowledge graph.
"""

import os
import logging
from typing import Dict, Any, List, Optional
from dataclasses import dataclass

from pydantic_ai import Agent, RunContext
from dotenv import load_dotenv

from .prompts import SYSTEM_PROMPT
from .providers import get_llm_model
from .tools import (
    # Core search tools
    graph_search_tool,
    hybrid_search_tool,
    get_document_tool,
    list_documents_tool,
    get_entity_relationships_tool,
    get_entity_timeline_tool,
    lookup_ontology_concept,
    # Input types
    GraphSearchInput,
    HybridSearchInput,
    DocumentInput,
    DocumentListInput,
    EntityRelationshipInput,
    EntityTimelineInput,
    OntologyLookupInput,
    # Ontology traversal (used by explore_ontology)
    search_chunks_by_element,
    search_chunks_by_modality,
    search_chunks_by_planet_rulership,
    search_chunks_by_theme,
    traverse_ontology_for_chunks,
    find_co_occurring_entities,
    find_chunks_with_entity_pair,
    ElementSearchInput,
    ModalitySearchInput,
    PlanetRulershipInput,
    ThemeSearchInput,
    OntologyTraversalInput,
    CoOccurrenceInput,
    EntityPairInput
)

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)


@dataclass
class AgentDependencies:
    """Dependencies for the agent."""
    session_id: str
    user_id: Optional[str] = None
    search_preferences: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.search_preferences is None:
            self.search_preferences = {
                "use_vector": True,
                "use_graph": True,
                "default_limit": 10
            }


# Initialize the agent with flexible model configuration
rag_agent = Agent(
    get_llm_model(),
    deps_type=AgentDependencies,
    system_prompt=SYSTEM_PROMPT
)


# =============================================================================
# Primary Search Tool
# =============================================================================

@rag_agent.tool
async def search(
    ctx: RunContext[AgentDependencies],
    query: str,
    limit: int = 10
) -> List[Dict[str, Any]]:
    """
    Search for relevant astrological content using semantic and keyword matching.
    
    This is your primary tool for finding information. It combines:
    - Semantic understanding (finds content by meaning)
    - Keyword matching (finds exact terms)
    
    Use this for any question that needs information from documents.
    
    Args:
        query: What to search for (e.g., "Saturn Rückkehr Bedeutung", "Mondphasen")
        limit: Maximum number of results (default 10)
    
    Returns:
        Relevant document chunks ranked by relevance
    """
    input_data = HybridSearchInput(
        query=query,
        limit=limit,
        text_weight=0.3
    )
    
    results = await hybrid_search_tool(input_data)
    
    return [
        {
            "content": r.content,
            "score": r.score,
            "document_title": r.document_title,
            "document_source": r.document_source,
            "chunk_id": r.chunk_id
        }
        for r in results
    ]


# =============================================================================
# Knowledge Graph Tools (Graphiti - LLM-extracted facts)
# =============================================================================

@rag_agent.tool
async def graph_search(
    ctx: RunContext[AgentDependencies],
    query: str
) -> List[Dict[str, Any]]:
    """
    Search the knowledge graph for astrological facts and relationships.
    
    Use this to find specific facts about:
    - Astrological events and their meanings
    - Relationships between celestial phenomena  
    - Historical or mythological context of astrology
    - Specific interpretations from ingested texts
    
    Args:
        query: Search query (e.g., "Saturn return meaning", "Pluto transformation")
    
    Returns:
        List of facts with source references and temporal data
    """
    input_data = GraphSearchInput(query=query)
    
    results = await graph_search_tool(input_data)
    
    return [
        {
            "fact": r.fact,
            "uuid": r.uuid,
            "valid_at": r.valid_at,
            "invalid_at": r.invalid_at,
            "source_node_uuid": r.source_node_uuid
        }
        for r in results
    ]


@rag_agent.tool
async def get_document(
    ctx: RunContext[AgentDependencies],
    document_id: str
) -> Optional[Dict[str, Any]]:
    """
    Retrieve the complete content of a specific document.
    
    This tool fetches the full document content along with all its chunks
    and metadata. Best for getting comprehensive information from a specific
    source when you need the complete context.
    
    Args:
        document_id: UUID of the document to retrieve
    
    Returns:
        Complete document data with content and metadata, or None if not found
    """
    input_data = DocumentInput(document_id=document_id)
    
    document = await get_document_tool(input_data)
    
    if document:
        # Format for agent consumption
        return {
            "id": document["id"],
            "title": document["title"],
            "source": document["source"],
            "content": document["content"],
            "chunk_count": len(document.get("chunks", [])),
            "created_at": document["created_at"]
        }
    
    return None


@rag_agent.tool
async def list_documents(
    ctx: RunContext[AgentDependencies],
    limit: int = 20,
    offset: int = 0
) -> List[Dict[str, Any]]:
    """
    List available documents with their metadata.
    
    This tool provides an overview of all documents in the knowledge base,
    including titles, sources, and chunk counts. Best for understanding
    what information sources are available.
    
    Args:
        limit: Maximum number of documents to return (1-100)
        offset: Number of documents to skip for pagination
    
    Returns:
        List of documents with metadata and chunk counts
    """
    input_data = DocumentListInput(limit=limit, offset=offset)
    
    documents = await list_documents_tool(input_data)
    
    # Convert to dict for agent
    return [
        {
            "id": d.id,
            "title": d.title,
            "source": d.source,
            "chunk_count": d.chunk_count,
            "created_at": d.created_at.isoformat()
        }
        for d in documents
    ]


@rag_agent.tool
async def search_document_facts(
    ctx: RunContext[AgentDependencies],
    entity_name: str,
    depth: int = 2
) -> Dict[str, Any]:
    """
    Search for facts about an entity EXTRACTED FROM INGESTED DOCUMENTS.
    
    ⚠️ THIS IS NOT FOR ASTROLOGY BASICS! For planets, signs, houses, aspects use:
    - lookup_concept() for definitions
    - explore_ontology() for relationships
    
    Only use this tool when you need to find what the USER'S DOCUMENTS say about something,
    not for standard astrological knowledge.
    
    Args:
        entity_name: Name to search in documents
        depth: Search depth (1-3)
    
    Returns:
        Facts extracted from user's documents (may be empty if not ingested)
    """
    input_data = EntityRelationshipInput(
        entity_name=entity_name,
        depth=depth
    )
    
    return await get_entity_relationships_tool(input_data)


@rag_agent.tool
async def get_entity_timeline(
    ctx: RunContext[AgentDependencies],
    entity_name: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Get chronological information about an astrological concept or event.
    
    Use this for:
    - Transit timelines and their effects
    - Retrograde periods and meanings
    - Lunar phase progressions
    - Seasonal astrological shifts
    
    Args:
        entity_name: Concept or event (e.g., "Merkur Rückläufig", "Vollmond")
        start_date: Optional start date filter (YYYY-MM-DD)
        end_date: Optional end date filter (YYYY-MM-DD)
    
    Returns:
        Chronological list of facts about the concept
    """
    input_data = EntityTimelineInput(
        entity_name=entity_name,
        start_date=start_date,
        end_date=end_date
    )
    
    return await get_entity_timeline_tool(input_data)


# =============================================================================
# Ontology Lookup Tool
# =============================================================================

@rag_agent.tool
async def lookup_concept(
    ctx: RunContext[AgentDependencies],
    concept: str
) -> Dict[str, Any]:
    """
    Look up an astrological concept for quick reference.
    
    Get details about planets, signs, houses, aspects, elements, and themes
    from the ontology. Use this for quick definitions and relationships.
    
    Examples: "Venus", "Skorpion", "8. Haus", "Quadrat", "Feuer", "Transformation"
    
    Args:
        concept: Name of the concept (German or English)
    
    Returns:
        Description, keywords, and related concepts
    """
    input_data = OntologyLookupInput(concept=concept)
    return lookup_ontology_concept(input_data)


# =============================================================================
# Comprehensive Lookup Tool (Combines ALL sources)
# =============================================================================

@rag_agent.tool
async def comprehensive_lookup(
    ctx: RunContext[AgentDependencies],
    concept: str
) -> Dict[str, Any]:
    """
    Get complete knowledge about an astrological concept from all sources.
    
    Use for any question about astrology concepts, people's charts, events, etc.
    
    Args:
        concept: Any astrological term (planet, sign, house, aspect, theme, transit, etc.)
    
    Returns:
        Structured data from ontology + relevant document excerpts
    """
    result = {
        "query": concept,
        "ontology": None,
        "documents": [],
        "usage_hint": None
    }
    
    # 1. Get ontology data (if concept exists there)
    try:
        ontology_data = lookup_ontology_concept(OntologyLookupInput(concept=concept))
        if ontology_data.get("found"):
            concept_info = ontology_data.get("concept", {})
            concept_type = concept_info.get("type", "unknown")
            
            result["ontology"] = {
                "name": concept_info.get("name"),
                "type": concept_type,
                "description": concept_info.get("description"),
                "keywords": concept_info.get("keywords", [])[:8],
                "attributes": concept_info.get("attributes", {}),
                "relationships": ontology_data.get("relationships", [])[:5],
                "related_concepts": ontology_data.get("related_concepts", [])[:5]
            }
            
            # Add type-specific usage hint to prevent attribute confusion
            if concept_type == "planet":
                result["usage_hint"] = "Dies ist ein PLANET. Planeten haben: Herrschaft über Zeichen, Erhöhung, Fall. Planeten haben KEINE Elemente oder Modalitäten."
            elif concept_type == "sign":
                result["usage_hint"] = "Dies ist ein TIERKREISZEICHEN. Zeichen haben: Element, Modalität, Herrscherplanet."
                
    except Exception as e:
        logger.warning(f"Ontology lookup failed for {concept}: {e}")
    
    # 2. Get relevant document content
    try:
        doc_results = await hybrid_search_tool(
            HybridSearchInput(query=concept, limit=3, text_weight=0.3)
        )
        if doc_results:
            result["documents"] = [
                {
                    "excerpt": r.content[:400] if r.content else "",
                    "source": r.document_title or "Unknown",
                    "relevance": round(r.score, 2)
                }
                for r in doc_results[:3]
            ]
    except Exception as e:
        logger.warning(f"Document search failed for {concept}: {e}")
    
    # 3. Summary
    result["summary"] = {
        "found_in_ontology": result["ontology"] is not None,
        "document_matches": len(result["documents"]),
        "concept_type": result["ontology"]["type"] if result["ontology"] else "unknown"
    }
    
    return result


# =============================================================================
# Ontology Exploration Tool (Consolidated)
# =============================================================================

@rag_agent.tool
async def explore_ontology(
    ctx: RunContext[AgentDependencies],
    query: str,
    mode: str = "connections",
    secondary_query: Optional[str] = None,
    max_hops: int = 2
) -> Dict[str, Any]:
    """
    Explore the astrological ontology graph for deep insights.
    
    This is your tool for understanding astrological relationships and finding
    content through the knowledge graph structure.
    
    Modes:
    - "element": Find content about signs of an element (query: "fire"/"earth"/"air"/"water")
    - "modality": Find content about signs of a modality (query: "cardinal"/"fixed"/"mutable")
    - "planet": Find content about signs ruled by a planet (query: "venus"/"saturn"/etc.)
    - "theme": Find content about a life theme (query: "transformation"/"healing"/"relationships")
    - "connections": Multi-hop traversal from a concept (query: entity ID like "pluto")
    - "co_occurrence": Find what's often mentioned with a concept (query: entity ID)
    - "pair": Find content mentioning two concepts together (query + secondary_query)
    
    Args:
        query: The search term - element name, planet, theme, or entity ID
        mode: Type of exploration (default: "connections")
        secondary_query: Second entity for "pair" mode only
        max_hops: Traversal depth for "connections" mode (1-3, default 2)
    
    Returns:
        Related chunks or entity connections based on mode
    """
    try:
        if mode == "element":
            input_data = ElementSearchInput(element=query)
            return await search_chunks_by_element(input_data)
        
        elif mode == "modality":
            input_data = ModalitySearchInput(modality=query)
            return await search_chunks_by_modality(input_data)
        
        elif mode == "planet":
            input_data = PlanetRulershipInput(planet=query)
            return await search_chunks_by_planet_rulership(input_data)
        
        elif mode == "theme":
            input_data = ThemeSearchInput(theme=query)
            return await search_chunks_by_theme(input_data)
        
        elif mode == "connections":
            input_data = OntologyTraversalInput(entity_id=query, max_hops=max_hops, limit=20)
            return await traverse_ontology_for_chunks(input_data)
        
        elif mode == "co_occurrence":
            input_data = CoOccurrenceInput(entity_id=query, min_count=1, limit=20)
            return await find_co_occurring_entities(input_data)
        
        elif mode == "pair":
            if not secondary_query:
                return {"error": "pair mode requires secondary_query parameter"}
            input_data = EntityPairInput(entity1=query, entity2=secondary_query)
            return await find_chunks_with_entity_pair(input_data)
        
        else:
            return {"error": f"Unknown mode: {mode}. Use: element, modality, planet, theme, connections, co_occurrence, pair"}
    
    except Exception as e:
        logger.error(f"explore_ontology failed: {e}")
        return {"error": str(e)}