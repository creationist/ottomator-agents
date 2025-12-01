"""
AG-UI enabled RAG agent with knowledge graph support and shared state.
"""

import os
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
from textwrap import dedent

from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext
from ag_ui.core import StateSnapshotEvent, EventType
from pydantic_ai.ag_ui import StateDeps
from dotenv import load_dotenv

from .prompts import SYSTEM_PROMPT, INSPIRATIONAL_CONTENT_TEMPLATE
from .providers import get_llm_model
from .tools import (
    generate_embedding,
    VectorSearchInput,
    GraphSearchInput,
    HybridSearchInput,
    EntityRelationshipInput,
    EntityTimelineInput,
    lookup_ontology_concept,
    OntologyLookupInput
)
from .db_utils import (
    initialize_database,
    close_database,
    vector_search,
    hybrid_search as db_hybrid_search
)
from .graph_utils import (
    initialize_graph,
    close_graph,
    search_knowledge_graph,
    get_entity_relationships as graph_get_entity_relationships,
    graph_client
)

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)


# Shared State Models
class RetrievedChunk(BaseModel):
    """Model for a retrieved chunk with metadata."""
    chunk_id: str = Field(description="Unique identifier for the chunk")
    document_id: str = Field(description="ID of the source document")
    content: str = Field(description="The actual text content of the chunk")
    similarity: float = Field(description="Similarity score to the query")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    document_title: str = Field(description="Title of the source document")
    document_source: str = Field(description="Source/path of the document")
    highlight: Optional[str] = Field(default=None, description="Highlighted matching text")


class GraphResult(BaseModel):
    """Model for a knowledge graph search result."""
    fact: str = Field(description="The fact from the knowledge graph")
    uuid: str = Field(description="Unique identifier for the fact")
    valid_at: Optional[str] = Field(default=None, description="When the fact became valid")
    invalid_at: Optional[str] = Field(default=None, description="When the fact became invalid")
    source_node_uuid: Optional[str] = Field(default=None, description="Source node UUID")


class SearchQuery(BaseModel):
    """Model for a search query."""
    query: str = Field(description="The search query text")
    timestamp: str = Field(description="When the query was made")
    match_count: int = Field(default=10, description="Number of results requested")
    search_type: str = Field(default="vector", description="Type of search performed")


class ToolCall(BaseModel):
    """Model for a tool call record."""
    tool_name: str = Field(description="Name of the tool that was called")
    arguments: Dict[str, Any] = Field(default_factory=dict, description="Arguments passed to the tool")
    timestamp: str = Field(description="When the tool was called")
    success: bool = Field(default=True, description="Whether the tool call succeeded")


class OntologyResult(BaseModel):
    """Model for an ontology lookup result."""
    concept_id: str = Field(description="ID of the concept")
    name: str = Field(description="Name of the concept")
    concept_type: str = Field(description="Type of concept (planet, sign, house, etc.)")
    description: str = Field(description="Description of the concept")
    keywords: List[str] = Field(default_factory=list, description="Related keywords")
    related_concepts: List[str] = Field(default_factory=list, description="Related concept IDs")


class RAGState(BaseModel):
    """Shared state for the RAG agent with knowledge graph support."""
    retrieved_chunks: List[RetrievedChunk] = Field(
        default_factory=list,
        description="List of chunks retrieved from vector search"
    )
    graph_results: List[GraphResult] = Field(
        default_factory=list,
        description="List of facts from knowledge graph search"
    )
    ontology_results: List[OntologyResult] = Field(
        default_factory=list,
        description="List of ontology lookup results"
    )
    current_query: Optional[SearchQuery] = Field(
        default=None,
        description="The current search query being processed"
    )
    search_history: List[SearchQuery] = Field(
        default_factory=list,
        description="History of search queries"
    )
    selected_chunk_id: Optional[str] = Field(
        default=None,
        description="ID of the currently selected/highlighted chunk"
    )
    total_chunks_in_kb: int = Field(
        default=0,
        description="Total number of chunks in the knowledge base"
    )
    knowledge_base_status: str = Field(
        default="ready",
        description="Status of the knowledge base (ready, indexing, error)"
    )
    graph_status: str = Field(
        default="ready",
        description="Status of the knowledge graph (ready, error)"
    )
    tools_used: List[ToolCall] = Field(
        default_factory=list,
        description="List of tools called during the current conversation turn"
    )


# Database connection state
_db_initialized = False
_graph_initialized = False


async def ensure_connections():
    """Ensure database and graph connections are initialized."""
    global _db_initialized, _graph_initialized
    
    if not _db_initialized:
        await initialize_database()
        _db_initialized = True
        logger.info("Database initialized for AG-UI agent")
    
    if not _graph_initialized:
        await initialize_graph()
        _graph_initialized = True
        logger.info("Graph database initialized for AG-UI agent")


# Create the RAG agent with AG-UI support
rag_agent = Agent(
    get_llm_model(),
    deps_type=StateDeps[RAGState],
    system_prompt=SYSTEM_PROMPT
)


@rag_agent.tool
async def vector_search_tool(
    ctx: RunContext[StateDeps[RAGState]],
    query: str,
    limit: Optional[int] = 10
) -> StateSnapshotEvent:
    """
    Search for relevant information using semantic similarity.
    
    This tool performs vector similarity search across document chunks
    to find semantically related content. Results are displayed in the UI.
    
    Args:
        ctx: Agent runtime context with state dependencies
        query: Search query to find similar content
        limit: Maximum number of results to return (1-50)
    
    Returns:
        StateSnapshotEvent with updated retrieved chunks
    """
    await ensure_connections()
    
    # Track tool usage
    tool_call = ToolCall(
        tool_name="vector_search",
        arguments={"query": query, "limit": limit or 10},
        timestamp=datetime.now().isoformat(),
        success=True
    )
    ctx.deps.state.tools_used.append(tool_call)
    
    # Create search query record
    search_query = SearchQuery(
        query=query,
        timestamp=datetime.now().isoformat(),
        match_count=limit or 10,
        search_type="vector"
    )
    
    ctx.deps.state.current_query = search_query
    ctx.deps.state.search_history.append(search_query)
    
    # Keep only last 10 queries
    if len(ctx.deps.state.search_history) > 10:
        ctx.deps.state.search_history = ctx.deps.state.search_history[-10:]
    
    try:
        # Generate embedding and perform search
        embedding = await generate_embedding(query)
        results = await vector_search(embedding=embedding, limit=limit or 10)
        
        # Convert to RetrievedChunk format
        chunks = [
            RetrievedChunk(
                chunk_id=str(r["chunk_id"]),
                document_id=str(r["document_id"]),
                content=r["content"],
                similarity=r["similarity"],
                metadata=r.get("metadata", {}),
                document_title=r["document_title"],
                document_source=r["document_source"]
            )
            for r in results
        ]
        
        ctx.deps.state.retrieved_chunks = chunks
        ctx.deps.state.knowledge_base_status = "ready"
        
    except Exception as e:
        logger.error(f"Vector search failed: {e}")
        ctx.deps.state.retrieved_chunks = []
        ctx.deps.state.knowledge_base_status = f"error: {str(e)}"
    
    return StateSnapshotEvent(
        type=EventType.STATE_SNAPSHOT,
        snapshot=ctx.deps.state.model_dump(),
    )


@rag_agent.tool
async def graph_search_tool(
    ctx: RunContext[StateDeps[RAGState]],
    query: str
) -> StateSnapshotEvent:
    """
    Search the knowledge graph for facts and relationships.
    
    This tool queries the knowledge graph to find specific facts, relationships 
    between entities, and temporal information. Best for finding specific facts,
    relationships between companies/people/technologies, and time-based information.
    
    Args:
        ctx: Agent runtime context with state dependencies
        query: Search query to find facts and relationships
    
    Returns:
        StateSnapshotEvent with graph search results
    """
    await ensure_connections()
    
    # Track tool usage
    tool_call = ToolCall(
        tool_name="graph_search",
        arguments={"query": query},
        timestamp=datetime.now().isoformat(),
        success=True
    )
    ctx.deps.state.tools_used.append(tool_call)
    
    # Create search query record
    search_query = SearchQuery(
        query=query,
        timestamp=datetime.now().isoformat(),
        match_count=0,  # Graph search returns variable results
        search_type="graph"
    )
    
    ctx.deps.state.current_query = search_query
    ctx.deps.state.search_history.append(search_query)
    
    if len(ctx.deps.state.search_history) > 10:
        ctx.deps.state.search_history = ctx.deps.state.search_history[-10:]
    
    try:
        results = await search_knowledge_graph(query=query)
        
        # Convert to GraphResult format
        graph_results = [
            GraphResult(
                fact=r["fact"],
                uuid=r["uuid"],
                valid_at=r.get("valid_at"),
                invalid_at=r.get("invalid_at"),
                source_node_uuid=r.get("source_node_uuid")
            )
            for r in results
        ]
        
        ctx.deps.state.graph_results = graph_results
        ctx.deps.state.graph_status = "ready"
        
    except Exception as e:
        logger.error(f"Graph search failed: {e}")
        ctx.deps.state.graph_results = []
        ctx.deps.state.graph_status = f"error: {str(e)}"
    
    return StateSnapshotEvent(
        type=EventType.STATE_SNAPSHOT,
        snapshot=ctx.deps.state.model_dump(),
    )


@rag_agent.tool
async def hybrid_search_tool(
    ctx: RunContext[StateDeps[RAGState]],
    query: str,
    limit: Optional[int] = 10,
    text_weight: Optional[float] = 0.3
) -> StateSnapshotEvent:
    """
    Perform both vector and keyword search for comprehensive results.
    
    This tool combines semantic similarity search with keyword matching
    for the best coverage. It ranks results using both vector similarity
    and text matching scores.
    
    Args:
        ctx: Agent runtime context with state dependencies
        query: Search query for hybrid search
        limit: Maximum number of results to return (1-50)
        text_weight: Weight for text similarity vs vector similarity (0.0-1.0)
    
    Returns:
        StateSnapshotEvent with hybrid search results
    """
    await ensure_connections()
    
    # Track tool usage
    tool_call = ToolCall(
        tool_name="hybrid_search",
        arguments={"query": query, "limit": limit or 10, "text_weight": text_weight or 0.3},
        timestamp=datetime.now().isoformat(),
        success=True
    )
    ctx.deps.state.tools_used.append(tool_call)
    
    # Create search query record
    search_query = SearchQuery(
        query=query,
        timestamp=datetime.now().isoformat(),
        match_count=limit or 10,
        search_type="hybrid"
    )
    
    ctx.deps.state.current_query = search_query
    ctx.deps.state.search_history.append(search_query)
    
    if len(ctx.deps.state.search_history) > 10:
        ctx.deps.state.search_history = ctx.deps.state.search_history[-10:]
    
    try:
        # Generate embedding and perform hybrid search
        embedding = await generate_embedding(query)
        results = await db_hybrid_search(
            embedding=embedding,
            query_text=query,
            limit=limit or 10,
            text_weight=text_weight or 0.3
        )
        
        # Convert to RetrievedChunk format
        chunks = [
            RetrievedChunk(
                chunk_id=str(r["chunk_id"]),
                document_id=str(r["document_id"]),
                content=r["content"],
                similarity=r["combined_score"],
                metadata={
                    **r.get("metadata", {}),
                    "vector_similarity": r.get("vector_similarity"),
                    "text_similarity": r.get("text_similarity")
                },
                document_title=r["document_title"],
                document_source=r["document_source"]
            )
            for r in results
        ]
        
        ctx.deps.state.retrieved_chunks = chunks
        ctx.deps.state.knowledge_base_status = "ready"
        
    except Exception as e:
        logger.error(f"Hybrid search failed: {e}")
        ctx.deps.state.retrieved_chunks = []
        ctx.deps.state.knowledge_base_status = f"error: {str(e)}"
    
    return StateSnapshotEvent(
        type=EventType.STATE_SNAPSHOT,
        snapshot=ctx.deps.state.model_dump(),
    )


@rag_agent.tool
async def get_entity_relationships_tool(
    ctx: RunContext[StateDeps[RAGState]],
    entity_name: str,
    depth: Optional[int] = 2
) -> StateSnapshotEvent:
    """
    Get all relationships for a specific entity in the knowledge graph.
    
    This tool explores the knowledge graph to find how a specific entity
    (company, person, technology) relates to other entities. Best for
    understanding how companies or technologies relate to each other.
    
    Args:
        ctx: Agent runtime context with state dependencies
        entity_name: Name of the entity to explore (e.g., "Google", "OpenAI")
        depth: Maximum traversal depth for relationships (1-5)
    
    Returns:
        StateSnapshotEvent with entity relationships
    """
    await ensure_connections()
    
    # Track tool usage
    tool_call = ToolCall(
        tool_name="get_entity_relationships",
        arguments={"entity_name": entity_name, "depth": depth or 2},
        timestamp=datetime.now().isoformat(),
        success=True
    )
    ctx.deps.state.tools_used.append(tool_call)
    
    # Create search query record
    search_query = SearchQuery(
        query=f"relationships: {entity_name}",
        timestamp=datetime.now().isoformat(),
        match_count=0,
        search_type="entity_relationships"
    )
    
    ctx.deps.state.current_query = search_query
    ctx.deps.state.search_history.append(search_query)
    
    if len(ctx.deps.state.search_history) > 10:
        ctx.deps.state.search_history = ctx.deps.state.search_history[-10:]
    
    try:
        result = await graph_get_entity_relationships(
            entity=entity_name,
            depth=depth or 2
        )
        
        # Convert related facts to GraphResult format
        related_facts = result.get("related_facts", [])
        graph_results = [
            GraphResult(
                fact=f["fact"],
                uuid=f["uuid"],
                valid_at=f.get("valid_at")
            )
            for f in related_facts
        ]
        
        ctx.deps.state.graph_results = graph_results
        ctx.deps.state.graph_status = "ready"
        
    except Exception as e:
        logger.error(f"Entity relationship query failed: {e}")
        ctx.deps.state.graph_results = []
        ctx.deps.state.graph_status = f"error: {str(e)}"
    
    return StateSnapshotEvent(
        type=EventType.STATE_SNAPSHOT,
        snapshot=ctx.deps.state.model_dump(),
    )


@rag_agent.tool
async def get_entity_timeline_tool(
    ctx: RunContext[StateDeps[RAGState]],
    entity_name: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None
) -> StateSnapshotEvent:
    """
    Get the timeline of facts for a specific entity.
    
    This tool retrieves chronological information about an entity,
    showing how information has evolved over time. Best for understanding
    how information about an entity has developed or changed.
    
    Args:
        ctx: Agent runtime context with state dependencies
        entity_name: Name of the entity (e.g., "Microsoft", "AI")
        start_date: Start date in ISO format (YYYY-MM-DD), optional
        end_date: End date in ISO format (YYYY-MM-DD), optional
    
    Returns:
        StateSnapshotEvent with timeline data
    """
    await ensure_connections()
    
    # Track tool usage
    tool_call = ToolCall(
        tool_name="get_entity_timeline",
        arguments={"entity_name": entity_name, "start_date": start_date, "end_date": end_date},
        timestamp=datetime.now().isoformat(),
        success=True
    )
    ctx.deps.state.tools_used.append(tool_call)
    
    # Create search query record
    search_query = SearchQuery(
        query=f"timeline: {entity_name}",
        timestamp=datetime.now().isoformat(),
        match_count=0,
        search_type="entity_timeline"
    )
    
    ctx.deps.state.current_query = search_query
    ctx.deps.state.search_history.append(search_query)
    
    if len(ctx.deps.state.search_history) > 10:
        ctx.deps.state.search_history = ctx.deps.state.search_history[-10:]
    
    try:
        # Parse dates if provided
        start_dt = None
        end_dt = None
        if start_date:
            start_dt = datetime.fromisoformat(start_date)
        if end_date:
            end_dt = datetime.fromisoformat(end_date)
        
        timeline = await graph_client.get_entity_timeline(
            entity_name=entity_name,
            start_date=start_dt,
            end_date=end_dt
        )
        
        # Convert to GraphResult format
        graph_results = [
            GraphResult(
                fact=item["fact"],
                uuid=item["uuid"],
                valid_at=item.get("valid_at"),
                invalid_at=item.get("invalid_at")
            )
            for item in timeline
        ]
        
        ctx.deps.state.graph_results = graph_results
        ctx.deps.state.graph_status = "ready"
        
    except Exception as e:
        logger.error(f"Entity timeline query failed: {e}")
        ctx.deps.state.graph_results = []
        ctx.deps.state.graph_status = f"error: {str(e)}"
    
    return StateSnapshotEvent(
        type=EventType.STATE_SNAPSHOT,
        snapshot=ctx.deps.state.model_dump(),
    )


@rag_agent.tool
async def clear_search_results(
    ctx: RunContext[StateDeps[RAGState]]
) -> StateSnapshotEvent:
    """
    Clear the current search results from the shared state.
    
    Args:
        ctx: Agent runtime context with state dependencies
    
    Returns:
        StateSnapshotEvent with cleared results
    """
    # Track tool usage (record before clearing)
    tool_call = ToolCall(
        tool_name="clear_search_results",
        arguments={},
        timestamp=datetime.now().isoformat(),
        success=True
    )
    
    ctx.deps.state.retrieved_chunks = []
    ctx.deps.state.graph_results = []
    ctx.deps.state.current_query = None
    ctx.deps.state.selected_chunk_id = None
    ctx.deps.state.tools_used = [tool_call]  # Reset with just this call
    
    return StateSnapshotEvent(
        type=EventType.STATE_SNAPSHOT,
        snapshot=ctx.deps.state.model_dump(),
    )


@rag_agent.tool
async def select_chunk(
    ctx: RunContext[StateDeps[RAGState]],
    chunk_id: str
) -> StateSnapshotEvent:
    """
    Select/highlight a specific chunk in the UI.
    
    Args:
        ctx: Agent runtime context with state dependencies
        chunk_id: ID of the chunk to select
    
    Returns:
        StateSnapshotEvent with updated selection
    """
    # Track tool usage
    tool_call = ToolCall(
        tool_name="select_chunk",
        arguments={"chunk_id": chunk_id},
        timestamp=datetime.now().isoformat(),
        success=True
    )
    ctx.deps.state.tools_used.append(tool_call)
    
    ctx.deps.state.selected_chunk_id = chunk_id
    
    return StateSnapshotEvent(
        type=EventType.STATE_SNAPSHOT,
        snapshot=ctx.deps.state.model_dump(),
    )


@rag_agent.tool
async def lookup_astrology_concept(
    ctx: RunContext[StateDeps[RAGState]],
    concept: str
) -> StateSnapshotEvent:
    """
    Look up an astrological concept to get its meaning and relationships.
    
    This tool queries the astrology ontology to provide detailed information
    about planets, zodiac signs, houses, aspects, elements, modalities,
    lunar phases, and astrological themes. Best for explaining astrological
    concepts or understanding how they relate to each other.
    
    Available concept types:
    - Planets: Sonne, Mond, Merkur, Venus, Mars, Jupiter, Saturn, Uranus, Neptun, Pluto, Chiron
    - Signs: Widder, Stier, Zwillinge, Krebs, Löwe, Jungfrau, Waage, Skorpion, Schütze, Steinbock, Wassermann, Fische
    - Houses: Erstes Haus through Zwölftes Haus
    - Aspects: Konjunktion, Sextil, Quadrat, Trigon, Opposition
    - Elements: Feuer, Erde, Luft, Wasser
    - Modalities: Kardinal, Fix, Veränderlich
    - Themes: Transformation, Heilung, Beziehungen, Karma, Spiritualität, etc.
    
    Args:
        ctx: Agent runtime context with state dependencies
        concept: Name of the astrological concept (German or English, e.g., "Venus", "Skorpion", "transformation")
    
    Returns:
        StateSnapshotEvent with ontology result in state
    """
    # Track tool usage
    tool_call = ToolCall(
        tool_name="lookup_astrology_concept",
        arguments={"concept": concept},
        timestamp=datetime.now().isoformat(),
        success=False
    )
    ctx.deps.state.tools_used.append(tool_call)
    
    # Create search query record
    search_query = SearchQuery(
        query=f"ontology: {concept}",
        timestamp=datetime.now().isoformat(),
        match_count=1,
        search_type="ontology"
    )
    ctx.deps.state.current_query = search_query
    ctx.deps.state.search_history.append(search_query)
    
    if len(ctx.deps.state.search_history) > 10:
        ctx.deps.state.search_history = ctx.deps.state.search_history[-10:]
    
    try:
        input_data = OntologyLookupInput(concept=concept)
        result = lookup_ontology_concept(input_data)
        
        if result and "error" not in result:
            # Extract related concept names (they come as dicts with id, name, type)
            related = result.get("related_concepts", [])
            related_names = []
            for r in related:
                if isinstance(r, dict):
                    related_names.append(r.get("name", r.get("id", "")))
                elif isinstance(r, str):
                    related_names.append(r)
            
            # Convert to OntologyResult and store in state
            ontology_result = OntologyResult(
                concept_id=result.get("id", concept),
                name=result.get("name", concept),
                concept_type=result.get("type", "unknown"),
                description=result.get("description", ""),
                keywords=result.get("keywords", []),
                related_concepts=related_names
            )
            ctx.deps.state.ontology_results = [ontology_result]
            tool_call.success = True
        else:
            ctx.deps.state.ontology_results = []
            
    except Exception as e:
        logger.error(f"Astrology concept lookup failed: {e}")
        ctx.deps.state.ontology_results = []
    
    return StateSnapshotEvent(
        type=EventType.STATE_SNAPSHOT,
        snapshot=ctx.deps.state.model_dump(),
    )


@rag_agent.tool
async def generate_inspirational_content(
    ctx: RunContext[StateDeps[RAGState]],
    topic: str,
    user_birthday: Optional[str] = None,
    sun_sign: Optional[str] = None,
    moon_sign: Optional[str] = None,
    rising_sign: Optional[str] = None,
    additional_context: Optional[str] = None
) -> StateSnapshotEvent:
    """
    Generate personalized inspirational astrology content based on user context.
    
    This tool creates meaningful, personalized insights by combining the user's
    astrological context with retrieved document knowledge. Use this when the user
    asks for personalized readings, inspirational texts, or content about their
    specific astrological situation.
    
    Args:
        ctx: Agent runtime context with state dependencies
        topic: The theme, question, or astrological event to create content about
               (e.g., "Neumond im Skorpion", "meine Woche", "Venus Transit")
        user_birthday: User's birthday in any format (optional)
        sun_sign: User's sun/zodiac sign - Sonnenzeichen (optional)
        moon_sign: User's moon sign - Mondzeichen (optional)
        rising_sign: User's rising/ascendant sign - Aszendent (optional)
        additional_context: Any additional astrological info like planets, houses, aspects (optional)
    
    Returns:
        StateSnapshotEvent with search results in state
    """
    await ensure_connections()
    
    # Track tool usage
    tool_call = ToolCall(
        tool_name="generate_inspirational_content",
        arguments={
            "topic": topic,
            "sun_sign": sun_sign,
            "moon_sign": moon_sign,
            "rising_sign": rising_sign
        },
        timestamp=datetime.now().isoformat(),
        success=False
    )
    ctx.deps.state.tools_used.append(tool_call)
    
    # Create search query record
    search_query = SearchQuery(
        query=f"inspirational: {topic}",
        timestamp=datetime.now().isoformat(),
        match_count=5,
        search_type="inspirational"
    )
    ctx.deps.state.current_query = search_query
    ctx.deps.state.search_history.append(search_query)
    
    if len(ctx.deps.state.search_history) > 10:
        ctx.deps.state.search_history = ctx.deps.state.search_history[-10:]
    
    try:
        # Search for relevant content about the topic
        embedding = await generate_embedding(topic)
        search_results = await vector_search(embedding=embedding, limit=5)
        
        # Update state with search results for UI display
        if search_results:
            chunks = [
                RetrievedChunk(
                    chunk_id=str(r["chunk_id"]),
                    document_id=str(r["document_id"]),
                    content=r["content"],
                    similarity=r["similarity"],
                    metadata=r.get("metadata", {}),
                    document_title=r["document_title"],
                    document_source=r["document_source"]
                )
                for r in search_results
            ]
            ctx.deps.state.retrieved_chunks = chunks
        else:
            ctx.deps.state.retrieved_chunks = []
        
        ctx.deps.state.knowledge_base_status = "ready"
        tool_call.success = True
        
    except Exception as e:
        logger.error(f"Inspirational content generation failed: {e}")
        ctx.deps.state.retrieved_chunks = []
        ctx.deps.state.knowledge_base_status = f"error: {str(e)}"
    
    return StateSnapshotEvent(
        type=EventType.STATE_SNAPSHOT,
        snapshot=ctx.deps.state.model_dump(),
    )


@rag_agent.instructions
async def rag_instructions(ctx: RunContext[StateDeps[RAGState]]) -> str:
    """
    Dynamic instructions for the RAG agent based on current state.
    """
    has_chunks = len(ctx.deps.state.retrieved_chunks) > 0
    has_graph_results = len(ctx.deps.state.graph_results) > 0
    has_ontology_results = len(ctx.deps.state.ontology_results) > 0
    current_query = ctx.deps.state.current_query
    
    base_instructions = dedent(
        f"""
        Du bist Nyah mit Zugang zu einer astrologischen Wissensbasis und Ontologie.
        
        **KRITISCH - IMMER ZUERST SUCHEN:**
        Du MUSST bei JEDER Frage mindestens ein Tool aufrufen bevor du antwortest!
        Antworte NIE nur aus deinem eigenen Wissen.
        
        **Tool-Nutzung:**
        1. `lookup_astrology_concept` - Für astrologische Konzepte (Planeten, Zeichen, Häuser, Aspekte)
        2. `vector_search_tool` - Für semantische Suche in Dokumenten
        3. `hybrid_search_tool` - Für kombinierte semantische + Keyword-Suche
        4. `generate_inspirational_content` - Für personalisierte inspirierende Texte
        5. `graph_search_tool` - Für Wissensverknüpfungen
        
        **Beispiele:**
        - Frage "Was ist Merkur?" → ZUERST `lookup_astrology_concept("Merkur")` aufrufen
        - Frage "Erzähl mir über Venus Transit" → ZUERST `vector_search_tool("Venus Transit")` aufrufen
        - Frage "Inspirierende Botschaft für Skorpion" → `generate_inspirational_content` nutzen
        
        Wissensbasis Status: {ctx.deps.state.knowledge_base_status}
        Graph Status: {ctx.deps.state.graph_status}
        """
    )
    
    if has_chunks or has_graph_results or has_ontology_results:
        context_info = "\n\n**AKTUELLER ZUSTAND:**"
        
        if current_query:
            context_info += f"\n- Aktuelle Suche: \"{current_query.query}\" ({current_query.search_type})"
        
        if has_ontology_results:
            context_info += f"\n- {len(ctx.deps.state.ontology_results)} Ontologie-Ergebnis(se) gefunden"
            context_info += "\n\n**ONTOLOGIE ERGEBNISSE (nutze diese für deine Antwort!):**"
            for i, result in enumerate(ctx.deps.state.ontology_results, 1):
                context_info += f"""
            Konzept {i}: {result.name} ({result.concept_type})
            Beschreibung: {result.description}
            Schlüsselwörter: {', '.join(result.keywords[:5])}
            Verwandte Konzepte: {', '.join(result.related_concepts[:5])}
            """
        
        if has_chunks:
            context_info += f"\n- {len(ctx.deps.state.retrieved_chunks)} Dokument-Chunks gefunden"
            context_info += "\n\n**GEFUNDENE INHALTE:**"
            for i, chunk in enumerate(ctx.deps.state.retrieved_chunks[:3], 1):
                context_info += f"""
            Chunk {i} (Ähnlichkeit: {chunk.similarity:.3f}):
            Quelle: {chunk.document_title}
            Inhalt: {chunk.content[:200]}...
            """
        
        if has_graph_results:
            context_info += f"\n- {len(ctx.deps.state.graph_results)} Graph-Fakten gefunden"
            context_info += "\n\n**GRAPH FAKTEN:**"
            for i, fact in enumerate(ctx.deps.state.graph_results[:3], 1):
                context_info += f"""
            Fakt {i}: {fact.fact}
            Gültig ab: {fact.valid_at or 'N/A'}
            """
        
        return base_instructions + context_info
    
    return base_instructions + dedent(
        """
        
        **AKTUELLER ZUSTAND:**
        - Noch keine Suchergebnisse
        - RUFE JETZT EIN SUCH-TOOL AUF bevor du antwortest!
        """
    )


# Convert agent to AG-UI app
app = rag_agent.to_ag_ui(deps=StateDeps(RAGState()))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)

