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

from .prompts import SYSTEM_PROMPT
from .providers import get_llm_model
from .tools import (
    generate_embedding,
    VectorSearchInput,
    GraphSearchInput,
    HybridSearchInput,
    EntityRelationshipInput,
    EntityTimelineInput
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
    ctx.deps.state.retrieved_chunks = []
    ctx.deps.state.graph_results = []
    ctx.deps.state.current_query = None
    ctx.deps.state.selected_chunk_id = None
    
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
    ctx.deps.state.selected_chunk_id = chunk_id
    
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
    current_query = ctx.deps.state.current_query
    
    base_instructions = dedent(
        f"""
        You are an intelligent RAG assistant with access to both a vector database 
        and a knowledge graph containing information about big tech companies and AI.
        
        IMPORTANT INSTRUCTIONS:
        1. Always use search tools to find relevant information before answering
        2. Use `vector_search_tool` for semantic similarity search
        3. Use `graph_search_tool` for facts, relationships, and entity connections
        4. Use `hybrid_search_tool` when you want both semantic and keyword matching
        5. Use `get_entity_relationships_tool` to explore how entities connect
        6. Use `get_entity_timeline_tool` for temporal/historical information
        7. Results will be displayed in the UI for the user to explore
        8. Use `clear_search_results` when starting a new topic
        
        Knowledge Base Status: {ctx.deps.state.knowledge_base_status}
        Graph Status: {ctx.deps.state.graph_status}
        Total chunks in KB: {ctx.deps.state.total_chunks_in_kb}
        """
    )
    
    if has_chunks or has_graph_results:
        context_info = "\n\nCURRENT STATE:"
        
        if current_query:
            context_info += f"\n- Current query: \"{current_query.query}\" ({current_query.search_type})"
        
        if has_chunks:
            context_info += f"\n- {len(ctx.deps.state.retrieved_chunks)} chunks retrieved from vector search"
            context_info += "\n\nTOP RETRIEVED CHUNKS:"
            for i, chunk in enumerate(ctx.deps.state.retrieved_chunks[:3], 1):
                context_info += f"""
            Chunk {i} (Score: {chunk.similarity:.3f}):
            Source: {chunk.document_title}
            Content: {chunk.content[:200]}...
            """
        
        if has_graph_results:
            context_info += f"\n- {len(ctx.deps.state.graph_results)} facts from knowledge graph"
            context_info += "\n\nTOP GRAPH FACTS:"
            for i, fact in enumerate(ctx.deps.state.graph_results[:3], 1):
                context_info += f"""
            Fact {i}: {fact.fact}
            Valid at: {fact.valid_at or 'N/A'}
            """
        
        return base_instructions + context_info
    
    return base_instructions + dedent(
        """
        
        CURRENT STATE:
        - No results retrieved yet
        - Use search tools to find relevant information
        """
    )


# Convert agent to AG-UI app
app = rag_agent.to_ag_ui(deps=StateDeps(RAGState()))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)

