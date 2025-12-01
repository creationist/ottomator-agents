"""
FastAPI endpoints for the agentic RAG system.
"""

import os
import asyncio
import json
import logging
from contextlib import asynccontextmanager
from typing import Dict, Any, List, Optional
from datetime import datetime
import uuid

from fastapi import FastAPI, HTTPException, Request, Depends
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
import uvicorn
from dotenv import load_dotenv

from .agent import rag_agent, AgentDependencies
from .db_utils import (
    initialize_database,
    close_database,
    create_session,
    get_session,
    add_message,
    get_session_messages,
    test_connection
)
from .graph_utils import initialize_graph, close_graph, test_graph_connection
from .models import (
    ChatRequest,
    ChatResponse,
    SearchRequest,
    SearchResponse,
    StreamDelta,
    ErrorResponse,
    HealthStatus,
    ToolCall,
    # Content generation models
    ContentTypeEnum,
    BirthData,
    UserProfileRequest,
    UserProfileResponse,
    ContentGenerateRequest,
    ContentResponse,
    BatchGenerateRequest,
    BatchJobStatusResponse,
    MonthlyContentResponse,
)
from .tools import (
    vector_search_tool,
    graph_search_tool,
    hybrid_search_tool,
    list_documents_tool,
    VectorSearchInput,
    GraphSearchInput,
    HybridSearchInput,
    DocumentListInput
)

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)

# Application configuration
APP_ENV = os.getenv("APP_ENV", "development")
APP_HOST = os.getenv("APP_HOST", "0.0.0.0")
APP_PORT = int(os.getenv("APP_PORT", 8000))
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

# Configure logging
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL.upper()),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

# Set debug level for our module during development
if APP_ENV == "development":
    logger.setLevel(logging.DEBUG)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for FastAPI app."""
    # Startup
    logger.info("Starting up agentic RAG API...")
    
    try:
        # Initialize database connections
        await initialize_database()
        logger.info("Database initialized")
        
        # Initialize graph database
        await initialize_graph()
        logger.info("Graph database initialized")
        
        # Test connections
        db_ok = await test_connection()
        graph_ok = await test_graph_connection()
        
        if not db_ok:
            logger.error("Database connection failed")
        if not graph_ok:
            logger.error("Graph database connection failed")
        
        logger.info("Agentic RAG API startup complete")
        
    except Exception as e:
        logger.error(f"Startup failed: {e}")
        raise
    
    yield
    
    # Shutdown
    logger.info("Shutting down agentic RAG API...")
    
    try:
        await close_database()
        await close_graph()
        logger.info("Connections closed")
    except Exception as e:
        logger.error(f"Shutdown error: {e}")


# Create FastAPI app
app = FastAPI(
    title="Agentic RAG with Knowledge Graph",
    description="AI agent combining vector search and knowledge graph for tech company analysis",
    version="0.1.0",
    lifespan=lifespan
)

# Add middleware with flexible CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

app.add_middleware(GZipMiddleware, minimum_size=1000)


# Helper functions for agent execution
async def get_or_create_session(request: ChatRequest) -> str:
    """Get existing session or create new one."""
    if request.session_id:
        session = await get_session(request.session_id)
        if session:
            return request.session_id
    
    # Create new session
    return await create_session(
        user_id=request.user_id,
        metadata=request.metadata
    )


async def get_conversation_context(
    session_id: str,
    max_messages: int = 10
) -> List[Dict[str, str]]:
    """
    Get recent conversation context.
    
    Args:
        session_id: Session ID
        max_messages: Maximum number of messages to retrieve
    
    Returns:
        List of messages
    """
    messages = await get_session_messages(session_id, limit=max_messages)
    
    return [
        {
            "role": msg["role"],
            "content": msg["content"]
        }
        for msg in messages
    ]


def extract_tool_calls(result) -> List[ToolCall]:
    """
    Extract tool calls from Pydantic AI result.
    
    Args:
        result: Pydantic AI result object
    
    Returns:
        List of ToolCall objects
    """
    tools_used = []
    
    try:
        # Get all messages from the result
        messages = result.all_messages()
        
        for message in messages:
            if hasattr(message, 'parts'):
                for part in message.parts:
                    # Check if this is a tool call part
                    if part.__class__.__name__ == 'ToolCallPart':
                        try:
                            # Debug logging to understand structure
                            logger.debug(f"ToolCallPart attributes: {dir(part)}")
                            logger.debug(f"ToolCallPart content: tool_name={getattr(part, 'tool_name', None)}")
                            
                            # Extract tool information safely
                            tool_name = str(part.tool_name) if hasattr(part, 'tool_name') else 'unknown'
                            
                            # Get args - the args field is a JSON string in Pydantic AI
                            tool_args = {}
                            if hasattr(part, 'args') and part.args is not None:
                                if isinstance(part.args, str):
                                    # Args is a JSON string, parse it
                                    try:
                                        import json
                                        tool_args = json.loads(part.args)
                                        logger.debug(f"Parsed args from JSON string: {tool_args}")
                                    except json.JSONDecodeError as e:
                                        logger.debug(f"Failed to parse args JSON: {e}")
                                        tool_args = {}
                                elif isinstance(part.args, dict):
                                    tool_args = part.args
                                    logger.debug(f"Args already a dict: {tool_args}")
                            
                            # Alternative: use args_as_dict method if available
                            if hasattr(part, 'args_as_dict'):
                                try:
                                    tool_args = part.args_as_dict()
                                    logger.debug(f"Got args from args_as_dict(): {tool_args}")
                                except:
                                    pass
                            
                            # Get tool call ID
                            tool_call_id = None
                            if hasattr(part, 'tool_call_id'):
                                tool_call_id = str(part.tool_call_id) if part.tool_call_id else None
                            
                            # Create ToolCall with explicit field mapping
                            tool_call_data = {
                                "tool_name": tool_name,
                                "args": tool_args,
                                "tool_call_id": tool_call_id
                            }
                            logger.debug(f"Creating ToolCall with data: {tool_call_data}")
                            tools_used.append(ToolCall(**tool_call_data))
                        except Exception as e:
                            logger.debug(f"Failed to parse tool call part: {e}")
                            continue
    except Exception as e:
        logger.warning(f"Failed to extract tool calls: {e}")
    
    return tools_used


async def save_conversation_turn(
    session_id: str,
    user_message: str,
    assistant_message: str,
    metadata: Optional[Dict[str, Any]] = None
):
    """
    Save a conversation turn to the database.
    
    Args:
        session_id: Session ID
        user_message: User's message
        assistant_message: Assistant's response
        metadata: Optional metadata
    """
    # Save user message
    await add_message(
        session_id=session_id,
        role="user",
        content=user_message,
        metadata=metadata or {}
    )
    
    # Save assistant message
    await add_message(
        session_id=session_id,
        role="assistant",
        content=assistant_message,
        metadata=metadata or {}
    )


async def execute_agent(
    message: str,
    session_id: str,
    user_id: Optional[str] = None,
    save_conversation: bool = True
) -> tuple[str, List[ToolCall]]:
    """
    Execute the agent with a message.
    
    Args:
        message: User message
        session_id: Session ID
        user_id: Optional user ID
        save_conversation: Whether to save the conversation
    
    Returns:
        Tuple of (agent response, tools used)
    """
    try:
        # Create dependencies
        deps = AgentDependencies(
            session_id=session_id,
            user_id=user_id
        )
        
        # Get conversation context
        context = await get_conversation_context(session_id)
        
        # Build prompt with context
        full_prompt = message
        if context:
            context_str = "\n".join([
                f"{msg['role']}: {msg['content']}"
                for msg in context[-6:]  # Last 3 turns
            ])
            full_prompt = f"Previous conversation:\n{context_str}\n\nCurrent question: {message}"
        
        # Run the agent
        result = await rag_agent.run(full_prompt, deps=deps)
        
        # Debug: log result attributes
        logger.debug(f"Agent result type: {type(result)}")
        logger.debug(f"Agent result attributes: {[a for a in dir(result) if not a.startswith('_')]}")
        
        # pydantic-ai uses .output (newer) or .data (older) for the response
        response = getattr(result, 'output', None) or getattr(result, 'data', None)
        
        # If neither works, try to get text from the result
        if response is None:
            # Try common attribute names
            for attr in ['text', 'content', 'message', 'response']:
                response = getattr(result, attr, None)
                if response:
                    break
        
        # Last resort: convert to string
        if response is None:
            response = str(result)
            logger.warning(f"Could not extract response, using str(): {response[:200]}")
        
        logger.debug(f"Final response type: {type(response)}, length: {len(str(response)) if response else 0}")
        
        tools_used = extract_tool_calls(result)
        
        # Save conversation if requested
        if save_conversation:
            await save_conversation_turn(
                session_id=session_id,
                user_message=message,
                assistant_message=response,
                metadata={
                    "user_id": user_id,
                    "tool_calls": len(tools_used)
                }
            )
        
        return response, tools_used
        
    except Exception as e:
        logger.error(f"Agent execution failed: {e}")
        error_response = f"I encountered an error while processing your request: {str(e)}"
        
        if save_conversation:
            await save_conversation_turn(
                session_id=session_id,
                user_message=message,
                assistant_message=error_response,
                metadata={"error": str(e)}
            )
        
        return error_response, []


# API Endpoints
@app.get("/health", response_model=HealthStatus)
async def health_check():
    """Health check endpoint."""
    try:
        # Test database connections
        db_status = await test_connection()
        graph_status = await test_graph_connection()
        
        # Determine overall status
        if db_status and graph_status:
            status = "healthy"
        elif db_status or graph_status:
            status = "degraded"
        else:
            status = "unhealthy"
        
        return HealthStatus(
            status=status,
            database=db_status,
            graph_database=graph_status,
            llm_connection=True,  # Assume OK if we can respond
            version="0.1.0",
            timestamp=datetime.now()
        )
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=500, detail="Health check failed")


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Non-streaming chat endpoint."""
    try:
        # Get or create session
        session_id = await get_or_create_session(request)
        
        # Execute agent
        response, tools_used = await execute_agent(
            message=request.message,
            session_id=session_id,
            user_id=request.user_id
        )
        
        return ChatResponse(
            message=response,
            session_id=session_id,
            tools_used=tools_used,
            metadata={"search_type": str(request.search_type)}
        )
        
    except Exception as e:
        logger.error(f"Chat endpoint failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    """Streaming chat endpoint using Server-Sent Events."""
    try:
        # Get or create session
        session_id = await get_or_create_session(request)
        
        async def generate_stream():
            """Generate streaming response using agent.iter() pattern."""
            try:
                yield f"data: {json.dumps({'type': 'session', 'session_id': session_id})}\n\n"
                
                # Create dependencies
                deps = AgentDependencies(
                    session_id=session_id,
                    user_id=request.user_id
                )
                
                # Get conversation context
                context = await get_conversation_context(session_id)
                
                # Build input with context
                full_prompt = request.message
                if context:
                    context_str = "\n".join([
                        f"{msg['role']}: {msg['content']}"
                        for msg in context[-6:]
                    ])
                    full_prompt = f"Previous conversation:\n{context_str}\n\nCurrent question: {request.message}"
                
                # Save user message immediately
                await add_message(
                    session_id=session_id,
                    role="user",
                    content=request.message,
                    metadata={"user_id": request.user_id}
                )
                
                full_response = ""
                
                # Stream using agent.iter() pattern
                async with rag_agent.iter(full_prompt, deps=deps) as run:
                    async for node in run:
                        if rag_agent.is_model_request_node(node):
                            # Stream tokens from the model
                            async with node.stream(run.ctx) as request_stream:
                                async for event in request_stream:
                                    from pydantic_ai.messages import PartStartEvent, PartDeltaEvent, TextPartDelta
                                    
                                    if isinstance(event, PartStartEvent) and event.part.part_kind == 'text':
                                        delta_content = event.part.content
                                        yield f"data: {json.dumps({'type': 'text', 'content': delta_content})}\n\n"
                                        full_response += delta_content
                                        
                                    elif isinstance(event, PartDeltaEvent) and isinstance(event.delta, TextPartDelta):
                                        delta_content = event.delta.content_delta
                                        yield f"data: {json.dumps({'type': 'text', 'content': delta_content})}\n\n"
                                        full_response += delta_content
                
                # Extract tools used from the final result
                result = run.result
                tools_used = extract_tool_calls(result)
                
                # Send tools used information
                if tools_used:
                    tools_data = [
                        {
                            "tool_name": tool.tool_name,
                            "args": tool.args,
                            "tool_call_id": tool.tool_call_id
                        }
                        for tool in tools_used
                    ]
                    yield f"data: {json.dumps({'type': 'tools', 'tools': tools_data})}\n\n"
                
                # Save assistant response
                await add_message(
                    session_id=session_id,
                    role="assistant",
                    content=full_response,
                    metadata={
                        "streamed": True,
                        "tool_calls": len(tools_used)
                    }
                )
                
                yield f"data: {json.dumps({'type': 'end'})}\n\n"
                
            except Exception as e:
                logger.error(f"Stream error: {e}")
                error_chunk = {
                    "type": "error",
                    "content": f"Stream error: {str(e)}"
                }
                yield f"data: {json.dumps(error_chunk)}\n\n"
        
        return StreamingResponse(
            generate_stream(),
            media_type="text/plain",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "Content-Type": "text/event-stream"
            }
        )
        
    except Exception as e:
        logger.error(f"Streaming chat failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/search/vector")
async def search_vector(request: SearchRequest):
    """Vector search endpoint."""
    try:
        input_data = VectorSearchInput(
            query=request.query,
            limit=request.limit
        )
        
        start_time = datetime.now()
        results = await vector_search_tool(input_data)
        end_time = datetime.now()
        
        query_time = (end_time - start_time).total_seconds() * 1000
        
        return SearchResponse(
            results=results,
            total_results=len(results),
            search_type="vector",
            query_time_ms=query_time
        )
        
    except Exception as e:
        logger.error(f"Vector search failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/search/graph")
async def search_graph(request: SearchRequest):
    """Knowledge graph search endpoint."""
    try:
        input_data = GraphSearchInput(
            query=request.query
        )
        
        start_time = datetime.now()
        results = await graph_search_tool(input_data)
        end_time = datetime.now()
        
        query_time = (end_time - start_time).total_seconds() * 1000
        
        return SearchResponse(
            graph_results=results,
            total_results=len(results),
            search_type="graph",
            query_time_ms=query_time
        )
        
    except Exception as e:
        logger.error(f"Graph search failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/search/hybrid")
async def search_hybrid(request: SearchRequest):
    """Hybrid search endpoint."""
    try:
        input_data = HybridSearchInput(
            query=request.query,
            limit=request.limit
        )
        
        start_time = datetime.now()
        results = await hybrid_search_tool(input_data)
        end_time = datetime.now()
        
        query_time = (end_time - start_time).total_seconds() * 1000
        
        return SearchResponse(
            results=results,
            total_results=len(results),
            search_type="hybrid",
            query_time_ms=query_time
        )
        
    except Exception as e:
        logger.error(f"Hybrid search failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/documents")
async def list_documents_endpoint(
    limit: int = 20,
    offset: int = 0
):
    """List documents endpoint."""
    try:
        input_data = DocumentListInput(limit=limit, offset=offset)
        documents = await list_documents_tool(input_data)
        
        return {
            "documents": documents,
            "total": len(documents),
            "limit": limit,
            "offset": offset
        }
        
    except Exception as e:
        logger.error(f"Document listing failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/sessions/{session_id}")
async def get_session_info(session_id: str):
    """Get session information."""
    try:
        session = await get_session(session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        
        return session
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Session retrieval failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# Content Generation Endpoints
# =============================================================================

# Lazy initialization for content services
_content_services = {}

async def get_content_services():
    """Get or initialize content services."""
    global _content_services
    
    if not _content_services:
        try:
            from content.content_types import ContentType
            from content.user_profile import UserProfileService
            from content.transit_service import TransitService
            from content.context_assembler import ContentContextAssembler
            from content.generator import PersonalizedContentGenerator
            from content.batch import BatchContentGenerator
            from .db_utils import db_pool
            from .providers import get_llm_client
            
            # Try to get ontology if available
            try:
                from knowledge.ontology_utils import AstrologyOntology
                ontology = AstrologyOntology()
            except Exception:
                ontology = None
            
            user_service = UserProfileService(db_pool)
            transit_service = TransitService()
            assembler = ContentContextAssembler(user_service, transit_service, ontology)
            
            llm_client = get_llm_client()
            generator = PersonalizedContentGenerator(assembler, llm_client, db_pool)
            batch_generator = BatchContentGenerator(generator, user_service, db_pool)
            
            _content_services = {
                "user_service": user_service,
                "transit_service": transit_service,
                "generator": generator,
                "batch_generator": batch_generator,
                "ContentType": ContentType,
            }
            logger.info("Content services initialized")
        except Exception as e:
            logger.error(f"Failed to initialize content services: {e}")
            raise HTTPException(status_code=500, detail=f"Content services not available: {e}")
    
    return _content_services


@app.post("/users/{user_id}/profile", response_model=UserProfileResponse)
async def create_user_profile(user_id: str, birth_data: BirthData):
    """Create or update user astrological profile."""
    try:
        services = await get_content_services()
        user_service = services["user_service"]
        
        profile = await user_service.create_or_update_profile(
            user_id=user_id,
            birth_datetime=birth_data.birth_datetime,
            birth_latitude=birth_data.latitude,
            birth_longitude=birth_data.longitude,
            birth_location_name=birth_data.location_name
        )
        
        return UserProfileResponse(
            user_id=profile.user_id,
            sun_sign=profile.sun_sign,
            moon_sign=profile.moon_sign,
            rising_sign=profile.rising_sign,
            birth_datetime=profile.birth_datetime,
            birth_location=profile.birth_location_name,
            natal_positions={
                k: {"sign": v.sign, "degree": v.degree_in_sign, "house": v.house}
                for k, v in profile.natal_positions.items()
            },
            chart_computed_at=profile.chart_computed_at
        )
        
    except Exception as e:
        logger.error(f"Profile creation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/users/{user_id}/profile", response_model=UserProfileResponse)
async def get_user_profile(user_id: str):
    """Get user astrological profile."""
    try:
        services = await get_content_services()
        user_service = services["user_service"]
        
        profile = await user_service.get_profile(user_id)
        
        if not profile:
            raise HTTPException(status_code=404, detail="Profile not found")
        
        return UserProfileResponse(
            user_id=profile.user_id,
            sun_sign=profile.sun_sign,
            moon_sign=profile.moon_sign,
            rising_sign=profile.rising_sign,
            birth_datetime=profile.birth_datetime,
            birth_location=profile.birth_location_name,
            natal_positions={
                k: {"sign": v.sign, "degree": v.degree_in_sign, "house": v.house}
                for k, v in profile.natal_positions.items()
            },
            chart_computed_at=profile.chart_computed_at
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Profile retrieval failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/content/generate", response_model=ContentResponse)
async def generate_content(request: ContentGenerateRequest):
    """Generate content (personalized if user_id provided)."""
    try:
        services = await get_content_services()
        generator = services["generator"]
        ContentType = services["ContentType"]
        
        # Map enum
        content_type = ContentType(request.content_type.value)
        
        result = await generator.generate_content(
            content_type=content_type,
            user_id=request.user_id,
            year=request.year,
            month=request.month,
            force_refresh=request.force_refresh
        )
        
        if not result:
            raise HTTPException(status_code=400, detail="Content generation failed")
        
        return ContentResponse(
            content_type=result.content_type.value,
            content=result.content,
            user_id=result.user_id,
            valid_from=result.valid_from,
            valid_until=result.valid_until,
            from_cache=result.from_cache,
            metadata=result.metadata
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Content generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/content/{user_id}/monthly", response_model=MonthlyContentResponse)
async def get_monthly_content(
    user_id: str,
    year: Optional[int] = None,
    month: Optional[int] = None
):
    """Get all monthly content for a user."""
    try:
        services = await get_content_services()
        generator = services["generator"]
        
        results = await generator.get_all_monthly_content(
            user_id=user_id,
            year=year,
            month=month
        )
        
        response = MonthlyContentResponse()
        
        if "general" in results:
            r = results["general"]
            response.general = ContentResponse(
                content_type=r.content_type.value,
                content=r.content,
                user_id=r.user_id,
                valid_from=r.valid_from,
                valid_until=r.valid_until,
                from_cache=r.from_cache,
                metadata=r.metadata
            )
        
        if "personal" in results:
            r = results["personal"]
            response.personal = ContentResponse(
                content_type=r.content_type.value,
                content=r.content,
                user_id=r.user_id,
                valid_from=r.valid_from,
                valid_until=r.valid_until,
                from_cache=r.from_cache,
                metadata=r.metadata
            )
        
        return response
        
    except Exception as e:
        logger.error(f"Monthly content retrieval failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/content/{user_id}/moon-reflection", response_model=ContentResponse)
async def get_moon_reflection(user_id: str, force_refresh: bool = False):
    """Get moon reflection questions for a user."""
    try:
        services = await get_content_services()
        generator = services["generator"]
        ContentType = services["ContentType"]
        
        result = await generator.generate_content(
            content_type=ContentType.MOON_REFLECTION,
            user_id=user_id,
            force_refresh=force_refresh
        )
        
        if not result:
            raise HTTPException(status_code=400, detail="Moon reflection generation failed")
        
        return ContentResponse(
            content_type=result.content_type.value,
            content=result.content,
            user_id=result.user_id,
            valid_from=result.valid_from,
            valid_until=result.valid_until,
            from_cache=result.from_cache,
            metadata=result.metadata
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Moon reflection generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/content/batch/generate", response_model=BatchJobStatusResponse)
async def batch_generate_content(request: BatchGenerateRequest):
    """Batch generate content for multiple users (called by cron)."""
    try:
        services = await get_content_services()
        batch_generator = services["batch_generator"]
        ContentType = services["ContentType"]
        
        content_type = ContentType(request.content_type.value)
        
        result = await batch_generator.generate_for_all_users(
            content_type=content_type,
            user_ids=request.user_ids,
            year=request.year,
            month=request.month
        )
        
        return BatchJobStatusResponse(**result.to_dict())
        
    except Exception as e:
        logger.error(f"Batch generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/content/batch/monthly", response_model=Dict[str, BatchJobStatusResponse])
async def batch_generate_monthly(
    year: Optional[int] = None,
    month: Optional[int] = None
):
    """Pre-generate all monthly content for all users."""
    try:
        services = await get_content_services()
        batch_generator = services["batch_generator"]
        
        results = await batch_generator.generate_monthly_content(
            year=year,
            month=month
        )
        
        return {
            key: BatchJobStatusResponse(**result.to_dict())
            for key, result in results.items()
        }
        
    except Exception as e:
        logger.error(f"Monthly batch generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/content/batch/status/{job_id}", response_model=BatchJobStatusResponse)
async def get_batch_job_status(job_id: str):
    """Check status of a batch generation job."""
    try:
        services = await get_content_services()
        batch_generator = services["batch_generator"]
        
        result = await batch_generator.get_job_status(job_id)
        
        if not result:
            raise HTTPException(status_code=404, detail="Job not found")
        
        return BatchJobStatusResponse(**result.to_dict())
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Job status retrieval failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/transits/current")
async def get_current_transits():
    """Get current planetary positions."""
    try:
        services = await get_content_services()
        transit_service = services["transit_service"]
        
        transits = transit_service.get_current_transits()
        return transits.to_dict()
        
    except Exception as e:
        logger.error(f"Transit retrieval failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/transits/monthly")
async def get_monthly_transits(
    year: Optional[int] = None,
    month: Optional[int] = None
):
    """Get major transits for a month."""
    try:
        from datetime import timezone
        now = datetime.now(timezone.utc)
        year = year or now.year
        month = month or now.month
        
        services = await get_content_services()
        transit_service = services["transit_service"]
        
        monthly = transit_service.get_monthly_transits(year, month)
        
        return {
            "year": monthly.year,
            "month": monthly.month,
            "events": [
                {
                    "date": e.date.isoformat(),
                    "event_type": e.event_type,
                    "description": e.description,
                    "planets": e.planets_involved,
                    "importance": e.importance
                }
                for e in monthly.events
            ],
            "retrogrades": monthly.retrogrades,
            "moon_phases": monthly.moon_phases
        }
        
    except Exception as e:
        logger.error(f"Monthly transit retrieval failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Exception handlers
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler."""
    logger.error(f"Unhandled exception: {exc}")
    
    return ErrorResponse(
        error=str(exc),
        error_type=type(exc).__name__,
        request_id=str(uuid.uuid4())
    )


# Development server
if __name__ == "__main__":
    uvicorn.run(
        "agent.api:app",
        host=APP_HOST,
        port=APP_PORT,
        reload=APP_ENV == "development",
        log_level=LOG_LEVEL.lower()
    )