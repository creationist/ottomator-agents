"""
Database utilities for Supabase operations.

Provides both Supabase client access and backward-compatible asyncpg pool
for operations that require raw SQL (like vector search with custom functions).
"""

import os
import json
import asyncio
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta, timezone
from contextlib import asynccontextmanager
from uuid import UUID
import logging

from dotenv import load_dotenv

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)

# =============================================================================
# Determine which backend to use
# =============================================================================

USE_SUPABASE = os.getenv("USE_SUPABASE", "true").lower() == "true"
DATABASE_URL = os.getenv("DATABASE_URL", "")
SUPABASE_URL = os.getenv("SUPABASE_URL", "")

# =============================================================================
# Supabase Client (Primary)
# =============================================================================

if USE_SUPABASE:
    from .supabase_client import (
        SupabaseDB, 
        supabase_db, 
        supabase_admin_db,
        get_supabase,
        get_supabase_admin
    )

# =============================================================================
# AsyncPG Pool (For raw SQL operations like vector search)
# =============================================================================

try:
    import asyncpg
    from asyncpg.pool import Pool
    ASYNCPG_AVAILABLE = True
except ImportError:
    ASYNCPG_AVAILABLE = False
    logger.warning("asyncpg not available - some operations may not work")


class DatabasePool:
    """Manages PostgreSQL connection pool for raw SQL operations."""
    
    def __init__(self, database_url: Optional[str] = None):
        self.database_url = database_url or DATABASE_URL
        self.pool: Optional["Pool"] = None
    
    async def initialize(self):
        """Create connection pool."""
        if not ASYNCPG_AVAILABLE:
            logger.warning("asyncpg not available, skipping pool initialization")
            return
        
        if not self.pool and self.database_url:
            try:
                self.pool = await asyncpg.create_pool(
                    self.database_url,
                    min_size=2,
                    max_size=10,
                    max_inactive_connection_lifetime=300,
                    command_timeout=60
                )
                logger.info("Database connection pool initialized")
            except Exception as e:
                logger.warning(f"Could not create asyncpg pool: {e}")
    
    async def close(self):
        """Close connection pool."""
        if self.pool:
            await self.pool.close()
            self.pool = None
            logger.info("Database connection pool closed")
    
    @asynccontextmanager
    async def acquire(self):
        """Acquire a connection from the pool."""
        if not self.pool:
            await self.initialize()
        
        if not self.pool:
            raise RuntimeError("Database pool not available")
        
        async with self.pool.acquire() as connection:
            yield connection


# Global database pool instance (for raw SQL)
db_pool = DatabasePool()


async def initialize_database():
    """Initialize database connections."""
    if USE_SUPABASE:
        # Supabase client initializes lazily
        try:
            client = get_supabase()
            logger.info("Supabase client ready")
        except Exception as e:
            logger.error(f"Supabase client initialization failed: {e}")
    
    # Also initialize asyncpg pool for vector operations
    if DATABASE_URL and ASYNCPG_AVAILABLE:
        await db_pool.initialize()


async def close_database():
    """Close database connections."""
    await db_pool.close()


# =============================================================================
# Session Management Functions
# =============================================================================

async def create_session(
    user_id: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
    timeout_minutes: int = 60
) -> str:
    """Create a new session."""
    if USE_SUPABASE:
        expires_at = datetime.now(timezone.utc) + timedelta(minutes=timeout_minutes)
        result = supabase_admin_db.client.table("sessions").insert({
            "user_id": user_id,
            "metadata": metadata or {},
            "expires_at": expires_at.isoformat()
        }).execute()
        return str(result.data[0]["id"]) if result.data else ""
    
    # Fallback to asyncpg
    async with db_pool.acquire() as conn:
        expires_at = datetime.now(timezone.utc) + timedelta(minutes=timeout_minutes)
        result = await conn.fetchrow(
            """
            INSERT INTO sessions (user_id, metadata, expires_at)
            VALUES ($1, $2, $3)
            RETURNING id::text
            """,
            user_id,
            json.dumps(metadata or {}),
            expires_at
        )
        return result["id"]


async def get_session(session_id: str) -> Optional[Dict[str, Any]]:
    """Get session by ID."""
    if USE_SUPABASE:
        result = supabase_admin_db.client.table("sessions") \
            .select("*") \
            .eq("id", session_id) \
            .maybe_single() \
            .execute()
        
        if result.data:
            row = result.data
            return {
                "id": str(row["id"]),
                "user_id": row.get("user_id"),
                "metadata": row.get("metadata", {}),
                "created_at": row.get("created_at"),
                "updated_at": row.get("updated_at"),
                "expires_at": row.get("expires_at")
            }
        return None
    
    # Fallback to asyncpg
    async with db_pool.acquire() as conn:
        result = await conn.fetchrow(
            """
            SELECT id::text, user_id, metadata, created_at, updated_at, expires_at
            FROM sessions
            WHERE id = $1::uuid
            AND (expires_at IS NULL OR expires_at > CURRENT_TIMESTAMP)
            """,
            session_id
        )
        
        if result:
            return {
                "id": result["id"],
                "user_id": result["user_id"],
                "metadata": json.loads(result["metadata"]),
                "created_at": result["created_at"].isoformat(),
                "updated_at": result["updated_at"].isoformat(),
                "expires_at": result["expires_at"].isoformat() if result["expires_at"] else None
            }
        return None


async def update_session(session_id: str, metadata: Dict[str, Any]) -> bool:
    """Update session metadata."""
    if USE_SUPABASE:
        # Get current metadata and merge
        current = await get_session(session_id)
        if not current:
            return False
        
        merged = {**current.get("metadata", {}), **metadata}
        result = supabase_admin_db.client.table("sessions") \
            .update({"metadata": merged}) \
            .eq("id", session_id) \
            .execute()
        return bool(result.data)
    
    # Fallback to asyncpg
    async with db_pool.acquire() as conn:
        result = await conn.execute(
            """
            UPDATE sessions
            SET metadata = metadata || $2::jsonb
            WHERE id = $1::uuid
            AND (expires_at IS NULL OR expires_at > CURRENT_TIMESTAMP)
            """,
            session_id,
            json.dumps(metadata)
        )
        return result.split()[-1] != "0"


# =============================================================================
# Message Management Functions
# =============================================================================

async def add_message(
    session_id: str,
    role: str,
    content: str,
    metadata: Optional[Dict[str, Any]] = None
) -> str:
    """Add a message to a session."""
    if USE_SUPABASE:
        result = supabase_admin_db.client.table("messages").insert({
            "session_id": session_id,
            "role": role,
            "content": content,
            "metadata": metadata or {}
        }).execute()
        return str(result.data[0]["id"]) if result.data else ""
    
    # Fallback to asyncpg
    async with db_pool.acquire() as conn:
        result = await conn.fetchrow(
            """
            INSERT INTO messages (session_id, role, content, metadata)
            VALUES ($1::uuid, $2, $3, $4)
            RETURNING id::text
            """,
            session_id,
            role,
            content,
            json.dumps(metadata or {})
        )
        return result["id"]


async def get_session_messages(
    session_id: str,
    limit: Optional[int] = None
) -> List[Dict[str, Any]]:
    """Get messages for a session."""
    if USE_SUPABASE:
        query = supabase_admin_db.client.table("messages") \
            .select("*") \
            .eq("session_id", session_id) \
            .order("created_at")
        
        if limit:
            query = query.limit(limit)
        
        result = query.execute()
        
        return [
            {
                "id": str(row["id"]),
                "role": row["role"],
                "content": row["content"],
                "metadata": row.get("metadata", {}),
                "created_at": row.get("created_at")
            }
            for row in (result.data or [])
        ]
    
    # Fallback to asyncpg
    async with db_pool.acquire() as conn:
        query = """
            SELECT id::text, role, content, metadata, created_at
            FROM messages
            WHERE session_id = $1::uuid
            ORDER BY created_at
        """
        if limit:
            query += f" LIMIT {limit}"
        
        results = await conn.fetch(query, session_id)
        
        return [
            {
                "id": row["id"],
                "role": row["role"],
                "content": row["content"],
                "metadata": json.loads(row["metadata"]),
                "created_at": row["created_at"].isoformat()
            }
            for row in results
        ]


# =============================================================================
# Document Management Functions
# =============================================================================

async def get_document(document_id: str) -> Optional[Dict[str, Any]]:
    """Get document by ID."""
    if USE_SUPABASE:
        result = supabase_db.client.table("documents") \
            .select("*") \
            .eq("id", document_id) \
            .maybe_single() \
            .execute()
        
        if result.data:
            row = result.data
            return {
                "id": str(row["id"]),
                "title": row["title"],
                "source": row["source"],
                "content": row.get("content", ""),
                "metadata": row.get("metadata", {}),
                "created_at": row.get("created_at"),
                "updated_at": row.get("updated_at")
            }
        return None
    
    # Fallback to asyncpg
    async with db_pool.acquire() as conn:
        result = await conn.fetchrow(
            """
            SELECT id::text, title, source, content, metadata, created_at, updated_at
            FROM documents
            WHERE id = $1::uuid
            """,
            document_id
        )
        
        if result:
            return {
                "id": result["id"],
                "title": result["title"],
                "source": result["source"],
                "content": result["content"],
                "metadata": json.loads(result["metadata"]),
                "created_at": result["created_at"].isoformat(),
                "updated_at": result["updated_at"].isoformat()
            }
        return None


async def list_documents(
    limit: int = 100,
    offset: int = 0,
    metadata_filter: Optional[Dict[str, Any]] = None
) -> List[Dict[str, Any]]:
    """List documents with optional filtering."""
    if USE_SUPABASE:
        query = supabase_db.client.table("documents") \
            .select("id, title, source, metadata, created_at, updated_at") \
            .order("created_at", desc=True) \
            .range(offset, offset + limit - 1)
        
        result = query.execute()
        
        return [
            {
                "id": str(row["id"]),
                "title": row["title"],
                "source": row["source"],
                "metadata": row.get("metadata", {}),
                "created_at": row.get("created_at"),
                "updated_at": row.get("updated_at"),
                "chunk_count": 0  # Would need separate query
            }
            for row in (result.data or [])
        ]
    
    # Fallback to asyncpg
    async with db_pool.acquire() as conn:
        query = """
            SELECT 
                d.id::text, d.title, d.source, d.metadata,
                d.created_at, d.updated_at,
                COUNT(c.id) AS chunk_count
            FROM documents d
            LEFT JOIN chunks c ON d.id = c.document_id
        """
        
        params = []
        if metadata_filter:
            query += " WHERE d.metadata @> $1::jsonb"
            params.append(json.dumps(metadata_filter))
        
        query += """
            GROUP BY d.id, d.title, d.source, d.metadata, d.created_at, d.updated_at
            ORDER BY d.created_at DESC
            LIMIT $%d OFFSET $%d
        """ % (len(params) + 1, len(params) + 2)
        
        params.extend([limit, offset])
        results = await conn.fetch(query, *params)
        
        return [
            {
                "id": row["id"],
                "title": row["title"],
                "source": row["source"],
                "metadata": json.loads(row["metadata"]),
                "created_at": row["created_at"].isoformat(),
                "updated_at": row["updated_at"].isoformat(),
                "chunk_count": row["chunk_count"]
            }
            for row in results
        ]


# =============================================================================
# Vector Search Functions (requires asyncpg for custom functions)
# =============================================================================

async def vector_search(
    embedding: List[float],
    limit: int = 10
) -> List[Dict[str, Any]]:
    """Perform vector similarity search."""
    if not ASYNCPG_AVAILABLE or not db_pool.pool:
        logger.error("asyncpg pool required for vector search")
        return []
    
    async with db_pool.acquire() as conn:
        embedding_str = '[' + ','.join(map(str, embedding)) + ']'
        
        results = await conn.fetch(
            "SELECT * FROM match_chunks($1::vector, $2)",
            embedding_str,
            limit
        )
        
        return [
            {
                "chunk_id": row["chunk_id"],
                "document_id": row["document_id"],
                "content": row["content"],
                "similarity": row["similarity"],
                "metadata": json.loads(row["metadata"]) if isinstance(row["metadata"], str) else row["metadata"],
                "document_title": row["document_title"],
                "document_source": row["document_source"]
            }
            for row in results
        ]


async def hybrid_search(
    embedding: List[float],
    query_text: str,
    limit: int = 10,
    text_weight: float = 0.3
) -> List[Dict[str, Any]]:
    """Perform hybrid search (vector + keyword)."""
    if not ASYNCPG_AVAILABLE or not db_pool.pool:
        logger.error("asyncpg pool required for hybrid search")
        return []
    
    async with db_pool.acquire() as conn:
        embedding_str = '[' + ','.join(map(str, embedding)) + ']'
        
        results = await conn.fetch(
            "SELECT * FROM hybrid_search($1::vector, $2, $3, $4)",
            embedding_str,
            query_text,
            limit,
            text_weight
        )
        
        return [
            {
                "chunk_id": row["chunk_id"],
                "document_id": row["document_id"],
                "content": row["content"],
                "combined_score": row["combined_score"],
                "vector_similarity": row["vector_similarity"],
                "text_similarity": row["text_similarity"],
                "metadata": json.loads(row["metadata"]) if isinstance(row["metadata"], str) else row["metadata"],
                "document_title": row["document_title"],
                "document_source": row["document_source"]
            }
            for row in results
        ]


# =============================================================================
# Chunk Management Functions
# =============================================================================

async def get_document_chunks(document_id: str) -> List[Dict[str, Any]]:
    """Get all chunks for a document."""
    if USE_SUPABASE:
        result = supabase_db.client.table("chunks") \
            .select("id, content, chunk_index, metadata") \
            .eq("document_id", document_id) \
            .order("chunk_index") \
            .execute()
        
        return [
            {
                "chunk_id": str(row["id"]),
                "content": row["content"],
                "chunk_index": row["chunk_index"],
                "metadata": row.get("metadata", {})
            }
            for row in (result.data or [])
        ]
    
    # Fallback to asyncpg
    async with db_pool.acquire() as conn:
        results = await conn.fetch(
            "SELECT * FROM get_document_chunks($1::uuid)",
            document_id
        )
        
        return [
            {
                "chunk_id": row["chunk_id"],
                "content": row["content"],
                "chunk_index": row["chunk_index"],
                "metadata": json.loads(row["metadata"]) if isinstance(row["metadata"], str) else row["metadata"]
            }
            for row in results
        ]


# =============================================================================
# Utility Functions
# =============================================================================

async def execute_query(query: str, *params) -> List[Dict[str, Any]]:
    """Execute a custom query (requires asyncpg)."""
    if not ASYNCPG_AVAILABLE or not db_pool.pool:
        logger.error("asyncpg pool required for raw SQL queries")
        return []
    
    async with db_pool.acquire() as conn:
        results = await conn.fetch(query, *params)
        return [dict(row) for row in results]


async def test_connection() -> bool:
    """Test database connection."""
    try:
        if USE_SUPABASE:
            client = get_supabase()
            # Simple query to test connection
            result = client.table("documents").select("id").limit(1).execute()
            return True
        
        if ASYNCPG_AVAILABLE and db_pool.pool:
            async with db_pool.acquire() as conn:
                await conn.fetchval("SELECT 1")
            return True
        
        return False
    except Exception as e:
        logger.error(f"Database connection test failed: {e}")
        return False


# =============================================================================
# Conversation Management (for chat history)
# =============================================================================

async def save_conversation_turn(
    session_id: str,
    user_message: str,
    assistant_message: str,
    metadata: Optional[Dict[str, Any]] = None
) -> None:
    """Save a conversation turn (user + assistant messages)."""
    await add_message(session_id, "user", user_message, metadata)
    await add_message(session_id, "assistant", assistant_message, metadata)


async def get_conversation_context(
    session_id: str,
    max_turns: int = 10
) -> List[Dict[str, str]]:
    """Get recent conversation context for the agent."""
    messages = await get_session_messages(session_id, limit=max_turns * 2)
    return [
        {"role": msg["role"], "content": msg["content"]}
        for msg in messages
    ]
