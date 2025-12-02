"""
Supabase client wrapper for database operations.

Provides a unified interface for Supabase operations including:
- Database queries with RLS
- Auth operations
- Service role access for batch operations
"""

import os
import logging
from typing import Optional, Dict, Any, List
from contextlib import asynccontextmanager

from supabase import create_client, Client
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

# =============================================================================
# Configuration
# =============================================================================

SUPABASE_URL = os.getenv("SUPABASE_URL", "http://localhost:54321")
SUPABASE_ANON_KEY = os.getenv("SUPABASE_ANON_KEY", "")
SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_KEY", "")

# =============================================================================
# Supabase Client Singleton
# =============================================================================

class SupabaseClientManager:
    """
    Manages Supabase client instances.
    
    Provides both anon client (respects RLS) and service client (bypasses RLS).
    """
    
    _anon_client: Optional[Client] = None
    _service_client: Optional[Client] = None
    
    @classmethod
    def get_anon_client(cls) -> Client:
        """
        Get the anonymous Supabase client.
        
        This client respects RLS policies - use for user-facing operations.
        """
        if cls._anon_client is None:
            if not SUPABASE_URL or not SUPABASE_ANON_KEY:
                raise ValueError("SUPABASE_URL and SUPABASE_ANON_KEY must be set")
            
            cls._anon_client = create_client(
                SUPABASE_URL,
                SUPABASE_ANON_KEY
            )
            logger.info("Supabase anon client initialized")
        
        return cls._anon_client
    
    @classmethod
    def get_service_client(cls) -> Client:
        """
        Get the service role Supabase client.
        
        This client bypasses RLS - use for batch operations and admin tasks.
        """
        if cls._service_client is None:
            if not SUPABASE_URL or not SUPABASE_SERVICE_KEY:
                raise ValueError("SUPABASE_URL and SUPABASE_SERVICE_KEY must be set")
            
            cls._service_client = create_client(
                SUPABASE_URL,
                SUPABASE_SERVICE_KEY
            )
            logger.info("Supabase service client initialized")
        
        return cls._service_client
    
    @classmethod
    def get_client_for_user(cls, access_token: Optional[str] = None) -> Client:
        """
        Get a client authenticated as a specific user.
        
        Args:
            access_token: User's JWT access token
            
        Returns:
            Supabase client with user's auth context
        """
        client = cls.get_anon_client()
        
        if access_token:
            # Set the auth header for this request
            client.auth.set_session(access_token, "")
        
        return client


# Convenience functions
def get_supabase() -> Client:
    """Get the default (anon) Supabase client."""
    return SupabaseClientManager.get_anon_client()


def get_supabase_admin() -> Client:
    """Get the service role Supabase client."""
    return SupabaseClientManager.get_service_client()


# =============================================================================
# Database Operations Helper
# =============================================================================

class SupabaseDB:
    """
    Database operations wrapper for Supabase.
    
    Provides async-like interface compatible with existing code structure.
    """
    
    def __init__(self, use_service_role: bool = False):
        """
        Initialize database wrapper.
        
        Args:
            use_service_role: If True, use service role client (bypasses RLS)
        """
        self.use_service_role = use_service_role
        self._client: Optional[Client] = None
    
    @property
    def client(self) -> Client:
        """Get the Supabase client."""
        if self._client is None:
            if self.use_service_role:
                self._client = get_supabase_admin()
            else:
                self._client = get_supabase()
        return self._client
    
    def table(self, name: str):
        """Access a table."""
        return self.client.table(name)
    
    # =========================================================================
    # User Profiles
    # =========================================================================
    
    async def get_user_profile(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Get a user profile by user_id."""
        response = self.client.table("user_profiles") \
            .select("*") \
            .eq("user_id", user_id) \
            .maybe_single() \
            .execute()
        return response.data
    
    async def upsert_user_profile(self, profile_data: Dict[str, Any]) -> Dict[str, Any]:
        """Insert or update a user profile."""
        response = self.client.table("user_profiles") \
            .upsert(profile_data, on_conflict="user_id") \
            .execute()
        return response.data[0] if response.data else {}
    
    async def get_all_user_ids(self) -> List[str]:
        """Get all user IDs with profiles."""
        response = self.client.table("user_profiles") \
            .select("user_id") \
            .execute()
        return [row["user_id"] for row in response.data] if response.data else []
    
    # =========================================================================
    # Generated Content
    # =========================================================================
    
    async def get_cached_content(
        self, 
        content_type: str, 
        user_id: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """Get valid cached content."""
        query = self.client.table("generated_content") \
            .select("*") \
            .eq("content_type", content_type) \
            .lte("valid_from", "now()") \
            .gte("valid_until", "now()") \
            .order("created_at", desc=True) \
            .limit(1)
        
        if user_id:
            query = query.eq("user_id", user_id)
        else:
            query = query.is_("user_id", "null")
        
        response = query.maybe_single().execute()
        return response.data
    
    async def save_generated_content(self, content_data: Dict[str, Any]) -> Dict[str, Any]:
        """Save generated content."""
        response = self.client.table("generated_content") \
            .insert(content_data) \
            .execute()
        return response.data[0] if response.data else {}
    
    # =========================================================================
    # Sessions and Messages
    # =========================================================================
    
    async def create_session(self, user_id: Optional[str] = None, metadata: Dict = None) -> Dict[str, Any]:
        """Create a new session."""
        data = {"user_id": user_id}
        if metadata:
            data["metadata"] = metadata
        
        response = self.client.table("sessions") \
            .insert(data) \
            .execute()
        return response.data[0] if response.data else {}
    
    async def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get a session by ID."""
        response = self.client.table("sessions") \
            .select("*") \
            .eq("id", session_id) \
            .maybe_single() \
            .execute()
        return response.data
    
    async def add_message(
        self, 
        session_id: str, 
        role: str, 
        content: str, 
        metadata: Dict = None
    ) -> Dict[str, Any]:
        """Add a message to a session."""
        data = {
            "session_id": session_id,
            "role": role,
            "content": content
        }
        if metadata:
            data["metadata"] = metadata
        
        response = self.client.table("messages") \
            .insert(data) \
            .execute()
        return response.data[0] if response.data else {}
    
    async def get_session_messages(
        self, 
        session_id: str, 
        limit: int = 50
    ) -> List[Dict[str, Any]]:
        """Get messages for a session."""
        response = self.client.table("messages") \
            .select("*") \
            .eq("session_id", session_id) \
            .order("created_at") \
            .limit(limit) \
            .execute()
        return response.data if response.data else []
    
    # =========================================================================
    # Documents and Chunks
    # =========================================================================
    
    async def list_documents(self, limit: int = 20, offset: int = 0) -> List[Dict[str, Any]]:
        """List documents."""
        response = self.client.table("documents") \
            .select("id, title, source, created_at, updated_at, metadata") \
            .order("created_at", desc=True) \
            .range(offset, offset + limit - 1) \
            .execute()
        return response.data if response.data else []
    
    async def get_document(self, document_id: str) -> Optional[Dict[str, Any]]:
        """Get a document by ID."""
        response = self.client.table("documents") \
            .select("*") \
            .eq("id", document_id) \
            .maybe_single() \
            .execute()
        return response.data
    
    # =========================================================================
    # Batch Jobs
    # =========================================================================
    
    async def create_batch_job(self, job_type: str, total_users: int) -> Dict[str, Any]:
        """Create a new batch job."""
        response = self.client.table("batch_jobs") \
            .insert({
                "job_type": job_type,
                "status": "running",
                "total_users": total_users,
                "started_at": "now()"
            }) \
            .execute()
        return response.data[0] if response.data else {}
    
    async def update_batch_job(self, job_id: str, updates: Dict[str, Any]) -> None:
        """Update a batch job."""
        self.client.table("batch_jobs") \
            .update(updates) \
            .eq("id", job_id) \
            .execute()
    
    async def get_batch_job(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Get a batch job by ID."""
        response = self.client.table("batch_jobs") \
            .select("*") \
            .eq("id", job_id) \
            .maybe_single() \
            .execute()
        return response.data


# =============================================================================
# Auth Operations
# =============================================================================

class SupabaseAuth:
    """Authentication operations wrapper."""
    
    def __init__(self):
        self.client = get_supabase()
    
    async def sign_up(self, email: str, password: str) -> Dict[str, Any]:
        """Sign up a new user."""
        response = self.client.auth.sign_up({
            "email": email,
            "password": password
        })
        return {
            "user": response.user.model_dump() if response.user else None,
            "session": response.session.model_dump() if response.session else None
        }
    
    async def sign_in(self, email: str, password: str) -> Dict[str, Any]:
        """Sign in a user."""
        response = self.client.auth.sign_in_with_password({
            "email": email,
            "password": password
        })
        return {
            "user": response.user.model_dump() if response.user else None,
            "session": response.session.model_dump() if response.session else None
        }
    
    async def sign_out(self) -> None:
        """Sign out the current user."""
        self.client.auth.sign_out()
    
    async def refresh_session(self, refresh_token: str) -> Dict[str, Any]:
        """Refresh a session using refresh token."""
        response = self.client.auth.refresh_session(refresh_token)
        return {
            "user": response.user.model_dump() if response.user else None,
            "session": response.session.model_dump() if response.session else None
        }
    
    async def get_user(self, access_token: str) -> Optional[Dict[str, Any]]:
        """Get user info from access token."""
        response = self.client.auth.get_user(access_token)
        return response.user.model_dump() if response.user else None


# =============================================================================
# Global instances for convenience
# =============================================================================

# Default DB instance (uses anon client, respects RLS)
supabase_db = SupabaseDB(use_service_role=False)

# Admin DB instance (uses service role, bypasses RLS)
supabase_admin_db = SupabaseDB(use_service_role=True)

# Auth instance
supabase_auth = SupabaseAuth()

