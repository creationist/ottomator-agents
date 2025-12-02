"""
Authentication module for Supabase JWT verification.

Provides FastAPI dependencies for protecting routes and extracting user info.
Supports development mode bypass for CLI testing.
"""

import os
import logging
from typing import Optional
from datetime import datetime, timezone

from fastapi import Depends, HTTPException, status, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import jwt
from pydantic import BaseModel

from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

# =============================================================================
# Configuration
# =============================================================================

APP_ENV = os.getenv("APP_ENV", "production")
SUPABASE_URL = os.getenv("SUPABASE_URL", "http://localhost:54321")
SUPABASE_JWT_SECRET = os.getenv("SUPABASE_JWT_SECRET", "")
SUPABASE_ANON_KEY = os.getenv("SUPABASE_ANON_KEY", "")

# In development, use a known test secret if not provided
if APP_ENV == "development" and not SUPABASE_JWT_SECRET:
    # Default local Supabase JWT secret
    SUPABASE_JWT_SECRET = "super-secret-jwt-token-with-at-least-32-characters-long"

# Security scheme
security = HTTPBearer(auto_error=False)


# =============================================================================
# Models
# =============================================================================

class AuthUser(BaseModel):
    """Authenticated user information."""
    id: str
    email: Optional[str] = None
    role: str = "authenticated"
    is_anonymous: bool = False
    
    @property
    def is_service_role(self) -> bool:
        return self.role == "service_role"


class DevUser(AuthUser):
    """Development mode user."""
    id: str = "dev-user"
    email: str = "dev@localhost"
    role: str = "authenticated"
    is_anonymous: bool = False


# =============================================================================
# JWT Verification
# =============================================================================

def decode_jwt(token: str) -> dict:
    """
    Decode and verify a Supabase JWT token.
    
    Args:
        token: JWT access token
        
    Returns:
        Decoded payload
        
    Raises:
        HTTPException: If token is invalid
    """
    try:
        # Supabase uses HS256 by default
        payload = jwt.decode(
            token,
            SUPABASE_JWT_SECRET,
            algorithms=["HS256"],
            audience="authenticated"
        )
        return payload
    except jwt.ExpiredSignatureError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token has expired",
            headers={"WWW-Authenticate": "Bearer"}
        )
    except jwt.InvalidTokenError as e:
        logger.warning(f"Invalid token: {e}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication token",
            headers={"WWW-Authenticate": "Bearer"}
        )


def extract_user_from_payload(payload: dict) -> AuthUser:
    """Extract user information from JWT payload."""
    return AuthUser(
        id=payload.get("sub", ""),
        email=payload.get("email"),
        role=payload.get("role", "authenticated"),
        is_anonymous=payload.get("is_anonymous", False)
    )


# =============================================================================
# FastAPI Dependencies
# =============================================================================

async def get_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)
) -> AuthUser:
    """
    Get the current authenticated user.
    
    In development mode (APP_ENV=development), returns a dev user if no token provided.
    In production, requires valid JWT token.
    
    Usage:
        @app.get("/protected")
        async def protected_route(user: AuthUser = Depends(get_current_user)):
            return {"user_id": user.id}
    """
    # Development mode bypass
    if APP_ENV == "development":
        if not credentials:
            logger.debug("Dev mode: using dev-user (no token provided)")
            return DevUser()
        # If token provided in dev mode, still validate it
        try:
            payload = decode_jwt(credentials.credentials)
            return extract_user_from_payload(payload)
        except HTTPException:
            # In dev mode, fall back to dev user on invalid token
            logger.debug("Dev mode: invalid token, falling back to dev-user")
            return DevUser()
    
    # Production mode - require valid token
    if not credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required",
            headers={"WWW-Authenticate": "Bearer"}
        )
    
    payload = decode_jwt(credentials.credentials)
    return extract_user_from_payload(payload)


async def get_optional_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)
) -> Optional[AuthUser]:
    """
    Get the current user if authenticated, None otherwise.
    
    Does not raise an error if no token is provided.
    Useful for routes that work differently for authenticated vs anonymous users.
    """
    if not credentials:
        if APP_ENV == "development":
            return DevUser()
        return None
    
    try:
        payload = decode_jwt(credentials.credentials)
        return extract_user_from_payload(payload)
    except HTTPException:
        if APP_ENV == "development":
            return DevUser()
        return None


async def require_service_role(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)
) -> AuthUser:
    """
    Require service role access.
    
    Use for admin/batch operations that should only be called by backend services.
    """
    if APP_ENV == "development":
        if not credentials:
            # In dev mode, allow service role operations without token
            return AuthUser(id="service", role="service_role")
    
    if not credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Service role authentication required",
            headers={"WWW-Authenticate": "Bearer"}
        )
    
    payload = decode_jwt(credentials.credentials)
    user = extract_user_from_payload(payload)
    
    if user.role != "service_role":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Service role required"
        )
    
    return user


def verify_user_access(user: AuthUser, resource_user_id: str) -> None:
    """
    Verify that a user can access a resource belonging to another user.
    
    Args:
        user: The authenticated user
        resource_user_id: The user_id of the resource being accessed
        
    Raises:
        HTTPException: If access is denied
    """
    # Service role can access anything
    if user.is_service_role:
        return
    
    # Users can only access their own resources
    if user.id != resource_user_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied"
        )


# =============================================================================
# Utility Functions
# =============================================================================

def get_user_id_for_rate_limit(request: Request) -> str:
    """
    Extract user ID for rate limiting.
    
    Used as key function for slowapi rate limiter.
    Falls back to IP address for unauthenticated requests.
    """
    # Try to get user from request state (set by middleware)
    user = getattr(request.state, "user", None)
    if user and hasattr(user, "id"):
        return f"user:{user.id}"
    
    # Fall back to IP address
    forwarded = request.headers.get("X-Forwarded-For")
    if forwarded:
        return f"ip:{forwarded.split(',')[0].strip()}"
    
    client = request.client
    if client:
        return f"ip:{client.host}"
    
    return "ip:unknown"


def is_development() -> bool:
    """Check if running in development mode."""
    return APP_ENV == "development"


