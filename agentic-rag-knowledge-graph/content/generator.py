"""
Content generator with caching.

Generates personalized astrology content using LLM and caches results.
"""

import os
import logging
import json
from typing import Dict, Any, Optional, List
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass

from .content_types import ContentType, CONTENT_TEMPLATES, get_template
from .context_assembler import ContentContextAssembler
from .prompts import (
    MONTHLY_GENERAL_SYSTEM_PROMPT,
    MONTHLY_PERSONAL_SYSTEM_PROMPT,
    MOON_REFLECTION_SYSTEM_PROMPT,
)

logger = logging.getLogger(__name__)

USE_SUPABASE = os.getenv("USE_SUPABASE", "true").lower() == "true"


@dataclass
class GeneratedContent:
    """Result of content generation."""
    content_type: ContentType
    content: str
    user_id: Optional[str]
    valid_from: datetime
    valid_until: datetime
    metadata: Dict[str, Any]
    from_cache: bool = False


class PersonalizedContentGenerator:
    """
    Generates personalized astrology content with caching.
    
    Workflow:
    1. Check cache for valid content
    2. If not cached, assemble context
    3. Generate with LLM
    4. Cache result
    """
    
    def __init__(
        self,
        assembler: ContentContextAssembler,
        llm_client,
        db_client=None,
        model: str = "gpt-4o-mini"
    ):
        """
        Initialize generator.
        
        Args:
            assembler: Context assembler instance
            llm_client: OpenAI-compatible async client
            db_client: Database client (asyncpg pool or Supabase)
            model: LLM model to use
        """
        self.assembler = assembler
        self.llm = llm_client
        self.db = db_client
        self.model = model
        self._use_supabase = USE_SUPABASE
    
    def _get_supabase_client(self):
        """Get Supabase client lazily."""
        if self._use_supabase:
            from agent.supabase_client import supabase_admin_db
            return supabase_admin_db.client
        return None
    
    def _get_system_prompt(self, content_type: ContentType) -> str:
        """Get system prompt for content type."""
        prompts = {
            ContentType.MONTHLY_GENERAL: MONTHLY_GENERAL_SYSTEM_PROMPT,
            ContentType.MONTHLY_PERSONAL: MONTHLY_PERSONAL_SYSTEM_PROMPT,
            ContentType.MOON_REFLECTION: MOON_REFLECTION_SYSTEM_PROMPT,
        }
        return prompts.get(content_type, "Du bist ein hilfreicher Astrologie-Assistent.")
    
    def _calculate_validity(
        self,
        content_type: ContentType,
        year: Optional[int] = None,
        month: Optional[int] = None
    ) -> tuple[datetime, datetime]:
        """Calculate validity period for content."""
        now = datetime.now(timezone.utc)
        template = get_template(content_type)
        duration_hours = template.cache_duration_hours
        
        if content_type == ContentType.MONTHLY_GENERAL:
            # Valid for entire month
            if year and month:
                valid_from = datetime(year, month, 1, tzinfo=timezone.utc)
                if month == 12:
                    valid_until = datetime(year + 1, 1, 1, tzinfo=timezone.utc)
                else:
                    valid_until = datetime(year, month + 1, 1, tzinfo=timezone.utc)
            else:
                valid_from = datetime(now.year, now.month, 1, tzinfo=timezone.utc)
                valid_until = valid_from + timedelta(days=32)
                valid_until = datetime(valid_until.year, valid_until.month, 1, tzinfo=timezone.utc)
        
        elif content_type == ContentType.MONTHLY_PERSONAL:
            # Valid for entire month
            if year and month:
                valid_from = datetime(year, month, 1, tzinfo=timezone.utc)
                if month == 12:
                    valid_until = datetime(year + 1, 1, 1, tzinfo=timezone.utc)
                else:
                    valid_until = datetime(year, month + 1, 1, tzinfo=timezone.utc)
            else:
                valid_from = datetime(now.year, now.month, 1, tzinfo=timezone.utc)
                valid_until = valid_from + timedelta(days=32)
                valid_until = datetime(valid_until.year, valid_until.month, 1, tzinfo=timezone.utc)
        
        elif content_type == ContentType.MOON_REFLECTION:
            # Valid for ~2.5 days (Moon changes sign)
            valid_from = now
            valid_until = now + timedelta(hours=duration_hours)
        
        else:
            # Default: use template duration
            valid_from = now
            valid_until = now + timedelta(hours=duration_hours)
        
        return valid_from, valid_until
    
    async def _get_cached_content(
        self,
        content_type: ContentType,
        user_id: Optional[str] = None
    ) -> Optional[GeneratedContent]:
        """Check cache for valid content."""
        if self._use_supabase:
            client = self._get_supabase_client()
            query = client.table("generated_content") \
                .select("*") \
                .eq("content_type", content_type.value) \
                .lte("valid_from", datetime.now(timezone.utc).isoformat()) \
                .gte("valid_until", datetime.now(timezone.utc).isoformat()) \
                .order("created_at", desc=True) \
                .limit(1)
            
            if user_id:
                query = query.eq("user_id", user_id)
            else:
                query = query.is_("user_id", "null")
            
            result = query.maybe_single().execute()
            
            if result.data:
                row = result.data
                metadata = row.get("metadata", {})
                if isinstance(metadata, str):
                    metadata = json.loads(metadata)
                
                valid_from = row["valid_from"]
                valid_until = row["valid_until"]
                if isinstance(valid_from, str):
                    valid_from = datetime.fromisoformat(valid_from.replace('Z', '+00:00'))
                if isinstance(valid_until, str):
                    valid_until = datetime.fromisoformat(valid_until.replace('Z', '+00:00'))
                
                return GeneratedContent(
                    content_type=content_type,
                    content=row["content"],
                    user_id=row.get("user_id"),
                    valid_from=valid_from,
                    valid_until=valid_until,
                    metadata=metadata,
                    from_cache=True
                )
            return None
        
        # Fallback to asyncpg
        async with self.db.acquire() as conn:
            if user_id:
                row = await conn.fetchrow(
                    """
                    SELECT * FROM generated_content
                    WHERE content_type = $1 
                      AND user_id = $2
                      AND CURRENT_TIMESTAMP BETWEEN valid_from AND valid_until
                    ORDER BY created_at DESC
                    LIMIT 1
                    """,
                    content_type.value,
                    user_id
                )
            else:
                row = await conn.fetchrow(
                    """
                    SELECT * FROM generated_content
                    WHERE content_type = $1 
                      AND user_id IS NULL
                      AND CURRENT_TIMESTAMP BETWEEN valid_from AND valid_until
                    ORDER BY created_at DESC
                    LIMIT 1
                    """,
                    content_type.value
                )
            
            if row:
                metadata = row.get("metadata", {})
                if isinstance(metadata, str):
                    metadata = json.loads(metadata)
                
                return GeneratedContent(
                    content_type=content_type,
                    content=row["content"],
                    user_id=row.get("user_id"),
                    valid_from=row["valid_from"],
                    valid_until=row["valid_until"],
                    metadata=metadata,
                    from_cache=True
                )
        
        return None
    
    async def _save_to_cache(
        self,
        content_type: ContentType,
        content: str,
        user_id: Optional[str],
        valid_from: datetime,
        valid_until: datetime,
        metadata: Dict[str, Any]
    ) -> None:
        """Save generated content to cache."""
        if self._use_supabase:
            client = self._get_supabase_client()
            data = {
                "content_type": content_type.value,
                "user_id": user_id,
                "content": content,
                "valid_from": valid_from.isoformat(),
                "valid_until": valid_until.isoformat(),
                "metadata": metadata
            }
            client.table("generated_content").insert(data).execute()
            return
        
        # Fallback to asyncpg
        async with self.db.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO generated_content (
                    content_type, user_id, content, 
                    valid_from, valid_until, metadata
                ) VALUES ($1, $2, $3, $4, $5, $6)
                """,
                content_type.value,
                user_id,
                content,
                valid_from,
                valid_until,
                json.dumps(metadata)
            )
    
    async def _generate_with_llm(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int = 2000
    ) -> str:
        """Generate content using LLM."""
        try:
            response = await self.llm.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=max_tokens,
                temperature=0.7
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            raise
    
    async def generate_content(
        self,
        content_type: ContentType,
        user_id: Optional[str] = None,
        year: Optional[int] = None,
        month: Optional[int] = None,
        force_refresh: bool = False
    ) -> Optional[GeneratedContent]:
        """
        Generate personalized content.
        
        Args:
            content_type: Type of content to generate
            user_id: User ID (required for personalized content)
            year: Year for monthly content
            month: Month for monthly content
            force_refresh: Bypass cache and regenerate
        
        Returns:
            GeneratedContent or None if generation failed
        """
        template = get_template(content_type)
        
        # Check if user is required
        if template.is_personalized and not user_id:
            logger.error(f"User ID required for {content_type.value}")
            return None
        
        # Check cache first (unless force refresh)
        if not force_refresh:
            cached = await self._get_cached_content(content_type, user_id)
            if cached:
                logger.info(f"Cache hit for {content_type.value}, user={user_id}")
                return cached
        
        # Assemble context
        logger.info(f"Generating {content_type.value} for user={user_id}")
        context = await self.assembler.assemble_context(
            content_type=content_type,
            user_id=user_id,
            year=year,
            month=month
        )
        
        if not context:
            logger.error(f"Failed to assemble context for {content_type.value}")
            return None
        
        # Get prompts
        system_prompt = self._get_system_prompt(content_type)
        user_prompt = context["prompt"]
        
        # Determine max tokens based on output length
        max_tokens_map = {"short": 500, "medium": 1000, "long": 2000}
        max_tokens = max_tokens_map.get(template.output_length, 1000)
        
        # Generate with LLM
        try:
            content = await self._generate_with_llm(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                max_tokens=max_tokens
            )
        except Exception as e:
            logger.error(f"Content generation failed: {e}")
            return None
        
        # Calculate validity
        valid_from, valid_until = self._calculate_validity(
            content_type, 
            year=context.get("year"),
            month=context.get("month")
        )
        
        # Prepare metadata
        metadata = context.get("metadata", {})
        metadata["generated_at"] = datetime.now(timezone.utc).isoformat()
        metadata["model"] = self.model
        
        # Save to cache
        await self._save_to_cache(
            content_type=content_type,
            content=content,
            user_id=user_id,
            valid_from=valid_from,
            valid_until=valid_until,
            metadata=metadata
        )
        
        logger.info(f"Generated and cached {content_type.value} for user={user_id}")
        
        return GeneratedContent(
            content_type=content_type,
            content=content,
            user_id=user_id,
            valid_from=valid_from,
            valid_until=valid_until,
            metadata=metadata,
            from_cache=False
        )
    
    async def generate_general_content(
        self,
        content_type: ContentType,
        year: Optional[int] = None,
        month: Optional[int] = None,
        force_refresh: bool = False
    ) -> Optional[GeneratedContent]:
        """
        Generate non-personalized content.
        
        Convenience method for general content types.
        """
        template = get_template(content_type)
        if template.is_personalized:
            logger.error(f"{content_type.value} requires user_id")
            return None
        
        return await self.generate_content(
            content_type=content_type,
            user_id=None,
            year=year,
            month=month,
            force_refresh=force_refresh
        )
    
    async def get_all_monthly_content(
        self,
        user_id: str,
        year: Optional[int] = None,
        month: Optional[int] = None
    ) -> Dict[str, GeneratedContent]:
        """
        Get all monthly content for a user.
        
        Returns dict with keys: 'general', 'personal'
        """
        results = {}
        
        # General monthly
        general = await self.generate_content(
            ContentType.MONTHLY_GENERAL,
            year=year,
            month=month
        )
        if general:
            results["general"] = general
        
        # Personal monthly
        personal = await self.generate_content(
            ContentType.MONTHLY_PERSONAL,
            user_id=user_id,
            year=year,
            month=month
        )
        if personal:
            results["personal"] = personal
        
        return results
    
    async def cleanup_expired_content(self) -> int:
        """
        Remove expired content from cache.
        
        Returns number of rows deleted.
        """
        if self._use_supabase:
            client = self._get_supabase_client()
            # Supabase doesn't return count directly, so we query first
            result = client.table("generated_content") \
                .delete() \
                .lt("valid_until", datetime.now(timezone.utc).isoformat()) \
                .execute()
            deleted = len(result.data) if result.data else 0
            logger.info(f"Cleaned up {deleted} expired content entries")
            return deleted
        
        # Fallback to asyncpg
        async with self.db.acquire() as conn:
            result = await conn.execute(
                """
                DELETE FROM generated_content
                WHERE valid_until < CURRENT_TIMESTAMP
                """
            )
            # Parse "DELETE N" response
            deleted = int(result.split()[-1]) if result else 0
            logger.info(f"Cleaned up {deleted} expired content entries")
            return deleted
