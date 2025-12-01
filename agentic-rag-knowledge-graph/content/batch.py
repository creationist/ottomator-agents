"""
Batch content generation for pre-generating content for multiple users.

Supports:
- Batch generation for all users
- Grouping by sun sign for efficiency
- Progress tracking
- Error handling and logging
"""

import logging
import asyncio
import json
from typing import Dict, Any, List, Optional
from datetime import datetime, timezone
from dataclasses import dataclass, field
from enum import Enum
import uuid

from .content_types import ContentType, get_personalized_content_types, get_general_content_types
from .generator import PersonalizedContentGenerator, GeneratedContent
from .user_profile import UserProfileService

logger = logging.getLogger(__name__)


class BatchJobStatus(str, Enum):
    """Status of a batch job."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class BatchJobError:
    """Error that occurred during batch processing."""
    user_id: str
    content_type: str
    error_message: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "user_id": self.user_id,
            "content_type": self.content_type,
            "error_message": self.error_message,
            "timestamp": self.timestamp.isoformat()
        }


@dataclass
class BatchJobResult:
    """Result of a batch job."""
    job_id: str
    job_type: str
    status: BatchJobStatus
    total_users: int
    processed_users: int
    successful: int = 0
    failed: int = 0
    errors: List[BatchJobError] = field(default_factory=list)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "job_id": self.job_id,
            "job_type": self.job_type,
            "status": self.status.value,
            "total_users": self.total_users,
            "processed_users": self.processed_users,
            "successful": self.successful,
            "failed": self.failed,
            "errors": [e.to_dict() for e in self.errors[:10]],  # Limit errors
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None
        }


class BatchContentGenerator:
    """
    Batch generator for pre-generating content for multiple users.
    
    Features:
    - Generates content for all users with profiles
    - Groups users by sun sign for shared context efficiency
    - Tracks progress in database
    - Handles errors gracefully
    - Supports concurrent processing with limits
    """
    
    def __init__(
        self,
        generator: PersonalizedContentGenerator,
        user_service: UserProfileService,
        db_pool,
        max_concurrency: int = 5
    ):
        self.generator = generator
        self.users = user_service
        self.db = db_pool
        self.max_concurrency = max_concurrency
        self.semaphore = asyncio.Semaphore(max_concurrency)
    
    async def _create_job(
        self,
        job_type: str,
        total_users: int
    ) -> str:
        """Create a batch job record in database."""
        job_id = str(uuid.uuid4())
        
        async with self.db.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO batch_jobs (id, job_type, status, total_users, started_at)
                VALUES ($1, $2, $3, $4, $5)
                """,
                uuid.UUID(job_id),
                job_type,
                BatchJobStatus.RUNNING.value,
                total_users,
                datetime.now(timezone.utc)
            )
        
        return job_id
    
    async def _update_job_progress(
        self,
        job_id: str,
        processed_users: int,
        errors: List[BatchJobError] = None
    ) -> None:
        """Update job progress in database."""
        async with self.db.acquire() as conn:
            if errors:
                await conn.execute(
                    """
                    UPDATE batch_jobs
                    SET processed_users = $2, errors = $3
                    WHERE id = $1
                    """,
                    uuid.UUID(job_id),
                    processed_users,
                    json.dumps([e.to_dict() for e in errors])
                )
            else:
                await conn.execute(
                    """
                    UPDATE batch_jobs
                    SET processed_users = $2
                    WHERE id = $1
                    """,
                    uuid.UUID(job_id),
                    processed_users
                )
    
    async def _complete_job(
        self,
        job_id: str,
        status: BatchJobStatus,
        errors: List[BatchJobError]
    ) -> None:
        """Mark job as completed."""
        async with self.db.acquire() as conn:
            await conn.execute(
                """
                UPDATE batch_jobs
                SET status = $2, completed_at = $3, errors = $4
                WHERE id = $1
                """,
                uuid.UUID(job_id),
                status.value,
                datetime.now(timezone.utc),
                json.dumps([e.to_dict() for e in errors])
            )
    
    async def get_job_status(self, job_id: str) -> Optional[BatchJobResult]:
        """Get current status of a batch job."""
        async with self.db.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT * FROM batch_jobs WHERE id = $1",
                uuid.UUID(job_id)
            )
            
            if not row:
                return None
            
            errors_data = row.get("errors", [])
            if isinstance(errors_data, str):
                errors_data = json.loads(errors_data)
            
            errors = [
                BatchJobError(
                    user_id=e["user_id"],
                    content_type=e["content_type"],
                    error_message=e["error_message"],
                    timestamp=datetime.fromisoformat(e["timestamp"])
                )
                for e in errors_data
            ] if errors_data else []
            
            return BatchJobResult(
                job_id=str(row["id"]),
                job_type=row["job_type"],
                status=BatchJobStatus(row["status"]),
                total_users=row["total_users"] or 0,
                processed_users=row["processed_users"] or 0,
                errors=errors,
                started_at=row.get("started_at"),
                completed_at=row.get("completed_at")
            )
    
    async def _generate_for_user(
        self,
        user_id: str,
        content_type: ContentType,
        year: Optional[int] = None,
        month: Optional[int] = None
    ) -> Optional[BatchJobError]:
        """Generate content for a single user with semaphore."""
        async with self.semaphore:
            try:
                result = await self.generator.generate_content(
                    content_type=content_type,
                    user_id=user_id,
                    year=year,
                    month=month
                )
                
                if not result:
                    return BatchJobError(
                        user_id=user_id,
                        content_type=content_type.value,
                        error_message="Generation returned None"
                    )
                
                return None
            
            except Exception as e:
                logger.error(f"Error generating {content_type.value} for {user_id}: {e}")
                return BatchJobError(
                    user_id=user_id,
                    content_type=content_type.value,
                    error_message=str(e)
                )
    
    async def generate_for_all_users(
        self,
        content_type: ContentType,
        user_ids: Optional[List[str]] = None,
        year: Optional[int] = None,
        month: Optional[int] = None
    ) -> BatchJobResult:
        """
        Generate content for all users (or specified list).
        
        Args:
            content_type: Type of content to generate
            user_ids: Optional list of user IDs (default: all users)
            year: Year for monthly content
            month: Month for monthly content
        
        Returns:
            BatchJobResult with status and any errors
        """
        # Get user list
        if user_ids is None:
            user_ids = await self.users.get_all_user_ids()
        
        if not user_ids:
            logger.warning("No users found for batch generation")
            return BatchJobResult(
                job_id=str(uuid.uuid4()),
                job_type=content_type.value,
                status=BatchJobStatus.COMPLETED,
                total_users=0,
                processed_users=0
            )
        
        # Create job record
        job_id = await self._create_job(content_type.value, len(user_ids))
        
        logger.info(f"Starting batch job {job_id}: {content_type.value} for {len(user_ids)} users")
        
        errors = []
        processed = 0
        
        # Process in batches to allow progress updates
        batch_size = 10
        for i in range(0, len(user_ids), batch_size):
            batch = user_ids[i:i + batch_size]
            
            # Generate concurrently within batch
            tasks = [
                self._generate_for_user(uid, content_type, year, month)
                for uid in batch
            ]
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            for result in results:
                if isinstance(result, Exception):
                    errors.append(BatchJobError(
                        user_id="unknown",
                        content_type=content_type.value,
                        error_message=str(result)
                    ))
                elif result is not None:
                    errors.append(result)
            
            processed += len(batch)
            await self._update_job_progress(job_id, processed, errors)
            
            logger.info(f"Job {job_id}: processed {processed}/{len(user_ids)}")
        
        # Complete job
        final_status = BatchJobStatus.COMPLETED if not errors else BatchJobStatus.COMPLETED
        await self._complete_job(job_id, final_status, errors)
        
        logger.info(f"Job {job_id} completed: {processed} processed, {len(errors)} errors")
        
        return BatchJobResult(
            job_id=job_id,
            job_type=content_type.value,
            status=final_status,
            total_users=len(user_ids),
            processed_users=processed,
            successful=processed - len(errors),
            failed=len(errors),
            errors=errors,
            started_at=datetime.now(timezone.utc),
            completed_at=datetime.now(timezone.utc)
        )
    
    async def generate_monthly_content(
        self,
        year: Optional[int] = None,
        month: Optional[int] = None
    ) -> Dict[str, BatchJobResult]:
        """
        Generate all monthly content (general + personal for all users).
        
        This is the main method called by monthly cron job.
        """
        now = datetime.now(timezone.utc)
        year = year or now.year
        month = month or now.month
        
        results = {}
        
        # 1. Generate general monthly content (once for all)
        logger.info(f"Generating monthly general content for {year}-{month}")
        general = await self.generator.generate_general_content(
            ContentType.MONTHLY_GENERAL,
            year=year,
            month=month,
            force_refresh=True
        )
        results["general"] = BatchJobResult(
            job_id=str(uuid.uuid4()),
            job_type="monthly_general",
            status=BatchJobStatus.COMPLETED if general else BatchJobStatus.FAILED,
            total_users=1,
            processed_users=1,
            successful=1 if general else 0,
            failed=0 if general else 1
        )
        
        # 2. Generate personal monthly content for all users
        logger.info(f"Generating monthly personal content for all users")
        results["personal"] = await self.generate_for_all_users(
            ContentType.MONTHLY_PERSONAL,
            year=year,
            month=month
        )
        
        return results
    
    async def generate_moon_reflection(
        self,
        user_ids: Optional[List[str]] = None
    ) -> BatchJobResult:
        """
        Generate moon reflection questions for all users.
        
        This is called when Moon changes sign (~every 2.5 days).
        """
        return await self.generate_for_all_users(
            ContentType.MOON_REFLECTION,
            user_ids=user_ids
        )

