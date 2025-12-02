"""
Pytest configuration for integration tests.

Provides fixtures for:
- Live API client
- LLM evaluator for response quality
- Skip markers when services unavailable
"""

import os
import pytest
import pytest_asyncio
import asyncio
import aiohttp
from typing import Dict, Any, Optional, List
from dataclasses import dataclass

# Load environment
from dotenv import load_dotenv
load_dotenv()

# Configuration
API_BASE_URL = os.getenv("TEST_API_URL", "http://localhost:8058")
API_TIMEOUT = int(os.getenv("TEST_API_TIMEOUT", "60"))


@dataclass
class APIResponse:
    """Structured API response."""
    message: str
    tools_used: List[Dict[str, Any]]
    session_id: Optional[str]
    raw: Dict[str, Any]


class AgentAPIClient:
    """Async client for the agent API."""
    
    def __init__(self, base_url: str = API_BASE_URL, timeout: int = API_TIMEOUT):
        self.base_url = base_url.rstrip('/')
        self.timeout = aiohttp.ClientTimeout(total=timeout)
        self.session_id: Optional[str] = None
    
    async def health_check(self) -> bool:
        """Check if API is available."""
        try:
            async with aiohttp.ClientSession(timeout=self.timeout) as session:
                async with session.get(f"{self.base_url}/health") as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        return data.get("status") == "healthy"
                    return False
        except Exception:
            return False
    
    async def chat(self, message: str, user_id: str = "test_user") -> APIResponse:
        """Send a chat message to the agent."""
        request_data = {
            "message": message,
            "session_id": self.session_id,
            "user_id": user_id,
            "search_type": "hybrid"
        }
        
        async with aiohttp.ClientSession(timeout=self.timeout) as session:
            async with session.post(
                f"{self.base_url}/chat",
                json=request_data,
                headers={"Content-Type": "application/json"}
            ) as resp:
                if resp.status != 200:
                    error_text = await resp.text()
                    raise RuntimeError(f"API error ({resp.status}): {error_text}")
                
                data = await resp.json()
                
                # Store session for conversation continuity
                if data.get("session_id"):
                    self.session_id = data["session_id"]
                
                return APIResponse(
                    message=data.get("message", data.get("response", "")),
                    tools_used=data.get("tools_used", []),
                    session_id=data.get("session_id"),
                    raw=data
                )
    
    def reset_session(self):
        """Clear the current session."""
        self.session_id = None


# Import evaluator (will be created next)
from .evaluator import LLMEvaluator, EvaluationResult


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture(scope="session")
def event_loop():
    """Create event loop for async tests."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest_asyncio.fixture(scope="session")
async def api_client() -> AgentAPIClient:
    """Provide API client, skip tests if API unavailable."""
    client = AgentAPIClient()
    
    if not await client.health_check():
        pytest.skip(f"API not available at {API_BASE_URL}")
    
    return client


@pytest_asyncio.fixture
async def fresh_api_client(api_client: AgentAPIClient) -> AgentAPIClient:
    """Provide API client with fresh session for each test."""
    api_client.reset_session()
    return api_client


@pytest.fixture(scope="session")
def evaluator() -> LLMEvaluator:
    """Provide LLM evaluator for response quality scoring."""
    api_key = os.getenv("OPENAI_API_KEY") or os.getenv("LLM_API_KEY")
    if not api_key:
        pytest.skip("No OpenAI API key available for evaluation")
    
    return LLMEvaluator(api_key=api_key, model="gpt-4o")


# =============================================================================
# Helper Functions
# =============================================================================

def get_tool_names(response: APIResponse) -> List[str]:
    """Extract tool names from API response."""
    return [t.get("tool_name", "") for t in response.tools_used]


def assert_tool_used(response: APIResponse, expected_tool: str):
    """Assert that a specific tool was used."""
    tools = get_tool_names(response)
    assert expected_tool in tools, f"Expected tool '{expected_tool}' not found. Used: {tools}"


def assert_any_tool_used(response: APIResponse, expected_tools: List[str]):
    """Assert that at least one of the expected tools was used."""
    tools = get_tool_names(response)
    found = [t for t in expected_tools if t in tools]
    assert found, f"Expected one of {expected_tools}, but used: {tools}"


def assert_tools_used(response: APIResponse, expected_tools: List[str]):
    """Assert that all expected tools were used (in any order)."""
    tools = get_tool_names(response)
    for expected in expected_tools:
        assert expected in tools, f"Expected tool '{expected}' not found. Used: {tools}"


# =============================================================================
# Markers
# =============================================================================

def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "requires_documents: marks tests that need ingested documents"
    )
    config.addinivalue_line(
        "markers", "requires_neo4j: marks tests that need Neo4j ontology"
    )

