"""
Flexible provider configuration for LLM and embedding models.

Supports:
- LLM: Groq, OpenAI, Anthropic, Ollama (local)
- Embeddings: Ollama (local), OpenAI
"""

import os
from typing import Optional
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.models.openai import OpenAIModel
import openai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


# =============================================================================
# Model Configurations
# =============================================================================

# Embedding model dimensions (important for database schema)
EMBEDDING_DIMENSIONS = {
    # Ollama / Local (recommended)
    "nomic-embed-text": 768,
    "mxbai-embed-large": 1024,
    "all-minilm": 384,
    "snowflake-arctic-embed": 1024,
    # OpenAI
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
    "text-embedding-ada-002": 1536,
}


def get_llm_model(model_choice: Optional[str] = None) -> OpenAIModel:
    """
    Get LLM model configuration based on environment variables.
    
    Supports Groq, OpenAI, and other OpenAI-compatible providers.
    
    Args:
        model_choice: Optional override for model choice
    
    Returns:
        Configured OpenAI-compatible model
    """
    llm_choice = model_choice or os.getenv('LLM_CHOICE', 'llama-3.3-70b-versatile')
    base_url = os.getenv('LLM_BASE_URL', 'https://api.groq.com/openai/v1')
    api_key = os.getenv('LLM_API_KEY')
    
    if not api_key:
        raise ValueError("LLM_API_KEY environment variable is required")
    
    provider = OpenAIProvider(base_url=base_url, api_key=api_key)
    return OpenAIModel(llm_choice, provider=provider)


def get_embedding_client() -> openai.AsyncOpenAI:
    """
    Get embedding client configuration based on environment variables.
    
    Supports Ollama (local) and OpenAI-compatible providers.
    
    Returns:
        Configured AsyncOpenAI client for embeddings
    """
    base_url = os.getenv('EMBEDDING_BASE_URL', 'http://localhost:11434/v1')
    api_key = os.getenv('EMBEDDING_API_KEY', 'ollama')  # Ollama doesn't need a real key
    
    return openai.AsyncOpenAI(
        base_url=base_url,
        api_key=api_key
    )


def get_embedding_model() -> str:
    """
    Get embedding model name from environment.
    
    Returns:
        Embedding model name
    """
    return os.getenv('EMBEDDING_MODEL', 'jina-embeddings-v4')


def get_embedding_dimensions() -> int:
    """
    Get the dimension of the current embedding model.
    
    Returns:
        Embedding dimension (e.g., 1024 for Jina v3)
    """
    model = get_embedding_model()
    # Check env override first
    env_dim = os.getenv('EMBEDDING_DIMENSIONS')
    if env_dim:
        return int(env_dim)
    # Use known dimensions
    return EMBEDDING_DIMENSIONS.get(model, 1024)


def get_ingestion_model() -> OpenAIModel:
    """
    Get ingestion-specific LLM model (can be faster/cheaper than main model).
    
    Returns:
        Configured model for ingestion tasks
    """
    ingestion_choice = os.getenv('INGESTION_LLM_CHOICE')
    
    # If no specific ingestion model, use the main model
    if not ingestion_choice:
        return get_llm_model()
    
    return get_llm_model(model_choice=ingestion_choice)


# Provider information functions
def get_llm_provider() -> str:
    """Get the LLM provider name."""
    return os.getenv('LLM_PROVIDER', 'openai')


def get_embedding_provider() -> str:
    """Get the embedding provider name."""
    return os.getenv('EMBEDDING_PROVIDER', 'openai')


def validate_configuration() -> bool:
    """
    Validate that required environment variables are set.
    
    Returns:
        True if configuration is valid
    """
    required_vars = [
        'LLM_API_KEY',
        'LLM_CHOICE',
        'EMBEDDING_API_KEY',
        'EMBEDDING_MODEL'
    ]
    
    missing_vars = []
    for var in required_vars:
        if not os.getenv(var):
            missing_vars.append(var)
    
    if missing_vars:
        print(f"Missing required environment variables: {', '.join(missing_vars)}")
        return False
    
    return True


def get_model_info() -> dict:
    """
    Get information about current model configuration.
    
    Returns:
        Dictionary with model configuration info
    """
    return {
        "llm_provider": get_llm_provider(),
        "llm_model": os.getenv('LLM_CHOICE'),
        "llm_base_url": os.getenv('LLM_BASE_URL'),
        "embedding_provider": get_embedding_provider(),
        "embedding_model": get_embedding_model(),
        "embedding_base_url": os.getenv('EMBEDDING_BASE_URL'),
        "ingestion_model": os.getenv('INGESTION_LLM_CHOICE', 'same as main'),
    }