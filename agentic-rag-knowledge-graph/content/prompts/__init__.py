"""
Prompt templates for content generation.
"""

from .monthly_general import MONTHLY_GENERAL_SYSTEM_PROMPT, format_monthly_general_prompt
from .monthly_personal import MONTHLY_PERSONAL_SYSTEM_PROMPT, format_monthly_personal_prompt
from .moon_reflection import MOON_REFLECTION_SYSTEM_PROMPT, format_moon_reflection_prompt

__all__ = [
    "MONTHLY_GENERAL_SYSTEM_PROMPT",
    "format_monthly_general_prompt",
    "MONTHLY_PERSONAL_SYSTEM_PROMPT", 
    "format_monthly_personal_prompt",
    "MOON_REFLECTION_SYSTEM_PROMPT",
    "format_moon_reflection_prompt",
]


