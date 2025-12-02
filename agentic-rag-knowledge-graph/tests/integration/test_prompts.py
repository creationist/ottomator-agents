"""
Integration tests for prompt correctness.

Tests that:
- System prompts contain required tool instructions
- Prompt templates include all placeholders
- German language consistency
- No truncation of critical instructions
"""

import pytest
from typing import List, Set

# Import prompts
from agent.prompts import SYSTEM_PROMPT, BASE_PERSONALITY
from content.content_types import (
    MONTHLY_GENERAL_TEMPLATE,
    MONTHLY_PERSONAL_TEMPLATE,
    MOON_REFLECTION_TEMPLATE,
)
from content.prompts import (
    MONTHLY_GENERAL_SYSTEM_PROMPT,
    MONTHLY_PERSONAL_SYSTEM_PROMPT,
    MOON_REFLECTION_SYSTEM_PROMPT,
)


# =============================================================================
# System Prompt Structure Tests
# =============================================================================

class TestSystemPromptStructure:
    """Tests for the main agent system prompt structure."""
    
    def test_system_prompt_includes_base_personality(self):
        """System prompt should include base personality."""
        assert BASE_PERSONALITY in SYSTEM_PROMPT, \
            "SYSTEM_PROMPT should include BASE_PERSONALITY"
    
    def test_system_prompt_has_tools_section(self):
        """System prompt should have a Tools section."""
        assert "## Tools" in SYSTEM_PROMPT or "## Werkzeuge" in SYSTEM_PROMPT, \
            "System prompt should have a Tools section"
    
    def test_system_prompt_has_workflow_section(self):
        """System prompt should have a Workflow section."""
        assert "## Workflow" in SYSTEM_PROMPT or "Workflow" in SYSTEM_PROMPT, \
            "System prompt should have a Workflow section"
    
    def test_system_prompt_mentions_key_tools(self):
        """System prompt should mention key tools."""
        key_tools = ["comprehensive_lookup", "search", "explore_ontology"]
        
        for tool in key_tools:
            assert tool in SYSTEM_PROMPT, \
                f"System prompt should mention tool: {tool}"
    
    def test_system_prompt_length_reasonable(self):
        """System prompt should not be excessively long or truncated."""
        # Should be substantial but not excessively long
        assert len(SYSTEM_PROMPT) > 500, "System prompt seems too short"
        assert len(SYSTEM_PROMPT) < 10000, "System prompt seems too long"


# =============================================================================
# Base Personality Tests
# =============================================================================

class TestBasePersonality:
    """Tests for the base personality prompt."""
    
    def test_personality_mentions_nyah(self):
        """Base personality should establish Nyah identity."""
        assert "Nyah" in BASE_PERSONALITY, \
            "Base personality should mention Nyah"
    
    def test_personality_is_german(self):
        """Base personality should be in German."""
        german_words = ["Du bist", "Persönlichkeit", "Planeten", "Zeichen"]
        has_german = any(word in BASE_PERSONALITY for word in german_words)
        assert has_german, "Base personality should be in German"
    
    def test_personality_mentions_knowledge_areas(self):
        """Base personality should list knowledge areas."""
        knowledge_areas = ["Planeten", "Zeichen", "Häuser", "Aspekte"]
        
        for area in knowledge_areas:
            assert area in BASE_PERSONALITY, \
                f"Base personality should mention knowledge area: {area}"
    
    def test_personality_defines_character_traits(self):
        """Base personality should define character traits."""
        traits = ["Warm", "poetisch", "inspirierend"]
        has_traits = any(trait.lower() in BASE_PERSONALITY.lower() for trait in traits)
        assert has_traits, "Base personality should define character traits"


# =============================================================================
# Tool Instructions Tests
# =============================================================================

class TestToolInstructions:
    """Tests for tool usage instructions in system prompt."""
    
    def test_comprehensive_lookup_instruction(self):
        """Should have clear comprehensive_lookup instruction."""
        assert "comprehensive_lookup" in SYSTEM_PROMPT
        # Should indicate it's the main tool for concepts
        assert "Konzept" in SYSTEM_PROMPT or "concept" in SYSTEM_PROMPT.lower()
    
    def test_search_instruction(self):
        """Should have search tool instruction."""
        assert "search" in SYSTEM_PROMPT
        # Should indicate it's for document search
        assert "Dokument" in SYSTEM_PROMPT or "Suche" in SYSTEM_PROMPT
    
    def test_explore_ontology_instruction(self):
        """Should have explore_ontology instruction."""
        assert "explore_ontology" in SYSTEM_PROMPT
    
    def test_tool_usage_warning(self):
        """Should warn about using tool data vs prior knowledge."""
        warning_indicators = [
            "Nutze die Daten",
            "nicht dein Vorwissen",
            "Erfinde nichts",
            "Tool"
        ]
        
        found_warnings = [w for w in warning_indicators if w in SYSTEM_PROMPT]
        assert len(found_warnings) >= 2, \
            f"System prompt should have tool usage warnings. Found: {found_warnings}"


# =============================================================================
# Content Generation System Prompts
# =============================================================================

class TestContentSystemPrompts:
    """Tests for content generation system prompts."""
    
    def test_monthly_general_system_prompt_exists(self):
        """MONTHLY_GENERAL_SYSTEM_PROMPT should be defined."""
        assert MONTHLY_GENERAL_SYSTEM_PROMPT is not None
        assert len(MONTHLY_GENERAL_SYSTEM_PROMPT) > 50
    
    def test_monthly_personal_system_prompt_exists(self):
        """MONTHLY_PERSONAL_SYSTEM_PROMPT should be defined."""
        assert MONTHLY_PERSONAL_SYSTEM_PROMPT is not None
        assert len(MONTHLY_PERSONAL_SYSTEM_PROMPT) > 50
    
    def test_moon_reflection_system_prompt_exists(self):
        """MOON_REFLECTION_SYSTEM_PROMPT should be defined."""
        assert MOON_REFLECTION_SYSTEM_PROMPT is not None
        assert len(MOON_REFLECTION_SYSTEM_PROMPT) > 50
    
    def test_system_prompts_mention_nyah(self):
        """All content system prompts should mention Nyah."""
        prompts = [
            ("MONTHLY_GENERAL", MONTHLY_GENERAL_SYSTEM_PROMPT),
            ("MONTHLY_PERSONAL", MONTHLY_PERSONAL_SYSTEM_PROMPT),
            ("MOON_REFLECTION", MOON_REFLECTION_SYSTEM_PROMPT),
        ]
        
        for name, prompt in prompts:
            assert "Nyah" in prompt, f"{name} should mention Nyah"
    
    def test_system_prompts_are_german(self):
        """Content system prompts should be in German."""
        prompts = [
            MONTHLY_GENERAL_SYSTEM_PROMPT,
            MONTHLY_PERSONAL_SYSTEM_PROMPT,
            MOON_REFLECTION_SYSTEM_PROMPT,
        ]
        
        german_indicators = ["Du bist", "Astrologie", "Schreibe", "Erstelle"]
        
        for prompt in prompts:
            has_german = any(ind in prompt for ind in german_indicators)
            assert has_german, "System prompt should be in German"


# =============================================================================
# Template Placeholder Tests
# =============================================================================

class TestTemplatePlaceholders:
    """Tests for prompt template placeholder completeness."""
    
    def _extract_placeholders(self, template: str) -> Set[str]:
        """Extract {placeholder} patterns from template."""
        import re
        return set(re.findall(r'\{(\w+)\}', template))
    
    def test_monthly_general_placeholders_complete(self):
        """MONTHLY_GENERAL template should have all required placeholders."""
        placeholders = self._extract_placeholders(MONTHLY_GENERAL_TEMPLATE)
        
        required = {"month_name", "year", "monthly_transits", "retrogrades", "moon_phases", "ontology_context"}
        
        missing = required - placeholders
        assert not missing, f"Missing placeholders in MONTHLY_GENERAL: {missing}"
    
    def test_monthly_personal_placeholders_complete(self):
        """MONTHLY_PERSONAL template should have all required placeholders."""
        placeholders = self._extract_placeholders(MONTHLY_PERSONAL_TEMPLATE)
        
        required = {
            "sun_sign", "moon_sign", "rising_sign", 
            "natal_positions", "transits_to_natal",
            "month_name", "year", "moon_phases_for_sign", "ontology_context"
        }
        
        missing = required - placeholders
        assert not missing, f"Missing placeholders in MONTHLY_PERSONAL: {missing}"
    
    def test_moon_reflection_placeholders_complete(self):
        """MOON_REFLECTION template should have all required placeholders."""
        placeholders = self._extract_placeholders(MOON_REFLECTION_TEMPLATE)
        
        required = {
            "transit_moon_sign", "moon_phase", "moon_aspects",
            "natal_moon_sign", "natal_moon_house", "natal_moon_aspects",
            "moon_sign_context"
        }
        
        missing = required - placeholders
        assert not missing, f"Missing placeholders in MOON_REFLECTION: {missing}"
    
    def test_no_unmatched_braces(self):
        """Templates should not have unmatched braces (malformed placeholders)."""
        import re
        
        templates = [
            ("MONTHLY_GENERAL", MONTHLY_GENERAL_TEMPLATE),
            ("MONTHLY_PERSONAL", MONTHLY_PERSONAL_TEMPLATE),
            ("MOON_REFLECTION", MOON_REFLECTION_TEMPLATE),
        ]
        
        for name, template in templates:
            # Count opening and closing braces
            open_count = template.count('{')
            close_count = template.count('}')
            
            assert open_count == close_count, \
                f"{name} has unbalanced braces: {open_count} {{ vs {close_count} }}"


# =============================================================================
# German Language Tests
# =============================================================================

class TestGermanLanguage:
    """Tests for German language consistency."""
    
    def test_system_prompt_german_instructions(self):
        """System prompt should give instructions in German."""
        # Check for German instruction verbs
        german_verbs = ["Nutze", "Beachte", "Workflow", "antwortest"]
        
        found = [v for v in german_verbs if v in SYSTEM_PROMPT]
        assert len(found) >= 2, \
            f"System prompt should have German instruction verbs. Found: {found}"
    
    def test_templates_specify_german_output(self):
        """Templates should explicitly request German output."""
        templates = [
            ("MONTHLY_GENERAL", MONTHLY_GENERAL_TEMPLATE),
            ("MONTHLY_PERSONAL", MONTHLY_PERSONAL_TEMPLATE),
            ("MOON_REFLECTION", MOON_REFLECTION_TEMPLATE),
        ]
        
        for name, template in templates:
            has_german_spec = (
                "auf Deutsch" in template or 
                "Deutsch" in template or
                "German" in template
            )
            assert has_german_spec, \
                f"{name} should specify German language output"
    
    def test_no_english_instructions_in_templates(self):
        """Templates should not have English instructions (except placeholders)."""
        templates = [
            MONTHLY_GENERAL_TEMPLATE,
            MONTHLY_PERSONAL_TEMPLATE,
            MOON_REFLECTION_TEMPLATE,
        ]
        
        english_phrases = ["You are", "Write a", "Create a", "The user"]
        
        for template in templates:
            for phrase in english_phrases:
                assert phrase not in template, \
                    f"Template contains English: '{phrase}'"


# =============================================================================
# Critical Instruction Tests
# =============================================================================

class TestCriticalInstructions:
    """Tests that critical instructions are not truncated or missing."""
    
    def test_system_prompt_ends_properly(self):
        """System prompt should end with complete instruction."""
        # Should not end with incomplete sentence or truncated text
        ending = SYSTEM_PROMPT.strip()[-50:]
        
        # Should end with quote, period, or closing bracket
        valid_endings = ['"', "'", ".", ")", "]", "an"]
        has_valid_ending = any(ending.endswith(e) for e in valid_endings)
        
        assert has_valid_ending, \
            f"System prompt may be truncated. Ends with: ...{ending}"
    
    def test_templates_have_task_section(self):
        """Templates should have complete task/aufgabe sections."""
        templates = [
            ("MONTHLY_GENERAL", MONTHLY_GENERAL_TEMPLATE),
            ("MONTHLY_PERSONAL", MONTHLY_PERSONAL_TEMPLATE),
            ("MOON_REFLECTION", MOON_REFLECTION_TEMPLATE),
        ]
        
        for name, template in templates:
            assert "**Aufgabe:**" in template or "Aufgabe:" in template, \
                f"{name} should have an Aufgabe section"
    
    def test_templates_have_numbered_requirements(self):
        """Templates should have numbered requirements."""
        templates = [
            ("MONTHLY_GENERAL", MONTHLY_GENERAL_TEMPLATE),
            ("MONTHLY_PERSONAL", MONTHLY_PERSONAL_TEMPLATE),
            ("MOON_REFLECTION", MOON_REFLECTION_TEMPLATE),
        ]
        
        for name, template in templates:
            has_numbers = "1." in template and "2." in template
            assert has_numbers, \
                f"{name} should have numbered requirements"
    
    def test_templates_end_properly(self):
        """Templates should not be truncated."""
        templates = [
            ("MONTHLY_GENERAL", MONTHLY_GENERAL_TEMPLATE),
            ("MONTHLY_PERSONAL", MONTHLY_PERSONAL_TEMPLATE),
            ("MOON_REFLECTION", MOON_REFLECTION_TEMPLATE),
        ]
        
        for name, template in templates:
            # Should end with complete instruction
            stripped = template.strip()
            assert len(stripped) > 500, f"{name} seems too short"
            
            # Should not end mid-sentence
            last_char = stripped[-1]
            assert last_char in ".)\"'\n", \
                f"{name} may be truncated, ends with: {last_char}"


# =============================================================================
# Prompt Consistency Tests
# =============================================================================

class TestPromptConsistency:
    """Tests for consistency across prompts."""
    
    def test_persona_consistent_across_prompts(self):
        """Nyah persona should be consistent across all prompts."""
        all_prompts = [
            SYSTEM_PROMPT,
            MONTHLY_GENERAL_TEMPLATE,
            MONTHLY_PERSONAL_TEMPLATE,
            MOON_REFLECTION_TEMPLATE,
        ]
        
        # All prompts mentioning Nyah should describe similar traits
        nyah_prompts = [p for p in all_prompts if "Nyah" in p]
        assert len(nyah_prompts) >= 3, "Nyah should be mentioned in most prompts"
    
    def test_tool_names_match_implementation(self):
        """Tool names in prompt should match actual implementation."""
        # These tools should exist based on agent.py
        expected_tools = [
            "comprehensive_lookup",
            "search",
            "explore_ontology",
        ]
        
        for tool in expected_tools:
            if tool in SYSTEM_PROMPT:
                # Verify it's described as a callable
                assert f"`{tool}" in SYSTEM_PROMPT or f"**{tool}" in SYSTEM_PROMPT or f"{tool}(" in SYSTEM_PROMPT, \
                    f"Tool {tool} should be formatted as code/callable"


# =============================================================================
# Edge Case Tests
# =============================================================================

class TestPromptEdgeCases:
    """Tests for edge cases in prompts."""
    
    def test_no_duplicate_sections(self):
        """Prompts should not have duplicate section headers."""
        # Check for duplicate ## headers in system prompt
        lines = SYSTEM_PROMPT.split('\n')
        headers = [l for l in lines if l.startswith('## ')]
        
        unique_headers = set(headers)
        assert len(headers) == len(unique_headers), \
            f"Duplicate headers found: {headers}"
    
    def test_placeholders_dont_contain_instructions(self):
        """Placeholders should be simple names, not instructions."""
        import re
        
        templates = [MONTHLY_GENERAL_TEMPLATE, MONTHLY_PERSONAL_TEMPLATE, MOON_REFLECTION_TEMPLATE]
        
        for template in templates:
            placeholders = re.findall(r'\{(\w+)\}', template)
            for ph in placeholders:
                # Placeholder names should be simple (no spaces or special chars)
                assert ph.isidentifier(), f"Invalid placeholder name: {ph}"
                # Should not be too long (likely an error)
                assert len(ph) < 50, f"Placeholder name too long: {ph}"


