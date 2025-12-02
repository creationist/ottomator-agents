"""
Integration tests for content generation system.

Tests the content_types.py feature including:
- ContentType enum
- ContentTemplate dataclass
- Template registry
- Validity period calculations
- Required data validation
"""

import pytest
from datetime import datetime, timezone, timedelta
from typing import Dict, Any

# Import content generation modules
from content.content_types import (
    ContentType,
    ContentTemplate,
    CONTENT_TEMPLATES,
    get_template,
    get_all_content_types,
    get_personalized_content_types,
    get_general_content_types,
    MONTHLY_GENERAL_TEMPLATE,
    MONTHLY_PERSONAL_TEMPLATE,
    MOON_REFLECTION_TEMPLATE,
)


# =============================================================================
# ContentType Enum Tests
# =============================================================================

class TestContentTypeEnum:
    """Tests for ContentType enumeration."""
    
    def test_all_content_types_exist(self):
        """Verify all expected content types are defined."""
        expected_types = ["monthly_general", "monthly_personal", "moon_reflection"]
        actual_types = [ct.value for ct in ContentType]
        
        for expected in expected_types:
            assert expected in actual_types, f"Missing content type: {expected}"
    
    def test_content_type_string_values(self):
        """Verify ContentType values are strings."""
        assert ContentType.MONTHLY_GENERAL.value == "monthly_general"
        assert ContentType.MONTHLY_PERSONAL.value == "monthly_personal"
        assert ContentType.MOON_REFLECTION.value == "moon_reflection"
    
    def test_content_type_is_string_enum(self):
        """Verify ContentType inherits from str."""
        for ct in ContentType:
            assert isinstance(ct.value, str)
            # String comparison should work directly
            assert ct == ct.value


# =============================================================================
# ContentTemplate Dataclass Tests
# =============================================================================

class TestContentTemplate:
    """Tests for ContentTemplate dataclass."""
    
    def test_template_required_fields(self):
        """Verify ContentTemplate has all required fields."""
        template = CONTENT_TEMPLATES[ContentType.MONTHLY_GENERAL]
        
        # Required fields
        assert hasattr(template, "content_type")
        assert hasattr(template, "name")
        assert hasattr(template, "description")
        assert hasattr(template, "required_user_data")
        assert hasattr(template, "required_transit_data")
        assert hasattr(template, "prompt_template")
        assert hasattr(template, "cache_duration_hours")
        assert hasattr(template, "is_personalized")
        assert hasattr(template, "output_length")
    
    def test_template_types(self):
        """Verify field types are correct."""
        for content_type, template in CONTENT_TEMPLATES.items():
            assert isinstance(template.content_type, ContentType)
            assert isinstance(template.name, str)
            assert isinstance(template.description, str)
            assert isinstance(template.required_user_data, list)
            assert isinstance(template.required_transit_data, list)
            assert isinstance(template.prompt_template, str)
            assert isinstance(template.cache_duration_hours, int)
            assert isinstance(template.is_personalized, bool)
            assert template.output_length in ["short", "medium", "long"]
    
    def test_template_cache_durations_positive(self):
        """Verify cache durations are positive."""
        for template in CONTENT_TEMPLATES.values():
            assert template.cache_duration_hours > 0, \
                f"Cache duration should be positive for {template.name}"


# =============================================================================
# Template Registry Tests
# =============================================================================

class TestTemplateRegistry:
    """Tests for CONTENT_TEMPLATES registry."""
    
    def test_all_content_types_have_templates(self):
        """Every ContentType should have a corresponding template."""
        for content_type in ContentType:
            assert content_type in CONTENT_TEMPLATES, \
                f"Missing template for {content_type}"
    
    def test_template_content_type_matches_key(self):
        """Template's content_type should match its registry key."""
        for key, template in CONTENT_TEMPLATES.items():
            assert template.content_type == key, \
                f"Template content_type mismatch: {key} vs {template.content_type}"
    
    def test_get_template_function(self):
        """get_template() should return correct templates."""
        for content_type in ContentType:
            template = get_template(content_type)
            assert template is not None
            assert template.content_type == content_type
    
    def test_get_all_content_types(self):
        """get_all_content_types() returns all types."""
        all_types = get_all_content_types()
        assert len(all_types) == len(ContentType)
        assert set(all_types) == set(ContentType)


# =============================================================================
# Personalization Classification Tests
# =============================================================================

class TestPersonalizationClassification:
    """Tests for personalized vs general content classification."""
    
    def test_monthly_general_is_not_personalized(self):
        """MONTHLY_GENERAL should not require user data."""
        template = CONTENT_TEMPLATES[ContentType.MONTHLY_GENERAL]
        assert not template.is_personalized
        assert len(template.required_user_data) == 0
    
    def test_monthly_personal_is_personalized(self):
        """MONTHLY_PERSONAL should require user data."""
        template = CONTENT_TEMPLATES[ContentType.MONTHLY_PERSONAL]
        assert template.is_personalized
        assert len(template.required_user_data) > 0
        assert "sun_sign" in template.required_user_data
    
    def test_moon_reflection_is_personalized(self):
        """MOON_REFLECTION should require user data."""
        template = CONTENT_TEMPLATES[ContentType.MOON_REFLECTION]
        assert template.is_personalized
        assert "natal_moon_sign" in template.required_user_data
    
    def test_get_personalized_content_types(self):
        """get_personalized_content_types() returns only personalized types."""
        personalized = get_personalized_content_types()
        
        for ct in personalized:
            template = CONTENT_TEMPLATES[ct]
            assert template.is_personalized, \
                f"{ct} returned by get_personalized_content_types but is_personalized=False"
    
    def test_get_general_content_types(self):
        """get_general_content_types() returns only non-personalized types."""
        general = get_general_content_types()
        
        for ct in general:
            template = CONTENT_TEMPLATES[ct]
            assert not template.is_personalized, \
                f"{ct} returned by get_general_content_types but is_personalized=True"
    
    def test_personalized_plus_general_equals_all(self):
        """Personalized + general types should equal all types."""
        personalized = set(get_personalized_content_types())
        general = set(get_general_content_types())
        all_types = set(get_all_content_types())
        
        assert personalized.union(general) == all_types
        assert personalized.intersection(general) == set()


# =============================================================================
# Required Data Tests
# =============================================================================

class TestRequiredData:
    """Tests for required data fields."""
    
    def test_monthly_personal_required_user_data(self):
        """MONTHLY_PERSONAL requires specific user data."""
        template = CONTENT_TEMPLATES[ContentType.MONTHLY_PERSONAL]
        required = template.required_user_data
        
        expected_fields = ["sun_sign", "moon_sign", "rising_sign", "natal_positions"]
        for field in expected_fields:
            assert field in required, f"Missing required field: {field}"
    
    def test_moon_reflection_required_user_data(self):
        """MOON_REFLECTION requires natal moon data."""
        template = CONTENT_TEMPLATES[ContentType.MOON_REFLECTION]
        required = template.required_user_data
        
        expected_fields = ["natal_moon_sign", "natal_moon_house", "natal_moon_aspects"]
        for field in expected_fields:
            assert field in required, f"Missing required field: {field}"
    
    def test_monthly_general_required_transit_data(self):
        """MONTHLY_GENERAL requires transit data."""
        template = CONTENT_TEMPLATES[ContentType.MONTHLY_GENERAL]
        required = template.required_transit_data
        
        expected_fields = ["monthly_transits", "retrogrades", "moon_phases"]
        for field in expected_fields:
            assert field in required, f"Missing required transit field: {field}"
    
    def test_moon_reflection_required_transit_data(self):
        """MOON_REFLECTION requires current moon data."""
        template = CONTENT_TEMPLATES[ContentType.MOON_REFLECTION]
        required = template.required_transit_data
        
        expected_fields = ["transit_moon_sign", "moon_phase", "moon_aspects"]
        for field in expected_fields:
            assert field in required, f"Missing required transit field: {field}"


# =============================================================================
# Prompt Template Tests
# =============================================================================

class TestPromptTemplates:
    """Tests for prompt template content."""
    
    def test_monthly_general_template_has_placeholders(self):
        """MONTHLY_GENERAL template should have required placeholders."""
        template = MONTHLY_GENERAL_TEMPLATE
        
        required_placeholders = [
            "{month_name}",
            "{year}",
            "{monthly_transits}",
            "{retrogrades}",
            "{moon_phases}",
            "{ontology_context}"
        ]
        
        for placeholder in required_placeholders:
            assert placeholder in template, \
                f"Missing placeholder in MONTHLY_GENERAL: {placeholder}"
    
    def test_monthly_personal_template_has_placeholders(self):
        """MONTHLY_PERSONAL template should have required placeholders."""
        template = MONTHLY_PERSONAL_TEMPLATE
        
        required_placeholders = [
            "{sun_sign}",
            "{moon_sign}",
            "{rising_sign}",
            "{natal_positions}",
            "{transits_to_natal}",
            "{month_name}",
            "{year}"
        ]
        
        for placeholder in required_placeholders:
            assert placeholder in template, \
                f"Missing placeholder in MONTHLY_PERSONAL: {placeholder}"
    
    def test_moon_reflection_template_has_placeholders(self):
        """MOON_REFLECTION template should have required placeholders."""
        template = MOON_REFLECTION_TEMPLATE
        
        required_placeholders = [
            "{transit_moon_sign}",
            "{moon_phase}",
            "{natal_moon_sign}",
            "{natal_moon_house}"
        ]
        
        for placeholder in required_placeholders:
            assert placeholder in template, \
                f"Missing placeholder in MOON_REFLECTION: {placeholder}"
    
    def test_templates_are_in_german(self):
        """Templates should be in German."""
        templates = [MONTHLY_GENERAL_TEMPLATE, MONTHLY_PERSONAL_TEMPLATE, MOON_REFLECTION_TEMPLATE]
        
        german_indicators = ["Du bist", "Schreibe", "Erstelle", "auf Deutsch"]
        
        for template in templates:
            has_german = any(indicator in template for indicator in german_indicators)
            assert has_german, "Template should contain German language indicators"
    
    def test_templates_mention_nyah(self):
        """Templates should mention the Nyah persona."""
        templates = [MONTHLY_GENERAL_TEMPLATE, MONTHLY_PERSONAL_TEMPLATE, MOON_REFLECTION_TEMPLATE]
        
        for template in templates:
            assert "Nyah" in template, "Template should mention Nyah persona"


# =============================================================================
# Cache Duration Tests
# =============================================================================

class TestCacheDurations:
    """Tests for cache duration settings."""
    
    def test_monthly_content_cache_duration(self):
        """Monthly content should cache for ~30 days."""
        general = CONTENT_TEMPLATES[ContentType.MONTHLY_GENERAL]
        personal = CONTENT_TEMPLATES[ContentType.MONTHLY_PERSONAL]
        
        # 30 days = 720 hours
        assert general.cache_duration_hours >= 720, "Monthly general should cache for at least 30 days"
        assert personal.cache_duration_hours >= 720, "Monthly personal should cache for at least 30 days"
    
    def test_moon_reflection_cache_duration(self):
        """Moon reflection should have shorter cache (~2-3 days)."""
        template = CONTENT_TEMPLATES[ContentType.MOON_REFLECTION]
        
        # Moon changes sign every ~2.5 days = 60 hours
        assert template.cache_duration_hours <= 72, "Moon reflection should cache for max 3 days"
        assert template.cache_duration_hours >= 48, "Moon reflection should cache for at least 2 days"


# =============================================================================
# Output Length Tests
# =============================================================================

class TestOutputLength:
    """Tests for output length specifications."""
    
    def test_monthly_content_is_long(self):
        """Monthly content should be long format."""
        general = CONTENT_TEMPLATES[ContentType.MONTHLY_GENERAL]
        personal = CONTENT_TEMPLATES[ContentType.MONTHLY_PERSONAL]
        
        assert general.output_length == "long"
        assert personal.output_length == "long"
    
    def test_moon_reflection_is_medium(self):
        """Moon reflection should be medium format."""
        template = CONTENT_TEMPLATES[ContentType.MOON_REFLECTION]
        assert template.output_length == "medium"
    
    def test_valid_output_lengths(self):
        """All templates should have valid output lengths."""
        valid_lengths = ["short", "medium", "long"]
        
        for template in CONTENT_TEMPLATES.values():
            assert template.output_length in valid_lengths, \
                f"Invalid output_length: {template.output_length}"


# =============================================================================
# Integration with Generator (if available)
# =============================================================================

class TestGeneratorIntegration:
    """Tests for integration with PersonalizedContentGenerator."""
    
    def test_can_import_generator(self):
        """Generator module should be importable."""
        try:
            from content.generator import PersonalizedContentGenerator, GeneratedContent
            assert PersonalizedContentGenerator is not None
            assert GeneratedContent is not None
        except ImportError as e:
            pytest.skip(f"Generator module not available: {e}")
    
    def test_generated_content_dataclass(self):
        """GeneratedContent should have expected fields."""
        from content.generator import GeneratedContent
        
        # Create a test instance
        now = datetime.now(timezone.utc)
        content = GeneratedContent(
            content_type=ContentType.MONTHLY_GENERAL,
            content="Test content",
            user_id=None,
            valid_from=now,
            valid_until=now + timedelta(days=30),
            metadata={"test": True},
            from_cache=False
        )
        
        assert content.content_type == ContentType.MONTHLY_GENERAL
        assert content.content == "Test content"
        assert content.user_id is None
        assert content.from_cache is False


# =============================================================================
# Template Consistency Tests
# =============================================================================

class TestTemplateConsistency:
    """Tests for template consistency and quality."""
    
    def test_all_templates_have_task_section(self):
        """Templates should have a clear task/aufgabe section."""
        templates = {
            "MONTHLY_GENERAL": MONTHLY_GENERAL_TEMPLATE,
            "MONTHLY_PERSONAL": MONTHLY_PERSONAL_TEMPLATE,
            "MOON_REFLECTION": MOON_REFLECTION_TEMPLATE
        }
        
        for name, template in templates.items():
            assert "Aufgabe" in template or "aufgabe" in template.lower(), \
                f"{name} should have an Aufgabe section"
    
    def test_templates_have_format_instructions(self):
        """Templates should specify output format."""
        templates = [MONTHLY_GENERAL_TEMPLATE, MONTHLY_PERSONAL_TEMPLATE, MOON_REFLECTION_TEMPLATE]
        
        format_indicators = ["Format", "Strukturiere", "###", "Überschriften"]
        
        for template in templates:
            has_format = any(indicator in template for indicator in format_indicators)
            assert has_format, "Template should have format instructions"
    
    def test_template_descriptions_are_descriptive(self):
        """Template descriptions should be meaningful."""
        for template in CONTENT_TEMPLATES.values():
            assert len(template.description) > 20, \
                f"Description too short for {template.name}"
            assert len(template.name) > 5, \
                f"Name too short for {template.content_type}"


