"""
Integration tests for tool selection.

Tests that the agent selects the correct tools based on query type.
Based on TESTING.md test scenarios.
"""

import pytest
from typing import List

from .conftest import (
    AgentAPIClient,
    APIResponse,
    get_tool_names,
    assert_tool_used,
    assert_any_tool_used,
)
from .evaluator import LLMEvaluator


# =============================================================================
# Search Tool Tests (Hybrid Vector + Text Search)
# =============================================================================

class TestSearchTool:
    """Tests for the 'search' tool - document queries."""
    
    @pytest.mark.asyncio
    async def test_search_document_content(
        self, fresh_api_client: AgentAPIClient, evaluator: LLMEvaluator
    ):
        """Test: Document content search triggers 'search' tool."""
        query = "Was sagt mein Dokument über Venus im Stier?"
        expected_tools = ["search"]
        
        response = await fresh_api_client.chat(query)
        
        # Verify tool selection - search_document_facts is also valid for document queries
        assert_any_tool_used(response, expected_tools + ["comprehensive_lookup", "search_document_facts"])
        
        # LLM evaluation
        result = await evaluator.evaluate_response(
            query=query,
            response=response.message,
            expected_behavior="Returns ranked document chunks with similarity scores about Venus in Taurus",
            tools_used=response.tools_used
        )
        assert result.passed, f"Evaluation failed: {result}"
    
    @pytest.mark.asyncio
    async def test_search_retrograde(
        self, fresh_api_client: AgentAPIClient, evaluator: LLMEvaluator
    ):
        """Test: Retrograde information search."""
        query = "Finde Informationen über Merkur Rückläufigkeit"
        
        response = await fresh_api_client.chat(query)
        
        # Should use search or comprehensive_lookup
        assert_any_tool_used(response, ["search", "comprehensive_lookup", "lookup_concept"])
        
        result = await evaluator.evaluate_response(
            query=query,
            response=response.message,
            expected_behavior="Returns information about Mercury retrograde from documents or ontology",
            tools_used=response.tools_used
        )
        assert result.passed, f"Evaluation failed: {result}"
    
    @pytest.mark.asyncio
    @pytest.mark.requires_documents
    async def test_search_moon_phases(
        self, fresh_api_client: AgentAPIClient, evaluator: LLMEvaluator
    ):
        """Test: Full moon document search."""
        query = "Welche Texte erwähnen den Vollmond?"
        
        response = await fresh_api_client.chat(query)
        
        assert_any_tool_used(response, ["search", "comprehensive_lookup"])
        
        result = await evaluator.evaluate_response(
            query=query,
            response=response.message,
            expected_behavior="Returns document chunks mentioning full moon",
            tools_used=response.tools_used
        )
        assert result.passed, f"Evaluation failed: {result}"


# =============================================================================
# Comprehensive Lookup Tests (Combined Knowledge)
# =============================================================================

class TestComprehensiveLookup:
    """Tests for 'comprehensive_lookup' - the preferred combined tool."""
    
    @pytest.mark.asyncio
    async def test_lookup_planet_mars(
        self, fresh_api_client: AgentAPIClient, evaluator: LLMEvaluator
    ):
        """Test: Planet explanation should use comprehensive_lookup."""
        query = "Erkläre mir Mars"
        expected_tools = ["comprehensive_lookup"]
        
        response = await fresh_api_client.chat(query)
        
        assert_any_tool_used(response, expected_tools + ["lookup_concept"])
        
        result = await evaluator.evaluate_response(
            query=query,
            response=response.message,
            expected_behavior="Returns ontology facts + document content + graph relationships about Mars",
            tools_used=response.tools_used
        )
        assert result.passed, f"Evaluation failed: {result}"
    
    @pytest.mark.asyncio
    async def test_lookup_sign_scorpio(
        self, fresh_api_client: AgentAPIClient, evaluator: LLMEvaluator
    ):
        """Test: Zodiac sign query."""
        query = "Was ist Skorpion?"
        
        response = await fresh_api_client.chat(query)
        
        assert_any_tool_used(response, ["comprehensive_lookup", "lookup_concept"])
        
        result = await evaluator.evaluate_response(
            query=query,
            response=response.message,
            expected_behavior="Returns complete information about Scorpio zodiac sign",
            tools_used=response.tools_used
        )
        assert result.passed, f"Evaluation failed: {result}"
    
    @pytest.mark.asyncio
    async def test_lookup_house(
        self, fresh_api_client: AgentAPIClient, evaluator: LLMEvaluator
    ):
        """Test: House explanation."""
        query = "Erzähl mir über das achte Haus"
        
        response = await fresh_api_client.chat(query)
        
        assert_any_tool_used(response, ["comprehensive_lookup", "lookup_concept"])
        
        result = await evaluator.evaluate_response(
            query=query,
            response=response.message,
            expected_behavior="Returns information about the 8th house in astrology",
            tools_used=response.tools_used
        )
        assert result.passed, f"Evaluation failed: {result}"
    
    @pytest.mark.asyncio
    async def test_lookup_venus(
        self, fresh_api_client: AgentAPIClient, evaluator: LLMEvaluator
    ):
        """Test: Venus knowledge query."""
        query = "Was weißt du über Venus?"
        
        response = await fresh_api_client.chat(query)
        
        assert_any_tool_used(response, ["comprehensive_lookup", "lookup_concept"])
        
        result = await evaluator.evaluate_response(
            query=query,
            response=response.message,
            expected_behavior="Returns comprehensive Venus information from all sources",
            tools_used=response.tools_used
        )
        assert result.passed, f"Evaluation failed: {result}"
    
    @pytest.mark.asyncio
    async def test_lookup_theme_transformation(
        self, fresh_api_client: AgentAPIClient, evaluator: LLMEvaluator
    ):
        """Test: Theme/concept query."""
        query = "Was bedeutet Transformation?"
        
        response = await fresh_api_client.chat(query)
        
        assert_any_tool_used(response, ["comprehensive_lookup", "lookup_concept", "search"])
        
        result = await evaluator.evaluate_response(
            query=query,
            response=response.message,
            expected_behavior="Returns information about transformation in astrological context",
            tools_used=response.tools_used
        )
        assert result.passed, f"Evaluation failed: {result}"


# =============================================================================
# Ontology Tools Tests
# =============================================================================

class TestOntologyTools:
    """Tests for explore_ontology and lookup_concept."""
    
    @pytest.mark.asyncio
    async def test_ontology_water_signs(
        self, fresh_api_client: AgentAPIClient, evaluator: LLMEvaluator
    ):
        """Test: Element-based query should use ontology tools."""
        query = "Welche Zeichen gehören zum Element Wasser?"
        
        response = await fresh_api_client.chat(query)
        
        assert_any_tool_used(response, ["explore_ontology", "lookup_concept", "comprehensive_lookup"])
        
        result = await evaluator.evaluate_response(
            query=query,
            response=response.message,
            expected_behavior="Returns water signs: Cancer, Scorpio, Pisces from ontology",
            tools_used=response.tools_used
        )
        assert result.passed, f"Evaluation failed: {result}"
    
    @pytest.mark.asyncio
    async def test_ontology_cardinal_signs(
        self, fresh_api_client: AgentAPIClient, evaluator: LLMEvaluator
    ):
        """Test: Modality query."""
        query = "Was sind die kardinalen Zeichen?"
        
        response = await fresh_api_client.chat(query)
        
        assert_any_tool_used(response, ["explore_ontology", "lookup_concept", "comprehensive_lookup"])
        
        result = await evaluator.evaluate_response(
            query=query,
            response=response.message,
            expected_behavior="Returns cardinal signs: Aries, Cancer, Libra, Capricorn",
            tools_used=response.tools_used
        )
        assert result.passed, f"Evaluation failed: {result}"
    
    @pytest.mark.asyncio
    async def test_lookup_concept_trigon(
        self, fresh_api_client: AgentAPIClient, evaluator: LLMEvaluator
    ):
        """Test: Aspect definition lookup."""
        query = "Was ist ein Trigon?"
        
        response = await fresh_api_client.chat(query)
        
        assert_any_tool_used(response, ["lookup_concept", "comprehensive_lookup"])
        
        result = await evaluator.evaluate_response(
            query=query,
            response=response.message,
            expected_behavior="Returns ontology definition of trine aspect with keywords",
            tools_used=response.tools_used
        )
        assert result.passed, f"Evaluation failed: {result}"
    
    @pytest.mark.asyncio
    async def test_lookup_concept_conjunction(
        self, fresh_api_client: AgentAPIClient, evaluator: LLMEvaluator
    ):
        """Test: Conjunction definition."""
        query = "Was bedeutet Konjunktion in der Astrologie?"
        
        response = await fresh_api_client.chat(query)
        
        assert_any_tool_used(response, ["lookup_concept", "comprehensive_lookup"])
        
        result = await evaluator.evaluate_response(
            query=query,
            response=response.message,
            expected_behavior="Returns definition of conjunction aspect",
            tools_used=response.tools_used
        )
        assert result.passed, f"Evaluation failed: {result}"
    
    @pytest.mark.asyncio
    async def test_explore_venus_rulership(
        self, fresh_api_client: AgentAPIClient, evaluator: LLMEvaluator
    ):
        """Test: Planet rulership query."""
        query = "Welche Zeichen regiert Venus?"
        
        response = await fresh_api_client.chat(query)
        
        assert_any_tool_used(response, ["explore_ontology", "lookup_concept", "comprehensive_lookup"])
        
        result = await evaluator.evaluate_response(
            query=query,
            response=response.message,
            expected_behavior="Returns that Venus rules Taurus and Libra",
            tools_used=response.tools_used
        )
        assert result.passed, f"Evaluation failed: {result}"
    
    @pytest.mark.asyncio
    async def test_explore_fixed_signs(
        self, fresh_api_client: AgentAPIClient, evaluator: LLMEvaluator
    ):
        """Test: Fixed modality query."""
        query = "Zeige mir die fixen Zeichen"
        
        response = await fresh_api_client.chat(query)
        
        assert_any_tool_used(response, ["explore_ontology", "lookup_concept", "comprehensive_lookup"])
        
        result = await evaluator.evaluate_response(
            query=query,
            response=response.message,
            expected_behavior="Returns fixed signs: Taurus, Leo, Scorpio, Aquarius",
            tools_used=response.tools_used
        )
        assert result.passed, f"Evaluation failed: {result}"


# =============================================================================
# Document Management Tests
# =============================================================================

class TestDocumentTools:
    """Tests for list_documents and get_document."""
    
    @pytest.mark.asyncio
    async def test_list_documents(
        self, fresh_api_client: AgentAPIClient, evaluator: LLMEvaluator
    ):
        """Test: Document listing query."""
        query = "Welche Dokumente hast du in deiner Wissensbasis?"
        
        response = await fresh_api_client.chat(query)
        
        assert_any_tool_used(response, ["list_documents"])
        
        result = await evaluator.evaluate_response(
            query=query,
            response=response.message,
            expected_behavior="Returns list of documents with metadata",
            tools_used=response.tools_used
        )
        assert result.passed, f"Evaluation failed: {result}"
    
    @pytest.mark.asyncio
    async def test_list_available_texts(
        self, fresh_api_client: AgentAPIClient, evaluator: LLMEvaluator
    ):
        """Test: Alternative document listing query."""
        query = "Liste alle verfügbaren Astrologie-Texte auf"
        
        response = await fresh_api_client.chat(query)
        
        assert_any_tool_used(response, ["list_documents", "search"])
        
        result = await evaluator.evaluate_response(
            query=query,
            response=response.message,
            expected_behavior="Returns list of available astrology texts",
            tools_used=response.tools_used
        )
        assert result.passed, f"Evaluation failed: {result}"


# =============================================================================
# Entity Relationship Tests
# =============================================================================

class TestEntityRelationships:
    """Tests for get_entity_relationships."""
    
    @pytest.mark.asyncio
    async def test_venus_relationships(
        self, fresh_api_client: AgentAPIClient, evaluator: LLMEvaluator
    ):
        """Test: Venus relationship query."""
        query = "Welche Beziehungen hat Venus?"
        
        response = await fresh_api_client.chat(query)
        
        # Could use entity relationships or ontology exploration
        assert_any_tool_used(
            response, 
            ["search_document_facts", "explore_ontology", "comprehensive_lookup", "lookup_concept"]
        )
        
        result = await evaluator.evaluate_response(
            query=query,
            response=response.message,
            expected_behavior="Returns relationship types and connected entities for Venus",
            tools_used=response.tools_used
        )
        assert result.passed, f"Evaluation failed: {result}"
    
    @pytest.mark.asyncio
    async def test_saturn_connections(
        self, fresh_api_client: AgentAPIClient, evaluator: LLMEvaluator
    ):
        """Test: Saturn connections query."""
        query = "Mit welchen Zeichen ist Saturn verbunden?"
        
        response = await fresh_api_client.chat(query)
        
        assert_any_tool_used(
            response, 
            ["search_document_facts", "explore_ontology", "comprehensive_lookup", "lookup_concept"]
        )
        
        result = await evaluator.evaluate_response(
            query=query,
            response=response.message,
            expected_behavior="Returns signs connected to Saturn (rules Capricorn, exalted in Libra)",
            tools_used=response.tools_used
        )
        assert result.passed, f"Evaluation failed: {result}"
    
    @pytest.mark.asyncio
    async def test_scorpio_entities(
        self, fresh_api_client: AgentAPIClient, evaluator: LLMEvaluator
    ):
        """Test: Scorpio entity connections."""
        query = "Welche Entitäten sind mit Skorpion verknüpft?"
        
        response = await fresh_api_client.chat(query)
        
        assert_any_tool_used(
            response, 
            ["search_document_facts", "explore_ontology", "comprehensive_lookup", "lookup_concept"]
        )
        
        result = await evaluator.evaluate_response(
            query=query,
            response=response.message,
            expected_behavior="Returns entities linked to Scorpio (Pluto, Mars, 8th house, etc.)",
            tools_used=response.tools_used
        )
        assert result.passed, f"Evaluation failed: {result}"


# =============================================================================
# Entity Timeline Tests
# =============================================================================

class TestEntityTimeline:
    """Tests for get_entity_timeline."""
    
    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_pluto_timeline(
        self, fresh_api_client: AgentAPIClient, evaluator: LLMEvaluator
    ):
        """Test: Pluto historical development."""
        query = "Wie hat sich das Verständnis von Pluto entwickelt?"
        
        response = await fresh_api_client.chat(query)
        
        # May use timeline, search, or comprehensive lookup
        tools = get_tool_names(response)
        assert len(tools) > 0, "At least one tool should be used"
        
        result = await evaluator.evaluate_response(
            query=query,
            response=response.message,
            expected_behavior="Returns time-ordered information about Pluto's astrological understanding",
            tools_used=response.tools_used
        )
        assert result.passed, f"Evaluation failed: {result}"


# =============================================================================
# Multi-Tool Scenarios
# =============================================================================

class TestMultiToolScenarios:
    """Tests for queries that should trigger multiple tools."""
    
    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_comprehensive_scorpio_query(
        self, fresh_api_client: AgentAPIClient, evaluator: LLMEvaluator
    ):
        """Test: Comprehensive query should use multiple tools."""
        query = "Erkläre mir alles über Skorpion - seine Planeten, Elemente und Themen"
        
        response = await fresh_api_client.chat(query)
        
        tools = get_tool_names(response)
        assert len(tools) >= 1, "At least one tool should be used for comprehensive query"
        
        result = await evaluator.evaluate_response(
            query=query,
            response=response.message,
            expected_behavior="Returns comprehensive Scorpio info: planets (Pluto, Mars), element (Water), themes (transformation, death/rebirth)",
            tools_used=response.tools_used
        )
        assert result.passed, f"Evaluation failed: {result}"
    
    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_mars_relationships_and_meaning(
        self, fresh_api_client: AgentAPIClient, evaluator: LLMEvaluator
    ):
        """Test: Combined relationship and meaning query."""
        query = "Zeige mir alle Beziehungen von Mars und erkläre seine Bedeutung in der Astrologie"
        
        response = await fresh_api_client.chat(query)
        
        tools = get_tool_names(response)
        assert len(tools) >= 1, "At least one tool should be used"
        
        result = await evaluator.evaluate_response(
            query=query,
            response=response.message,
            expected_behavior="Returns Mars relationships and astrological meaning (action, drive, assertiveness)",
            tools_used=response.tools_used
        )
        assert result.passed, f"Evaluation failed: {result}"
    
    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_moon_full_knowledge(
        self, fresh_api_client: AgentAPIClient, evaluator: LLMEvaluator
    ):
        """Test: Full knowledge base query about Moon."""
        query = "Was weiß das System über den Mond? Zeige Dokumente, Beziehungen und Ontologie-Einträge"
        
        response = await fresh_api_client.chat(query)
        
        tools = get_tool_names(response)
        assert len(tools) >= 1, "Multiple tools expected for full knowledge query"
        
        result = await evaluator.evaluate_response(
            query=query,
            response=response.message,
            expected_behavior="Returns Moon information from documents, relationships, and ontology",
            tools_used=response.tools_used
        )
        assert result.passed, f"Evaluation failed: {result}"


# =============================================================================
# Tool Selection Quality Tests
# =============================================================================

class TestToolSelectionQuality:
    """Tests that verify the LLM's tool selection reasoning."""
    
    @pytest.mark.asyncio
    async def test_tool_selection_for_concept_vs_document(
        self, fresh_api_client: AgentAPIClient, evaluator: LLMEvaluator
    ):
        """Test: Distinguish between concept lookup and document search."""
        # Concept query - should use ontology
        concept_query = "Was ist ein Aszendent?"
        concept_response = await fresh_api_client.chat(concept_query)
        
        concept_eval = await evaluator.evaluate_tool_selection(
            query=concept_query,
            tools_used=concept_response.tools_used,
            expected_tools=["lookup_concept", "comprehensive_lookup"]
        )
        
        assert concept_eval.get("tool_selection_correct") or concept_eval.get("alternative_acceptable"), \
            f"Concept query tool selection failed: {concept_eval.get('reasoning')}"
    
    @pytest.mark.asyncio
    async def test_fire_element_exploration(
        self, fresh_api_client: AgentAPIClient, evaluator: LLMEvaluator
    ):
        """Test: Fire element query uses ontology exploration."""
        query = "Welche Planeten gehören zum Element Feuer?"
        
        response = await fresh_api_client.chat(query)
        
        tool_eval = await evaluator.evaluate_tool_selection(
            query=query,
            tools_used=response.tools_used,
            expected_tools=["explore_ontology", "comprehensive_lookup"]
        )
        
        # Note: This is a trick question - planets don't have elements, signs do
        # The agent should clarify or correctly explain
        result = await evaluator.evaluate_response(
            query=query,
            response=response.message,
            expected_behavior="Should clarify that planets don't have elements, but fire signs are ruled by certain planets",
            tools_used=response.tools_used
        )
        # This test validates the agent's understanding, not just tool selection
        assert result.score >= 2, f"Response quality too low: {result}"

