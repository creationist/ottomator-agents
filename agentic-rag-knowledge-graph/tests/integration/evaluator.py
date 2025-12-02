"""
LLM-based evaluator for response quality.

Uses GPT-4o to score agent responses for:
- Relevance to the query
- Coherence and quality
- Language correctness (German)
- Appropriate tool usage
"""

import json
import logging
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, field
from openai import AsyncOpenAI

logger = logging.getLogger(__name__)


@dataclass
class EvaluationResult:
    """Result of LLM evaluation."""
    relevant: bool
    score: int  # 1-5
    coherent: bool
    language_correct: bool
    tool_appropriate: bool
    reasoning: str
    raw_response: Optional[str] = None
    
    @property
    def passed(self) -> bool:
        """Overall pass/fail based on key criteria."""
        return self.relevant and self.score >= 3 and self.coherent
    
    def __str__(self) -> str:
        status = "✅ PASS" if self.passed else "❌ FAIL"
        return f"{status} (score={self.score}/5, relevant={self.relevant}): {self.reasoning}"


EVALUATION_PROMPT = """Du bist ein Qualitätsprüfer für einen Astrologie-Assistenten namens Nyah.

Bewerte die folgende Antwort anhand dieser Kriterien:

## Anfrage
{query}

## Erwartetes Verhalten
{expected_behavior}

## Verwendete Tools
{tools_used}

## Antwort des Assistenten
{response}

---

Bewerte auf einer Skala von 1-5:
- 1 = Völlig unpassend oder falsch
- 2 = Größtenteils unpassend
- 3 = Akzeptabel, aber verbesserungswürdig
- 4 = Gut und hilfreich
- 5 = Exzellent und vollständig

Antworte NUR mit einem JSON-Objekt (keine Markdown-Formatierung):
{{
    "relevant": true/false,
    "score": 1-5,
    "coherent": true/false,
    "language_correct": true/false,
    "tool_appropriate": true/false,
    "reasoning": "Kurze Begründung auf Deutsch"
}}"""


TOOL_EVALUATION_PROMPT = """Analysiere, ob die richtigen Tools für diese Anfrage verwendet wurden.

## Anfrage
{query}

## Verwendete Tools
{tools_used}

## Erwartete Tools (laut Testspezifikation)
{expected_tools}

## Verfügbare Tools im System
- search: Hybrid-Suche in Dokumenten
- comprehensive_lookup: Kombinierte Wissensabfrage (Ontologie + Dokumente)
- lookup_concept: Ontologie-Konzeptabfrage
- explore_ontology: Graph-Traversierung der Ontologie
- graph_search: Wissensgraph-Suche
- get_document: Einzelnes Dokument abrufen
- list_documents: Dokumentenliste abrufen
- get_entity_relationships: Entitätsbeziehungen
- get_entity_timeline: Zeitliche Entitätsinformationen
- search_document_facts: Fakten aus Dokumenten suchen

Antworte NUR mit einem JSON-Objekt:
{{
    "tool_selection_correct": true/false,
    "reasoning": "Begründung",
    "alternative_acceptable": true/false
}}"""


class LLMEvaluator:
    """Evaluates agent responses using GPT-4o."""
    
    def __init__(self, api_key: str, model: str = "gpt-4o"):
        self.client = AsyncOpenAI(api_key=api_key)
        self.model = model
    
    async def evaluate_response(
        self,
        query: str,
        response: str,
        expected_behavior: str,
        tools_used: Optional[List[Dict[str, Any]]] = None
    ) -> EvaluationResult:
        """
        Evaluate an agent response for quality.
        
        Args:
            query: The user's original query
            response: The agent's response text
            expected_behavior: Description of what the response should do
            tools_used: List of tools that were used
        
        Returns:
            EvaluationResult with scores and reasoning
        """
        tools_str = self._format_tools(tools_used) if tools_used else "Keine Tools verwendet"
        
        prompt = EVALUATION_PROMPT.format(
            query=query,
            expected_behavior=expected_behavior,
            tools_used=tools_str,
            response=response[:2000]  # Truncate very long responses
        )
        
        try:
            completion = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "Du bist ein präziser Qualitätsprüfer. Antworte nur mit validem JSON."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=500
            )
            
            raw_response = completion.choices[0].message.content
            result = self._parse_evaluation(raw_response)
            result.raw_response = raw_response
            return result
            
        except Exception as e:
            logger.error(f"Evaluation failed: {e}")
            return EvaluationResult(
                relevant=False,
                score=0,
                coherent=False,
                language_correct=False,
                tool_appropriate=False,
                reasoning=f"Evaluation error: {str(e)}"
            )
    
    async def evaluate_tool_selection(
        self,
        query: str,
        tools_used: List[Dict[str, Any]],
        expected_tools: List[str]
    ) -> Dict[str, Any]:
        """
        Evaluate if the correct tools were selected.
        
        Args:
            query: The user's query
            tools_used: List of tools that were used
            expected_tools: List of expected tool names
        
        Returns:
            Dict with tool_selection_correct, reasoning, alternative_acceptable
        """
        tools_str = self._format_tools(tools_used)
        
        prompt = TOOL_EVALUATION_PROMPT.format(
            query=query,
            tools_used=tools_str,
            expected_tools=", ".join(expected_tools)
        )
        
        try:
            completion = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "Du bist ein präziser Qualitätsprüfer. Antworte nur mit validem JSON."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=300
            )
            
            raw_response = completion.choices[0].message.content
            return self._parse_json_response(raw_response)
            
        except Exception as e:
            logger.error(f"Tool evaluation failed: {e}")
            return {
                "tool_selection_correct": False,
                "reasoning": f"Evaluation error: {str(e)}",
                "alternative_acceptable": False
            }
    
    def _format_tools(self, tools: List[Dict[str, Any]]) -> str:
        """Format tools list for prompt."""
        if not tools:
            return "Keine Tools verwendet"
        
        formatted = []
        for t in tools:
            name = t.get("tool_name", "unknown")
            args = t.get("args", {})
            args_str = ", ".join(f"{k}={repr(v)[:50]}" for k, v in args.items())
            formatted.append(f"- {name}({args_str})")
        
        return "\n".join(formatted)
    
    def _parse_evaluation(self, raw_response: str) -> EvaluationResult:
        """Parse LLM response into EvaluationResult."""
        try:
            # Clean up response (remove markdown code blocks if present)
            cleaned = raw_response.strip()
            if cleaned.startswith("```"):
                cleaned = cleaned.split("```")[1]
                if cleaned.startswith("json"):
                    cleaned = cleaned[4:]
            cleaned = cleaned.strip()
            
            data = json.loads(cleaned)
            
            return EvaluationResult(
                relevant=data.get("relevant", False),
                score=int(data.get("score", 0)),
                coherent=data.get("coherent", False),
                language_correct=data.get("language_correct", True),
                tool_appropriate=data.get("tool_appropriate", False),
                reasoning=data.get("reasoning", "No reasoning provided")
            )
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            logger.warning(f"Failed to parse evaluation response: {e}")
            return EvaluationResult(
                relevant=False,
                score=0,
                coherent=False,
                language_correct=False,
                tool_appropriate=False,
                reasoning=f"Parse error: {raw_response[:200]}"
            )
    
    def _parse_json_response(self, raw_response: str) -> Dict[str, Any]:
        """Parse JSON response from LLM."""
        try:
            cleaned = raw_response.strip()
            if cleaned.startswith("```"):
                cleaned = cleaned.split("```")[1]
                if cleaned.startswith("json"):
                    cleaned = cleaned[4:]
            cleaned = cleaned.strip()
            
            return json.loads(cleaned)
        except json.JSONDecodeError:
            return {
                "error": "Failed to parse response",
                "raw": raw_response[:200]
            }


