"""
Graphiti-based knowledge graph builder for astrological entities.

Uses LLM-based entity extraction via Graphiti for conversational memory.
Extracts: planets, zodiac signs, houses, aspects, and astrological themes.

For zero-LLM document ingestion, use OntologyGraphBuilder from ontology_builder.py instead.
"""

import os
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime, timezone
import asyncio
import re

from dotenv import load_dotenv

from .chunker import DocumentChunk

# Import graph utilities
try:
    from ..agent.graph_utils import GraphitiClient
except ImportError:
    # For direct execution or testing
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from agent.graph_utils import GraphitiClient

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)


class GraphBuilder:
    """
    Builds knowledge graph from document chunks using Graphiti (LLM-based).
    
    Best for: Conversational memory, rich entity extraction.
    Trade-off: Expensive (LLM API calls), slower processing.
    """
    
    def __init__(self):
        """Initialize graph builder."""
        self.graph_client = GraphitiClient()
        self._initialized = False
    
    async def initialize(self):
        """Initialize graph client."""
        if not self._initialized:
            await self.graph_client.initialize()
            self._initialized = True
    
    async def close(self):
        """Close graph client."""
        if self._initialized:
            await self.graph_client.close()
            self._initialized = False
    
    async def add_document_to_graph(
        self,
        chunks: List[DocumentChunk],
        document_title: str,
        document_source: str,
        document_metadata: Optional[Dict[str, Any]] = None,
        batch_size: int = 3  # Reduced batch size for Graphiti
    ) -> Dict[str, Any]:
        """
        Add document chunks to the knowledge graph.
        
        Args:
            chunks: List of document chunks
            document_title: Title of the document
            document_source: Source of the document
            document_metadata: Additional metadata
            batch_size: Number of chunks to process in each batch
        
        Returns:
            Processing results
        """
        if not self._initialized:
            await self.initialize()
        
        if not chunks:
            return {"episodes_created": 0, "errors": []}
        
        logger.info(f"Adding {len(chunks)} chunks to knowledge graph for document: {document_title}")
        logger.info("⚠️ Large chunks will be truncated to avoid Graphiti token limits.")
        
        # Check for oversized chunks and warn
        oversized_chunks = [i for i, chunk in enumerate(chunks) if len(chunk.content) > 6000]
        if oversized_chunks:
            logger.warning(f"Found {len(oversized_chunks)} chunks over 6000 chars that will be truncated: {oversized_chunks}")
        
        episodes_created = 0
        errors = []
        
        # Process chunks one by one with retry logic
        for i, chunk in enumerate(chunks):
            max_retries = 3
            retry_count = 0
            success = False
            
            while not success and retry_count < max_retries:
                try:
                    # Create episode ID
                    episode_id = f"{document_source}_{chunk.index}_{datetime.now().timestamp()}"
                    
                    # Prepare episode content with size limits
                    episode_content = self._prepare_episode_content(
                        chunk,
                        document_title,
                        document_metadata
                    )
                    
                    # Create source description (shorter)
                    source_description = f"Document: {document_title} (Chunk: {chunk.index})"
                    
                    # Add episode to graph
                    await self.graph_client.add_episode(
                        episode_id=episode_id,
                        content=episode_content,
                        source=source_description,
                        timestamp=datetime.now(timezone.utc),
                        metadata={
                            "document_title": document_title,
                            "document_source": document_source,
                            "chunk_index": chunk.index,
                            "original_length": len(chunk.content),
                            "processed_length": len(episode_content)
                        }
                    )
                    
                    episodes_created += 1
                    logger.info(f"✓ Added episode {episode_id} to knowledge graph ({episodes_created}/{len(chunks)})")
                    success = True
                    
                    # Delay between episodes to avoid rate limiting
                    # gpt-4o has stricter rate limits - use 5 seconds
                    if i < len(chunks) - 1:
                        await asyncio.sleep(5.0)
                    
                except Exception as e:
                    retry_count += 1
                    error_str = str(e).lower()
                    
                    if "rate limit" in error_str:
                        wait_time = 60 * retry_count  # 60s, 120s, 180s
                        logger.warning(f"Rate limit hit on chunk {chunk.index}, waiting {wait_time}s (retry {retry_count}/{max_retries})...")
                        await asyncio.sleep(wait_time)
                    else:
                        # Non-rate-limit error
                        logger.error(f"Error on chunk {chunk.index}: {e}")
                        if retry_count < max_retries:
                            logger.info(f"Retrying chunk {chunk.index} in 10s (retry {retry_count}/{max_retries})...")
                            await asyncio.sleep(10)
            
            if not success:
                error_msg = f"Failed to add chunk {chunk.index} after {max_retries} retries"
                logger.error(error_msg)
                errors.append(error_msg)
        
        result = {
            "episodes_created": episodes_created,
            "total_chunks": len(chunks),
            "errors": errors
        }
        
        logger.info(f"Graph building complete: {episodes_created} episodes created, {len(errors)} errors")
        return result
    
    def _prepare_episode_content(
        self,
        chunk: DocumentChunk,
        document_title: str,
        document_metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Prepare episode content with minimal context to avoid token limits.
        
        Args:
            chunk: Document chunk
            document_title: Title of the document
            document_metadata: Additional metadata
        
        Returns:
            Formatted episode content (optimized for Graphiti)
        """
        # Limit chunk content to avoid Graphiti's 8192 token limit
        # Estimate ~4 chars per token, keep content under 6000 chars to leave room for processing
        max_content_length = 6000
        
        content = chunk.content
        if len(content) > max_content_length:
            # Truncate content but try to end at a sentence boundary
            truncated = content[:max_content_length]
            last_sentence_end = max(
                truncated.rfind('. '),
                truncated.rfind('! '),
                truncated.rfind('? ')
            )
            
            if last_sentence_end > max_content_length * 0.7:  # If we can keep 70% and end cleanly
                content = truncated[:last_sentence_end + 1] + " [TRUNCATED]"
            else:
                content = truncated + "... [TRUNCATED]"
            
            logger.warning(f"Truncated chunk {chunk.index} from {len(chunk.content)} to {len(content)} chars for Graphiti")
        
        # Add minimal context (just document title for now)
        if document_title and len(content) < max_content_length - 100:
            episode_content = f"[Doc: {document_title[:50]}]\n\n{content}"
        else:
            episode_content = content
        
        return episode_content
    
    def _estimate_tokens(self, text: str) -> int:
        """Rough estimate of token count (4 chars per token)."""
        return len(text) // 4
    
    def _is_content_too_large(self, content: str, max_tokens: int = 7000) -> bool:
        """Check if content is too large for Graphiti processing."""
        return self._estimate_tokens(content) > max_tokens
    
    async def extract_entities_from_chunks(
        self,
        chunks: List[DocumentChunk],
        extract_planets: bool = True,
        extract_signs: bool = True,
        extract_concepts: bool = True
    ) -> List[DocumentChunk]:
        """
        Extract astrological entities from chunks and add to metadata.
        
        Args:
            chunks: List of document chunks
            extract_planets: Whether to extract planet names
            extract_signs: Whether to extract zodiac sign names
            extract_concepts: Whether to extract astrological concepts
        
        Returns:
            Chunks with entity metadata added
        """
        logger.info(f"Extracting astrological entities from {len(chunks)} chunks")
        
        enriched_chunks = []
        
        for chunk in chunks:
            entities = {
                "planets": [],
                "signs": [],
                "houses": [],
                "aspects": [],
                "themes": []
            }
            
            content = chunk.content
            
            # Extract planets
            if extract_planets:
                entities["planets"] = self._extract_planets(content)
            
            # Extract zodiac signs
            if extract_signs:
                entities["signs"] = self._extract_zodiac_signs(content)
            
            # Extract astrological concepts
            if extract_concepts:
                entities["houses"] = self._extract_houses(content)
                entities["aspects"] = self._extract_aspects(content)
                entities["themes"] = self._extract_themes(content)
            
            # Create enriched chunk
            enriched_chunk = DocumentChunk(
                content=chunk.content,
                index=chunk.index,
                start_char=chunk.start_char,
                end_char=chunk.end_char,
                metadata={
                    **chunk.metadata,
                    "entities": entities,
                    "entity_extraction_date": datetime.now().isoformat()
                },
                token_count=chunk.token_count
            )
            
            # Preserve embedding if it exists
            if hasattr(chunk, 'embedding'):
                enriched_chunk.embedding = chunk.embedding
            
            enriched_chunks.append(enriched_chunk)
        
        logger.info("Astrological entity extraction complete")
        return enriched_chunks
    
    def _extract_planets(self, text: str) -> List[str]:
        """Extract planet names from text (German and English)."""
        planets = {
            # German names
            "Sonne", "Mond", "Merkur", "Venus", "Mars", "Jupiter", 
            "Saturn", "Uranus", "Neptun", "Pluto", "Chiron",
            # English names
            "Sun", "Moon", "Mercury", "Neptune",
            # Lunar nodes
            "Nordknoten", "Südknoten", "North Node", "South Node",
            "Mondknoten"
        }
        
        found_planets = set()
        text_lower = text.lower()
        
        for planet in planets:
            pattern = r'\b' + re.escape(planet.lower()) + r'\b'
            if re.search(pattern, text_lower):
                found_planets.add(planet)
        
        return list(found_planets)
    
    def _extract_zodiac_signs(self, text: str) -> List[str]:
        """Extract zodiac sign names from text (German and English)."""
        signs = {
            # German names
            "Widder", "Stier", "Zwillinge", "Krebs", "Löwe", "Jungfrau",
            "Waage", "Skorpion", "Schütze", "Steinbock", "Wassermann", "Fische",
            # English names
            "Aries", "Taurus", "Gemini", "Cancer", "Leo", "Virgo",
            "Libra", "Scorpio", "Sagittarius", "Capricorn", "Aquarius", "Pisces"
        }
        
        found_signs = set()
        text_lower = text.lower()
        
        for sign in signs:
            pattern = r'\b' + re.escape(sign.lower()) + r'\b'
            if re.search(pattern, text_lower):
                found_signs.add(sign)
        
        return list(found_signs)
    
    def _extract_houses(self, text: str) -> List[str]:
        """Extract astrological house references from text."""
        houses = {
            "erstes Haus", "zweites Haus", "drittes Haus", "viertes Haus",
            "fünftes Haus", "sechstes Haus", "siebtes Haus", "achtes Haus",
            "neuntes Haus", "zehntes Haus", "elftes Haus", "zwölftes Haus",
            "1. Haus", "2. Haus", "3. Haus", "4. Haus", "5. Haus", "6. Haus",
            "7. Haus", "8. Haus", "9. Haus", "10. Haus", "11. Haus", "12. Haus",
            "Aszendent", "Deszendent", "Medium Coeli", "MC", "Imum Coeli", "IC"
        }
        
        found_houses = set()
        text_lower = text.lower()
        
        for house in houses:
            if house.lower() in text_lower:
                found_houses.add(house)
        
        return list(found_houses)
    
    def _extract_aspects(self, text: str) -> List[str]:
        """Extract astrological aspect names from text."""
        aspects = {
            "Konjunktion", "Sextil", "Quadrat", "Trigon", "Opposition",
            "Halbsextil", "Quinkunx", "Quincunx",
            "Conjunction", "Sextile", "Square", "Trine"
        }
        
        found_aspects = set()
        text_lower = text.lower()
        
        for aspect in aspects:
            pattern = r'\b' + re.escape(aspect.lower()) + r'\b'
            if re.search(pattern, text_lower):
                found_aspects.add(aspect)
        
        return list(found_aspects)
    
    def _extract_themes(self, text: str) -> List[str]:
        """Extract astrological themes and concepts from text."""
        themes = {
            # Life themes (German)
            "Transformation", "Heilung", "Beziehung", "Partnerschaft",
            "Kreativität", "Spiritualität", "Karma", "Berufung",
            "Familie", "Kommunikation", "Finanzen", "Gesundheit",
            # Concepts
            "Rückläufigkeit", "Retrograde", "Vollmond", "Neumond",
            "Zunehmender Mond", "Abnehmender Mond",
            # Elements & modalities
            "Feuer", "Erde", "Luft", "Wasser",
            "Kardinal", "Fix", "Veränderlich",
            # Additional concepts
            "Horoskop", "Geburtshoroskop", "Transit", "Synastrie",
            "Aspekt", "Planetenstellung", "Tierkreis"
        }
        
        found_themes = set()
        text_lower = text.lower()
        
        for theme in themes:
            pattern = r'\b' + re.escape(theme.lower()) + r'\b'
            if re.search(pattern, text_lower):
                found_themes.add(theme)
        
        return list(found_themes)
    
    async def clear_graph(self):
        """Clear all data from the knowledge graph."""
        if not self._initialized:
            await self.initialize()
        
        logger.warning("Clearing knowledge graph...")
        await self.graph_client.clear_graph()
        logger.info("Knowledge graph cleared")


class SimpleEntityExtractor:
    """Simple rule-based entity extractor for astrology content."""
    
    def __init__(self):
        """Initialize extractor with astrology patterns."""
        # Planet patterns (German and English)
        self.planet_patterns = [
            r'\b(?:Sonne|Mond|Merkur|Venus|Mars|Jupiter|Saturn|Uranus|Neptun|Pluto|Chiron)\b',
            r'\b(?:Sun|Moon|Mercury|Neptune)\b',
            r'\b(?:Nordknoten|Südknoten|Mondknoten)\b'
        ]
        
        # Zodiac sign patterns (German and English)
        self.sign_patterns = [
            r'\b(?:Widder|Stier|Zwillinge|Krebs|Löwe|Jungfrau|Waage|Skorpion|Schütze|Steinbock|Wassermann|Fische)\b',
            r'\b(?:Aries|Taurus|Gemini|Cancer|Leo|Virgo|Libra|Scorpio|Sagittarius|Capricorn|Aquarius|Pisces)\b'
        ]
        
        # Aspect patterns
        self.aspect_patterns = [
            r'\b(?:Konjunktion|Sextil|Quadrat|Trigon|Opposition|Quinkunx)\b',
            r'\b(?:Conjunction|Sextile|Square|Trine)\b'
        ]
        
        # Theme patterns
        self.theme_patterns = [
            r'\b(?:Transformation|Heilung|Karma|Spiritualität|Kreativität)\b',
            r'\b(?:Vollmond|Neumond|Rückläufigkeit|Retrograde)\b',
            r'\b(?:Feuer|Erde|Luft|Wasser)(?:zeichen)?\b'
        ]
    
    def extract_entities(self, text: str) -> Dict[str, List[str]]:
        """Extract astrological entities using patterns."""
        entities = {
            "planets": [],
            "signs": [],
            "aspects": [],
            "themes": []
        }
        
        # Extract planets
        for pattern in self.planet_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            entities["planets"].extend(matches)
        
        # Extract zodiac signs
        for pattern in self.sign_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            entities["signs"].extend(matches)
        
        # Extract aspects
        for pattern in self.aspect_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            entities["aspects"].extend(matches)
        
        # Extract themes
        for pattern in self.theme_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            entities["themes"].extend(matches)
        
        # Remove duplicates
        for key in entities:
            entities[key] = list(set(entities[key]))
        
        return entities


# Factory function
def create_graph_builder() -> GraphBuilder:
    """Create Graphiti-based graph builder (LLM-heavy, use for conversational memory)."""
    return GraphBuilder()


# Example usage
async def main():
    """Example usage of the graph builder with astrology content."""
    from .chunker import ChunkingConfig, create_chunker
    
    # Create chunker and graph builder
    config = ChunkingConfig(chunk_size=300, use_semantic_splitting=False)
    chunker = create_chunker(config)
    graph_builder = create_graph_builder()
    
    sample_text = """
    Venus im Stier bringt eine Zeit der Sinnlichkeit und Genussfreude. Diese 
    Planetenstellung verstärkt das Bedürfnis nach Sicherheit und materiellen 
    Werten. Beziehungen werden stabiler, aber auch besitzergreifender.
    
    Der Vollmond im Skorpion bildet eine Opposition zur Sonne im Stier und 
    bringt tiefe Transformation. Emotionen kommen an die Oberfläche, und 
    Heilung auf seelischer Ebene wird möglich. Das achte Haus wird aktiviert,
    was Themen wie Intimität und gemeinsame Ressourcen betont.
    
    Bei Merkur rückläufig sollte man besonders auf die Kommunikation achten.
    Missverständnisse können leichter entstehen, aber es ist eine gute Zeit
    für Reflexion und Überarbeitung alter Projekte.
    """
    
    # Chunk the document
    chunks = chunker.chunk_document(
        content=sample_text,
        title="Astrologische Einflüsse",
        source="astro_example.md"
    )
    
    print(f"Created {len(chunks)} chunks")
    
    # Extract entities
    enriched_chunks = await graph_builder.extract_entities_from_chunks(chunks)
    
    for i, chunk in enumerate(enriched_chunks):
        print(f"Chunk {i}: {chunk.metadata.get('entities', {})}")
    
    # Add to knowledge graph
    try:
        result = await graph_builder.add_document_to_graph(
            chunks=enriched_chunks,
            document_title="Astrologische Einflüsse",
            document_source="astro_example.md",
            document_metadata={"topic": "Astrologie", "date": "2024"}
        )
        
        print(f"Graph building result: {result}")
        
    except Exception as e:
        print(f"Graph building failed: {e}")
    
    finally:
        await graph_builder.close()


if __name__ == "__main__":
    asyncio.run(main())

