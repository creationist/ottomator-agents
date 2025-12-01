"""
Ontology-based knowledge graph builder for astrological entities.

ZERO LLM API calls - uses regex matching against predefined astrology ontology.
Creates MENTIONS and CO_OCCURS relationships in Neo4j.

Best for: Document ingestion, predictable costs, fast processing.
For conversational memory with rich extraction, use GraphBuilder from graphiti_builder.py.
"""

import os
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime

from neo4j import AsyncGraphDatabase
from dotenv import load_dotenv

from .chunker import DocumentChunk

# Import ontology
try:
    from ..knowledge.ontology_utils import AstrologyOntology
except ImportError:
    # For direct execution or testing
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from knowledge.ontology_utils import AstrologyOntology

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)

# Suppress neo4j notifications about existing indexes/constraints
logging.getLogger("neo4j.notifications").setLevel(logging.WARNING)


class OntologyGraphBuilder:
    """
    Graph builder using predefined ontology for entity extraction.
    
    ZERO LLM API calls - uses regex matching against ontology keywords.
    Creates MENTIONS relationships between chunks and ontology entities in Neo4j.
    
    Best for: Document ingestion with predictable costs and fast processing.
    """
    
    def __init__(self):
        """Initialize ontology-based graph builder."""
        self.ontology = AstrologyOntology()
        self._driver = None
        self._initialized = False
        
        # Neo4j connection config
        self.neo4j_uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
        self.neo4j_user = os.getenv("NEO4J_USER", "neo4j")
        self.neo4j_password = os.getenv("NEO4J_PASSWORD")
        
        logger.info(f"OntologyGraphBuilder initialized with {self.ontology.entity_count} entities")
    
    async def initialize(self):
        """Initialize Neo4j connection."""
        if self._initialized:
            return
        
        if not self.neo4j_password:
            raise ValueError("NEO4J_PASSWORD environment variable not set")
        
        self._driver = AsyncGraphDatabase.driver(
            self.neo4j_uri, 
            auth=(self.neo4j_user, self.neo4j_password)
        )
        self._initialized = True
        logger.info("OntologyGraphBuilder connected to Neo4j")
    
    async def close(self):
        """Close Neo4j connection."""
        if self._driver:
            await self._driver.close()
            self._driver = None
            self._initialized = False
    
    async def extract_entities_from_chunks(
        self,
        chunks: List[DocumentChunk],
        **kwargs  # Accept but ignore old parameters for compatibility
    ) -> List[DocumentChunk]:
        """
        Extract entities from chunks using ontology matching (ZERO LLM calls).
        
        Args:
            chunks: List of document chunks
            
        Returns:
            Chunks with ontology entity metadata added
        """
        logger.info(f"Extracting ontology entities from {len(chunks)} chunks (zero LLM calls)")
        
        enriched_chunks = []
        total_entities_found = 0
        
        for chunk in chunks:
            # Match chunk content against ontology keywords
            matched_entities = self.ontology.match_keywords(chunk.content)
            
            # Group entities by type
            entities_by_type = {}
            for entity in matched_entities:
                entity_type = entity.type
                if entity_type not in entities_by_type:
                    entities_by_type[entity_type] = []
                entities_by_type[entity_type].append({
                    "id": entity.id,
                    "name": entity.name,
                    "description": entity.description
                })
            
            total_entities_found += len(matched_entities)
            
            # Create enriched chunk
            enriched_chunk = DocumentChunk(
                content=chunk.content,
                index=chunk.index,
                start_char=chunk.start_char,
                end_char=chunk.end_char,
                metadata={
                    **chunk.metadata,
                    "ontology_entities": entities_by_type,
                    "ontology_entity_ids": [e.id for e in matched_entities],
                    "ontology_entity_count": len(matched_entities),
                    "entity_extraction_method": "ontology_matching",
                    "entity_extraction_date": datetime.now().isoformat()
                },
                token_count=chunk.token_count
            )
            
            # Preserve embedding if it exists
            if hasattr(chunk, 'embedding'):
                enriched_chunk.embedding = chunk.embedding
            
            enriched_chunks.append(enriched_chunk)
        
        logger.info(f"Ontology entity extraction complete: {total_entities_found} total entities found across {len(chunks)} chunks")
        return enriched_chunks
    
    async def add_document_to_graph(
        self,
        chunks: List[DocumentChunk],
        document_title: str,
        document_source: str,
        document_metadata: Optional[Dict[str, Any]] = None,
        track_co_occurrences: bool = True,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Link document chunks to ontology entities in Neo4j via MENTIONS relationships.
        
        This creates:
        - Chunk nodes (for each document chunk)
        - MENTIONS relationships from chunks to OntologyEntity nodes
        - CO_OCCURS relationships between entities that appear in the same chunk
        
        ZERO LLM API calls - all entity matching is done via ontology keywords.
        
        Args:
            chunks: List of document chunks with entity metadata
            document_title: Title of the document
            document_source: Source path of the document
            document_metadata: Additional metadata
            track_co_occurrences: Whether to create CO_OCCURS relationships
        """
        if not self._initialized:
            await self.initialize()
        
        if not chunks:
            return {"chunks_linked": 0, "mentions_created": 0, "co_occurrences_created": 0, "errors": []}
        
        logger.info(f"Linking {len(chunks)} chunks to ontology entities for: {document_title}")
        
        chunks_linked = 0
        mentions_created = 0
        co_occurrences_created = 0
        errors = []
        
        async with self._driver.session() as session:
            for chunk in chunks:
                try:
                    # Get ontology entity IDs from chunk metadata
                    entity_ids = chunk.metadata.get("ontology_entity_ids", [])
                    
                    if not entity_ids:
                        # No entities found in this chunk
                        chunks_linked += 1
                        continue
                    
                    # Create a Chunk node and link it to ontology entities
                    chunk_id = f"{document_source}_chunk_{chunk.index}"
                    
                    # Create Chunk node
                    await session.run("""
                        MERGE (c:Chunk {id: $chunk_id})
                        SET c.document_title = $doc_title,
                            c.document_source = $doc_source,
                            c.chunk_index = $chunk_index,
                            c.content_preview = $preview,
                            c.entity_count = $entity_count,
                            c.created_at = datetime()
                    """, {
                        "chunk_id": chunk_id,
                        "doc_title": document_title,
                        "doc_source": document_source,
                        "chunk_index": chunk.index,
                        "preview": chunk.content[:200] + "..." if len(chunk.content) > 200 else chunk.content,
                        "entity_count": len(entity_ids)
                    })
                    
                    # Create MENTIONS relationships to each matched entity
                    for entity_id in entity_ids:
                        await session.run("""
                            MATCH (c:Chunk {id: $chunk_id})
                            MATCH (e:OntologyEntity {id: $entity_id})
                            MERGE (c)-[r:MENTIONS]->(e)
                            SET r.document_source = $doc_source
                        """, {
                            "chunk_id": chunk_id,
                            "entity_id": entity_id,
                            "doc_source": document_source
                        })
                        mentions_created += 1
                    
                    # Create CO_OCCURS relationships between entity pairs in this chunk
                    if track_co_occurrences and len(entity_ids) >= 2:
                        co_occurrences = await self._create_co_occurrences(
                            session, entity_ids, chunk_id
                        )
                        co_occurrences_created += co_occurrences
                    
                    chunks_linked += 1
                    
                except Exception as e:
                    error_msg = f"Failed to link chunk {chunk.index}: {str(e)}"
                    logger.error(error_msg)
                    errors.append(error_msg)
        
        result = {
            "chunks_linked": chunks_linked,
            "mentions_created": mentions_created,
            "co_occurrences_created": co_occurrences_created,
            "total_chunks": len(chunks),
            "errors": errors
        }
        
        logger.info(f"Graph linking complete: {chunks_linked} chunks linked, {mentions_created} MENTIONS, {co_occurrences_created} CO_OCCURS relationships")
        return result
    
    async def _create_co_occurrences(
        self,
        session,
        entity_ids: List[str],
        chunk_id: str
    ) -> int:
        """
        Create CO_OCCURS relationships between entities that appear together in a chunk.
        
        For each pair of entities in the chunk, creates or updates a CO_OCCURS relationship
        with an incremented count. This enables queries like "what concepts frequently
        appear together in the knowledge base".
        
        Args:
            session: Neo4j session
            entity_ids: List of entity IDs found in the chunk
            chunk_id: ID of the chunk where co-occurrence was found
            
        Returns:
            Number of co-occurrence relationships created/updated
        """
        co_occurrences_created = 0
        
        # Create pairs (avoid duplicates by ensuring e1 < e2 alphabetically)
        entity_ids_sorted = sorted(entity_ids)
        
        for i, entity1 in enumerate(entity_ids_sorted):
            for entity2 in entity_ids_sorted[i + 1:]:
                try:
                    # Create or update CO_OCCURS relationship with count
                    # Using MERGE to increment count if relationship exists
                    await session.run("""
                        MATCH (e1:OntologyEntity {id: $entity1})
                        MATCH (e2:OntologyEntity {id: $entity2})
                        MERGE (e1)-[r:CO_OCCURS]-(e2)
                        ON CREATE SET r.count = 1, r.chunk_ids = [$chunk_id]
                        ON MATCH SET r.count = r.count + 1,
                                     r.chunk_ids = CASE 
                                         WHEN size(r.chunk_ids) < 100 
                                         THEN r.chunk_ids + $chunk_id 
                                         ELSE r.chunk_ids 
                                     END
                    """, {
                        "entity1": entity1,
                        "entity2": entity2,
                        "chunk_id": chunk_id
                    })
                    co_occurrences_created += 1
                except Exception as e:
                    logger.warning(f"Failed to create CO_OCCURS for {entity1}-{entity2}: {e}")
        
        return co_occurrences_created
    
    async def clear_graph(self, clear_co_occurrences: bool = True):
        """
        Clear chunk nodes and relationships (preserves ontology entities).
        
        Args:
            clear_co_occurrences: If True, also clears CO_OCCURS relationships
        """
        if not self._initialized:
            await self.initialize()
        
        logger.warning("Clearing Chunk nodes and MENTIONS relationships...")
        async with self._driver.session() as session:
            # Clear chunk nodes (automatically removes MENTIONS relationships)
            await session.run("MATCH (c:Chunk) DETACH DELETE c")
            
            # Optionally clear CO_OCCURS relationships between ontology entities
            if clear_co_occurrences:
                logger.warning("Clearing CO_OCCURS relationships...")
                await session.run("MATCH (:OntologyEntity)-[r:CO_OCCURS]-(:OntologyEntity) DELETE r")
        
        logger.info("Chunk nodes and relationships cleared (ontology entities preserved)")


# Factory function
def create_ontology_graph_builder() -> OntologyGraphBuilder:
    """Create ontology-based graph builder (zero LLM calls, use for document ingestion)."""
    return OntologyGraphBuilder()

