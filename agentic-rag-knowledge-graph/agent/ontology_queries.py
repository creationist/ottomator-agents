"""
Direct Neo4j Cypher queries for ontology traversal.

These queries leverage the pre-seeded ontology structure to enable
multi-hop graph traversal that Graphiti's semantic search cannot do.

Usage:
    from agent.ontology_queries import OntologyQueryClient
    
    client = OntologyQueryClient()
    await client.initialize()
    
    # Find chunks about fire signs
    chunks = await client.get_chunks_by_element("fire")
    
    # Find chunks about signs ruled by Venus
    chunks = await client.get_chunks_by_planet_rulership("venus")
"""

import os
import logging
from typing import List, Dict, Any, Optional, Set
from neo4j import AsyncGraphDatabase
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

# Suppress neo4j notifications about existing indexes/constraints
logging.getLogger("neo4j.notifications").setLevel(logging.WARNING)


class OntologyQueryClient:
    """
    Client for direct Neo4j queries against the ontology structure.
    
    Provides multi-hop traversal queries that leverage the pre-seeded
    ontology relationships (RULES, HAS_ELEMENT, HAS_MODALITY, etc.).
    """
    
    def __init__(self):
        """Initialize the query client."""
        self.neo4j_uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
        self.neo4j_user = os.getenv("NEO4J_USER", "neo4j")
        self.neo4j_password = os.getenv("NEO4J_PASSWORD")
        
        if not self.neo4j_password:
            raise ValueError("NEO4J_PASSWORD environment variable not set")
        
        self._driver = None
        self._initialized = False
    
    async def initialize(self):
        """Initialize Neo4j connection."""
        if self._initialized:
            return
        
        self._driver = AsyncGraphDatabase.driver(
            self.neo4j_uri,
            auth=(self.neo4j_user, self.neo4j_password)
        )
        self._initialized = True
        logger.info("OntologyQueryClient connected to Neo4j")
    
    async def close(self):
        """Close Neo4j connection."""
        if self._driver:
            await self._driver.close()
            self._driver = None
            self._initialized = False
    
    async def _ensure_initialized(self):
        """Ensure client is initialized before queries."""
        if not self._initialized:
            await self.initialize()
    
    # =========================================================================
    # Element-based queries
    # =========================================================================
    
    async def get_chunks_by_element(self, element: str) -> List[Dict[str, Any]]:
        """
        Find chunks mentioning zodiac signs of a specific element.
        
        Traversal: Chunk -[MENTIONS]-> Sign -[HAS_ELEMENT]-> Element
        
        Args:
            element: Element ID (fire, earth, air, water)
            
        Returns:
            List of chunks with their mentioned signs
        """
        await self._ensure_initialized()
        
        query = """
        MATCH (c:Chunk)-[m:MENTIONS]->(sign:OntologyEntity)-[he:HAS_ELEMENT]->(elem:OntologyEntity {id: $element})
        WHERE sign.type = 'sign'
        RETURN c.id as chunk_id,
               c.document_title as document_title,
               c.content_preview as content_preview,
               c.chunk_index as chunk_index,
               collect(DISTINCT sign.name) as mentioned_signs,
               elem.name as element_name
        ORDER BY c.document_title, c.chunk_index
        """
        
        async with self._driver.session() as session:
            result = await session.run(query, {"element": element.lower()})
            records = await result.data()
            
        logger.info(f"Found {len(records)} chunks mentioning {element} signs")
        return records
    
    async def get_chunks_by_modality(self, modality: str) -> List[Dict[str, Any]]:
        """
        Find chunks mentioning zodiac signs of a specific modality.
        
        Traversal: Chunk -[MENTIONS]-> Sign -[HAS_MODALITY]-> Modality
        
        Args:
            modality: Modality ID (cardinal, fixed, mutable)
            
        Returns:
            List of chunks with their mentioned signs
        """
        await self._ensure_initialized()
        
        query = """
        MATCH (c:Chunk)-[m:MENTIONS]->(sign:OntologyEntity)-[hm:HAS_MODALITY]->(mod:OntologyEntity {id: $modality})
        WHERE sign.type = 'sign'
        RETURN c.id as chunk_id,
               c.document_title as document_title,
               c.content_preview as content_preview,
               c.chunk_index as chunk_index,
               collect(DISTINCT sign.name) as mentioned_signs,
               mod.name as modality_name
        ORDER BY c.document_title, c.chunk_index
        """
        
        async with self._driver.session() as session:
            result = await session.run(query, {"modality": modality.lower()})
            records = await result.data()
            
        logger.info(f"Found {len(records)} chunks mentioning {modality} signs")
        return records
    
    # =========================================================================
    # Planet rulership queries
    # =========================================================================
    
    async def get_chunks_by_planet_rulership(self, planet: str) -> List[Dict[str, Any]]:
        """
        Find chunks mentioning signs ruled by a specific planet.
        
        Traversal: Chunk -[MENTIONS]-> Sign <-[RULES]- Planet
        
        Args:
            planet: Planet ID (sun, moon, mercury, venus, mars, jupiter, saturn, uranus, neptune, pluto)
            
        Returns:
            List of chunks with their mentioned signs
        """
        await self._ensure_initialized()
        
        query = """
        MATCH (planet:OntologyEntity {id: $planet})-[r:RULES|TRADITIONAL_RULES]->(sign:OntologyEntity)
        WHERE sign.type = 'sign'
        WITH sign
        MATCH (c:Chunk)-[m:MENTIONS]->(sign)
        RETURN c.id as chunk_id,
               c.document_title as document_title,
               c.content_preview as content_preview,
               c.chunk_index as chunk_index,
               collect(DISTINCT sign.name) as mentioned_signs,
               $planet as ruling_planet
        ORDER BY c.document_title, c.chunk_index
        """
        
        async with self._driver.session() as session:
            result = await session.run(query, {"planet": planet.lower()})
            records = await result.data()
            
        logger.info(f"Found {len(records)} chunks mentioning signs ruled by {planet}")
        return records
    
    async def get_chunks_mentioning_planet_and_ruled_signs(self, planet: str) -> List[Dict[str, Any]]:
        """
        Find chunks that mention both a planet AND any of its ruled signs.
        
        This finds content where the planet-sign relationship is likely discussed.
        
        Args:
            planet: Planet ID
            
        Returns:
            List of chunks mentioning both the planet and its signs
        """
        await self._ensure_initialized()
        
        query = """
        MATCH (planet:OntologyEntity {id: $planet})-[r:RULES|TRADITIONAL_RULES]->(sign:OntologyEntity)
        WITH planet, collect(sign) as ruled_signs
        MATCH (c:Chunk)-[:MENTIONS]->(planet)
        MATCH (c)-[:MENTIONS]->(sign)
        WHERE sign IN ruled_signs
        RETURN c.id as chunk_id,
               c.document_title as document_title,
               c.content_preview as content_preview,
               c.chunk_index as chunk_index,
               planet.name as planet_name,
               collect(DISTINCT sign.name) as co_mentioned_signs
        ORDER BY c.document_title, c.chunk_index
        """
        
        async with self._driver.session() as session:
            result = await session.run(query, {"planet": planet.lower()})
            records = await result.data()
            
        logger.info(f"Found {len(records)} chunks mentioning {planet} with its ruled signs")
        return records
    
    # =========================================================================
    # Theme-based queries
    # =========================================================================
    
    async def get_chunks_by_theme(self, theme: str) -> List[Dict[str, Any]]:
        """
        Find chunks related to a specific theme via ontology relationships.
        
        Traversal: Chunk -[MENTIONS]-> Entity <-[RELATES_TO]- Theme
        
        Args:
            theme: Theme ID (transformation, healing, relationships, creativity, etc.)
            
        Returns:
            List of chunks related to the theme
        """
        await self._ensure_initialized()
        
        query = """
        MATCH (theme:OntologyEntity {id: $theme})-[rt:RELATES_TO]->(entity:OntologyEntity)
        WITH entity
        MATCH (c:Chunk)-[m:MENTIONS]->(entity)
        RETURN c.id as chunk_id,
               c.document_title as document_title,
               c.content_preview as content_preview,
               c.chunk_index as chunk_index,
               collect(DISTINCT entity.name) as related_entities,
               $theme as theme
        ORDER BY c.document_title, c.chunk_index
        """
        
        async with self._driver.session() as session:
            result = await session.run(query, {"theme": theme.lower()})
            records = await result.data()
            
        logger.info(f"Found {len(records)} chunks related to theme: {theme}")
        return records
    
    # =========================================================================
    # House-based queries
    # =========================================================================
    
    async def get_chunks_by_house_theme(self, house: str) -> List[Dict[str, Any]]:
        """
        Find chunks mentioning entities related to a specific house.
        
        Traversal: Chunk -[MENTIONS]-> Entity <-[NATURAL_RULER|RELATES_TO]- House
        
        Args:
            house: House ID (house_1 through house_12)
            
        Returns:
            List of chunks related to the house
        """
        await self._ensure_initialized()
        
        query = """
        MATCH (house:OntologyEntity {id: $house})
        OPTIONAL MATCH (house)-[:NATURAL_RULER]->(sign:OntologyEntity)
        OPTIONAL MATCH (theme:OntologyEntity)-[:RELATES_TO]->(house)
        WITH house, sign, theme
        MATCH (c:Chunk)-[:MENTIONS]->(entity:OntologyEntity)
        WHERE entity = house OR entity = sign OR entity = theme
        RETURN c.id as chunk_id,
               c.document_title as document_title,
               c.content_preview as content_preview,
               c.chunk_index as chunk_index,
               collect(DISTINCT entity.name) as related_entities,
               house.name as house_name
        ORDER BY c.document_title, c.chunk_index
        """
        
        async with self._driver.session() as session:
            result = await session.run(query, {"house": house.lower()})
            records = await result.data()
            
        logger.info(f"Found {len(records)} chunks related to {house}")
        return records
    
    # =========================================================================
    # Generic multi-hop traversal
    # =========================================================================
    
    async def get_related_chunks_via_ontology(
        self,
        entity_id: str,
        max_hops: int = 2,
        limit: int = 20
    ) -> List[Dict[str, Any]]:
        """
        Find chunks connected to an entity through ontology relationships.
        
        Performs variable-length path traversal through the ontology,
        then finds chunks mentioning any connected entity.
        
        Args:
            entity_id: Starting entity ID
            max_hops: Maximum relationship hops (1-3)
            limit: Maximum chunks to return
            
        Returns:
            List of chunks with their connection path
        """
        await self._ensure_initialized()
        
        # Clamp hops to reasonable range
        max_hops = min(max(1, max_hops), 3)
        
        query = f"""
        MATCH (start:OntologyEntity {{id: $entity_id}})
        MATCH path = (start)-[*1..{max_hops}]-(related:OntologyEntity)
        WHERE related <> start
        WITH DISTINCT related, length(path) as distance
        MATCH (c:Chunk)-[:MENTIONS]->(related)
        RETURN c.id as chunk_id,
               c.document_title as document_title,
               c.content_preview as content_preview,
               c.chunk_index as chunk_index,
               related.name as connected_via,
               related.type as entity_type,
               distance as hops_away
        ORDER BY distance, c.document_title, c.chunk_index
        LIMIT $limit
        """
        
        async with self._driver.session() as session:
            result = await session.run(query, {"entity_id": entity_id.lower(), "limit": limit})
            records = await result.data()
            
        logger.info(f"Found {len(records)} chunks within {max_hops} hops of {entity_id}")
        return records
    
    async def get_entity_neighborhood(
        self,
        entity_id: str,
        max_hops: int = 2
    ) -> Dict[str, Any]:
        """
        Get the ontology neighborhood of an entity.
        
        Returns all entities and relationships within N hops.
        
        Args:
            entity_id: Starting entity ID
            max_hops: Maximum relationship hops
            
        Returns:
            Dictionary with entities and relationships
        """
        await self._ensure_initialized()
        
        max_hops = min(max(1, max_hops), 3)
        
        query = f"""
        MATCH (start:OntologyEntity {{id: $entity_id}})
        MATCH path = (start)-[r*1..{max_hops}]-(related:OntologyEntity)
        WITH start, related, relationships(path) as rels, length(path) as distance
        UNWIND rels as rel
        WITH start, related, rel, distance,
             startNode(rel) as rel_start, endNode(rel) as rel_end
        RETURN DISTINCT
               related.id as entity_id,
               related.name as entity_name,
               related.type as entity_type,
               distance as hops,
               type(rel) as relationship_type,
               rel_start.name as from_entity,
               rel_end.name as to_entity
        ORDER BY distance, related.name
        """
        
        async with self._driver.session() as session:
            result = await session.run(query, {"entity_id": entity_id.lower()})
            records = await result.data()
        
        # Organize results
        entities = {}
        relationships = []
        
        for record in records:
            eid = record["entity_id"]
            if eid not in entities:
                entities[eid] = {
                    "id": eid,
                    "name": record["entity_name"],
                    "type": record["entity_type"],
                    "hops": record["hops"]
                }
            
            rel = {
                "type": record["relationship_type"],
                "from": record["from_entity"],
                "to": record["to_entity"]
            }
            if rel not in relationships:
                relationships.append(rel)
        
        return {
            "center": entity_id,
            "entities": list(entities.values()),
            "relationships": relationships,
            "total_entities": len(entities),
            "total_relationships": len(relationships)
        }
    
    # =========================================================================
    # Co-occurrence queries
    # =========================================================================
    
    async def get_co_occurring_entities(
        self,
        entity_id: str,
        min_co_occurrences: int = 1,
        limit: int = 20
    ) -> List[Dict[str, Any]]:
        """
        Find entities that frequently co-occur with a given entity in chunks.
        
        Args:
            entity_id: Entity to find co-occurrences for
            min_co_occurrences: Minimum times entities must appear together
            limit: Maximum results to return
            
        Returns:
            List of co-occurring entities with counts
        """
        await self._ensure_initialized()
        
        query = """
        MATCH (e1:OntologyEntity {id: $entity_id})<-[:MENTIONS]-(c:Chunk)-[:MENTIONS]->(e2:OntologyEntity)
        WHERE e1 <> e2
        WITH e2, count(DISTINCT c) as co_occurrence_count
        WHERE co_occurrence_count >= $min_count
        RETURN e2.id as entity_id,
               e2.name as entity_name,
               e2.type as entity_type,
               co_occurrence_count
        ORDER BY co_occurrence_count DESC
        LIMIT $limit
        """
        
        async with self._driver.session() as session:
            result = await session.run(query, {
                "entity_id": entity_id.lower(),
                "min_count": min_co_occurrences,
                "limit": limit
            })
            records = await result.data()
            
        logger.info(f"Found {len(records)} entities co-occurring with {entity_id}")
        return records
    
    async def get_chunks_with_entity_pair(
        self,
        entity1: str,
        entity2: str
    ) -> List[Dict[str, Any]]:
        """
        Find chunks that mention both entities.
        
        Args:
            entity1: First entity ID
            entity2: Second entity ID
            
        Returns:
            List of chunks mentioning both entities
        """
        await self._ensure_initialized()
        
        query = """
        MATCH (e1:OntologyEntity {id: $entity1})<-[:MENTIONS]-(c:Chunk)-[:MENTIONS]->(e2:OntologyEntity {id: $entity2})
        RETURN c.id as chunk_id,
               c.document_title as document_title,
               c.content_preview as content_preview,
               c.chunk_index as chunk_index,
               e1.name as entity1_name,
               e2.name as entity2_name
        ORDER BY c.document_title, c.chunk_index
        """
        
        async with self._driver.session() as session:
            result = await session.run(query, {
                "entity1": entity1.lower(),
                "entity2": entity2.lower()
            })
            records = await result.data()
            
        logger.info(f"Found {len(records)} chunks mentioning both {entity1} and {entity2}")
        return records
    
    # =========================================================================
    # Aspect queries
    # =========================================================================
    
    async def get_chunks_by_aspect(self, aspect: str) -> List[Dict[str, Any]]:
        """
        Find chunks mentioning a specific aspect type.
        
        Args:
            aspect: Aspect ID (conjunction, sextile, square, trine, opposition, etc.)
            
        Returns:
            List of chunks mentioning the aspect
        """
        await self._ensure_initialized()
        
        query = """
        MATCH (c:Chunk)-[:MENTIONS]->(aspect:OntologyEntity {id: $aspect})
        WHERE aspect.type = 'aspect'
        RETURN c.id as chunk_id,
               c.document_title as document_title,
               c.content_preview as content_preview,
               c.chunk_index as chunk_index,
               aspect.name as aspect_name,
               aspect.description as aspect_description
        ORDER BY c.document_title, c.chunk_index
        """
        
        async with self._driver.session() as session:
            result = await session.run(query, {"aspect": aspect.lower()})
            records = await result.data()
            
        logger.info(f"Found {len(records)} chunks mentioning aspect: {aspect}")
        return records
    
    # =========================================================================
    # Statistics and overview
    # =========================================================================
    
    async def get_ontology_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about the ontology in Neo4j.
        
        Returns:
            Dictionary with entity counts, relationship counts, and chunk coverage
        """
        await self._ensure_initialized()
        
        async with self._driver.session() as session:
            # Entity counts by type
            entity_result = await session.run("""
                MATCH (e:OntologyEntity)
                RETURN e.type as type, count(*) as count
                ORDER BY count DESC
            """)
            entity_counts = {r["type"]: r["count"] for r in await entity_result.data()}
            
            # Relationship counts by type
            rel_result = await session.run("""
                MATCH (:OntologyEntity)-[r]->(:OntologyEntity)
                RETURN type(r) as type, count(*) as count
                ORDER BY count DESC
            """)
            rel_counts = {r["type"]: r["count"] for r in await rel_result.data()}
            
            # Chunk statistics
            chunk_result = await session.run("""
                MATCH (c:Chunk)
                OPTIONAL MATCH (c)-[m:MENTIONS]->(:OntologyEntity)
                WITH c, count(m) as mention_count
                RETURN count(c) as total_chunks,
                       sum(mention_count) as total_mentions,
                       avg(mention_count) as avg_mentions_per_chunk
            """)
            chunk_stats = await chunk_result.single()
            
            # Most mentioned entities
            top_entities_result = await session.run("""
                MATCH (e:OntologyEntity)<-[m:MENTIONS]-(:Chunk)
                RETURN e.name as entity, e.type as type, count(m) as mention_count
                ORDER BY mention_count DESC
                LIMIT 10
            """)
            top_entities = await top_entities_result.data()
        
        return {
            "entity_counts_by_type": entity_counts,
            "relationship_counts_by_type": rel_counts,
            "total_chunks": chunk_stats["total_chunks"] if chunk_stats else 0,
            "total_mentions": chunk_stats["total_mentions"] if chunk_stats else 0,
            "avg_mentions_per_chunk": round(chunk_stats["avg_mentions_per_chunk"] or 0, 2) if chunk_stats else 0,
            "top_mentioned_entities": top_entities
        }


# Global instance for convenience
_ontology_query_client: Optional[OntologyQueryClient] = None


async def get_ontology_query_client() -> OntologyQueryClient:
    """Get or create the global ontology query client."""
    global _ontology_query_client
    if _ontology_query_client is None:
        _ontology_query_client = OntologyQueryClient()
        await _ontology_query_client.initialize()
    return _ontology_query_client


async def close_ontology_query_client():
    """Close the global ontology query client."""
    global _ontology_query_client
    if _ontology_query_client is not None:
        await _ontology_query_client.close()
        _ontology_query_client = None
