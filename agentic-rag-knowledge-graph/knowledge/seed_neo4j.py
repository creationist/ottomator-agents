"""
Seed Neo4j with the astrology ontology.
This creates all entities and relationships WITHOUT using Graphiti's LLM processing.

Usage:
    python -m knowledge.seed_neo4j
    
    # To clear existing data first:
    python -m knowledge.seed_neo4j --clear
"""

import os
import asyncio
import argparse
import logging
from neo4j import AsyncGraphDatabase
from dotenv import load_dotenv

from .ontology_utils import AstrologyOntology

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Suppress neo4j notifications about existing indexes/constraints
logging.getLogger("neo4j.notifications").setLevel(logging.WARNING)


async def seed_ontology_to_neo4j(clear_first: bool = False, clear_all: bool = False):
    """
    Seed the astrology ontology directly into Neo4j.
    
    This bypasses Graphiti's LLM-based entity extraction and creates
    nodes/relationships directly from the predefined ontology.
    """
    # Neo4j connection
    uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    user = os.getenv("NEO4J_USER", "neo4j")
    password = os.getenv("NEO4J_PASSWORD")
    
    if not password:
        raise ValueError("NEO4J_PASSWORD environment variable not set")
    
    # Load ontology
    ontology = AstrologyOntology()
    logger.info(f"Loaded ontology: {ontology.entity_count} entities, {ontology.relationship_count} relationships")
    
    # Connect to Neo4j
    driver = AsyncGraphDatabase.driver(uri, auth=(user, password))
    
    try:
        async with driver.session() as session:
            if clear_all:
                logger.warning("⚠️ Clearing ALL Neo4j data (including Graphiti)...")
                await session.run("MATCH (n) DETACH DELETE n")
                logger.info("Cleared ALL data from Neo4j")
            elif clear_first:
                logger.warning("Clearing existing ontology nodes...")
                await session.run("""
                    MATCH (n:OntologyEntity)
                    DETACH DELETE n
                """)
                logger.info("Cleared existing ontology nodes")
            
            # Create entities as nodes
            logger.info("Creating entity nodes...")
            entities_created = 0
            
            for entity_id, entity in ontology._entities.items():
                await session.run("""
                    MERGE (e:OntologyEntity {id: $id})
                    SET e.name = $name,
                        e.type = $type,
                        e.description = $description,
                        e.keywords = $keywords
                """, {
                    "id": entity.id,
                    "name": entity.name,
                    "type": entity.type,
                    "description": entity.description,
                    "keywords": entity.keywords
                })
                entities_created += 1
            
            logger.info(f"Created {entities_created} entity nodes")
            
            # Create relationships
            logger.info("Creating relationships...")
            relationships_created = 0
            
            for rel in ontology._relationships:
                await session.run(f"""
                    MATCH (source:OntologyEntity {{id: $source}})
                    MATCH (target:OntologyEntity {{id: $target}})
                    MERGE (source)-[r:{rel.type}]->(target)
                    SET r.description = $description
                """, {
                    "source": rel.source,
                    "target": rel.target,
                    "description": rel.description
                })
                relationships_created += 1
            
            logger.info(f"Created {relationships_created} relationships")
            
            # Create indexes for better query performance
            logger.info("Creating indexes...")
            await session.run("CREATE INDEX ontology_id IF NOT EXISTS FOR (e:OntologyEntity) ON (e.id)")
            await session.run("CREATE INDEX ontology_type IF NOT EXISTS FOR (e:OntologyEntity) ON (e.type)")
            await session.run("CREATE INDEX ontology_name IF NOT EXISTS FOR (e:OntologyEntity) ON (e.name)")
            
            logger.info("Indexes created")
            
            # Print summary
            result = await session.run("""
                MATCH (e:OntologyEntity)
                RETURN e.type as type, count(*) as count
                ORDER BY count DESC
            """)
            
            print("\n" + "="*50)
            print("ONTOLOGY SEEDING COMPLETE")
            print("="*50)
            print("\nEntities by type:")
            async for record in result:
                print(f"  {record['type']}: {record['count']}")
            
            # Count relationships
            rel_result = await session.run("""
                MATCH ()-[r]->()
                WHERE startNode(r):OntologyEntity AND endNode(r):OntologyEntity
                RETURN type(r) as type, count(*) as count
                ORDER BY count DESC
                LIMIT 10
            """)
            
            print("\nTop relationship types:")
            async for record in rel_result:
                print(f"  {record['type']}: {record['count']}")
            
            print("\n✅ Neo4j graph is ready!")
            print("Open Neo4j Browser and run: MATCH (n:OntologyEntity)-[r]->(m) RETURN n, r, m LIMIT 100")
            
    finally:
        await driver.close()


async def main():
    parser = argparse.ArgumentParser(description="Seed Neo4j with astrology ontology")
    parser.add_argument("--clear", action="store_true", help="Clear existing ontology nodes first")
    parser.add_argument("--clear-all", action="store_true", help="Clear ALL Neo4j data (including Graphiti)")
    args = parser.parse_args()
    
    await seed_ontology_to_neo4j(clear_first=args.clear, clear_all=args.clear_all)


if __name__ == "__main__":
    asyncio.run(main())