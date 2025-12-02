"""
Clean all data from Supabase and Neo4j.

Usage:
    python scripts/clean_databases.py           # Clean all data
    python scripts/clean_databases.py --keep-ontology  # Keep OntologyEntity nodes
"""

import asyncio
import argparse
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv

load_dotenv()


async def clean_supabase():
    """Clean all data from Supabase/PostgreSQL."""
    from agent.db_utils import initialize_database, close_database, db_pool
    
    await initialize_database()
    async with db_pool.acquire() as conn:
        async with conn.transaction():
            await conn.execute('DELETE FROM messages')
            print('  ✓ Cleared messages')
            await conn.execute('DELETE FROM sessions')
            print('  ✓ Cleared sessions')
            await conn.execute('DELETE FROM chunks')
            print('  ✓ Cleared chunks')
            await conn.execute('DELETE FROM documents')
            print('  ✓ Cleared documents')
    await close_database()
    print('✅ Supabase cleaned!')


async def clean_neo4j(keep_ontology: bool = False):
    """Clean data from Neo4j."""
    from neo4j import AsyncGraphDatabase
    
    driver = AsyncGraphDatabase.driver(
        os.getenv('NEO4J_URI'),
        auth=(os.getenv('NEO4J_USER'), os.getenv('NEO4J_PASSWORD'))
    )
    
    async with driver.session() as session:
        if keep_ontology:
            # Keep OntologyEntity nodes, delete everything else
            result = await session.run('MATCH (n) WHERE NOT n:OntologyEntity DETACH DELETE n')
            print('  ✓ Cleared non-ontology nodes')
            # Clear CO_OCCURS relationships between ontology entities
            await session.run('MATCH (:OntologyEntity)-[r:CO_OCCURS]-(:OntologyEntity) DELETE r')
            print('  ✓ Cleared CO_OCCURS relationships')
            print('✅ Neo4j cleaned (ontology preserved)!')
        else:
            # Delete everything
            result = await session.run('MATCH (n) RETURN count(n) as count')
            record = await result.single()
            print(f'  Nodes before: {record["count"]}')
            
            await session.run('MATCH (n) DETACH DELETE n')
            print('  ✓ All nodes and relationships deleted')
            print('✅ Neo4j cleaned completely!')
    
    await driver.close()


async def main(keep_ontology: bool = False):
    print('\n' + '='*50)
    print('CLEANING DATABASES')
    print('='*50)
    
    print('\n--- Supabase ---')
    await clean_supabase()
    
    print('\n--- Neo4j ---')
    await clean_neo4j(keep_ontology=keep_ontology)
    
    print('\n' + '='*50)
    print('DONE!')
    print('='*50 + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Clean Supabase and Neo4j databases')
    parser.add_argument('--keep-ontology', action='store_true', 
                       help='Keep OntologyEntity nodes in Neo4j')
    args = parser.parse_args()
    
    asyncio.run(main(keep_ontology=args.keep_ontology))

