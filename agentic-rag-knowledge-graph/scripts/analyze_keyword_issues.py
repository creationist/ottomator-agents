"""
Deep analysis of keyword matching issues in the ontology-based ingestion.

Identifies:
- Overly broad keywords causing false positives
- Missing entities from ontology
- German text matching problems
"""

import os
import sys
import asyncio
import json
from collections import defaultdict

from neo4j import AsyncGraphDatabase
from dotenv import load_dotenv

# Add parent to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from knowledge.ontology_utils import AstrologyOntology

load_dotenv()


async def analyze_keyword_issues():
    """Analyze potential keyword matching problems."""
    
    uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    user = os.getenv("NEO4J_USER", "neo4j")
    password = os.getenv("NEO4J_PASSWORD")
    
    if not password:
        print("❌ NEO4J_PASSWORD not set")
        return
    
    driver = AsyncGraphDatabase.driver(uri, auth=(user, password))
    ontology = AstrologyOntology()
    
    try:
        async with driver.session() as session:
            print("\n" + "="*70)
            print("KEYWORD MATCHING DEEP ANALYSIS")
            print("="*70)
            
            # ============================================================
            # 1. IDENTIFY PROBLEMATIC KEYWORDS
            # ============================================================
            print("\n" + "-"*70)
            print("1. POTENTIALLY PROBLEMATIC KEYWORDS")
            print("-"*70)
            
            # Keywords that are common German words (false positive risk)
            common_german_words = {
                'ich': 'I (first person)',
                'du': 'you',
                'wir': 'we', 
                'sie': 'they/she',
                'sein': 'to be / his',
                'haben': 'to have',
                'werden': 'to become',
                'zeit': 'time',
                'leben': 'life',
                'haus': 'house',
                'welt': 'world',
                'tag': 'day',
                'jahr': 'year',
                'ende': 'end',
                'anfang': 'beginning',
                'start': 'start',
                'ziel': 'goal',
                'kraft': 'power/strength',
                'mut': 'courage',
                'glück': 'luck/happiness',
                'liebe': 'love',
            }
            
            print("\n  Ontology entities with common German word keywords:")
            problematic_keywords = []
            
            for entity in ontology._entities.values():
                for keyword in entity.keywords:
                    kw_lower = keyword.lower()
                    if kw_lower in common_german_words:
                        problematic_keywords.append((entity.id, entity.name, keyword, common_german_words[kw_lower]))
            
            if problematic_keywords:
                for entity_id, name, keyword, meaning in problematic_keywords:
                    print(f"    ⚠️ {name} ({entity_id}): keyword '{keyword}' means '{meaning}'")
            
            # Short keywords (2-3 chars) - high false positive risk
            print("\n  Very short keywords (high false positive risk):")
            short_keywords = []
            for entity in ontology._entities.values():
                for keyword in entity.keywords:
                    if len(keyword) <= 3:
                        short_keywords.append((entity.id, entity.name, keyword))
            
            for entity_id, name, keyword in short_keywords:
                print(f"    ⚠️ {name} ({entity_id}): keyword '{keyword}' (only {len(keyword)} chars)")
            
            # ============================================================
            # 2. CHECK ONTOLOGY vs NEO4J ENTITY COUNT
            # ============================================================
            print("\n" + "-"*70)
            print("2. ONTOLOGY vs NEO4J ENTITY MISMATCH")
            print("-"*70)
            
            ontology_ids = set(ontology._entities.keys())
            print(f"  Ontology file has {len(ontology_ids)} entities")
            
            # Get Neo4j entity IDs
            result = await session.run("""
                MATCH (e:OntologyEntity)
                RETURN e.id AS id
            """)
            neo4j_ids = set()
            async for record in result:
                neo4j_ids.add(record['id'])
            
            print(f"  Neo4j has {len(neo4j_ids)} OntologyEntity nodes")
            
            missing_in_neo4j = ontology_ids - neo4j_ids
            extra_in_neo4j = neo4j_ids - ontology_ids
            
            if missing_in_neo4j:
                print(f"\n  ❌ Entities in ontology but MISSING from Neo4j ({len(missing_in_neo4j)}):")
                for eid in sorted(missing_in_neo4j):
                    entity = ontology.get_entity(eid)
                    print(f"    - {eid}: {entity.name if entity else 'unknown'}")
            
            if extra_in_neo4j:
                print(f"\n  ⚠️ Entities in Neo4j but NOT in ontology ({len(extra_in_neo4j)}):")
                for eid in sorted(extra_in_neo4j):
                    print(f"    - {eid}")
            
            if not missing_in_neo4j and not extra_in_neo4j:
                print("  ✅ Ontology and Neo4j entities match perfectly")
            
            # ============================================================
            # 3. ANALYZE HOUSE OVER-REPRESENTATION
            # ============================================================
            print("\n" + "-"*70)
            print("3. HOUSE ENTITY OVER-REPRESENTATION ANALYSIS")
            print("-"*70)
            
            # Get house entity mentions
            result = await session.run("""
                MATCH (c:Chunk)-[:MENTIONS]->(e:OntologyEntity)
                WHERE e.type = 'house'
                RETURN e.name AS house, e.id AS house_id, count(*) AS mentions
                ORDER BY mentions DESC
            """)
            
            house_mentions = []
            async for record in result:
                house_mentions.append({
                    'name': record['house'],
                    'id': record['house_id'],
                    'mentions': record['mentions']
                })
            
            total_chunks = 1821  # from previous analysis
            print(f"\n  House mentions (out of {total_chunks} total chunks):")
            for h in house_mentions:
                pct = (h['mentions'] / total_chunks) * 100
                entity = ontology.get_entity(h['id'])
                keywords = entity.keywords if entity else []
                print(f"    {h['name']}: {h['mentions']} mentions ({pct:.1f}%)")
                print(f"      Keywords: {keywords}")
            
            # The first and seventh house are suspiciously high
            # Check their keywords
            print("\n  Analysis: 'Erstes Haus' and 'Siebtes Haus' keywords:")
            house1 = ontology.get_entity('house_1')
            house7 = ontology.get_entity('house_7')
            
            if house1:
                print(f"    house_1 keywords: {house1.keywords}")
            if house7:
                print(f"    house_7 keywords: {house7.keywords}")
            
            # ============================================================
            # 4. VERIFY SAMPLE MATCHES
            # ============================================================
            print("\n" + "-"*70)
            print("4. SAMPLE CHUNK KEYWORD VERIFICATION")
            print("-"*70)
            
            # Get a few chunks and manually check if the entity matches make sense
            result = await session.run("""
                MATCH (c:Chunk)-[:MENTIONS]->(e:OntologyEntity {id: 'house_1'})
                RETURN c.content_preview AS preview, c.chunk_index AS idx, c.document_title AS doc
                LIMIT 5
            """)
            
            print("\n  Checking chunks that MENTION 'house_1' (Erstes Haus):")
            async for record in result:
                preview = record['preview'][:150] if record['preview'] else ''
                print(f"\n    [{record['doc']} - Chunk {record['idx']}]")
                print(f"    {preview}...")
                
                # Check which keyword triggered the match
                house1_keywords = house1.keywords if house1 else []
                found_keywords = []
                preview_lower = preview.lower()
                for kw in house1_keywords:
                    if kw.lower() in preview_lower:
                        found_keywords.append(kw)
                print(f"    → Matched keywords: {found_keywords}")
            
            # ============================================================
            # 5. RECOMMENDATIONS
            # ============================================================
            print("\n" + "-"*70)
            print("5. RECOMMENDATIONS")
            print("-"*70)
            
            recommendations = [
                "1. Remove overly generic keywords from house entities:",
                "   - house_1: Remove 'ich' (German 'I'), 'selbst' (self)",
                "   - house_7: Remove 'du' (German 'you'), 'gegenüber' (opposite)",
                "",
                "2. Consider using multi-word phrases instead of single words:",
                "   - Instead of 'haus' use 'erstes haus', 'astrological house'",
                "   - Instead of 'ich' use 'selbstbild', 'persönlichkeit'",
                "",
                "3. Add minimum keyword length (3+ characters) in ontology_builder.py",
                "",
                "4. Consider case-sensitive matching for acronyms (IC, MC)",
                "",
                "5. Re-run ingestion after fixing keywords:",
                "   python -m knowledge.seed_neo4j --clear",
                "   Then re-ingest documents",
            ]
            
            for rec in recommendations:
                print(f"  {rec}")
            
            print("\n" + "="*70)
            print("ANALYSIS COMPLETE")
            print("="*70 + "\n")
            
    finally:
        await driver.close()


if __name__ == "__main__":
    asyncio.run(analyze_keyword_issues())


