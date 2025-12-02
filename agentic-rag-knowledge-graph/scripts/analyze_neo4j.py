"""
Neo4j Graph Diagnostics Script.

Analyzes the knowledge graph for potential issues in the ingestion process:
- Entity/relationship counts and distribution
- Orphaned nodes and missing relationships  
- Data quality issues
- Consistency checks against the ontology

Usage:
    python scripts/analyze_neo4j.py
"""

import os
import asyncio
import logging
from collections import defaultdict
from typing import Dict, Any, List

from neo4j import AsyncGraphDatabase
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


async def analyze_graph():
    """Run comprehensive graph analysis."""
    
    uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    user = os.getenv("NEO4J_USER", "neo4j")
    password = os.getenv("NEO4J_PASSWORD")
    
    if not password:
        print("❌ NEO4J_PASSWORD environment variable not set")
        return
    
    driver = AsyncGraphDatabase.driver(uri, auth=(user, password))
    
    try:
        async with driver.session() as session:
            print("\n" + "="*70)
            print("NEO4J KNOWLEDGE GRAPH DIAGNOSTIC REPORT")
            print("="*70)
            
            # ============================================================
            # 1. OVERALL NODE STATISTICS
            # ============================================================
            print("\n" + "-"*70)
            print("1. NODE STATISTICS")
            print("-"*70)
            
            result = await session.run("""
                MATCH (n)
                RETURN labels(n) AS labels, count(*) AS count
                ORDER BY count DESC
            """)
            
            node_counts = {}
            total_nodes = 0
            async for record in result:
                labels = record['labels']
                count = record['count']
                label_str = ':'.join(labels) if labels else '(no label)'
                node_counts[label_str] = count
                total_nodes += count
                print(f"  {label_str}: {count}")
            
            print(f"\n  TOTAL NODES: {total_nodes}")
            
            # ============================================================
            # 2. RELATIONSHIP STATISTICS
            # ============================================================
            print("\n" + "-"*70)
            print("2. RELATIONSHIP STATISTICS")
            print("-"*70)
            
            result = await session.run("""
                MATCH ()-[r]->()
                RETURN type(r) AS type, count(*) AS count
                ORDER BY count DESC
            """)
            
            rel_counts = {}
            total_rels = 0
            async for record in result:
                rel_type = record['type']
                count = record['count']
                rel_counts[rel_type] = count
                total_rels += count
                print(f"  {rel_type}: {count}")
            
            print(f"\n  TOTAL RELATIONSHIPS: {total_rels}")
            
            # ============================================================
            # 3. ONTOLOGY ENTITY ANALYSIS
            # ============================================================
            print("\n" + "-"*70)
            print("3. ONTOLOGY ENTITY ANALYSIS")
            print("-"*70)
            
            # Check ontology entities by type
            result = await session.run("""
                MATCH (e:OntologyEntity)
                RETURN e.type AS type, count(*) AS count
                ORDER BY count DESC
            """)
            
            ontology_types = {}
            async for record in result:
                etype = record['type']
                count = record['count']
                ontology_types[etype] = count
                print(f"  {etype}: {count}")
            
            if not ontology_types:
                print("  ⚠️ WARNING: No OntologyEntity nodes found!")
                print("     → Run: python -m knowledge.seed_neo4j")
            
            # ============================================================
            # 4. CHUNK ANALYSIS
            # ============================================================
            print("\n" + "-"*70)
            print("4. CHUNK NODE ANALYSIS")
            print("-"*70)
            
            # Count chunks
            result = await session.run("""
                MATCH (c:Chunk)
                RETURN count(c) AS chunk_count
            """)
            record = await result.single()
            chunk_count = record['chunk_count'] if record else 0
            print(f"  Total Chunk nodes: {chunk_count}")
            
            if chunk_count > 0:
                # Chunks by document
                result = await session.run("""
                    MATCH (c:Chunk)
                    RETURN c.document_title AS doc, count(*) AS chunks
                    ORDER BY chunks DESC
                    LIMIT 10
                """)
                
                print("\n  Chunks by document (top 10):")
                async for record in result:
                    print(f"    {record['doc']}: {record['chunks']} chunks")
                
                # Chunks with entity counts
                result = await session.run("""
                    MATCH (c:Chunk)
                    RETURN 
                        avg(c.entity_count) AS avg_entities,
                        min(c.entity_count) AS min_entities,
                        max(c.entity_count) AS max_entities
                """)
                record = await result.single()
                if record and record['avg_entities'] is not None:
                    print(f"\n  Entities per chunk: avg={record['avg_entities']:.1f}, min={record['min_entities']}, max={record['max_entities']}")
            else:
                print("  ⚠️ WARNING: No Chunk nodes found!")
                print("     → Run ingestion to create chunks and link them to ontology")
            
            # ============================================================
            # 5. MENTIONS RELATIONSHIP ANALYSIS
            # ============================================================
            print("\n" + "-"*70)
            print("5. MENTIONS RELATIONSHIP ANALYSIS (Chunk → OntologyEntity)")
            print("-"*70)
            
            # Count MENTIONS relationships
            result = await session.run("""
                MATCH (c:Chunk)-[r:MENTIONS]->(e:OntologyEntity)
                RETURN count(r) AS mentions_count
            """)
            record = await result.single()
            mentions_count = record['mentions_count'] if record else 0
            print(f"  Total MENTIONS relationships: {mentions_count}")
            
            if mentions_count > 0:
                # Most mentioned entities
                result = await session.run("""
                    MATCH (c:Chunk)-[:MENTIONS]->(e:OntologyEntity)
                    RETURN e.name AS entity, e.type AS type, count(*) AS mentions
                    ORDER BY mentions DESC
                    LIMIT 15
                """)
                
                print("\n  Most frequently mentioned entities:")
                async for record in result:
                    print(f"    {record['entity']} ({record['type']}): {record['mentions']} mentions")
                
                # Entities NEVER mentioned (potential gap)
                result = await session.run("""
                    MATCH (e:OntologyEntity)
                    WHERE NOT (e)<-[:MENTIONS]-(:Chunk)
                    RETURN e.name AS entity, e.type AS type
                    ORDER BY e.type, e.name
                """)
                
                never_mentioned = []
                async for record in result:
                    never_mentioned.append((record['entity'], record['type']))
                
                if never_mentioned:
                    print(f"\n  ⚠️ Entities NEVER mentioned in any chunk ({len(never_mentioned)}):")
                    # Group by type
                    by_type = defaultdict(list)
                    for name, etype in never_mentioned:
                        by_type[etype].append(name)
                    
                    for etype, names in sorted(by_type.items()):
                        if len(names) <= 5:
                            print(f"    {etype}: {', '.join(names)}")
                        else:
                            print(f"    {etype}: {', '.join(names[:5])}... ({len(names)} total)")
            else:
                print("  ⚠️ WARNING: No MENTIONS relationships found!")
                print("     → Ingestion may not have linked chunks to ontology entities")
            
            # ============================================================
            # 6. CO_OCCURS RELATIONSHIP ANALYSIS
            # ============================================================
            print("\n" + "-"*70)
            print("6. CO_OCCURS RELATIONSHIP ANALYSIS")
            print("-"*70)
            
            result = await session.run("""
                MATCH (e1:OntologyEntity)-[r:CO_OCCURS]-(e2:OntologyEntity)
                RETURN count(r)/2 AS co_occurs_count
            """)
            record = await result.single()
            co_occurs_count = record['co_occurs_count'] if record else 0
            print(f"  Total CO_OCCURS relationships: {co_occurs_count}")
            
            if co_occurs_count > 0:
                # Top co-occurring pairs
                result = await session.run("""
                    MATCH (e1:OntologyEntity)-[r:CO_OCCURS]-(e2:OntologyEntity)
                    WHERE id(e1) < id(e2)
                    RETURN e1.name AS entity1, e2.name AS entity2, r.count AS co_count
                    ORDER BY r.count DESC
                    LIMIT 10
                """)
                
                print("\n  Top co-occurring entity pairs:")
                async for record in result:
                    print(f"    {record['entity1']} <-> {record['entity2']}: {record['co_count']} times")
            
            # ============================================================
            # 7. GRAPHITI ENTITIES (if using Graphiti for ingestion)
            # ============================================================
            print("\n" + "-"*70)
            print("7. GRAPHITI ENTITIES (from LLM-based extraction)")
            print("-"*70)
            
            # Check for Graphiti's EntityNode
            result = await session.run("""
                MATCH (e:Entity)
                RETURN count(e) AS entity_count
            """)
            record = await result.single()
            graphiti_entity_count = record['entity_count'] if record else 0
            
            if graphiti_entity_count > 0:
                print(f"  Graphiti Entity nodes: {graphiti_entity_count}")
                
                # Sample entities
                result = await session.run("""
                    MATCH (e:Entity)
                    RETURN e.name AS name, e.summary AS summary
                    LIMIT 10
                """)
                
                print("\n  Sample Graphiti entities:")
                async for record in result:
                    name = record['name'] or '(unnamed)'
                    summary = (record['summary'] or '')[:60]
                    print(f"    - {name}: {summary}...")
                
                # Check for Graphiti's episodic edges
                result = await session.run("""
                    MATCH ()-[r:RELATES_TO]->()
                    RETURN count(r) AS rel_count
                """)
                record = await result.single()
                relates_to_count = record['rel_count'] if record else 0
                print(f"\n  Graphiti RELATES_TO edges: {relates_to_count}")
            else:
                print("  No Graphiti Entity nodes found (LLM extraction not used or empty)")
            
            # Check for Episodes
            result = await session.run("""
                MATCH (ep:Episodic)
                RETURN count(ep) AS episode_count
            """)
            record = await result.single()
            episode_count = record['episode_count'] if record else 0
            print(f"  Graphiti Episodic nodes: {episode_count}")
            
            # ============================================================
            # 8. POTENTIAL ISSUES / RECOMMENDATIONS
            # ============================================================
            print("\n" + "-"*70)
            print("8. POTENTIAL ISSUES & RECOMMENDATIONS")
            print("-"*70)
            
            issues = []
            
            # Check for orphaned chunks (no MENTIONS)
            result = await session.run("""
                MATCH (c:Chunk)
                WHERE NOT (c)-[:MENTIONS]->()
                RETURN count(c) AS orphan_count
            """)
            record = await result.single()
            orphan_chunks = record['orphan_count'] if record else 0
            
            if orphan_chunks > 0:
                issues.append(f"⚠️ {orphan_chunks} Chunk nodes have no MENTIONS relationships")
                issues.append("   → These chunks didn't match any ontology keywords")
            
            # Check for isolated ontology entities
            result = await session.run("""
                MATCH (e:OntologyEntity)
                WHERE NOT (e)-[]-()
                RETURN count(e) AS isolated_count
            """)
            record = await result.single()
            isolated_entities = record['isolated_count'] if record else 0
            
            if isolated_entities > 0:
                issues.append(f"⚠️ {isolated_entities} OntologyEntity nodes are completely isolated")
                issues.append("   → These have no relationships at all (ontology may be incomplete)")
            
            # Check for missing ontology relationships
            expected_ont_rels = {'RULES', 'HAS_ELEMENT', 'HAS_MODALITY', 'NATURAL_RULER', 
                                'RELATES_TO', 'OPPOSITE', 'OPPOSITE_SIGN', 'PHASE_OF',
                                'TRADITIONAL_RULES', 'ASSOCIATED_WITH', 'AFFECTS', 
                                'VARIANT_OF', 'INSTANCE_OF'}
            
            missing_rels = expected_ont_rels - set(rel_counts.keys())
            if missing_rels:
                issues.append(f"⚠️ Expected ontology relationships not found: {missing_rels}")
                issues.append("   → Re-run: python -m knowledge.seed_neo4j --clear")
            
            # Check ratio of chunks to mentions
            if chunk_count > 0 and mentions_count > 0:
                avg_mentions_per_chunk = mentions_count / chunk_count
                if avg_mentions_per_chunk < 2:
                    issues.append(f"⚠️ Low average mentions per chunk: {avg_mentions_per_chunk:.1f}")
                    issues.append("   → Chunks may be too short or ontology keywords not matching content")
            
            # Check if both Graphiti and Ontology were used
            if graphiti_entity_count > 0 and chunk_count > 0:
                issues.append("ℹ️ Both Graphiti (LLM) and Ontology (regex) ingestion were used")
                issues.append("   → This is fine, but be aware of duplicate information")
            
            # Check for low entity coverage
            if ontology_types:
                total_ontology = sum(ontology_types.values())
                if mentions_count > 0:
                    result = await session.run("""
                        MATCH (e:OntologyEntity)
                        WHERE (e)<-[:MENTIONS]-(:Chunk)
                        RETURN count(DISTINCT e) AS mentioned_entities
                    """)
                    record = await result.single()
                    mentioned_entities = record['mentioned_entities'] if record else 0
                    coverage = (mentioned_entities / total_ontology) * 100
                    
                    if coverage < 50:
                        issues.append(f"⚠️ Low ontology coverage: only {coverage:.1f}% of entities are mentioned")
                        issues.append("   → Content may be too narrow or keywords don't match")
            
            # Print issues
            if issues:
                for issue in issues:
                    print(f"  {issue}")
            else:
                print("  ✅ No major issues detected")
            
            # ============================================================
            # 9. SAMPLE DATA INSPECTION
            # ============================================================
            print("\n" + "-"*70)
            print("9. SAMPLE DATA INSPECTION")
            print("-"*70)
            
            # Sample a chunk with its entities
            result = await session.run("""
                MATCH (c:Chunk)-[:MENTIONS]->(e:OntologyEntity)
                WITH c, collect(e.name) AS entities
                WHERE size(entities) >= 3
                RETURN c.document_title AS doc, c.chunk_index AS idx, 
                       c.content_preview AS preview, entities
                LIMIT 3
            """)
            
            print("\n  Sample chunks with their linked entities:")
            samples_found = False
            async for record in result:
                samples_found = True
                print(f"\n    [{record['doc']} - Chunk {record['idx']}]")
                print(f"    Preview: {record['preview'][:100]}...")
                print(f"    Entities: {', '.join(record['entities'][:8])}")
                if len(record['entities']) > 8:
                    print(f"              + {len(record['entities']) - 8} more")
            
            if not samples_found:
                print("    No chunks with 3+ entities found")
            
            print("\n" + "="*70)
            print("DIAGNOSTIC COMPLETE")
            print("="*70 + "\n")
            
    finally:
        await driver.close()


if __name__ == "__main__":
    asyncio.run(analyze_graph())



