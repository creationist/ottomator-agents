"""
Merge Graphiti entities with Ontology entities.

Creates SAME_AS relationships between Graphiti's Entity nodes and 
OntologyEntity nodes when they represent the same concept.

This enables unified queries across both data sources.

Usage:
    python scripts/merge_graphiti_ontology.py
    python scripts/merge_graphiti_ontology.py --dry-run  # Preview without changes
"""

import asyncio
import argparse
import os
import sys
from typing import Dict, List, Tuple

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from neo4j import AsyncGraphDatabase
from dotenv import load_dotenv

load_dotenv()


# Name mappings for German/English variations
NAME_MAPPINGS = {
    # German to English planet names
    "sonne": "sun",
    "mond": "moon",
    "merkur": "mercury",
    "neptun": "neptune",
    # German to English sign names
    "widder": "aries",
    "stier": "taurus",
    "zwillinge": "gemini",
    "krebs": "cancer",
    "löwe": "leo",
    "jungfrau": "virgo",
    "waage": "libra",
    "skorpion": "scorpio",
    "schütze": "sagittarius",
    "steinbock": "capricorn",
    "wassermann": "aquarius",
    "fische": "pisces",
    # Houses
    "erstes haus": "house_1",
    "zweites haus": "house_2",
    "drittes haus": "house_3",
    "viertes haus": "house_4",
    "fünftes haus": "house_5",
    "sechstes haus": "house_6",
    "siebtes haus": "house_7",
    "achtes haus": "house_8",
    "neuntes haus": "house_9",
    "zehntes haus": "house_10",
    "elftes haus": "house_11",
    "zwölftes haus": "house_12",
    # Elements
    "feuer": "fire",
    "erde": "earth",
    "luft": "air",
    "wasser": "water",
    # Modalities
    "kardinal": "cardinal",
    "fix": "fixed",
    "veränderlich": "mutable",
    # Aspects
    "konjunktion": "conjunction",
    "sextil": "sextile",
    "quadrat": "square",
    "trigon": "trine",
    "quinkunx": "quincunx",
    # Lunar phases
    "neumond": "new_moon",
    "vollmond": "full_moon",
    "zunehmender mond": "waxing_crescent",
    "abnehmender mond": "waning_moon",
    # Themes
    "heilung": "healing",
    "beziehungen": "relationships",
    "kreativität": "creativity",
    "spiritualität": "spirituality",
    "berufung": "vocation",
    "kommunikation": "communication",
    "finanzen": "finances",
    "gesundheit": "health",
    # Concepts
    "rückläufigkeit": "retrograde",
    "merkur rückläufig": "mercury_retrograde",
}


def normalize_name(name: str) -> str:
    """Normalize a name for matching."""
    return name.lower().strip()


def get_possible_matches(graphiti_name: str) -> List[Tuple[str, int]]:
    """
    Get possible ontology IDs that might match a Graphiti entity name.
    
    Returns list of (possible_id, priority) tuples.
    Priority: 1 = exact match, 2 = mapped, 3 = partial
    """
    normalized = normalize_name(graphiti_name)
    
    matches = []
    
    # Priority 1: Direct/exact match
    matches.append((normalized, 1))
    matches.append((normalized.replace(" ", "_"), 1))
    
    # Priority 2: Direct mapping
    if normalized in NAME_MAPPINGS:
        matches.append((NAME_MAPPINGS[normalized], 2))
    
    # Priority 3: Partial matches (only if no exact match exists)
    # Skip partial matching - it causes issues like Merkur → Merkur rückläufig
    
    return matches


async def merge_entities(dry_run: bool = False):
    """Merge Graphiti entities with Ontology entities."""
    
    driver = AsyncGraphDatabase.driver(
        os.getenv('NEO4J_URI'),
        auth=(os.getenv('NEO4J_USER'), os.getenv('NEO4J_PASSWORD'))
    )
    
    try:
        async with driver.session() as session:
            print('\n' + '='*60)
            print('MERGING GRAPHITI AND ONTOLOGY ENTITIES')
            print('='*60)
            
            # Get all Graphiti entities
            result = await session.run("""
                MATCH (e:Entity)
                RETURN e.name AS name, elementId(e) AS element_id
            """)
            
            graphiti_entities = []
            async for record in result:
                graphiti_entities.append({
                    'name': record['name'],
                    'element_id': record['element_id']
                })
            
            print(f'\nFound {len(graphiti_entities)} Graphiti Entity nodes')
            
            # Get all Ontology entities
            result = await session.run("""
                MATCH (o:OntologyEntity)
                RETURN o.id AS id, o.name AS name, o.type AS type
            """)
            
            ontology_entities = {}
            async for record in result:
                ont_id = record['id']
                ontology_entities[ont_id] = {
                    'id': ont_id,
                    'name': record['name'],
                    'type': record['type']
                }
                # Also index by lowercase name
                ontology_entities[normalize_name(record['name'])] = ontology_entities[ont_id]
            
            print(f'Found {len(ontology_entities) // 2} OntologyEntity nodes')
            
            # Match and create relationships
            matches = []
            no_matches = []
            
            for ge in graphiti_entities:
                ge_name = ge['name'] or ''
                possible_matches = get_possible_matches(ge_name)
                
                # Sort by priority (exact matches first)
                possible_matches.sort(key=lambda x: x[1])
                
                matched = False
                for possible_id, priority in possible_matches:
                    if possible_id in ontology_entities:
                        ont = ontology_entities[possible_id]
                        # For exact matches (priority 1), verify it's truly exact
                        # This prevents "Merkur" matching "Merkur rückläufig"
                        if priority == 1:
                            ont_name_lower = normalize_name(ont['name'])
                            ge_name_lower = normalize_name(ge_name)
                            # Must be exact or very close
                            if ont_name_lower != ge_name_lower and ont['id'] != ge_name_lower:
                                continue
                        
                        matches.append({
                            'graphiti_name': ge_name,
                            'graphiti_id': ge['element_id'],
                            'ontology_id': ont['id'],
                            'ontology_name': ont['name'],
                            'ontology_type': ont['type'],
                            'match_priority': priority
                        })
                        matched = True
                        break
                
                if not matched:
                    no_matches.append(ge_name)
            
            # Print matches
            print(f'\n--- MATCHES ({len(matches)}) ---')
            for m in matches[:20]:  # Show first 20
                print(f"  ✓ '{m['graphiti_name']}' → {m['ontology_name']} ({m['ontology_type']})")
            if len(matches) > 20:
                print(f"  ... and {len(matches) - 20} more")
            
            # Print non-matches
            print(f'\n--- NO MATCH ({len(no_matches)}) ---')
            for name in no_matches[:15]:  # Show first 15
                print(f"  ✗ '{name}'")
            if len(no_matches) > 15:
                print(f"  ... and {len(no_matches) - 15} more")
            
            # Create SAME_AS relationships
            if not dry_run and matches:
                print(f'\n--- CREATING SAME_AS RELATIONSHIPS ---')
                
                created = 0
                for m in matches:
                    await session.run("""
                        MATCH (ge:Entity)
                        WHERE elementId(ge) = $graphiti_id
                        MATCH (oe:OntologyEntity {id: $ontology_id})
                        MERGE (ge)-[r:SAME_AS]->(oe)
                        SET r.matched_by = 'name_mapping'
                    """, {
                        'graphiti_id': m['graphiti_id'],
                        'ontology_id': m['ontology_id']
                    })
                    created += 1
                
                print(f'  Created {created} SAME_AS relationships')
                
                # Verify
                result = await session.run("""
                    MATCH (e:Entity)-[r:SAME_AS]->(o:OntologyEntity)
                    RETURN count(r) AS count
                """)
                record = await result.single()
                print(f'  Total SAME_AS relationships: {record["count"]}')
            
            elif dry_run:
                print(f'\n--- DRY RUN - No changes made ---')
                print(f'Would create {len(matches)} SAME_AS relationships')
            
            # Summary
            print('\n' + '='*60)
            print('SUMMARY')
            print('='*60)
            print(f'  Graphiti entities: {len(graphiti_entities)}')
            print(f'  Matched to ontology: {len(matches)} ({len(matches)*100//max(len(graphiti_entities),1)}%)')
            print(f'  No match: {len(no_matches)}')
            
            if no_matches:
                print('\n  Unmatched entities are likely:')
                print('  - People names (Astrologin, Fee, etc.)')
                print('  - Specific concepts not in ontology')
                print('  - Compound terms (Merkur rückläufig in Schütze)')
            
            print('\n' + '='*60 + '\n')
            
    finally:
        await driver.close()


async def main():
    parser = argparse.ArgumentParser(description='Merge Graphiti and Ontology entities')
    parser.add_argument('--dry-run', action='store_true', 
                       help='Preview matches without creating relationships')
    args = parser.parse_args()
    
    await merge_entities(dry_run=args.dry_run)


if __name__ == '__main__':
    asyncio.run(main())

