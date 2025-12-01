# Document Ingestion Guide

This guide explains how to ingest documents into the Astrology RAG system with **zero or minimal LLM API costs**.

---

## Quick Start

```bash
# 1. Seed Neo4j with the astrology ontology (run once)
python -m knowledge.seed_neo4j --clear-all

# 2. Ingest documents (fast, zero LLM for entity extraction)
python -m ingestion.ingest --documents documents/ --clean --no-semantic
```

That's it! Your documents are now searchable via vector search and linked to the knowledge graph.

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                         INGESTION PIPELINE                          │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  Documents (.txt, .md)                                              │
│        │                                                            │
│        ▼                                                            │
│  ┌──────────────┐                                                   │
│  │   Chunking   │  Split into ~1000 char chunks                     │
│  └──────────────┘                                                   │
│        │                                                            │
│        ▼                                                            │
│  ┌──────────────┐                                                   │
│  │  Embedding   │  OpenAI text-embedding-3-small (batched)          │
│  └──────────────┘                                                   │
│        │                                                            │
│        ▼                                                            │
│  ┌──────────────┐     ┌─────────────────────────────────────┐       │
│  │   Entity     │────▶│  Ontology Matching (70 entities)    │       │
│  │  Extraction  │     │  ZERO LLM calls - regex matching    │       │
│  └──────────────┘     └─────────────────────────────────────┘       │
│        │                                                            │
│        ▼                                                            │
│  ┌──────────────┐     ┌──────────────────┐                          │
│  │  PostgreSQL  │     │      Neo4j       │                          │
│  │  (pgvector)  │     │  (knowledge      │                          │
│  │              │     │   graph)         │                          │
│  │  • documents │     │  • OntologyEntity│                          │
│  │  • chunks    │     │  • Chunk nodes   │                          │
│  │  • embeddings│     │  • MENTIONS rels │                          │
│  └──────────────┘     └──────────────────┘                          │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Two Ingestion Modes

### 1. Ontology Mode (Default) - RECOMMENDED

Uses the predefined astrology ontology for entity extraction.

| Feature | Details |
|---------|---------|
| **LLM Calls** | Zero for entity extraction |
| **Speed** | ~10 seconds for 3 documents |
| **Entities** | Matched from 70 predefined (planets, signs, houses, aspects, themes) |
| **Relationships** | MENTIONS links from chunks to ontology entities |

```bash
python -m ingestion.ingest --documents documents/ --no-semantic
```

### 2. Graphiti Mode - EXPENSIVE

Uses Graphiti's LLM-based entity extraction.

| Feature | Details |
|---------|---------|
| **LLM Calls** | 4-10 per chunk (120+ for 3 docs) |
| **Speed** | Several minutes |
| **Entities** | Dynamically extracted by LLM |
| **Relationships** | Inferred by LLM |

```bash
# Not recommended for bulk ingestion
python -m ingestion.ingest --documents documents/ --use-graphiti
```

---

## Command Reference

### Seed Neo4j with Ontology

```bash
# Seed ontology (keeps existing data)
python -m knowledge.seed_neo4j

# Clear ontology nodes only, then reseed
python -m knowledge.seed_neo4j --clear

# Clear ALL Neo4j data (including Graphiti data), then seed
python -m knowledge.seed_neo4j --clear-all
```

### Ingest Documents

```bash
python -m ingestion.ingest [OPTIONS]
```

| Option | Description | Default |
|--------|-------------|---------|
| `--documents`, `-d` | Path to documents folder | `documents/` |
| `--clean`, `-c` | Clear existing data before ingestion | `False` |
| `--chunk-size` | Target chunk size in characters | `1000` |
| `--chunk-overlap` | Overlap between chunks | `200` |
| `--no-semantic` | Use simple chunking (faster) | `False` |
| `--no-entities` | Skip entity extraction | `False` |
| `--fast`, `-f` | Skip graph building entirely | `False` |
| `--verbose`, `-v` | Enable debug logging | `False` |

### Common Commands

```bash
# Full ingestion with graph linking (recommended)
python -m ingestion.ingest -d documents/ --clean --no-semantic

# Fast mode: vector search only, no graph
python -m ingestion.ingest -d documents/ --clean --fast

# With semantic chunking (uses LLM for split points)
python -m ingestion.ingest -d documents/ --clean

# Verbose output for debugging
python -m ingestion.ingest -d documents/ --clean --no-semantic -v
```

---

## What Gets Created

### PostgreSQL (pgvector)

| Table | Content |
|-------|---------|
| `documents` | Full document content, title, source, metadata |
| `chunks` | Document chunks with embeddings (1536-dim vectors) |
| `sessions` | Chat sessions |
| `messages` | Chat history |

### Neo4j

| Node Type | Count | Description |
|-----------|-------|-------------|
| `OntologyEntity` | 70 | Predefined astrology entities |
| `Chunk` | Per doc | Document chunks (linked to ontology) |

| Relationship | Description |
|--------------|-------------|
| `RULES` | Planet rules a sign (e.g., Venus → Taurus) |
| `HAS_ELEMENT` | Sign has element (e.g., Aries → Fire) |
| `HAS_MODALITY` | Sign has modality (e.g., Aries → Cardinal) |
| `RELATES_TO` | Theme relates to entity |
| `MENTIONS` | Chunk mentions an ontology entity |

---

## Ontology Entities

The astrology ontology includes:

| Type | Count | Examples |
|------|-------|----------|
| `planet` | 10 | Sonne, Mond, Venus, Mars, Merkur, Jupiter, Saturn, Uranus, Neptun, Pluto |
| `sign` | 12 | Widder, Stier, Zwillinge, Krebs, Löwe, Jungfrau, Waage, Skorpion, Schütze, Steinbock, Wassermann, Fische |
| `house` | 12 | Erstes Haus through Zwölftes Haus |
| `aspect` | 7 | Konjunktion, Sextil, Quadrat, Trigon, Opposition, Halbsextil, Quinkunx |
| `element` | 4 | Feuer, Erde, Luft, Wasser |
| `modality` | 3 | Kardinal, Fix, Veränderlich |
| `lunar_phase` | 6 | Neumond, Vollmond, etc. |
| `theme` | 11 | Transformation, Heilung, Beziehungen, Kreativität, etc. |
| `lunar_node` | 2 | Nordknoten, Südknoten |
| `concept` | 2 | Rückläufigkeit, Merkur rückläufig |
| `asteroid` | 1 | Chiron |

Each entity has German keywords for matching. For example, "Venus" matches:
- `venus`, `liebe`, `schönheit`, `harmonie`, `werte`, `beziehung`, `genuss`, `kunst`, `ästhetik`

---

## Neo4j Queries

After ingestion, explore your data:

```cypher
-- See all ontology entities
MATCH (e:OntologyEntity)
RETURN e.name, e.type, e.description
ORDER BY e.type, e.name

-- See chunks and their entity mentions
MATCH (c:Chunk)-[:MENTIONS]->(e:OntologyEntity)
RETURN c.document_title, c.chunk_index, collect(e.name) as entities
LIMIT 20

-- Find most mentioned entities
MATCH (c:Chunk)-[:MENTIONS]->(e:OntologyEntity)
RETURN e.name, e.type, count(c) as mentions
ORDER BY mentions DESC
LIMIT 20

-- See relationships between ontology entities
MATCH (e1:OntologyEntity)-[r]->(e2:OntologyEntity)
RETURN e1.name, type(r), e2.name
LIMIT 50

-- Find all chunks mentioning a specific entity
MATCH (c:Chunk)-[:MENTIONS]->(e:OntologyEntity {name: 'Venus'})
RETURN c.document_title, c.content_preview

-- Graph visualization (run in Neo4j Browser)
MATCH (n)-[r]->(m)
RETURN n, r, m LIMIT 200
```

---

## Performance Comparison

| Metric | Graphiti Mode | Ontology Mode |
|--------|---------------|---------------|
| **3 Documents** | | |
| Time | ~5 minutes | **9 seconds** |
| OpenAI API calls | 120+ | **~4** (embeddings) |
| Cost | ~$0.50 | **~$0.01** |
| | | |
| **100 Documents** | | |
| Time | ~2 hours | **~5 minutes** |
| OpenAI API calls | 4000+ | **~100** |
| Cost | ~$15 | **~$0.30** |

---

## Troubleshooting

### "NEO4J_PASSWORD not set"
```bash
export NEO4J_PASSWORD=your_password
# Or add to .env file
```

### "python-dotenv could not parse statement"
This is a warning about your `.env` file format. The script still works. To fix, ensure your `.env` file uses simple `KEY=value` format without complex values.

### "No chunks created"
Check that your documents:
- Are in `.txt`, `.md`, or `.markdown` format
- Are in the correct folder
- Have content (not empty)

### Neo4j connection refused
```bash
# Start Neo4j (if using Docker)
docker-compose up -d neo4j
```

---

## Files Reference

| File | Purpose |
|------|---------|
| `knowledge/astrology_ontology.json` | Predefined entities and relationships |
| `knowledge/ontology_utils.py` | Ontology loading and keyword matching |
| `knowledge/seed_neo4j.py` | Seeds Neo4j with ontology |
| `ingestion/ingest.py` | Main ingestion script |
| `ingestion/chunker.py` | Document chunking |
| `ingestion/embedder.py` | Embedding generation |
| `ingestion/graph_builder.py` | Graph building (ontology + Graphiti modes) |

---

## Next Steps

After ingestion, you can:

1. **Start the chat interface:**
   ```bash
   ./start_agui.sh
   ```

2. **Query via API:**
   ```bash
   curl -X POST http://localhost:8000/chat \
     -H "Content-Type: application/json" \
     -d '{"message": "Was bedeutet Venus im Stier?"}'
   ```

3. **Explore the knowledge graph** in Neo4j Browser at `http://localhost:7474`

