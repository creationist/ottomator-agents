# Tool Testing Guide

This document contains targeted test questions for each agent tool. Use these to verify the system correctly selects and executes the appropriate tools.

---

## Available Tools

| Tool | Purpose | LLM Calls |
|------|---------|-----------|
| `search` | Hybrid vector + text search | Embedding only |
| `graph_search` | Knowledge graph traversal | Embedding only |
| `get_document` | Retrieve specific document | None |
| `list_documents` | List available documents | None |
| `get_entity_relationships` | Get entity connections | None |
| `get_entity_timeline` | Get entity history | None |
| `lookup_concept` | Ontology concept lookup | None |
| `generate_personalized_content` | Create inspirational text | LLM |
| `explore_ontology` | Traverse ontology relationships | None |

---

## Test Questions by Tool

### 🔍 search (Hybrid Vector + Text Search)

Best for: Finding relevant content from ingested documents.

```
Was sagt mein Dokument über Venus im Stier?
Finde Informationen über Merkur Rückläufigkeit
Welche Texte erwähnen den Vollmond?
Was steht in meinen Dokumenten über Transformation?
Suche nach Informationen über das achte Haus
```

**Expected behavior:** Returns ranked document chunks with similarity scores.

---

### 🕸️ graph_search (Knowledge Graph)

Best for: Finding connections and relationships in the knowledge graph.

```
Welche Beziehungen hat Mars zu anderen Planeten?
Wie sind Skorpion und Pluto verbunden?
Zeige mir alle Verbindungen zum achten Haus
Was ist mit dem Thema Heilung im Graph verbunden?
Finde Graphverbindungen zu Venus
```

**Expected behavior:** Returns entities and relationships from Neo4j.

---

### 📄 get_document / list_documents

Best for: Document management and overview.

```
Welche Dokumente hast du in deiner Wissensbasis?
Liste alle verfügbaren Astrologie-Texte auf
Zeige mir die Dokumentübersicht
Wie viele Dokumente wurden ingested?
Was für Quellen hast du?
```

**Expected behavior:** Returns list of documents with metadata.

---

### 🔗 get_entity_relationships

Best for: Understanding specific entity connections.

```
Welche Beziehungen hat Venus?
Mit welchen Zeichen ist Saturn verbunden?
Was sind die Beziehungen des Mondes?
Zeige mir alle Verbindungen von Jupiter
Welche Entitäten sind mit Skorpion verknüpft?
```

**Expected behavior:** Returns relationship types and connected entities.

---

### 📅 get_entity_timeline

Best for: Historical/temporal entity information.

```
Wie hat sich das Verständnis von Pluto entwickelt?
Zeige mir die Zeitlinie von Jupiter-Themen
Was ist die Geschichte von Uranus in der Astrologie?
Chronologische Entwicklung des Neptun-Konzepts
```

**Expected behavior:** Returns time-ordered information about an entity.

---

### 🔮 lookup_concept (Ontology Lookup)

Best for: Quick concept definitions from the astrology ontology.

```
Was ist ein Trigon?
Erkläre mir das Konzept Karma
Was bedeutet Konjunktion in der Astrologie?
Definition von Quadrat-Aspekt
Was ist ein Aszendent?
Was bedeutet Retrograde?
Erkläre das Element Feuer
```

**Expected behavior:** Returns ontology definition with keywords and attributes.

---

### ✨ generate_personalized_content

Best for: Creative, inspirational astrology content.

```
Schreibe mir einen inspirierenden Text über Neumond
Generiere einen motivierenden Absatz über Transformation
Erstelle einen poetischen Text über Venus und Liebe
Schreibe etwas Inspirierendes zum Thema Heilung
Verfasse einen kurzen Text über den Vollmond
Kreiere einen spirituellen Text über Karma
```

**Expected behavior:** Returns creative, personalized astrology content.

---

### 🧭 explore_ontology

Best for: Exploring ontology structure and relationships.

```
Welche Planeten gehören zum Element Feuer?
Zeige mir alle kardinalen Zeichen
Welche Themen sind mit dem 7. Haus verbunden?
Was sind die Wasserzeichen?
Welche Zeichen regiert Venus?
Zeige mir die fixen Zeichen
Welche Aspekte gibt es in der Astrologie?
```

**Expected behavior:** Returns structured ontology data with relationships.

---

## Multi-Tool Test Scenarios

These questions should trigger multiple tools in sequence:

### Comprehensive Entity Query
```
Erkläre mir alles über Skorpion - seine Planeten, Elemente und schreibe mir dazu einen inspirierenden Text
```
**Expected tools:** `lookup_concept` → `explore_ontology` → `generate_personalized_content`

### Research + Creation
```
Suche in meinen Dokumenten nach Venus-Themen und erstelle daraus einen inspirierenden Text
```
**Expected tools:** `search` → `generate_personalized_content`

### Graph + Ontology Exploration
```
Zeige mir alle Beziehungen von Mars und erkläre seine Bedeutung in der Astrologie
```
**Expected tools:** `get_entity_relationships` → `lookup_concept`

### Full Knowledge Base Query
```
Was weiß das System über den Mond? Zeige Dokumente, Beziehungen und Ontologie-Einträge
```
**Expected tools:** `search` → `get_entity_relationships` → `lookup_concept`

---

## Testing Checklist

- [ ] Each tool can be triggered individually
- [ ] Tools return expected data format
- [ ] Multi-tool scenarios work correctly
- [ ] Error handling works (invalid entities, empty results)
- [ ] Response times are acceptable
- [ ] German language queries work correctly

---

## Running Tests

### CLI Testing (Instant Mode)
```bash
python cli.py --no-stream
```

### CLI Testing (Streaming Mode)
```bash
python cli.py
```

### API Testing
```bash
curl -X POST http://localhost:8058/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Was ist ein Trigon?", "user_id": "test"}'
```

---

## Troubleshooting

### Tool not being selected
- Check if the question is clear enough
- Try more explicit phrasing
- Verify the tool is registered in `agent.py`

### Empty results
- Verify documents are ingested
- Check Neo4j has ontology seeded
- Verify embeddings were created

### Wrong tool selected
- The LLM chooses tools based on the system prompt
- Adjust `prompts.py` if needed for better tool selection

