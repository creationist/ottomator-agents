#!/bin/bash
# =============================================================================
# Local Development Startup Script
# Starts all services and initializes databases
# =============================================================================

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}"
echo "=================================================="
echo "🚀 Agentic RAG Local Development Setup"
echo "=================================================="
echo -e "${NC}"

# Check if we're in the right directory
if [ ! -f "requirements.txt" ]; then
    echo -e "${RED}❌ Error: Please run this script from the project root directory${NC}"
    exit 1
fi

# Check for .env file
if [ ! -f ".env" ]; then
    echo -e "${YELLOW}⚠️  No .env file found. Creating from template...${NC}"
    if [ -f ".env.example" ]; then
        cp .env.example .env
        echo -e "${YELLOW}📝 Please edit .env with your API keys before continuing${NC}"
        echo ""
        echo "Required keys:"
        echo "  - LLM_API_KEY (Groq): https://console.groq.com/keys"
        echo "  - EMBEDDING_API_KEY (Jina): https://jina.ai/embeddings/"
        echo ""
        read -p "Press Enter after you've added your API keys..."
    else
        echo -e "${RED}❌ No .env.example found. Please create .env manually${NC}"
        exit 1
    fi
fi

# Check for Docker
if ! command -v docker &> /dev/null; then
    echo -e "${RED}❌ Docker is not installed. Please install Docker first.${NC}"
    exit 1
fi

if ! docker info &> /dev/null; then
    echo -e "${RED}❌ Docker daemon is not running. Please start Docker.${NC}"
    exit 1
fi

# =============================================================================
# Step 1: Start Docker Services
# =============================================================================
echo -e "${BLUE}📦 Step 1: Starting Docker services...${NC}"

# Check if containers are already running
if docker ps | grep -q "agentic-rag-postgres"; then
    echo -e "${GREEN}✓ PostgreSQL already running${NC}"
else
    echo "Starting PostgreSQL with pgvector..."
    docker-compose up -d postgres
fi

if docker ps | grep -q "agentic-rag-neo4j"; then
    echo -e "${GREEN}✓ Neo4j already running${NC}"
else
    echo "Starting Neo4j..."
    docker-compose up -d neo4j
fi

# =============================================================================
# Step 2: Wait for services to be healthy
# =============================================================================
echo -e "${BLUE}⏳ Step 2: Waiting for services to be ready...${NC}"

# Wait for PostgreSQL
echo -n "Waiting for PostgreSQL"
for i in {1..30}; do
    if docker exec agentic-rag-postgres pg_isready -U postgres &> /dev/null; then
        echo -e " ${GREEN}✓${NC}"
        break
    fi
    echo -n "."
    sleep 1
done

# Wait for Neo4j
echo -n "Waiting for Neo4j"
for i in {1..60}; do
    if curl -s http://localhost:7474 &> /dev/null; then
        echo -e " ${GREEN}✓${NC}"
        break
    fi
    echo -n "."
    sleep 1
done

# =============================================================================
# Step 3: Initialize PostgreSQL Schema
# =============================================================================
echo -e "${BLUE}🗄️  Step 3: Initializing PostgreSQL schema...${NC}"

# Check if tables exist
TABLE_EXISTS=$(docker exec agentic-rag-postgres psql -U postgres -d agentic_rag -tAc "SELECT EXISTS (SELECT FROM information_schema.tables WHERE table_name = 'chunks');" 2>/dev/null || echo "f")

if [ "$TABLE_EXISTS" = "t" ]; then
    echo -e "${YELLOW}⚠️  Tables already exist. Reinitialize? (y/N)${NC}"
    read -r REINIT
    if [ "$REINIT" = "y" ] || [ "$REINIT" = "Y" ]; then
        echo "Reinitializing schema..."
        docker exec -i agentic-rag-postgres psql -U postgres -d agentic_rag < sql/schema.sql
        echo -e "${GREEN}✓ Schema reinitialized${NC}"
    else
        echo -e "${GREEN}✓ Keeping existing schema${NC}"
    fi
else
    echo "Creating schema..."
    docker exec -i agentic-rag-postgres psql -U postgres -d agentic_rag < sql/schema.sql
    echo -e "${GREEN}✓ Schema created${NC}"
fi

# =============================================================================
# Step 4: Setup Python Environment
# =============================================================================
echo -e "${BLUE}🐍 Step 4: Setting up Python environment...${NC}"

# Check for virtual environment
if [ -d "venv" ]; then
    echo "Activating virtual environment..."
    source venv/bin/activate
elif [ -d ".venv" ]; then
    echo "Activating virtual environment..."
    source .venv/bin/activate
else
    echo -e "${YELLOW}No virtual environment found. Creating one...${NC}"
    python3 -m venv venv
    source venv/bin/activate
    pip install --upgrade pip
fi

# Install dependencies
if ! python -c "import pydantic_ai" 2>/dev/null; then
    echo "Installing Python dependencies..."
    pip install -r requirements.txt
else
    echo -e "${GREEN}✓ Python dependencies installed${NC}"
fi

# =============================================================================
# Step 5: Seed Neo4j Ontology
# =============================================================================
echo -e "${BLUE}🌐 Step 5: Seeding Neo4j ontology...${NC}"

# Check if ontology is already seeded
ONTOLOGY_COUNT=$(python -c "
from neo4j import GraphDatabase
import os
driver = GraphDatabase.driver(
    os.getenv('NEO4J_URI', 'bolt://localhost:7687'),
    auth=(os.getenv('NEO4J_USER', 'neo4j'), os.getenv('NEO4J_PASSWORD', 'password123'))
)
with driver.session() as session:
    result = session.run('MATCH (n:OntologyEntity) RETURN count(n) as count')
    print(result.single()['count'])
driver.close()
" 2>/dev/null || echo "0")

if [ "$ONTOLOGY_COUNT" -gt "0" ]; then
    echo -e "${GREEN}✓ Ontology already seeded ($ONTOLOGY_COUNT entities)${NC}"
else
    echo "Seeding ontology..."
    python -m knowledge.seed_neo4j
    echo -e "${GREEN}✓ Ontology seeded${NC}"
fi

# =============================================================================
# Step 6: Verify Configuration
# =============================================================================
echo -e "${BLUE}🔍 Step 6: Verifying configuration...${NC}"

python << 'EOF'
import os
from dotenv import load_dotenv
load_dotenv()

checks = []

# Check LLM
llm_key = os.getenv('LLM_API_KEY')
if llm_key and len(llm_key) > 10:
    checks.append(("LLM API Key", True, os.getenv('LLM_CHOICE', 'not set')))
else:
    checks.append(("LLM API Key", False, "Missing or invalid"))

# Check Embeddings
emb_key = os.getenv('EMBEDDING_API_KEY')
if emb_key and len(emb_key) > 10:
    checks.append(("Embedding API Key", True, os.getenv('EMBEDDING_MODEL', 'not set')))
else:
    checks.append(("Embedding API Key", False, "Missing or invalid"))

# Check Database
db_url = os.getenv('DATABASE_URL') or os.getenv('POSTGRES_HOST')
checks.append(("Database Config", bool(db_url), db_url[:30] + "..." if db_url and len(db_url) > 30 else db_url or "Not set"))

# Check Neo4j
neo4j_uri = os.getenv('NEO4J_URI')
checks.append(("Neo4j Config", bool(neo4j_uri), neo4j_uri or "Not set"))

print()
for name, ok, detail in checks:
    status = "✓" if ok else "✗"
    color = "\033[92m" if ok else "\033[91m"
    print(f"  {color}{status}\033[0m {name}: {detail}")
print()

if not all(c[1] for c in checks):
    print("\033[93m⚠️  Some configuration is missing. Check your .env file.\033[0m")
EOF

# =============================================================================
# Done!
# =============================================================================
echo -e "${GREEN}"
echo "=================================================="
echo "✅ Local development environment ready!"
echo "=================================================="
echo -e "${NC}"
echo ""
echo "Services running:"
echo "  📦 PostgreSQL: localhost:5432"
echo "  🌐 Neo4j:      localhost:7687 (Browser: http://localhost:7474)"
echo ""
echo "Next steps:"
echo "  1. Add documents to ./documents/ folder"
echo "  2. Run ingestion:  python -m ingestion.ingest --folder documents"
echo "  3. Start the app:  ./start_agui.sh"
echo ""
echo "Useful commands:"
echo "  docker-compose logs -f     # View service logs"
echo "  docker-compose down        # Stop all services"
echo "  docker-compose down -v     # Stop and remove data"
echo ""
