#!/bin/bash
# =============================================================================
# Local Development Startup Script
# Starts Supabase and Neo4j, initializes databases
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
        echo -e "${YELLOW}📝 Please edit .env with your API keys${NC}"
        echo ""
        echo "Required configuration will be shown after starting Supabase."
        echo ""
    else
        echo -e "${RED}❌ No .env.example found. Please create .env manually${NC}"
        exit 1
    fi
fi

# =============================================================================
# Step 1: Check for Supabase CLI
# =============================================================================
echo -e "${BLUE}📦 Step 1: Checking Supabase CLI...${NC}"

if ! command -v supabase &> /dev/null; then
    echo -e "${YELLOW}⚠️  Supabase CLI not found. Installing...${NC}"
    if command -v brew &> /dev/null; then
        brew install supabase/tap/supabase
    else
        echo -e "${RED}❌ Please install Supabase CLI: https://supabase.com/docs/guides/cli${NC}"
        echo "  macOS: brew install supabase/tap/supabase"
        echo "  npm:   npm install -g supabase"
        exit 1
    fi
fi

echo -e "${GREEN}✓ Supabase CLI installed${NC}"

# Check for Docker (required by Supabase)
if ! command -v docker &> /dev/null; then
    echo -e "${RED}❌ Docker is not installed. Please install Docker first.${NC}"
    exit 1
fi

if ! docker info &> /dev/null; then
    echo -e "${RED}❌ Docker daemon is not running. Please start Docker.${NC}"
    exit 1
fi

# =============================================================================
# Step 2: Start Supabase
# =============================================================================
echo -e "${BLUE}🗄️  Step 2: Starting Supabase...${NC}"

# Check if Supabase is already running
if supabase status 2>/dev/null | grep -q "API URL"; then
    echo -e "${GREEN}✓ Supabase already running${NC}"
else
    echo "Starting Supabase local development stack..."
    supabase start
    
    echo ""
    echo -e "${GREEN}✓ Supabase started${NC}"
    echo -e "${YELLOW}📝 IMPORTANT: Copy these values to your .env file:${NC}"
    supabase status
    echo ""
fi

# =============================================================================
# Step 3: Start Neo4j
# =============================================================================
echo -e "${BLUE}🌐 Step 3: Starting Neo4j...${NC}"

if docker ps | grep -q "agentic-rag-neo4j"; then
    echo -e "${GREEN}✓ Neo4j already running${NC}"
else
    echo "Starting Neo4j..."
    docker-compose up -d neo4j
    
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
fi

# =============================================================================
# Step 4: Apply Supabase Migrations
# =============================================================================
echo -e "${BLUE}📊 Step 4: Applying database migrations...${NC}"

# Check if migrations have been applied
MIGRATION_COUNT=$(supabase db query "SELECT COUNT(*) FROM schema_migrations" --data-only 2>/dev/null || echo "0")

if [ "$MIGRATION_COUNT" != "0" ] && [ "$MIGRATION_COUNT" != "" ]; then
    echo -e "${GREEN}✓ Migrations already applied${NC}"
else
    echo "Applying migrations..."
    supabase db push
    echo -e "${GREEN}✓ Migrations applied${NC}"
fi

# =============================================================================
# Step 5: Setup Python Environment
# =============================================================================
echo -e "${BLUE}🐍 Step 5: Setting up Python environment...${NC}"

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
# Step 6: Seed Neo4j Ontology
# =============================================================================
echo -e "${BLUE}🌐 Step 6: Seeding Neo4j ontology...${NC}"

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
# Step 7: Verify Configuration
# =============================================================================
echo -e "${BLUE}🔍 Step 7: Verifying configuration...${NC}"

python << 'EOF'
import os
from dotenv import load_dotenv
load_dotenv()

checks = []

# Check LLM
llm_key = os.getenv('OPENAI_API_KEY') or os.getenv('LLM_API_KEY')
llm_model = os.getenv('LLM_MODEL', 'not set')
if llm_key and len(llm_key) > 10:
    checks.append(("LLM API Key", True, llm_model))
else:
    checks.append(("LLM API Key", False, "Missing or invalid"))

# Check Embeddings
emb_base = os.getenv('EMBEDDING_BASE_URL', '')
emb_model = os.getenv('EMBEDDING_MODEL', 'not set')
if 'ollama' in emb_base.lower() or os.getenv('EMBEDDING_API_KEY'):
    checks.append(("Embedding Config", True, emb_model))
else:
    checks.append(("Embedding Config", False, "Missing configuration"))

# Check Supabase
supabase_url = os.getenv('SUPABASE_URL')
checks.append(("Supabase URL", bool(supabase_url), supabase_url[:40] + "..." if supabase_url and len(supabase_url) > 40 else supabase_url or "Not set"))

supabase_key = os.getenv('SUPABASE_ANON_KEY')
checks.append(("Supabase Anon Key", bool(supabase_key), "Set" if supabase_key else "Not set"))

# Check Neo4j
neo4j_uri = os.getenv('NEO4J_URI')
checks.append(("Neo4j Config", bool(neo4j_uri), neo4j_uri or "Not set"))

# Check DATABASE_URL (for vector operations)
db_url = os.getenv('DATABASE_URL')
checks.append(("DATABASE_URL (vectors)", bool(db_url), db_url[:40] + "..." if db_url and len(db_url) > 40 else db_url or "Not set"))

print()
for name, ok, detail in checks:
    status = "✓" if ok else "✗"
    color = "\033[92m" if ok else "\033[91m"
    print(f"  {color}{status}\033[0m {name}: {detail}")
print()

if not all(c[1] for c in checks):
    print("\033[93m⚠️  Some configuration is missing. Check your .env file.\033[0m")
    print("\033[93m   Run 'supabase status' to get Supabase credentials.\033[0m")
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
echo "  📦 Supabase:   http://localhost:54321 (API)"
echo "                 http://localhost:54323 (Studio)"
echo "  🗄️  PostgreSQL: localhost:54322"
echo "  🌐 Neo4j:      localhost:7687 (Browser: http://localhost:7474)"
echo ""
echo "Next steps:"
echo "  1. Ensure your .env has Supabase credentials (run: supabase status)"
echo "  2. Add documents to ./documents/ folder"
echo "  3. Run ingestion:  python -m ingestion.ingest -d documents"
echo "  4. Start the API:  python -m agent.api"
echo "  5. Or start AGUI:  ./start_agui.sh"
echo ""
echo "Useful commands:"
echo "  supabase status       # View Supabase info & credentials"
echo "  supabase stop         # Stop Supabase"
echo "  supabase db reset     # Reset database and reapply migrations"
echo "  docker-compose logs   # View Neo4j logs"
echo ""
