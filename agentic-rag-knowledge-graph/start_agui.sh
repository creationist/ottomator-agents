#!/bin/bash
# Start the AG-UI system (frontend + backend)
# This runs the Next.js frontend on port 3000 and the AG-UI agent on port 8001

set -e

echo "=================================================="
echo "🚀 Starting AG-UI RAG System"
echo "=================================================="

# Check if we're in the right directory
if [ ! -f "requirements.txt" ]; then
    echo "❌ Error: Please run this script from the project root directory"
    exit 1
fi

# Check for Python virtual environment
if [ -d "venv" ]; then
    echo "📦 Activating Python virtual environment..."
    source venv/bin/activate
fi

# Check for required Python packages
echo "📋 Checking Python dependencies..."
if ! python -c "import pydantic_ai" 2>/dev/null; then
    echo "⚠️  Installing Python dependencies..."
    pip install -r requirements.txt
fi

# Check for Node.js
if ! command -v node &> /dev/null; then
    echo "❌ Error: Node.js is not installed"
    exit 1
fi

# Check for frontend dependencies
if [ ! -d "frontend/node_modules" ]; then
    echo "📦 Installing frontend dependencies..."
    cd frontend
    npm install
    cd ..
fi

echo ""
echo "=================================================="
echo "📡 Starting services..."
echo "=================================================="
echo ""
echo "  🔧 AG-UI Backend: http://localhost:8001"
echo "  🌐 Frontend:      http://localhost:3000"
echo "  📚 Original API:  http://localhost:8000 (run separately with 'python -m agent.api')"
echo ""
echo "Press Ctrl+C to stop all services"
echo "=================================================="
echo ""

# Function to cleanup on exit
cleanup() {
    echo ""
    echo "🛑 Stopping services..."
    kill $BACKEND_PID 2>/dev/null || true
    kill $FRONTEND_PID 2>/dev/null || true
    exit 0
}

trap cleanup SIGINT SIGTERM

# Start the AG-UI backend
echo "🔧 Starting AG-UI backend..."
python -m agent.agui_agent &
BACKEND_PID=$!

# Wait a moment for the backend to start
sleep 2

# Start the frontend
echo "🌐 Starting Next.js frontend..."
cd frontend
npm run dev &
FRONTEND_PID=$!
cd ..

# Wait for both processes
wait $BACKEND_PID $FRONTEND_PID

