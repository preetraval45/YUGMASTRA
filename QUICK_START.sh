#!/bin/bash

# YUGMĀSTRA Quick Start Script
# This script helps you get the platform running quickly

set -e  # Exit on error

echo "======================================"
echo "   YUGMĀSTRA Quick Start"
echo "======================================"
echo ""

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_step() {
    echo -e "${BLUE}[STEP]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

# Check prerequisites
print_step "Checking prerequisites..."

# Check Node.js
if command -v node &> /dev/null; then
    NODE_VERSION=$(node --version)
    print_success "Node.js $NODE_VERSION installed"
else
    print_warning "Node.js not found. Please install Node.js 20+"
    exit 1
fi

# Check Python
if command -v python3 &> /dev/null; then
    PYTHON_VERSION=$(python3 --version)
    print_success "Python $PYTHON_VERSION installed"
else
    print_warning "Python not found. Please install Python 3.11+"
    exit 1
fi

# Check Docker
if command -v docker &> /dev/null; then
    DOCKER_VERSION=$(docker --version)
    print_success "Docker installed"
else
    print_warning "Docker not found. Please install Docker"
    exit 1
fi

echo ""
print_step "Installing Node.js dependencies..."
npm install
print_success "Node.js dependencies installed"

echo ""
print_step "Installing Python dependencies..."
pip install -r requirements.txt
print_success "Python dependencies installed"

echo ""
print_step "Starting Docker services..."
docker-compose up -d
print_success "Docker services started"

echo ""
print_step "Waiting for services to be ready (30 seconds)..."
sleep 30

echo ""
print_step "Checking service health..."
docker-compose ps

echo ""
echo "======================================"
echo "   Setup Complete!"
echo "======================================"
echo ""
echo "Available services:"
echo "  - PostgreSQL:     localhost:5432"
echo "  - Neo4j:          localhost:7474"
echo "  - Redis:          localhost:6379"
echo "  - Elasticsearch:  localhost:9200"
echo "  - Prometheus:     localhost:9090"
echo "  - Grafana:        localhost:3001 (admin/admin)"
echo ""
echo "Next steps:"
echo ""
echo "1. Start the web app:"
echo "   cd apps/web && npm run dev"
echo "   Then visit: http://localhost:3000"
echo ""
echo "2. Start the API:"
echo "   cd apps/api && uvicorn main:app --reload"
echo "   Then visit: http://localhost:8000/docs"
echo ""
echo "3. Start training:"
echo "   cd services/red-team-ai"
echo "   python training/train_red_agent.py"
echo ""
echo "For detailed instructions, see:"
echo "  - GETTING_STARTED.md"
echo "  - IMPLEMENTATION_GUIDE.md"
echo ""
echo "Happy researching! 🚀"
