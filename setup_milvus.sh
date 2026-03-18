#!/bin/bash

# Automated Milvus Docker Setup Script
# Run from: ~/RAGs/workflow

echo "=========================================="
echo "MILVUS DOCKER SETUP - AUTOMATED"
echo "=========================================="
echo ""

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "ERROR: Docker is not installed"
    echo "Please install Docker first: https://docs.docker.com/install/"
    exit 1
fi

echo "1. Checking Docker daemon..."
if ! docker info > /dev/null 2>&1; then
    echo "ERROR: Docker daemon is not running"
    echo "Please start Docker Desktop or Docker daemon"
    exit 1
fi
echo "   Docker is running - OK"
echo ""

# Get the workflow directory
WORKFLOW_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MILVUS_DATA="$WORKFLOW_DIR/milvus_data"

echo "2. Workflow directory: $WORKFLOW_DIR"
echo "   Milvus data folder: $MILVUS_DATA"
echo ""

# Clean up old container if exists (optional)
echo "3. Checking for existing Milvus container..."
if docker ps -a | grep -q milvus-local; then
    echo "   Found existing container. Stopping..."
    docker stop milvus-local > /dev/null 2>&1
    docker rm milvus-local > /dev/null 2>&1
    echo "   Removed old container"
fi
echo ""

# Create milvus_data directory if it doesn't exist
echo "4. Preparing milvus_data directory..."
mkdir -p "$MILVUS_DATA"
chmod 755 "$MILVUS_DATA"
echo "   Directory ready"
echo ""

# Start Milvus with Docker Compose
echo "5. Starting Milvus server..."
cd "$WORKFLOW_DIR"

if command -v docker-compose &> /dev/null; then
    echo "   Using docker-compose..."
    docker-compose up -d
    COMPOSE_USED=1
else
    echo "   Using docker run command..."
    docker run -d --name milvus-local \
        -p 19530:19530 \
        -p 9091:9091 \
        -v "$MILVUS_DATA:/var/lib/milvus" \
        -e COMMON_STORAGETYPE=local \
        milvusdb/milvus:latest
    COMPOSE_USED=0
fi

echo ""
echo "6. Waiting for Milvus to start (15 seconds)..."
for i in {1..15}; do
    echo -n "."
    sleep 1
done
echo ""
echo ""

# Verify Milvus is running
echo "7. Verifying Milvus is running..."
if docker ps | grep -q milvus-local; then
    echo "   SUCCESS: Milvus container is running"
else
    echo "   ERROR: Milvus container is not running"
    if [ $COMPOSE_USED -eq 1 ]; then
        docker-compose logs
    else
        docker logs milvus-local
    fi
    exit 1
fi
echo ""

# Test connection
echo "8. Testing Milvus connection..."
if curl -s http://localhost:19530/healthz > /dev/null; then
    echo "   SUCCESS: Milvus server is responding"
else
    echo "   WARNING: Could not reach Milvus yet (it may still be starting)"
fi
echo ""

# Display next steps
echo "=========================================="
echo "SETUP COMPLETE!"
echo "=========================================="
echo ""
echo "Next steps:"
echo ""
echo "1. Activate your virtual environment:"
echo "   cd ~/RAGs/workflow"
echo "   source ../../venv/bin/activate"
echo ""
echo "2. Run your pipeline:"
echo "   python run_pipeline.py --input_dir pdfs"
echo ""
echo "3. Your data is saved at:"
echo "   $MILVUS_DATA"
echo ""
echo "To stop Milvus:"
if [ $COMPOSE_USED -eq 1 ]; then
    echo "   cd ~/RAGs/workflow && docker-compose down"
else
    echo "   docker stop milvus-local"
fi
echo ""
