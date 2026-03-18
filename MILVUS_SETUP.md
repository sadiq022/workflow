MILVUS DOCKER SETUP GUIDE - EASY STEPS
=====================================

Your Data Location: ~/RAGs/workflow/milvus_data/ (stays on your PC)
Milvus Server Port: 19530
Health Check Port: 9091

STEP 1: Clean Up Old Database Files
===================================

Run in workflow folder:

cd ~/RAGs/workflow

# Remove all old database files (optional backup if you have old data)
rm -rf milvus_data/*


STEP 2: Start Milvus Docker Container
======================================

Option A - Using Docker Commands (Manual):

docker run -d --name milvus-local \
  -p 19530:19530 \
  -p 9091:9091 \
  -v ~/RAGs/workflow/milvus_data:/var/lib/milvus \
  -e COMMON_STORAGETYPE=local \
  milvusdb/milvus:v0.4.9

# Wait for it to start (about 10-15 seconds)
sleep 15

# Check if it's running
docker logs milvus-local


Option B - Using Docker Compose (Easier - Recommended):

cd ~/RAGs/workflow

# Start Milvus from docker-compose.yml
docker-compose up -d

# Wait for it to be healthy
sleep 15

# Check status
docker-compose logs -f milvus


STEP 3: Verify Milvus is Running
=================================

# Check if container is running
docker ps | grep milvus

# Expected output shows "milvus-local" container running

# Test connection
curl http://localhost:19530/healthz

# Should return status information


STEP 4: Activate Your Virtual Environment and Run Pipeline
===========================================================

cd ~/RAGs/workflow

# Activate virtual environment (parent in RAGs folder)
source ../../venv/bin/activate

# OR just use the activate from current directory if it exists
source .venv/bin/activate

# Verify you're in the virtual environment (should show (venv) in prompt)

# Run the pipeline
python run_pipeline.py --input_dir pdfs

# You should see:
# Loading PDFs
# Chunking
# Embedding
# Storing in Milvus
# Pipeline completed successfully


STEP 5: Stop Milvus (When Done)
================================

Option A - With Docker Commands:
docker stop milvus-local

Option B - With Docker Compose:
docker-compose down


STEP 6: Restart Milvus (Next Time)
===================================

Option A - With Docker Commands:
docker start milvus-local

Option B - With Docker Compose:
docker-compose up -d


WHAT HAPPENS WITH YOUR DATA:
============================

- All chunks are stored in Milvus database
- Database files are saved in ~/RAGs/workflow/milvus_data
- Your data persists between runs
- Docker container is just the server, data stays on your PC
- You can stop/restart container without losing data


TROUBLESHOOTING:
================

If "Connection refused" error:

1. Check if Docker is running:
   docker info

2. Check if Milvus container is running:
   docker ps

3. Check Milvus logs:
   docker logs milvus-local

4. Wait a bit longer (takes 10-15 seconds to start):
   sleep 20
   docker logs milvus-local

5. If container won't start, restart it:
   docker stop milvus-local
   docker rm milvus-local
   docker-compose up -d -build


If permission errors:

chmod 755 milvus_data/


If port already in use:

# See what's using port 19530
lsof -i :19530

# Kill the process or use different port
