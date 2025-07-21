#!/bin/bash

# Activate API environment
source venv_fastapi/bin/activate

# Start API server
uvicorn main:app --host 0.0.0.0 --port 8000 --reload