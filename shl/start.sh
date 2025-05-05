#!/bin/bash

# Start FastAPI app using uvicorn
uvicorn api.main:app --host 0.0.0.0 --port $PORT
