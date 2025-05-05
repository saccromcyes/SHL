#!/bin/bash

# Start the Flask app
export FLASK_APP=api.main
export FLASK_RUN_HOST=0.0.0.0
export FLASK_RUN_PORT=$PORT

python -m flask run
