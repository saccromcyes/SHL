#!/bin/bash

# Navigate to the directory where your Flask app is located (optional, if needed)
# cd path/to/your/app


# Start the Flask app
export FLASK_APP="C:/Users/ndsha/Desktop/shl/api/main.py" # update this path to your actual main Flask file
export FLASK_RUN_HOST=0.0.0.0
export FLASK_RUN_PORT=$PORT

python -m flask run