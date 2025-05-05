import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI
from app.recommender import recommend_tests

app = FastAPI()
app = FastAPI()

# ✅ Allow your HTML page to talk to this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # allow all (for development)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def read_root():
    return {"message": "Welcome to SHL Test Recommender"}

@app.get("/recommend")
def get_recommendation(query: str):
    return {"recommendations": recommend_tests(query)}

