import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from app.recommender import recommend_tests
results = recommend_tests("AI Developer")
for r in results:
    print(r)
