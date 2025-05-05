import pandas as pd
from sentence_transformers import SentenceTransformer, util
import re

# Load CSV
df = pd.read_csv("C:/Users/ndsha/Desktop/shl/data/shl_assessments.csv")

# Drop rows where 'name' is missing
df = df.dropna(subset=["name"])

# Fill missing 'duration' and 'roles' with 'NA'
df['duration'] = df['duration'].fillna('NA')
df['roles'] = df['roles'].fillna('NA')

# Convert duration text to minutes (keep 'NA' as is)
def convert_duration(value):
    if value == 'NA' or pd.isna(value):
        return 'NA'
    value = value.lower()
    match = re.search(r'(\d+)', value)
    if not match:
        return 'NA'
    num = int(match.group(1))
    return num * 60 if 'hour' in value else num

df['duration'] = df['duration'].apply(convert_duration)

# Mapping test type keys
test_type_map = {
    'A': 'Ability & Aptitude',
    'B': 'Biodata & Situational Judgement',
    'C': 'Competencies',
    'D': 'Development & 360',
    'E': 'Assessment Exercises',
    'K': 'Knowledge & Skills',
    'P': 'Personality & Behavior',
    'S': 'Simulations'
}

# Load the embedding model
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# Decode test type keys into human-readable format
def decode_test_types(codes):
    if pd.isna(codes):
        return ""
    separators = [",", " "]
    for sep in separators:
        if sep in codes:
            parts = [part.strip() for part in codes.split(sep) if part.strip()]
            break
    else:
        parts = [codes.strip()]
    return ", ".join([test_type_map.get(code, code) for code in parts])

# Main recommender function
def recommend_tests(query, top_k=5):
    print(f"Received query: {query}")
    query_embedding = model.encode(query, convert_to_tensor=True)
    product_embeddings = model.encode(df["name"].tolist(), convert_to_tensor=True)

    scores = util.pytorch_cos_sim(query_embedding, product_embeddings)[0]
    top_results = scores.topk(k=top_k)

    results = []
    for score, idx in zip(top_results[0], top_results[1]):
        row = df.iloc[int(idx)]
        results.append({
            "Name": row["name"],
            "Link": row.get("url", "N/A"),
            "Duration (mins)": row["duration"],
            "Roles": row["roles"],
            "Remote Testing": row.get("remote_support", "N/A"),
            "Adaptive/IRT": row.get("adaptive_support", "N/A"),
            "Test Types": decode_test_types(row.get("test_type", "")),
            "Similarity Score": float(score),
        })

    print(f"Returning {len(results)} results.")
    return results
