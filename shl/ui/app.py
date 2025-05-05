import streamlit as st 
import pandas as pd
import re
from sentence_transformers import SentenceTransformer, util

# Load data
df = pd.read_csv("C:/Users/ndsha/Desktop/shl/data/shl_assessments.csv")

# 🧹 Clean and preprocess data
def extract_minutes(duration_text):
    if pd.isna(duration_text):
        return 0
    duration_text = str(duration_text).lower()
    minutes = 0
    hours_match = re.search(r'(\d+)\s*hour', duration_text)
    mins_match = re.search(r'(\d+)\s*min', duration_text)
    if hours_match:
        minutes += int(hours_match.group(1)) * 60
    if mins_match:
        minutes += int(mins_match.group(1))
    return minutes

df['duration'] = df['duration'].apply(extract_minutes)
df['roles'] = df['roles'].fillna('Not specified')
df['duration'] = df['duration'].fillna(0).astype(int)
df = df.dropna(subset=["name"])

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

def decode_test_types(codes):
    if pd.isna(codes):
        return ""
    return ", ".join([test_type_map.get(c.strip(), c.strip()) for c in codes.split(",")])

df["Full Test Types"] = df["test_type"].apply(decode_test_types)

# Load model
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

def recommend_tests(query, top_k=5):
    query_embedding = model.encode(query, convert_to_tensor=True)
    product_embeddings = model.encode(df["description"].tolist(), convert_to_tensor=True)

    scores = util.pytorch_cos_sim(query_embedding, product_embeddings)[0]
    top_results = scores.topk(k=top_k)

    results = []
    for score, idx in zip(top_results[0], top_results[1]):
        row = df.iloc[int(idx)]
        results.append({
            "Product Name": row["name"],
            "Test Types": row["Full Test Types"],
            "Remote Testing": row["remote_support"],
            "Adaptive/IRT": row["adaptive_support"],
            "Roles": row["roles"],
            "Duration (min)": row["duration"],
            "Link": row["url"],
            "Similarity": float(score)
        })

    return results

# ------------------ UI PART ------------------

st.set_page_config(page_title="⚡ SHL AI Test Recommender", layout="centered")

# 💅 Custom CSS Styling
st.markdown("""
    <style>
    body {
        background: linear-gradient(to right, #141e30, #243b55);
        color: #fff !important;
    }
    .main {
        background-color: transparent;
    }
    h1, h2, h3, h4, h5 {
        color: #f1f1f1;
    }
    .block {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(15px);
        padding: 20px;
        border-radius: 15px;
        margin-bottom: 15px;
        box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.37);
    }
    a {
        color: #00ffff !important;
        font-weight: bold;
    }
    .stTextArea textarea {
        background-color: #2c3e50;
        color: #ecf0f1;
        border-radius: 10px;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown("<h1 style='text-align: center;'>⚡ SHL AI Test Recommender</h1>", unsafe_allow_html=True)
st.markdown("<h5 style='text-align: center; color: #bdc3c7;'>Paste your job description and get AI-powered test recommendations instantly.</h5>", unsafe_allow_html=True)
st.write("")

# 🧠 Job Description Input
job_desc = st.text_area("📝 Job Description", height=200)

# 🔘 Recommendation Button
if st.button("🚀 Get Recommendations"):
    if job_desc.strip():
        with st.spinner("🤖 Thinking..."):
            results = recommend_tests(job_desc)

        st.success("✅ Top SHL Tests for Your Job Description:")

        for r in results:
            st.markdown(f"""
                <div class="block">
                    <h4>📌 {r['Product Name']}</h4>
                    <p><strong>🧬 Test Types:</strong> {r['Test Types']}</p>
                    <p><strong>🎯 Roles:</strong> {r['Roles']}</p>
                    <p><strong>⏱ Duration:</strong> {r['Duration (min)']} minutes</p>
                    <p><strong>🌐 Remote Testing:</strong> {r['Remote Testing']}</p>
                    <p><strong>📊 Adaptive/IRT:</strong> {r['Adaptive/IRT']}</p>
                    <p><strong>💡 Similarity:</strong> {r['Similarity']:.2f}</p>
                    <a href="{r['Link']}" target="_blank">🔗 View Test</a>
                </div>
            """, unsafe_allow_html=True)
    else:
        st.warning("⚠️ Please paste a job description.")
