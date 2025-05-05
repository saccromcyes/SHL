import pandas as pd

# Read the CSV
df = pd.read_csv("C:/Users/ndsha/Desktop/shl/data/shl_assessments.csv")

# Mapping for test type keys
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

# Create a new column with full type names
# Some rows may have multiple codes like "A,P", so we split and map
def decode_test_types(codes):
    if pd.isna(codes):
        return ""
    return ", ".join([test_type_map.get(code.strip(), code.strip()) for code in codes.split(",")])

df['Full Test Types'] = df['Test Type Keys'].apply(decode_test_types)

# Show first few rows
print(df[['Product Name', 'Test Type Keys', 'Full Test Types']].head())
