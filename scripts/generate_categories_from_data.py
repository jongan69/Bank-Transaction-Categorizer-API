import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import pandas as pd
from collections import defaultdict

DATA_PATH = 'data/main.csv'

df = pd.read_csv(DATA_PATH)
cat_dict = defaultdict(set)

for _, row in df.iterrows():
    cat = str(row['Category']).strip()
    sub = str(row['Sub_Category']).strip() if 'Sub_Category' in row else None
    if cat and cat.lower() != 'nan':
        if sub and sub.lower() != 'nan':
            cat_dict[cat].add(sub)

# Convert sets to sorted lists for readability
final_dict = {cat: sorted(list(subs)) for cat, subs in cat_dict.items()}

print("categories = {")
for cat, subs in final_dict.items():
    print(f"    '{cat}': {subs},")
print("}") 