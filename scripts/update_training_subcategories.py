import sys
import os
import pandas as pd
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.dicts import categories

# Mapping of (Category, old Sub_Category) to new unique Sub_Category
rename_map = {
    ('Entertainment', 'Sports_Outdoors'): 'Entertainment_Sports_Outdoors',
    ('Personal_Care', 'Sports_Outdoors'): 'Personal_Care_Sports_Outdoors',
    ('Clothes', 'Clothes'): 'Clothes_Clothes',
    ('Baby', 'Clothes'): 'Baby_Clothes',
    ('Home', 'Gym'): 'Home_Gym',
    ('Subscriptions_Memberships', 'Gym'): 'Subscriptions_Memberships_Gym',
    ('Home', 'Maintenance'): 'Home_Maintenance',
    ('Auto', 'Maintenance'): 'Auto_Maintenance',
}

def update_subcategories(input_csv, output_csv):
    df = pd.read_csv(input_csv)
    changes = 0
    for idx, row in df.iterrows():
        cat = row['Category']
        sub = row['Sub_Category']
        key = (cat, sub)
        if key in rename_map:
            df.at[idx, 'Sub_Category'] = rename_map[key]
            changes += 1
    df.to_csv(output_csv, index=False)
    print(f"Updated {changes} subcategory values.")
    print(f"Output written to {output_csv}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python update_training_subcategories.py <input_csv> [<output_csv>]")
        sys.exit(1)
    input_csv = sys.argv[1]
    output_csv = sys.argv[2] if len(sys.argv) > 2 else input_csv.replace('.csv', '_updated.csv')
    update_subcategories(input_csv, output_csv) 