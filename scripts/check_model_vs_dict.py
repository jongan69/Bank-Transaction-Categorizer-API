import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
from utils.dicts import categories
from sklearn.preprocessing import LabelEncoder

# Path to your trained model
MODEL_PATH = 'models/pt_cat_modelV1'

# 1. Get expected output size from dict
num_categories = len(categories)
num_subcategories = sum(len(v) for v in categories.values())
expected_size = num_categories + num_subcategories
print(f'Expected output size (from dict): {expected_size}')

# 2. Get actual output size from model file
state_dict = torch.load(MODEL_PATH, map_location='cpu')
actual_size = state_dict['bert_model.classifier.weight'].shape[0]
print(f'Actual output size (from model): {actual_size}')

# 3. Print all categories and subcategories
print('\nCategories:')
for cat in categories:
    print(f'  {cat}')
print('\nSubcategories:')
for cat, subs in categories.items():
    for sub in subs:
        print(f'  {cat} -> {sub}')

# 4. If there is a mismatch, help user find the difference
if expected_size != actual_size:
    print(f'\nWARNING: Mismatch detected!')
    print(f'Your model was trained for {actual_size} outputs, but your code expects {expected_size}.')
    print('This usually means categories or subcategories were added/removed after training.')
    print('To fix:')
    print('- Edit utils/dicts.py to match the set used during training, or')
    print('- Retrain the model with the current dict.')
else:
    print('\nYour model and code are in sync!')

# Get subcategories from dicts.py
subcategories_dict = [item for sublist in categories.values() for item in sublist]

# Try to load the model's label encoder (if you saved it), otherwise reconstruct as in training
# Here, we reconstruct as in your code
label_encoder = LabelEncoder()
label_encoder.fit(subcategories_dict)
model_subcategories = list(label_encoder.classes_)

print(f"Subcategories in dicts.py ({len(subcategories_dict)}):")
print(sorted(subcategories_dict))
print()
print(f"Subcategories in model label encoder ({len(model_subcategories)}):")
print(sorted(model_subcategories))
print()

# Check for missing/extra
missing_in_dict = set(model_subcategories) - set(subcategories_dict)
missing_in_model = set(subcategories_dict) - set(model_subcategories)

if missing_in_dict:
    print("Subcategories present in model but missing in dicts.py:")
    print(sorted(missing_in_dict))
else:
    print("No subcategories missing in dicts.py.")

if missing_in_model:
    print("Subcategories present in dicts.py but missing in model:")
    print(sorted(missing_in_model))
else:
    print("No subcategories missing in model.")

print(f"\nTotal subcategories in dicts.py: {len(subcategories_dict)}")
print(f"Total subcategories in model label encoder: {len(model_subcategories)}")

if len(subcategories_dict) != len(model_subcategories):
    print("WARNING: The number of subcategories in dicts.py and the model do not match!")
else:
    print("Subcategory counts match.") 