import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.dicts import categories

# List of subcategories/categories to remove (from earlier script output)
REMOVE = [
    ('Auto', 'Upgrades'),
    ('Electronics', 'Camera'),
    ('Electronics', 'Gaming'),
    ('Electronics', 'Phone'),
    ('Entertainment', 'Books'),
    ('Entertainment', 'DateNights'),
    ('Entertainment', 'E_Other'),
    ('Entertainment', 'Movies_TV'),
    ('Food', 'FastFood_Restaurants'),
    ('Personal_Care', 'Hair'),
    ('Personal_Care', 'Massage'),
    ('Subscriptions_Memberships', None),
    ('Subscriptions_Memberships', 'Entertainment'),
    ('Subscriptions_Memberships', 'Gym'),
    ('Subscriptions_Memberships', 'Sub_Other'),
    ('Travel', None),
    ('Travel', 'Activities'),
    ('Travel', 'Car_Rental'),
    ('Travel', 'Flights'),
    ('Travel', 'Hotels'),
]

# Flatten dict and remove as needed until sum is 71
trimmed = {cat: subs.copy() for cat, subs in categories.items()}
removed_count = 0
for cat, sub in REMOVE:
    if cat in trimmed:
        if sub is None:
            removed_count += 1 + len(trimmed[cat])
            del trimmed[cat]
        elif sub in trimmed[cat]:
            trimmed[cat].remove(sub)
            removed_count += 1
        if removed_count >= 4:  # Remove at least 4 to go from 75 to 71
            break

# Print the trimmed dict and its total
total = len(trimmed) + sum(len(v) for v in trimmed.values())
print(f"Trimmed total outputs: {total}")
print("categories = {")
for cat, subs in trimmed.items():
    print(f"    '{cat}': {subs},")
print("}") 