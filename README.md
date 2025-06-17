# Bank Transaction Categorizer API

A privacy-friendly, self-hosted backend API that uses a neural network (BERT) to categorize bank transaction descriptions into categories and subcategories.

---

## 💡 Motivation

- **Self-hosting:** Run your own transaction categorization service, with no reliance on third-party APIs or cloud services.
- **Privacy:** Your financial data never leaves your infrastructure.

---

## 📋 Overview

This project provides a FastAPI backend for categorizing bank transaction descriptions. It leverages a fine-tuned BERT model to classify transactions into customizable categories and subcategories.

---

## ✨ Features

- **REST API:** Accepts POST requests with transaction descriptions and returns predicted categories/subcategories.
- **BERT-based:** Uses a state-of-the-art NLP model for high accuracy.
- **Customizable:** Easily update categories/subcategories and retrain the model.
- **No GUI:** Pure backend, ready for integration with your apps or services.

---

## 🚀 Getting Started

### 1. **Install Dependencies**

```bash
pip install -r requirements.txt
```

### 2. **Train the Model**

Train on your own data (CSV with `Description`, `Category`, `Sub_Category` columns):

```bash
python scripts/train_models.py
```
- By default, this uses `data/main.csv` and saves models to `models/pt_cat_modelV1` and `models/pt_sub_modelV1`.
- Adjust `BATCH_SIZE` in `scripts/train_models.py` if you have limited RAM.

### 3. **Run the API**

```bash
uvicorn api:app --reload
```

### 4. **Make a Request**

Send a POST request to `/categorize`:

```json
POST /categorize
{
  "transactions": [
    {"Description": "Starbucks Coffee"},
    {"Description": "Shell Gas Station"}
  ]
}
```

**Response:**
```json
{
  "results": [
    {"Description": "Starbucks Coffee", "Category": "Food", "Subcategory": "FastFood_Restaurants"},
    {"Description": "Shell Gas Station", "Category": "Auto", "Subcategory": "Gas"}
  ]
}
```

---

## 🏷️ Categories & Subcategories

Default categories and subcategories (see `utils/dicts.py`):

```python
categories = {
    'Auto': ['Gas','Maintenance', 'Upgrades', 'Other_Auto'],
    'Baby': ['Diapers', 'Formula', 'Clothes', 'Toys', 'Other_Baby'],
    'Clothes': ['Clothes', 'Shoes', 'Jewelry', 'Bags_Accessories'],
    'Entertainment': ['Sports_Outdoors', 'Movies_TV', 'DateNights', 'Arts_Crafts', 'Books', 'Games', 'Guns', 'E_Other'],
    'Electronics': ['Accessories', 'Computer', 'TV', 'Camera', 'Phone','Tablet_Watch', 'Gaming', 'Electronics_misc'],
    'Food': ['Groceries', 'FastFood_Restaurants'],
    'Home': ['Maintenance', 'Furniture_Appliances', 'Hygiene', 'Gym',
        'Home_Essentials', 'Kitchen', 'Decor', 'Security', 'Yard_Garden', 'Tools'],
    'Medical': ['Health_Wellness'],
    'Kids': ['K_Toys'],
    'Personal_Care': ['Hair', 'Makeup_Nails', 'Beauty', 'Massage','Vitamins_Supplements', 'PC_Other'],
    'Pets': ['Pet_Food', 'Pet_Toys', 'Pet_Med', 'Pet_Grooming', 'Pet_Other'],
    'Subscriptions_Memberships': ['Entertainment', 'Gym', 'Sub_Other'],
    'Travel': ['Hotels', 'Flights', 'Car_Rental', 'Activities']
}
```

---

## 🛠️ Customization

1. **Edit `utils/dicts.py`** to add/remove categories or subcategories.
2. **Update your training data** to include the new labels.
3. **Retrain the model** with `python scripts/train_models.py`.
4. **Restart the API** to use the new model.

---

## 🏗️ Project Structure

- `api.py` — FastAPI backend
- `utils/` — Model, data preprocessing, and category logic
- `models/` — Trained model weights
- `data/` — Training data (CSV)
- `scripts/` — Training scripts

---

## ⚡ Performance

- Trained on 62,000+ transactions, BERT-base model.
- Inference on a Raspberry Pi 4 (8GB): ~2–8 seconds per transaction (batching possible).
- For faster inference, consider using DistilBERT or TinyBERT.

---

## 🔮 Future Improvements

- Support for lightweight models (DistilBERT, TinyBERT)
- ONNX/TensorFlow Lite export for edge devices
- More granular categories (e.g., Utilities, Insurance, Income)
- Docker deployment

---

## 🤝 Contributing

- PRs and issues welcome!
- Please star the repo if you find it useful.

---

## 📄 License

MIT

# Bank-Transaction-Categorizer-API
