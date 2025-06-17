# Hugging Face Bank Transaction Categorizer API

A FastAPI backend that uses models hosted on Hugging Face Hub to categorize bank transaction descriptions into categories and subcategories.

---

## 🚀 Getting Started

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the API

```bash
uvicorn main:app --reload
```

### 3. Make a Request

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

## 🧠 How it Works
- Loads two DistilBERT models from Hugging Face Hub (`jonngan/trans-cat` and `jonngan/trans-subcat`).
- Tokenizes and preprocesses transaction descriptions.
- Returns predicted category and subcategory for each transaction.

---

## 📦 Folder Structure
- `main.py` — FastAPI backend
- `utils/` — Model, data preprocessing, and category logic (copied from main repo)
- `requirements.txt` — Python dependencies

---

## ⚡ Notes
- No local model files are needed; everything loads from Hugging Face Hub.
- You can copy this folder anywhere and run the API as long as Python dependencies are installed.

---

## 📄 License
MIT 