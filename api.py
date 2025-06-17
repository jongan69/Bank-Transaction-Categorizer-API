from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import pandas as pd
import torch
import os
from utils.api_predict import BankTransactionCategorizer

app = FastAPI()

class Transaction(BaseModel):
    Description: str

class TransactionsRequest(BaseModel):
    transactions: List[Transaction]

class CategorizedTransaction(BaseModel):
    Description: str
    Category: str
    Subcategory: str

class CategorizedResponse(BaseModel):
    results: List[CategorizedTransaction]

# Initialize the categorizer once (loads models)
categorizer = BankTransactionCategorizer()

@app.post("/categorize", response_model=CategorizedResponse)
def categorize_transactions(request: TransactionsRequest):
    if not request.transactions:
        raise HTTPException(status_code=400, detail="No transactions provided.")
    # Prepare DataFrame
    df = pd.DataFrame([t.dict() for t in request.transactions])
    # Predict
    results_df = categorizer.predict(df)
    # Build response
    results = [CategorizedTransaction(
        Description=row["Description"],
        Category=row["Category"],
        Subcategory=row["Subcategory"]
    ) for _, row in results_df.iterrows()]
    return CategorizedResponse(results=results) 