import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import numpy as np
from utils.model import DistilBertModel, train_model
from utils.data_prep import DataPreprocessor
from torch.utils.data import TensorDataset, DataLoader

# Parameters for fast training on MacBook Air (Apple Silicon, no CUDA)
DATA_PATH = 'data/main_clean.csv'  # Use the full dataset for training
CAT_MODEL_PATH = 'models/pt_cat_modelV1'
SUB_MODEL_PATH = 'models/pt_sub_modelV1'
BATCH_SIZE = 8  # Faster, fits in RAM
EPOCHS = 1  # For quick testing
LEARNING_RATE = 2e-5
# Use MPS for Apple Silicon if available
if torch.backends.mps.is_available():
    DEVICE = torch.device('mps')
else:
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Data Preparation
books_obj = DataPreprocessor(DATA_PATH)
num_categories, num_subcategories = books_obj.get_cat_sub_numbers()
books_obj.clean_dataframe()
books_obj.tokenize_data(max_len=32)
X_train, X_test, y_cat_train, y_cat_test, y_sub_train, y_sub_test = books_obj.prepare_data()

y_cat_train = torch.tensor(np.array(y_cat_train).argmax(axis=1), dtype=torch.long)
y_cat_test = torch.tensor(np.array(y_cat_test).argmax(axis=1), dtype=torch.long)
y_sub_train = torch.tensor(np.array(y_sub_train).argmax(axis=1), dtype=torch.long)
y_sub_test = torch.tensor(np.array(y_sub_test).argmax(axis=1), dtype=torch.long)
train_input_ids = torch.tensor(X_train)
val_input_ids = torch.tensor(X_test)

# Category model
cat_train_dataset = TensorDataset(train_input_ids, y_cat_train)
cat_val_dataset = TensorDataset(val_input_ids, y_cat_test)
cat_train_dataloader = DataLoader(cat_train_dataset, batch_size=BATCH_SIZE, shuffle=True)
cat_val_dataloader = DataLoader(cat_val_dataset, batch_size=BATCH_SIZE)
print('Category train batches:', len(cat_train_dataloader))
print('Category val batches:', len(cat_val_dataloader))
cat_model = DistilBertModel(num_categories, num_subcategories)
cat_model.to(DEVICE)
print('Training category model...')
train_model(cat_model, 'category', cat_train_dataloader, cat_val_dataloader, EPOCHS, LEARNING_RATE, DEVICE)
torch.save(cat_model.state_dict(), CAT_MODEL_PATH)
print(f'Category model saved to {CAT_MODEL_PATH}')

# Subcategory model
sub_train_dataset = TensorDataset(train_input_ids, y_sub_train)
sub_val_dataset = TensorDataset(val_input_ids, y_sub_test)
sub_train_dataloader = DataLoader(sub_train_dataset, batch_size=BATCH_SIZE, shuffle=True)
sub_val_dataloader = DataLoader(sub_val_dataset, batch_size=BATCH_SIZE)
print('Subcategory train batches:', len(sub_train_dataloader))
print('Subcategory val batches:', len(sub_val_dataloader))
sub_model = DistilBertModel(num_categories, num_subcategories)
sub_model.to(DEVICE)
print('Training subcategory model...')
train_model(sub_model, 'subcategory', sub_train_dataloader, sub_val_dataloader, EPOCHS, LEARNING_RATE, DEVICE)
torch.save(sub_model.state_dict(), SUB_MODEL_PATH)
print(f'Subcategory model saved to {SUB_MODEL_PATH}') 