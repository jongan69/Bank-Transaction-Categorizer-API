import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader
from transformers import DistilBertTokenizer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from .data_prep import DataPreprocessor
from .model import DistilBertModel
from .dicts import categories

class BankTransactionCategorizer:
    def __init__(self, cat_model_path='models/pt_cat_modelV1', sub_model_path='models/pt_sub_modelV1'):
        self.cat_model_path = cat_model_path
        self.sub_model_path = sub_model_path
        self.category_keys = list(categories.keys())
        self.category_values = [item for sublist in categories.values() for item in sublist]
        self.num_categories = len(self.category_keys)
        self.num_subcategories = len(self.category_values)
        self.tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        # Use MPS for Apple Silicon if available
        if torch.backends.mps.is_available():
            self.device = torch.device('mps')
        else:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.label_encoder_cat = LabelEncoder()
        self.label_encoder_subcat = LabelEncoder()
        self.label_encoder_cat.fit(self.category_keys)
        self.label_encoder_subcat.fit(self.category_values)
        self.cat_model = self._load_model(self.cat_model_path)
        self.sub_model = self._load_model(self.sub_model_path)

    def _load_model(self, model_path):
        model = DistilBertModel(self.num_categories, self.num_subcategories)
        state_dict = torch.load(model_path, map_location=self.device)
        model.load_state_dict(state_dict)
        model.to(self.device)
        model.eval()
        return model

    def predict(self, df: pd.DataFrame):
        # Prepare Data
        df_obj = DataPreprocessor(df)
        df_obj.clean_dataframe()
        X_predict = df_obj.tokenize_predict_data(max_len=32)
        predict_input_ids = torch.tensor(X_predict, dtype=torch.long)
        predict_dataset = TensorDataset(predict_input_ids)
        predict_dataloader = DataLoader(predict_dataset, batch_size=1, shuffle=False)
        # Predict
        categories, subcategories, descriptions = [], [], []
        for batch in predict_dataloader:
            input_ids = batch[0].to(self.device)
            with torch.no_grad():
                category_probs, _ = self.cat_model(input_ids)
                category_predictions = category_probs.argmax(dim=-1)
                _, subcategory_probs = self.sub_model(input_ids)
                subcategory_predictions = subcategory_probs.argmax(dim=-1)
            for i in range(input_ids.size(0)):
                category_name = self.label_encoder_cat.inverse_transform([category_predictions[i].item()])[0]
                subcategory_name = self.label_encoder_subcat.inverse_transform([subcategory_predictions[i].item()])[0]
                single_input_ids = input_ids[i].to('cpu')
                tokens = self.tokenizer.convert_ids_to_tokens(single_input_ids)
                description = self.tokenizer.convert_tokens_to_string([token for token in tokens if token != "[PAD]"]).strip()
                categories.append(category_name)
                subcategories.append(subcategory_name)
                descriptions.append(description)
        # Return as DataFrame
        result_df = pd.DataFrame({
            'Description': descriptions,
            'Category': categories,
            'Subcategory': subcategories
        })
        return result_df 