import torch
from transformers import BertForSequenceClassification
from .data_processing import MultiLabelDataset

# in progress
def predict_categories(model_path, tokenizer, new_sentences):
    model = BertForSequenceClassification.from_pretrained(model_path)
    new_encodings = tokenizer(new_sentences, truncation=True, padding=True, max_length=128)
    new_dataset = MultiLabelDataset(new_encodings, [[0]*len(category_columns)])
    predictions = model.predict(new_dataset)
    predicted_labels = (torch.sigmoid(torch.tensor(predictions[0])) > 0.4).int()
    predicted_categories = [category_columns[i] for i in range(len(category_columns)) if predicted_labels[0][i] == 1]
    return predicted_categories