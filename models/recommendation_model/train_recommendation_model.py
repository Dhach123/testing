import pandas as pd
from transformers import BertTokenizer, BertModel
import torch
from sklearn.metrics.pairwise import cosine_similarity

# Load the dataset
data = pd.read_csv('./data/enhanced_dataset_with_synthetic_negotiations.csv')

# Prepare the BERT model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertModel.from_pretrained('bert-base-uncased')

def get_bert_embedding(text):
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True)
    with torch.no_grad():
        outputs = bert_model(**inputs)
    return outputs.last_hidden_state.mean(dim=1)

data['bert_embedding'] = data['clean_description'].apply(lambda x: get_bert_embedding(x).numpy().flatten())
