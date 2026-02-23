# evaluate_from_hub.py

from transformers import DistilBertForSequenceClassification, DistilBertTokenizerFast

model = DistilBertForSequenceClassification.from_pretrained("sps1001/distilbert-goodreads")
tokenizer = DistilBertTokenizerFast.from_pretrained("sps1001/distilbert-goodreads")

print("Model loaded from HF Hub successfully")