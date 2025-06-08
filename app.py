import gradio as gr
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import contractions
import html
import re

# Load model and tokenizer
model_path = "my_model"  # directory with your trained model
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)

# Labels
id2label = {
    0: 'Anxiety',
    1: 'BPD',
    2: 'Normal',
    3: 'bipolar',
    4: 'depression',
    5: 'mentalillness',
    6: 'schizophrenia'
}

def clean_text(text):
    text = str(text)
    text = text.lower()
    text = contractions.fix(text)
    text = html.unescape(text)
    text = re.sub(r'http\S+', '', text)  # Remove URLs
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Remove special characters
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Inference function
def predict(text):
    text = clean_text(text)
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    pred_id = outputs.logits.argmax(dim=1).item()
    return id2label[pred_id]

# Gradio UI
demo = gr.Interface(
    fn=predict,
    inputs=gr.Textbox(lines=3, placeholder="Enter text here..."),
    outputs=gr.Label(),
    title="Mental Health Text Classifier",
    description="Predicts mental health category based on text input."
)

if __name__ == "__main__":
    demo.launch()
