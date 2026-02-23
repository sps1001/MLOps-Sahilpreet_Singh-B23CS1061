# Goodreads Genre Classification with DistilBERT

This project demonstrates fine-tuning a DistilBERT model to classify book reviews from Goodreads into their respective genres.

## Project Overview

The core of this project is fine-tuning a BERT-based model (DistilBERT) on the [UCSD Book Graph](https://mengtingwan.github.io/data/goodreads.html) dataset. It compares the transformer-based approach against a baseline Logistic Regression model.

### Supported Genres:
- Poetry
- Comics & Graphic
- Fantasy & Paranormal
- History & Biography
- Mystery, Thriller, & Crime
- Romance
- Young Adult

## Features

- **Data Processing**: Downloads and samples reviews from the UCSD dataset.
- **Baseline Model**: Implements a TF-IDF + Logistic Regression baseline for comparison.
- **Transformer Fine-tuning**: Uses HuggingFace `transformers` to fine-tune `distilbert-base-cased`.
- **Evaluation**: Includes detailed classification reports and misclassification heatmaps.
- **Docker Support**: Provided `Dockerfile` for containerized training and evaluation.
- **HuggingFace Integration**: Scripts to push and pull models from the HuggingFace Hub.

## Setup and Installation

1. **Clone the repository**:
   ```bash
   git clone <your-repo-url>
   cd MLOps-Sahilpreet_Singh-B23CS1061
   ```

2. **Create a virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Training and Fine-Tuning
Run the main script to train the model:
```bash
python ml_dl_ops_ass_3_fine_tuning_classification.py
```

### Evaluation
To verify the model can be loaded from the HuggingFace Hub:
```bash
python evaluate.py
```

### Docker
Build and run with Docker:
```bash
# Build the image
docker build -t goodreads-classifier .

# Run the container
docker run goodreads-classifier
```

## Repository Structure

- `ml_dl_ops_ass_3_fine_tuning_classification.py`: Main training and evaluation script.
- `evaluate.py`: Script to load and test the model from HuggingFace.
- `Dockerfile`: Container configuration.
- `requirements.txt`: Python dependencies.
- `.gitignore`: Configured to exclude large model files and virtual environments.

## Model Hub
The fine-tuned model is available on the HuggingFace Hub: [sps1001/distilbert-goodreads](https://huggingface.co/sps1001/distilbert-goodreads)
