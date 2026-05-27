# LLM Chatbot

A simple chatbot built with Flask, PyTorch, and NLTK that uses a neural network to respond to user messages based on training data.

## Features

- Neural network-based response generation
- Natural language processing with NLTK
- Web interface with Flask
- Customizable training data via intents.json

## Installation

1. Create a virtual environment:
```bash
python -m venv myenv
source myenv/bin/activate  # On Windows: myenv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Training the Model

1. Ensure `intents.json` contains your training data with patterns and responses
2. Run the training script:
```bash
python train.py
```

This will generate:
- `data.pth` - Trained model state
- `words.pkl` - Vocabulary
- `classes.pkl` - Intent classes

### Running the Chatbot

#### Command Line Interface
```bash
python chat.py
```

#### Web Interface
```bash
python app.py
```

Then open `http://localhost:5000` in your browser.

## Project Structure

- `app.py` - Flask web application
- `chat.py` - Chat logic and model inference
- `train.py` - Model training script
- `NLTK_UTILS.py` - NLP utility functions
- `intents.json` - Training data with patterns and responses
- `requirements.txt` - Python dependencies

## Customization

Edit `intents.json` to add new intents, patterns, and responses for your chatbot. After modifying, re-run `train.py` to update the model.

## Error Handling

The code includes error handling for:
- Missing `intents.json` file
- Missing trained model files (`data.pth`)
- Invalid user input in web interface

## Requirements

- Python 3.7+
- Flask 3.0.0
- TensorFlow 2.15.0
- NLTK 3.8.1
- NumPy 1.24.3
- PyTorch (installed with TensorFlow)
