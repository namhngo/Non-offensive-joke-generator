# Deep Learning Final Project

This project implements two complementary natural language processing models:

1. **Hate Speech Detection Model** - Identifies and filters offensive content
2. **Non-Offensive Joke Generator** - Generates clean, appropriate jokes

## Project Overview

The project addresses the challenge of creating safe, positive online content by combining content moderation with creative text generation. The hate speech detection model can filter harmful content, while the joke generator creates appropriate, family-friendly humor.

## Models

### 1. Hate Speech and Offensive Language Detection

**File:** `dlFinal.ipynb`

- **Architecture:** Bidirectional LSTM with embedding layer
- **Dataset:** Twitter hate speech dataset (24,783 tweets)
- **Classes:** Binary classification (0: neutral, 1: offensive/hate speech)
- **Features:**
  - Text preprocessing (URL removal, user tag cleaning, stopword removal)
  - Bidirectional LSTM with dropout for regularization
  - BinaryFocalCrossentropy loss with class balancing
  - Early stopping to prevent overfitting

**Model Performance:**
- Test accuracy achieved through BiLSTM architecture
- Handles class imbalance using focal loss
- Robust text preprocessing pipeline

### 2. Non-Offensive Joke Generator

**File:** `non-offensive-joke-generator.ipynb`

- **Architecture:** Fine-tuned GPT-2 model
- **Dataset:** Q&A jokes dataset from Kaggle
- **Features:**
  - Question-answer format joke generation
  - GPT-2 tokenizer with padding support
  - Temperature control for creativity
  - Model checkpointing for best performance

## Datasets

### Hate Speech Detection
- **Source:** Twitter hate speech and offensive language dataset
- **Size:** 24,783 labeled tweets
- **Format:** CSV with tweet text and classification labels
- **Classes:** Originally 3 classes (hate speech, offensive language, neither), merged to binary

### Joke Generation
- **Source:** `joke_dataset/qajokes1.1.2.csv`
- **Format:** Question-answer pairs
- **Size:** Multiple CSV files with joke data
- **Additional datasets:** Light bulb jokes and subject-specific humor

## Requirements

```python
tensorflow
transformers==4.39
keras==2.15.0
pandas
numpy
scikit-learn
nltk
matplotlib
```

## Usage

### Hate Speech Detection

```python
# Load and preprocess data
data = pd.read_csv("train.csv")
clean_tweets = preprocess(tweet_data)

# Train model
model = create_bilstm_model()
model.fit(X_train, y_train, validation_data=(X_test, y_test))

# Make predictions
predictions = model.predict(test_texts)
```

### Joke Generation

```python
# Load pre-trained model
model = TFGPT2LMHeadModel.from_pretrained('gpt2')

# Generate joke
def generate_joke(question):
    input_ids = tokenizer.encode("Q: " + question + " A:", return_tensors='tf')
    output = model.generate(input_ids, max_length=200)
    return tokenizer.decode(output[0], skip_special_tokens=True)
```

## File Structure

```
├── dlFinal.ipynb                              # Hate speech detection model
├── non-offensive-joke-generator.ipynb         # Joke generation model
├── joke_dataset/                              # Joke datasets
│   ├── qajokes1.1.2.csv
│   ├── t_lightbulbs.csv
│   └── t_nosubject.csv
├── Normal_Hate_and_Offensive_Speeches/        # Hate speech datasets
│   ├── Hate_Speeches_*.csv
│   ├── Normal_Speeches_*.csv
│   └── Offensive_Speeches_*.csv
└── Offensive_Language_Detection/              # Additional detection resources
```

## Key Features

### Text Preprocessing Pipeline
- HTML entity removal
- User mention cleaning (@username removal)
- URL removal
- Noise symbol filtering
- Stopword removal
- NLTK tokenization

### Model Architecture Highlights
- **BiLSTM:** Captures bidirectional context for better understanding
- **Focal Loss:** Addresses class imbalance in hate speech data
- **Dropout Regularization:** Prevents overfitting
- **Early Stopping:** Optimizes training time and performance

### Safety and Ethics
- Focuses on detecting and preventing harmful content
- Generates appropriate, family-friendly humor
- Addresses online safety through content moderation

## Results

The hate speech detection model successfully identifies offensive content with high accuracy, while the joke generator produces coherent, appropriate humor. Together, these models demonstrate the potential for AI to create safer, more positive online environments.

## Future Improvements

- Expand joke categories and diversity
- Improve multilingual support
- Enhance real-time processing capabilities
- Add more sophisticated content filtering
- Implement user feedback mechanisms