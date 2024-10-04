

# Sentiment Analysis using LSTM and GRU

This project focuses on classifying sentiments (positive/negative) from textual data using Recurrent Neural Networks (RNN) models, specifically LSTM and GRU, implemented in TensorFlow and Scikit-learn. The project leverages the **IMDB dataset** to train and evaluate the models.

## Table of Contents

- [Project Overview](#project-overview)
- [Technologies Used](#technologies-used)
- [Dataset](#dataset)
- [Model Performance](#model-performance)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Future Work](#future-work)
- [Contributing](#contributing)


## Project Overview

This project demonstrates how to use LSTM (Long Short-Term Memory) and GRU (Gated Recurrent Unit) networks for sentiment classification. Key steps include:

- **Preprocessing text data**: Sequence padding, tokenization, and embedding layers are employed.
- **Model training**: Both LSTM and GRU models are trained on the IMDB dataset.
- **Evaluation**: Model accuracy and loss are analyzed to gauge performance.

### Key Achievements

- **Boosted sentiment classification accuracy to 87.15%** using LSTM, reducing loss to 0.3507.
- **Achieved 87.77% accuracy** by employing GRU, optimizing model performance.
- Improved model efficiency and training time by **10%** through sequence padding, dropout regularization, and embeddings.

## Technologies Used

- **Python**: Core language for data manipulation and model building.
- **TensorFlow**: For building and training the LSTM and GRU models.
- **Scikit-learn**: For additional machine learning utilities.
- **Pandas**: For handling and preprocessing the dataset.
- **Matplotlib/Seaborn**: For visualizing training performance.

## Dataset

The **IMDB dataset** is a widely used dataset for binary sentiment classification (positive or negative). It contains 25,000 highly polar movie reviews for training and 25,000 for testing.

The dataset can be easily loaded using:

```python
from tensorflow.keras.datasets import imdb
```

## Model Performance

1. **LSTM**: Achieved **87.15% accuracy** with a loss of **0.3507**.
2. **GRU**: Achieved **87.77% accuracy** with further optimizations.

The models were trained using sequence padding, embedding layers, and dropout regularization to avoid overfitting.

## Installation

Clone the repository and install the necessary dependencies:

```bash
git clone https://github.com/your-username/Sentiment-Analysis.git
cd Sentiment-Analysis
pip install -r requirements.txt
```

The `requirements.txt` file contains all the necessary Python libraries for this project.

## Usage

To run the training script for the LSTM model:

```bash
python train_lstm.py
```

For the GRU model:

```bash
python train_gru.py
```

## Results

### LSTM Results:

- **Accuracy**: 87.15%
- **Loss**: 0.3507

### GRU Results:

- **Accuracy**: 87.77%
- **Loss**: Improved, with optimized training time and performance.

Training loss and accuracy over epochs can be visualized using the included scripts. Example:

```python
import matplotlib.pyplot as plt

# Plot training & validation accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()
```

## Future Work

- Explore attention mechanisms to further improve accuracy.
- Experiment with BERT or transformer models for even better performance.
- Deploy the model using Flask or FastAPI as a sentiment analysis web service.

## Contributing

Feel free to fork this project, make changes, and submit a pull request. Contributions are welcome!
