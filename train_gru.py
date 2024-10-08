# -*- coding: utf-8 -*-
"""2.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1r5YpJyWyPo828ILELtmaXVLKYaUZDzlJ
"""

from google.colab import drive
drive.mount('/content/drive')

cd /content/drive/MyDrive/100_days_of_Deep_Learning/DL-PROJECTS-MISCLINOUS/Sentiment-analysis-IMDB

ls

!pip install kaggle

import os
import json

from zipfile import ZipFile
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Embedding, GRU
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt



"""## **Load the Dataset**"""

data = pd.read_csv("IMDB Dataset.csv")

data.shape

data.head()



"""## **Data analysis & pre-processing**"""

data["sentiment"].value_counts()

# getting some information about the data
data.info()

data.isnull().sum()

data.replace({"sentiment": {"positive": 1, "negative": 0}}, inplace=True)

data.head()

data["sentiment"].value_counts()

# Step 1: Calculate the number of words in each review and store in a NumPy array
review_word_counts = data['review'].apply(lambda x: len(x.split())).to_numpy()

# Step 2: Find the maximum number of words
max_word_count = review_word_counts.max()

print(f"The maximum number of words in a review is: {max_word_count}")

review_word_counts.shape

review_word_counts # this is the number of words in each review ...

from collections import Counter

# Step 3: Count the frequency of each number in the array
frequency_counter = Counter(review_word_counts)

print(frequency_counter)

# Step 4: Get the number of unique keys (unique word counts)
num_unique_word_counts = len(frequency_counter.keys())

print(f"The number of unique word counts is: {num_unique_word_counts}")

# Step 5: Verify the highest key in the frequency counter
max_key_in_counter = max(frequency_counter.keys())
print(f"The highest word count in the frequency counter is: {max_key_in_counter}")

# Step 6: Verify the lowest key in the frequency counter
min_key_in_counter = min(frequency_counter.keys())
print(f"The lowest word count in the frequency counter is: {min_key_in_counter}")

# Step 7: Find the key with the highest frequency
most_frequent_key = max(frequency_counter, key=frequency_counter.get)
most_frequent_key_count = frequency_counter[most_frequent_key]
print(f"The word count that appears most frequently is: {most_frequent_key} with a frequency of: {most_frequent_key_count}")

sns.set()

# Convert the frequency counter to a DataFrame for easier plotting
frequency_df = pd.DataFrame(frequency_counter.items(), columns=['word_count', 'frequency'])

# Plot the count plot
plt.figure(figsize=(34, 7))
sns.barplot(x='word_count', y='frequency', data=frequency_df)
plt.title('Frequency of Word Counts in Reviews')
plt.xlabel('Number of Words')
plt.ylabel('Frequency')
plt.show()



"""## **Train-Test Split**"""

# split data into training data and test data
train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

print(train_data.shape)
print(test_data.shape)

# Tokenize text data
tokenizer = Tokenizer(num_words=5000)  # this will take only most frequent 5000 words
tokenizer.fit_on_texts(train_data["review"])  # fit on train data
X_train = pad_sequences(tokenizer.texts_to_sequences(train_data["review"]), maxlen=200)  # transform on train data
X_test = pad_sequences(tokenizer.texts_to_sequences(test_data["review"]), maxlen=200)    # transform on test data

print(X_train)
print(X_test)

Y_train = train_data["sentiment"]
Y_test = test_data["sentiment"]

print(Y_train)
print(Y_test)



"""## **Model Architecture & Evaluation**"""

# Build the model

model = Sequential()
model.add(Embedding(input_dim=5000, output_dim=128, input_length=200))
model.add(GRU(128, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(1, activation="sigmoid"))

model.summary()

# compile the model
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

history = model.fit(X_train, Y_train, epochs=10, batch_size=64, validation_split=0.2)

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])

plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')

plt.legend(['training data', 'validation data'], loc = 'upper right')

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])

plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')

plt.legend(['training data', 'validation data'], loc = 'lower right')

# testing on test data ...
loss, accuracy = model.evaluate(X_test, Y_test)
print(f"Test Loss: {loss}")
print(f"Test Accuracy: {accuracy}")



"""## **Accuracy Score**"""

# make predictions ...
y_log = model.predict(X_test)

# the predicted values in not in 0 1 form , because we use sigmoid function it returns a probability, so we have to convert it into 0 1 using some threshold value ..
y_pred = np.where(y_log>0.5,1,0)

from sklearn.metrics import accuracy_score
accuracy_score(Y_test,y_pred)



"""## **Building a Predictive System**"""

def predict_sentiment(review):
  # tokenize and pad the review
  sequence = tokenizer.texts_to_sequences([review])
  padded_sequence = pad_sequences(sequence, maxlen=200)
  prediction = model.predict(padded_sequence)
  sentiment = "positive" if prediction[0][0] > 0.5 else "negative"
  return sentiment

# example usage
new_review = "This movie was fantastic. I loved it."
sentiment = predict_sentiment(new_review)
print(f"The sentiment of the review is: {sentiment}")

# example usage
new_review = "This movie was not that good"
sentiment = predict_sentiment(new_review)
print(f"The sentiment of the review is: {sentiment}")

# example usage
new_review = "This movie was ok but not that good."
sentiment = predict_sentiment(new_review)
print(f"The sentiment of the review is: {sentiment}")

# example usage
new_review = "This movie sound was not good, but the way of represent the story was better,but cast not that good, so overall it is not good."
sentiment = predict_sentiment(new_review)
print(f"The sentiment of the review is: {sentiment}")

# example usage
new_review = "This movie sound was not good, but the way of represent the story was better,but cast not that good, but overall it is good."
sentiment = predict_sentiment(new_review)
print(f"The sentiment of the review is: {sentiment}")

