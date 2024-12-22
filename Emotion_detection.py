import pandas as pd
from datasets import load_dataset

dataset = load_dataset('dair-ai/emotion')
train_data = dataset['train']
test_data = dataset['test']

from sklearn.model_selection import train_test_split

train_data_reduced = train_data.select(range(0, len(train_data) // 2))
test_data_reduced = test_data.select(range(0, len(test_data) // 2))

import re

def preprocess_text(text):
  text = text.lower()
  text = re.sub(r'[^\w\s]', '', text)
  return text

train_data_reduced = train_data_reduced.map(lambda x: {'text': preprocess_text(x['text'])})
test_data_reduced = test_data_reduced.map(lambda x: {'text': preprocess_text(x['text'])})

from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

def tokenize_function(examples):
  return tokenizer(examples['text'], padding = 'max_length', truncation = True)

train_encodings = train_data_reduced.map(tokenize_function, batched = True)
test_encodings = test_data_reduced.map(tokenize_function, batched = True)

import numpy as np

train_features = {
    'input_ids': np.array(train_encodings['input_ids']),
    'attention_mask': np.array(train_encodings['attention_mask']),
    # Uncomment if using token type ids (for BERT)
    # 'token_type_ids': np.array(train_encodings['token_type_ids'])
}

train_labels = np.array(train_data_reduced['label'])

test_features = {
    'input_ids': np.array(test_encodings['input_ids']),
    'attention_mask': np.array(test_encodings['attention_mask']),
    # Uncomment if using token type ids (for BERT)
    # 'token_type_ids': np.array(test_encodings['token_type_ids'])
}
test_labels = np.array(test_data_reduced['label'])

import tensorflow as tf

train_dataset = tf.data.Dataset.from_tensor_slices((train_features, train_labels)).shuffle(500).batch(16)
test_dataset = tf.data.Dataset.from_tensor_slices((test_features, test_labels)).shuffle(500).batch(16)

from transformers import TFBertForSequenceClassification

model = TFBertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels = 6)

model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate = 5e-5),
              loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits = True),
              metrics = ['accuracy'])

model.fit(train_dataset, epochs = 3, validation_data = test_dataset)

results = model.evaluate(test_data)
print(f'Test Loss: {results[0]}, Test Accuracy: {results[1]}')

def predict_emotion(text):
    inputs = tokenizer(text, return_tensors='tf', padding = True, truncation = True)
    logits = model(inputs).logits
    predicted_class = tf.argmax(logits, axis = 1)
    return predicted_class.numpy()

print(predict_emotion("I love this movie!"))

from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

y_true = test_data_reduced['label']
y_pred = model.predict(test_dataset).logits.argmax(axis = -1)

cm = confusion_matrix(y_true, y_pred)
sns.heatmap(cm, annot= True)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()