#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pycrfsuite
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split


# In[2]:


def load_dataset(file_path):
    dataset = []
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()
        sentence = []
        for line in lines:
            line = line.strip()
            if line:
                fields = line.split('\t')
                word = fields[1]
                pos = fields[2]  # Adjust the index based on the format
                sentence.append((word, pos))
            else:
                dataset.append(sentence)
                sentence = []
    return dataset

file_path = '/home/niket/Desktop/test.utf.conll'
dataset = load_dataset(file_path)


# In[3]:


#Split the dataset into training and testing sets
train_data, test_data = train_test_split(dataset, test_size=0.2, random_state=42)


# In[4]:


def word2features(sent, i):
    word = sent[i][0]

    features = {
        'bias': 1.0,
        'word.lower()': word.lower(),
        'word[-3:]': word[-3:],
        'word[-2:]': word[-2:],
        'word.isupper()': word.isupper(),
        'word.istitle()': word.istitle(),
        'word.isdigit()': word.isdigit(),
    }
    if i > 0:
        prev_word = sent[i-1][0]
        features.update({
            'prev_word.lower()': prev_word.lower(),
            'prev_word.isupper()': prev_word.isupper(),
            'prev_word.istitle()': prev_word.istitle(),
            'prev_word.isdigit()': prev_word.isdigit(),
        })
    else:
        features['BOS'] = True

    if i < len(sent)-1:
        next_word = sent[i+1][0]
        features.update({
            'next_word.lower()': next_word.lower(),
            'next_word.isupper()': next_word.isupper(),
            'next_word.istitle()': next_word.istitle(),
            'next_word.isdigit()': next_word.isdigit(),
        })
    else:
        features['EOS'] = True

    return features

def extract_features(dataset):
    X = []
    for sentence in dataset:
        features = []
        for i in range(len(sentence)):
            feature = word2features(sentence, i)
            features.append(feature)
        X.append(features)
    return X

def extract_labels(dataset):
    y = []
    for sentence in dataset:
        labels = [pos for _, pos in sentence]
        y.append(labels)
    return y

X_train = extract_features(train_data)
y_train = extract_labels(train_data)

X_test = extract_features(test_data)
y_test = extract_labels(test_data)


# In[5]:


# Train the CRF model
trainer = pycrfsuite.Trainer()

# Add your training data to the trainer
for x_seq, y_seq in zip(X_train, y_train):
    trainer.append(x_seq, y_seq)

# Set the model parameters
trainer.set_params({
    'c1': 0.1,
    'c2': 0.1,
    'max_iterations': 100,
    'feature.possible_transitions': True
})

# Train the CRF model and save it to a file
model_file = '/home/niket/Desktop/trained_model.crfsuite'
trainer.train(model_file)


# In[6]:


# Load the trained model
model_file = '/home/niket/Desktop/trained_model.crfsuite'
tagger = pycrfsuite.Tagger()
tagger.open(model_file)


# In[7]:


# Sample sentence
sentence = "वह प्रतिदिन कड़ी मेहनत करता है"

# Split the sentence into individual words
words = sentence.split()

# Extract features for the sample sentence
features = [word2features(words, i) for i in range(len(words))]

# Predict labels for the sample sentence
predicted_labels = tagger.tag(features)

# Print the predicted labels
for word, label in zip(words, predicted_labels):
    print(f'{word}: {label}')


# In[9]:


#Evaluation
model_file = '/home/niket/Desktop/trained_model.crfsuite'
tagger = pycrfsuite.Tagger()
tagger.open(model_file)

# Predict labels for the test data
y_pred = [tagger.tag(x_seq) for x_seq in X_test]

# Convert the true labels and predicted labels to a flattened list
y_true_flat = [label for sequence in y_test for label in sequence]
y_pred_flat = [label for sequence in y_pred for label in sequence]


# Generate the classification report
report = classification_report(y_true_flat, y_pred_flat)
print(report)


# In[ ]:




