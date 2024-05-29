import numpy as np
import json

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# CUSTOM MODULES
from nltk_utils import bag_of_words, tokenize, stem
from model import NeuralNet

# LOAD INTENTS DATA FROM JSON
with open("intensts.json", 'r') as f:
    intents = json.load(f)

# INITIALIZE EMPTY LISTS FOR WORDS, TAGS, AND (PATTERNS, TAGS) PAIRS
all_words = []
tags = []
xy = []

# LOOP THROUGH EACH INTENT IN THE INTENTS DATA
for intent in intents['intents']:
    tag = intent['tag']
    tags.append(tag)  # ADD INTENT TAG TO THE TAGS LIST
    
    # LOOP THROUGH EACH PATTERN IN THE CURRENT INTENT
    for pattern in intent['patterns']:
        # TOKENIZE THE PATTERN INTO WORDS
        w = tokenize(pattern)
        all_words.extend(w)  # ADD WORDS TO THE ALL_WORDS LIST
        xy.append((w, tag))  # ADD (WORDS, TAG) PAIR TO THE XY LIST

# DEFINE WORDS TO IGNORE IN STEMMING
ignore_words = ['?', '.', '!']

# STEM AND LOWERCASE EACH WORD, REMOVE DUPLICATES, AND SORT
all_words = [stem(w) for w in all_words if w not in ignore_words]
all_words = sorted(set(all_words))
tags = sorted(set(tags))

# PRINT INFORMATION ABOUT THE DATA
# print(len(xy), "patterns")
# print(len(tags), "tags:", tags)
# print(len(all_words), "unique stemmed words:", all_words)

# INITIALIZE TRAINING DATA ARRAYS
X_train = []
y_train = []

# PROCESS EACH (PATTERN, TAG) PAIR
for (pattern_sentence, tag) in xy:
    # CREATE BAG OF WORDS REPRESENTATION FOR THE PATTERN
    bag = bag_of_words(pattern_sentence, all_words)
    X_train.append(bag)  # ADD BAG OF WORDS TO X_TRAIN
    
    # CONVERT TAG TO NUMERICAL LABEL
    label = tags.index(tag)
    y_train.append(label)  # ADD LABEL TO Y_TRAIN

# CONVERT TRAINING DATA ARRAYS TO NUMPY ARRAYS
X_train = np.array(X_train)
y_train = np.array(y_train)

# HYPERPARAMETERS
num_epochs = 2000
batch_size = 8
learning_rate = 0.0005
input_size = len(X_train[0])
hidden_size = 16
output_size = len(tags)

# DISPLAY INPUT AND OUTPUT SIZES
print(input_size, output_size)

# DEFINE A CUSTOM DATASET CLASS FOR PYTORCH
class ChatDataset(Dataset):
    def __init__(self):
        self.n_samples = len(X_train)
        self.x_data = X_train
        self.y_data = y_train
    
    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]
    
    def __len__(self):
        return self.n_samples

# CREATE AN INSTANCE OF THE CUSTOM DATASET
dataset = ChatDataset()

# CREATE A DATALOADER FOR EFFICIENTLY LOADING DATA IN BATCHES
train_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=0)

# CHECK GPU AVAILABILITY AND DEFINE THE DEVICE
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# CREATE AN INSTANCE OF THE NEURALNET MODEL AND MOVE IT TO THE APPROPRIATE DEVICE
model = NeuralNet(input_size, hidden_size, output_size).to(device)

# DEFINE LOSS FUNCTION AND OPTIMIZER
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# TRAINING LOOP
for epoch in range(num_epochs):
    for (words, labels) in train_loader:
        words = words.to(device)  # MOVE DATA TO DEVICE
        labels = labels.to(dtype=torch.long).to(device)
        
        # FORWARD PASS
        outputs = model(words)
        loss = criterion(outputs, labels)  # CALCULATE LOSS
        
        # BACKPROPAGATION AND OPTIMIZATION
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    # PRINT LOSS AT CERTAIN INTERVALS
    if (epoch+1) % 100 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# PRINT FINAL LOSS
print(f'final loss: {loss.item():.4f}')


# CREATE A DICTIONARY TO SAVE MODEL DATA
data = {
    "model_state": model.state_dict(),
    "input_size": input_size,
    "hidden_size": hidden_size,
    "output_size": output_size,
    "all_words": all_words,
    "tags": tags
}

# SAVE THE MODEL DATA TO A FILE
FILE = "data.pth"
torch.save(data, FILE)
print(f'training complete. file saved to {FILE}')