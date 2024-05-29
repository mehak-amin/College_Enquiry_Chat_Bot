import random
import json
import torch

# IMPORT CUSTOM MODULES
from college.model import NeuralNet
from college.nltk_utils import bag_of_words, tokenize


# CHECK FOR GPU AVAILABILITY
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# LOAD INTENTS DATA FROM JSON
with open("intensts.json", 'r') as json_data:
    intents = json.load(json_data)

# LOAD PREVIOUSLY SAVED MODEL DATA
FILE = "data.pth"
data = torch.load(FILE)

# EXTRACT MODEL PARAMETERS FROM THE LOADED DATA
input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data['all_words']
tags = data['tags']
model_state = data["model_state"]

# CREATE INSTANCE OF THE NEURAL NETWORK MODEL
model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)  # LOAD MODEL PARAMETERS
model.eval()  # SET MODEL TO EVALUATION MODE

# FALLBACK RESPONSES
fallback = [
    "I apologize, but I am having difficulty comprehending your query",
    "Regrettably, due to certain limitations inherent to my bot functionality I cant answer the question"
]

# THE RESPONSE IF THE LEN OF QUESTION IN LESS THAN THE USUAL
x = "Could you please rephrase your question? It should be at least long enough to be understand what actually you want."

# BOT NAME
bot_name = "EduNav"

# FUNCTION TO GENERATE BOT RESPONSE
def get_response(msg, logged_in):
    if not logged_in:
        limited_tags = ['greetings', 'goodbye']  
        # Modify this list based on the tags you want to include for non-logged in users
        
        sentence = tokenize(msg)  
        X = bag_of_words(sentence, all_words)  
        X = X.reshape(1, X.shape[0])
        X = torch.from_numpy(X).to(device)

        output = model(X)
        _, predicted = torch.max(output, dim=1)

        tag = tags[predicted.item()]

        if tag in limited_tags:
            for intent in intents["intents"]:
                if tag == intent["tag"]:
                    return random.choice(intent['responses'])
        else:
            return "Please login to have access a lot more questions."
    else:
        sentence = tokenize(msg)  # TOKENIZE THE USER INPUT
        X = bag_of_words(sentence, all_words)  # CREATE BAG OF WORDS
        X = X.reshape(1, X.shape[0])
        X = torch.from_numpy(X).to(device)

        # FORWARD PASS THROUGH THE MODEL
        output = model(X)
        _, predicted = torch.max(output, dim=1)

        tag = tags[predicted.item()]

        probs = torch.softmax(output, dim=1)
        prob = probs[0][predicted.item()]
        if len(sentence) <= 2:
            return x
        elif prob.item() > 0.99:  
            for intent in intents["intents"]:
                if tag == intent["tag"]:
                    return random.choice(intent['responses'])
        
        return random.choice(fallback)

# MAIN INTERACTION LOOP
if __name__ == "__main__":
    print("Let's chat! (type 'quit' to exit)")
    logged_in = False
    while True:
        sentence = input("You: ").lower()
        if sentence == "quit":
            break

        resp = get_response(sentence, logged_in)
        print(f"{bot_name}: {resp}")