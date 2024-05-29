import numpy as np
import nltk
# nltk.download('punkt')  # UNCOMMENT THIS LINE IF YOU NEED TO DOWNLOAD 'punkt'
from nltk.stem.porter import PorterStemmer
stemmer = PorterStemmer()

# TOKENIZE A SENTENCE INTO WORDS/TOKENS
def tokenize(sentence):
    """
    Split sentence into an array of words/tokens.
    A token can be a word, punctuation character, or number.
    """
    return nltk.word_tokenize(sentence)

# PERFORM STEMMING ON A WORD
def stem(word):
    """
    Stemming: Find the root form of the word.
    Example:
    words = ["organize", "organizes", "organizing"]
    words = [stem(w) for w in words]
    -> ["organ", "organ", "organ"]
    """
    return stemmer.stem(word.lower())

# CREATE A BAG OF WORDS REPRESENTATION
def bag_of_words(tokenized_sentence, words):
    """
    Return a bag of words array:
    1 for each known word that exists in the sentence, 0 otherwise.
    Example:
    sentence = ["hello", "how", "are", "you"]
    words = ["hi", "hello", "I", "you", "bye", "thank", "cool"]
    bag   = [  0 ,    1 ,    0 ,   1 ,    0 ,    0 ,      0]
    """
    # STEM EACH WORD IN THE TOKENIZED SENTENCE
    sentence_words = [stem(word) for word in tokenized_sentence]
    # INITIALIZE A BAG WITH 0 FOR EACH WORD
    bag = np.zeros(len(words), dtype=np.float32)
    
    # SET BAG ELEMENTS TO 1 IF THE WORD IS PRESENT IN THE SENTENCE
    for idx, w in enumerate(words):
        if w in sentence_words: 
            bag[idx] = 1

    return bag