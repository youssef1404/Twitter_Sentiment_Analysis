import re, os
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import joblib
from typing import List, Literal
from gensim.models import Word2Vec, FastText
import numpy as np
from nltk import word_tokenize

# Vectorizer 
bow_vectorizer = joblib.load(os.path.join(os.getcwd(), 'artifacts', 'bow_vectorizer.pkl'))
tfidf_vectorizer = joblib.load(os.path.join(os.getcwd(), 'artifacts', 'tfidf_vectorizer.pkl'))
w2v_vectorizer = Word2Vec.load(os.path.join(os.getcwd(), 'artifacts', 'word2vec_model.model')) 

# Model
bow_svc_model = joblib.load(os.path.join(os.getcwd(), 'artifacts', 'svm-bow.pkl'))
tfidf_svc_model = joblib.load(os.path.join(os.getcwd(), 'artifacts', 'svm-tfidf.pkl'))
w2v_svc_model = joblib.load(os.path.join(os.getcwd(), 'artifacts', 'svm-w2v.pkl'))


## Remove unwanted text patterns from the tweets
def remove_pattern(input_txt: str, pattern: str):
    ''' This Function takes the input and pattern you want to remove

    Args:
    *****
        (input_text: str) --> The text you want to apply the function to it.
        (pattern: str) --> The pattern you want to remove from the text.
    '''
    input_txt = re.sub(pattern, '', input_txt)
    return input_txt


## A Function to remove excessive repeated chars while preserving correct words
def remove_excessive_repeated_characters(input_string, max_repeats=2):
    ## Define a regular expression pattern to match consecutive repeated characters
    pattern = f"(\\w)\\1{{{max_repeats},}}"
    ## Replace the matched pattern with a single occurrence of the character
    cleaned_string = re.sub(pattern, r"\1", input_string)
    
    return cleaned_string



emoticon_meanings = {
    ":)": "Happy",
    ":(": "Sad",
    ":D": "Very Happy",
    ":|": "Neutral",
    ":O": "Surprised",
    "<3": "Love",
    ";)": "Wink",
    ":P": "Playful",
    ":/": "Confused",
    ":*": "Kiss",
    ":')": "Touched",
    "XD": "Laughing",
    ":3": "Cute",
    ">:(": "Angry",
    ":-O": "Shocked",
    ":|]": "Robot",
    ":>": "Sly",
    "^_^": "Happy",
    "O_o": "Confused",
    ":-|": "Straight Face",
    ":X": "Silent",
    "B-)": "Cool",
    "<(‘.'<)": "Dance",
    "(-_-)": "Bored",
    "(>_<)": "Upset",
    "(¬‿¬)": "Sarcastic",
    "(o_o)": "Surprised",
    "(o.O)": "Shocked",
    ":0": "Shocked",
    ":*(": "Crying",
    ":v": "Pac-Man",
    "(^_^)v": "Double Victory",
    ":-D": "Big Grin",
    ":-*": "Blowing a Kiss",
    ":^)": "Nosey",
    ":-((": "Very Sad",
    ":-(": "Frowning",
}

## Function to replace emoticons with their meanings
def convert_emoticons(text: str):
    ''' This Function is to replace the emoticons with thier meaning instead 
    '''
    for emoticon, meaning in emoticon_meanings.items():
        text = text.replace(emoticon, meaning)
    return text




## A Function to remove redundant words like (I've, You'll)
def remove_redundant_words_extra_spaces(text: str):
    ## Remove contractions using regular expressions
    contraction_pattern = re.compile(r"'\w+|\w+'\w+|\w+'")
    text = contraction_pattern.sub('', text)

    ## Define a set of stopwords
    stop_words = set(stopwords.words("english"))

    ## Remove stopwords and extra spaces
    words = text.split()
    filtered_words = [word for word in words if word.lower() not in stop_words]
    clean_text = ' '.join(filtered_words)

    ## Remove extra spaces
    clean_text = ' '.join(clean_text.split())
    
    return clean_text




# --------------------------------------------------------------------------------------------------------------- #

def text_cleaning(text: str) -> str:
    """ This function will be passed to main.py for text cleaning
    """

    # Removing tags for the new text
    text = remove_pattern(input_txt=text, pattern=r'@[\w]*')

    # Remove URLS
    text = remove_pattern(input_txt=text, pattern=r'https?://\S+|www\.\S+')

    # remove excessive repeated chars
    text = remove_excessive_repeated_characters(input_string=text)

    # Convert Emotions
    text = convert_emoticons(text=text)

    # Removing Punctuations, Numbers, and Special Characters
    text = text.replace('[^a-zA-Z#]', ' ')

    # Remove Short Words
    text = ' '.join([w for w in text.split() if len(w)>3])

    # Remove the numbers from words 
    text = remove_pattern(input_txt=text, pattern=r'(?<=\w)\d+|\d+(?=\w)')

    # Remove special characters
    text = remove_pattern(input_txt=text, pattern=r'[!@#$%^&*()_+{}\[\]:;<>,.?~\\|\/]')

    # remove redundant words 
    cleaned_text = remove_redundant_words_extra_spaces(text=text)

    return cleaned_text



def text_lemamtizing(text: str) -> str:
    """ This function will be passed to main.py for text lemmatiozation
    """

    ## Tokenization
    tokenized_text = text.split()

    ## Lemmatization
    lemmatizer = WordNetLemmatizer()

    # Lemmatize each word
    lemmatized_words = [lemmatizer.lemmatize(word) for word in tokenized_text]

    # Join the lemmatized words back into a sentence
    lemmatized_text = ' '.join(lemmatized_words)

    return lemmatized_text

def preprocess_text(doc: str):
    """ Function for tokenizing for Word2Vec Model"""
    
    # Tokenizing
    tokens = word_tokenize(doc)
    # Remove not-alphabetic
    tokens = [word for word in tokens if word.isalpha()]
    # Remove stop words
    tokens = [word for word in tokens if word not in stopwords.words('english')]

    return tokens


def doc_to_w2v(doc):
    """Transform the text data to Word2Vec embeddings"""

    # Call the preprocessing function
    tokens = preprocess_text(doc=doc)

    # Get the embeddings
    embeddings = [w2v_vectorizer.wv[word] for word in tokens if word in w2v_vectorizer.wv]

    # Get the document overall embeddings
    if embeddings:
        return np.mean(embeddings, axis=0)
    else:
        return np.zeros(w2v_vectorizer.vector_size)
    

def text_vectorizing(text: str, method:str):
    """ This function will be passed to main.py for text vectorizing
    """
    # Apply Vectorizing
    if method.lower() == 'bow':
        X_processed = bow_vectorizer.transform([text]).toarray()

    elif method.lower() == 'w2v':
        X_processed = doc_to_w2v(doc=text).toarray()

    else:
        X_processed = tfidf_vectorizer.transform([text]).toarray()

    return X_processed


def predict_new(X_new, method:str):
    """ This function will be passed to main.py for predicting class
    """
    if method.lower() == 'bow':
        y_pred = bow_svc_model.predict(X_new)[0]

    elif method.lower() == 'w2v':
        y_pred = w2v_svc_model.predict(X_new)[0]

    else:
        y_pred = tfidf_svc_model.predict(X_new)[0]

    return y_pred