from fastapi import FastAPI, Form
from pydantic import BaseModel
from typing import List, Literal
import uvicorn

# My custom functions
from utils import text_cleaning, text_lemamtizing, text_vectorizing, predict_new

# Intialzie the app
app = FastAPI(debug=True)

# Dictionary for mapping the label to text
map_label = {
            0: 'Negative',
            1: 'Neutral',
            2: 'Positive'
        }

class DataInput(BaseModel):
    text: str
    method: Literal['BOW', 'TF-IDF', 'W2V', 'FT', 'GloVe'] = 'TF-IDF'

@app.get('/')
async def home():
    return 'Hello I am youssef kamel'

@app.get('/predict')
async def tweet_clf(data: DataInput):
    
    # Cleaning
    cleaned_text = text_cleaning(text=data.text)

    # Lemmatization
    cleaned_text = text_lemamtizing(text=cleaned_text)

    # Vectorizing
    X_processed = text_vectorizing(text=cleaned_text, method=data.method)

    # Model
    y_pred = predict_new(X_new=X_processed, method=data.method)

    # Map integer to Class Text
    final_pred = map_label.get(y_pred)

    return {f'Prediction is: {final_pred}'}

if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)