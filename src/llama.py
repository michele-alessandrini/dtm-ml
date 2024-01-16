import nltk
import nltk.corpus
import pandas as pd
import boto3
import json
import config
import numpy
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

bedrock = boto3.client('bedrock-runtime' , 'us-west-2', endpoint_url='https://bedrock-runtime.us-west-2.amazonaws.com')

def preprocess_text(text):
    # Tokenize the text
    tokens = word_tokenize(text.lower())
    # Remove stop words
    filtered_tokens = [token for token in tokens if token not in stopwords.words('english')]
    # Lemmatize the tokens
    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in filtered_tokens]
    # Join the tokens back into a string
    processed_text = ' '.join(lemmatized_tokens)
    return processed_text

def get_sentiment(text):
        
    body = json.dumps({
        "prompt": "Can you say whether the user who wrote the following review liked the movie? Your answer should be include the word POSITIVE if you feel the review is good or the answer shoud include the word NEGATIVE if the user did not like the move.  It is not meant to be a definitive assessement. The review is:" + "\n\n" +  text,
        "max_gen_len": 128,
        "temperature": 0.5,
        "top_p": 0.5
    })
    
    modelId = 'meta.llama2-13b-chat-v1'
    response = bedrock.invoke_model(body=body, modelId=modelId)
    response_body = json.loads(response.get('body').read())

    ret_str = response_body["generation"]

    moderated = 0
    if "positive" in ret_str.lower():
            moderated= 1

    return moderated


df = pd.read_csv('../data/train_data_short.csv')

df['Review'] = df['Review'].apply(preprocess_text)
df['SentimentPython'] = df['Review'].apply(get_sentiment)

print(accuracy_score(df['SentimentPython'], df['Sentiment']))