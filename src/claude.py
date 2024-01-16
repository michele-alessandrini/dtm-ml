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
    
    prompt = "Can you say whether the following text  contains content that must be moderated? Example of that content include but not limited to sexual terms, hate, violence, and weapons. Your answer should be POSITIVE for content that you feel should be moderated or that woudl go against the majority of policies for content distribution and NEGATIVE for content that is good.  It is not a defintive assessement. It is more a likelyhood to happen. The text is:\n\n" + text

    body = json.dumps({
            "prompt": "\n\nHuman: " + prompt +"\n\nAssistant:",
            "max_tokens_to_sample": 3000,
            "temperature": 0.1,
            "top_p": 0.9,
        })
    
    
    modelId = 'anthropic.claude-v2'
    accept = 'application/json'
    contentType = 'application/json'
    response = bedrock.invoke_model(body=body, modelId=modelId, accept=accept, contentType=contentType)
    response_body = json.loads(response.get('body').read())

    ret_str = response_body.get('completion')
    
    moderated = 0
    if "positive" in ret_str.lower():
            moderated= 1

    return moderated


df = pd.read_csv('../data/train_data_short.csv')

df['Review'] = df['Review'].apply(preprocess_text)
df['SentimentPython'] = df['Review'].apply(get_sentiment)

print(accuracy_score(df['SentimentPython'], df['Sentiment']))