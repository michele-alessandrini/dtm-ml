import nltk
import nltk.corpus
import pandas as pd
import openai
import config
import numpy
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

openai.api_key = config.OPEN_AI_KEY

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
    final_prompt = "Can you say whether the following text  contains content that must be moderated? Example of that content include but not limited to sexual terms, hate, violence, and weapons. Your answer should be POSITIVE for content that you feel should be moderated or that woudl go against the majority of policies for content distribution and NEGATIVE for content that is good.  It is not a defintive assessement. It is more a likelyhood to happen. The text is:" + "\n\n" +  text    
    # Define the system message
    system_msg = 'You are a content moderator.'
    # Define the user message
    user_msg = final_prompt
    # Create a dataset using GPT
    response = openai.ChatCompletion.create(model="gpt-4",
                                            messages=[{"role": "system", "content": system_msg},
                                             {"role": "user", "content": user_msg}])

    ret_str = response["choices"][0]["message"]["content"]
    moderated = 0
    if "positive" in ret_str.lower():
            moderated= 1

    return moderated


df = pd.read_csv('../data/train_data_short.csv')

df['Review'] = df['Review'].apply(preprocess_text)
df['SentimentPython'] = df['Review'].apply(get_sentiment)

print(accuracy_score(df['SentimentPython'], df['Sentiment']))