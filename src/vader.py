import nltk
import nltk.corpus
import pandas as pd
import numpy
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score


# create preprocess_text function
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
    scores = analyzer.polarity_scores(text)
    #sentiment = 1 if scores['pos'] > 1 else 0
    sentiment = 0
    k_neutrality = 0.5

    if scores['pos'] > (scores['neg']+(k_neutrality*scores['neu'])):
        sentiment = 1
    return sentiment


analyzer = SentimentIntensityAnalyzer()

#df = pd.read_csv('data/sample_short.csv')
df = pd.read_csv('../data/train_data_short.csv')

df['Review'] = df['Review'].apply(preprocess_text)
df['SentimentPython'] = df['Review'].apply(get_sentiment)

print(accuracy_score(df['SentimentPython'], df['Sentiment']))


