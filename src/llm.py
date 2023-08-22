import pandas as pd
import config
from skllm import ZeroShotGPTClassifier
from skllm.datasets import get_classification_dataset
from skllm.config import SKLLMConfig
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

SKLLMConfig.set_openai_key(config.OPEN_AI_KEY)
SKLLMConfig.set_openai_org(config.OPEN_AI_ORG)

df = pd.read_csv('../data/train_data_short.csv')
data = df["Review"]
datalabels = df["Sentiment"]

x_train, x_test, y_train, y_test  = train_test_split(
        data, 
        datalabels,
        train_size=0.60, 
        random_state=1234)

clf = ZeroShotGPTClassifier(openai_model="gpt-3.5-turbo")
clf.fit(X=x_train, y=y_train)

predicted_movie_review_labels = clf.predict(X=x_test)

print(accuracy_score(y_test, predicted_movie_review_labels))

#for review, real_sentiment, sentiment in zip(x_test, y_test, predicted_movie_review_labels):
#    print(f"Review: {review}\nReal Sentiment: {real_sentiment}\nPredicted Sentiment: {sentiment}\n\n")