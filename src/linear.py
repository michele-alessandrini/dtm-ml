import pandas as pd
import random as rand
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

df = pd.read_csv('../data/train_data_short.csv')

data = df["Review"]
datalabels = df["Sentiment"]


vectorizers = CountVectorizer(
    analyzer = 'word',
    lowercase = False,
)
features = vectorizers.fit_transform(
    data
)
features_nd = features.toarray()

x_train, x_test, y_train, y_test  = train_test_split(
        features_nd, 
        datalabels,
        train_size=0.60, 
        random_state=1234)

logreg_model = LogisticRegression()
logreg_model = logreg_model.fit(X=x_train, y=y_train)
y_pred = logreg_model.predict(x_test)

print(accuracy_score(y_test, y_pred))
