from flask import Flask
from flask import request
from flask_cors import CORS
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.naive_bayes import MultinomialNB
from sklearn.externals import joblib
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
import pandas as pd
import pickle
# heroku git:remote -a {your-project-name}

import pickle

from textDataCleaning import text_data_cleaning


from sklearn.externals import joblib

classifier = joblib.load(" news_classifier.pkl")
app = Flask(__name__)
CORS(app)


@app.route('/')
def hello_world():
    return 'Hello World!'


@app.route('/classifyNews', methods=['GET', 'POST'])
def classifyNews():

    data = request.json
    text = data['text']

    result = dict()
    result['prediction'] = classifier.predict([text])[0]

    return result


@app.route('/classifySpam', methods=['GET', 'POST'])
def classifySpam():

    df = pd.read_csv('spam2.csv', encoding='latin-1')
    df.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], axis=1,
            inplace=True)
    df['label'] = df['class'].map({'ham': 0, 'spam': 1})
    X = df['message']
    y = df['label']

    # Extract feature with CountVectorize
    cv = CountVectorizer()
    X = cv.fit_transform(X)
    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size=0.33,
                                                        random_state=42)
    from sklearn.naive_bayes import MultinomialNB

    clf = MultinomialNB()
    clf.fit(X_train, y_train)

    data = request.json
    text = data['text']
    vect = cv.transform([text]).toarray()
    pred = clf.predict(vect)
    print(pred)

    result = dict()
    result['prediction'] = str(pred[0])

    return result


@app.route('/sentiment', methods=['GET', 'POST'])
def sentiment():

    data = request.json
    text = data['text']
    stop_words = stopwords.words('english')
    text1 = text.lower()

    processed_doc1 = ' '.join([word for word in text1.split() if word not in stop_words])
    sa = SentimentIntensityAnalyzer()
    dd = sa.polarity_scores(text=text1)
    compound = round((1 + dd['compound'])/2, 2)
    result = dict()
    result['prediction'] = compound

    return result


if __name__ == '__main__':

    app.run()
