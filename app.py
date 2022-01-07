from flask import Flask
from flask import request
from flask_cors import CORS
from sklearn.feature_extraction.text import CountVectorizer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import joblib

# heroku git:remote -a {your-project-name}

news_classifier = joblib.load("news_classifier.pkl")
spam_classifier = joblib.load("spam_classifier.pkl")
app = Flask(__name__)
CORS(app)


@app.after_request
def after_request(response):
  response.headers.add('Access-Control-Allow-Origin', '*')
  response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
  response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
  return response


@app.route('/')
def hello_world():
  return 'Hello World!'


@app.route('/classifyNews', methods=['GET', 'POST'])
def classify_news():
  data = request.json
  text = data['text']

  result = dict()
  result['prediction'] = news_classifier.predict([text])[0]

  return result


@app.route('/classifySpam', methods=['GET', 'POST'])
def classify_spam():
  # Extract feature with CountVectorize
  cv = CountVectorizer()
  data = request.json
  text = data['text']
  vect = cv.transform([text]).toarray()
  pred = spam_classifier.predict(vect)
  print(pred)

  result = dict()
  result['prediction'] = str(pred[0])

  return result


@app.route('/sentiment', methods=['GET', 'POST'])
def sentiment():
  data = request.json
  text = data['text']
  text1 = text.lower()

  sa = SentimentIntensityAnalyzer()
  dd = sa.polarity_scores(text=text1)
  compound = round((1 + dd['compound']) / 2, 2)
  result = dict()
  result['prediction'] = compound

  return result


if __name__ == '__main__':
  app.run(host='0.0.0.0', debug=True, port=80, threaded=False)
