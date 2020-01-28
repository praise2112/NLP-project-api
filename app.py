from flask import Flask
from flask import request
from flask_cors import CORS


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

    text = request.form.get('text')
    result = dict()

    result['prediction'] = classifier.predict([text])[0]

    return result


if __name__ == '__main__':

    app.run()
