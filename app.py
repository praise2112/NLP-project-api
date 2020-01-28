from flask import Flask
from flask import request
import pickle

from textDataCleaning import text_data_cleaning


from sklearn.externals import joblib

classifier = joblib.load(" news_classifier.pkl")
app = Flask(__name__)


@app.route('/')
def hello_world():
    return 'Hello World!'


@app.route('/classifyNews', methods=['GET', 'POST'])
def classifyNews():

    text = request.args.get('text')
    result = dict()

    result['prediction'] = classifier.predict([text])[0]

    return result


if __name__ == '__main__':

    app.run()
