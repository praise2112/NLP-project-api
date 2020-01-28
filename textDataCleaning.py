from spacy.lang.en.stop_words import STOP_WORDS
import spacy

import en_core_web_sm
nlp = en_core_web_sm.load()
# nlp = spacy.load('en_core_web_sm')
stopwords = list(STOP_WORDS)
import string
punct = string.punctuation


def text_data_cleaning(sentence):
    doc = nlp(sentence)

    tokens = []
    for token in doc:
        if token.lemma_ != '-PRON-':
            temp = token.lemma_.lower().strip()
        else:
            temp = token.lower_
        tokens.append(temp)

    cleaned_tokens = []
    for token in tokens:
        if token not in stopwords and token not in punct:
            cleaned_tokens.append(token)
    return cleaned_tokens