from time import time
from flask import Flask, render_template, request, url_for, redirect
import re
from textblob import Word
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from nltk.corpus import stopwords
import pickle
from trnlp import TrnlpWord

obj = TrnlpWord()

app = Flask(__name__)

with open('main_model.science', 'rb') as f:
    log_model = pickle.load(f)

with open('sentiment_model.science', 'rb') as f:
    sentiment_log_model = pickle.load(f)

with open('main_model_vectorizer.vector', 'rb') as f:
    vectorizer = pickle.load(f)

with open('sentiment_model_vectorizer.vector', 'rb') as f:
    sentiment_vectorizer = pickle.load(f)



@app.route('/', methods=['GET', 'POST'])
def home_page():
    if request.method == "POST":
        tweet_content = request.form['tweet']

        predicted = predict(tweet_content)
        predicted_sentiment = predict_sentiment(tweet_content)

        return redirect(url_for('home_page', predicted=predicted[0], predicted_sentiment=predicted_sentiment[0]))

    try:
        predictedd_sentiment = request.args['predicted_sentiment']
        if predictedd_sentiment == '0':
            predictedd_sentiment = 'Olumlu'
        elif predictedd_sentiment == '1':
            predictedd_sentiment = 'Olumsuz'
        predictedd = request.args['predicted']
        print(type(predictedd))
        if predictedd == '0':
            predictedd = 'Cinsiyet'+' - '+predictedd_sentiment
        elif predictedd == '1':
            predictedd = 'Dini'+' - '+predictedd_sentiment
        elif predictedd == '2':
            predictedd = 'Etnik'+' - '+predictedd_sentiment
        elif predictedd == '3':
            predictedd = 'Hiçbiri'+' - '+predictedd_sentiment
        else:
            predictedd = ''
    except:
        predictedd=''

    return render_template('index.html', result=predictedd)

def lemmetization(sentence):
    words = []
    for word in sentence.split():
        obj.setword(word)
        words.append(obj.get_base)
    result = " ".join(words)
    return result

def prepare_tweet(tweet):
    tweet = tweet.lower() # küçük harf
    tweet = re.sub(r'/^#\w+$/', '', tweet)  # numbers
    tweet = re.sub(r'[^\w\s]', '', tweet) # noktalama şeylerinden
    tweet = re.sub('\s\s+', '', tweet) # multiple spaces to single space
    tweet = re.sub('\d', '', tweet) # numbers
    sw = stopwords.words('turkish')
    tweet = " ".join(x for x in str(tweet).split() if x not in sw)
    tweet = lemmetization(tweet)
    vector = vectorizer.transform([tweet])
    tweet = vector.toarray()
    return tweet

def prepare_tweet_sentiment(tweet):
    tweet = tweet.lower() # küçük harf
    tweet = re.sub(r'/^#\w+$/', '', tweet)  # numbers
    tweet = re.sub(r'[^\w\s]', '', tweet) # noktalama şeylerinden
    tweet = re.sub('\s\s+', '', tweet) # multiple spaces to single space
    tweet = re.sub('\d', '', tweet) # numbers
    sw = stopwords.words('turkish')
    tweet = " ".join(x for x in str(tweet).split() if x not in sw)
    tweet = lemmetization(tweet)
    vector = sentiment_vectorizer.transform([tweet])
    tweet = vector.toarray()
    return tweet

def predict(tweet):
    tweet = prepare_tweet(tweet)
    result = log_model.predict(tweet)
    print(result)
    return result

def predict_sentiment(tweet):
    tweet = prepare_tweet_sentiment(tweet)
    result = sentiment_log_model.predict(tweet)
    print(result)
    return result


if __name__ == '__main__':
  app.run(debug=True)