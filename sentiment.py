import pandas as pd
from nltk.corpus import stopwords
from textblob import Word
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import cross_validate, RandomizedSearchCV
from sklearn.metrics import classification_report, make_scorer, precision_score, recall_score, f1_score, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
import numpy as np
import re
from trnlp import TrnlpWord

obj = TrnlpWord()

# pd.set_option('display.max_columns', None)
# pd.set_option('display.width', 200)
# pd.set_option('display.float_format', lambda x: '%.2f' % x)

## READING DATASET ##

df = pd.read_csv('sentiment_dataset.csv', index_col=0, encoding='utf-8')
df.reset_index(inplace=True)
df.head()
df.columns = ['sentence', 'label']
df.dropna(inplace=True)
df = df[df['label'] != 'Tarafsız']
df.drop_duplicates(subset=['sentence'], inplace=True)
df['label'].value_counts()
df.drop('index', inplace=True, axis=1)

## PREPROCESSING ##
def lemmetization(sentence):
    words = []
    for word in sentence.split():
        obj.setword(word)
        words.append(obj.get_base)
    result = " ".join(words)
    return result


def preprocess(df, column):
    df[column] = df[column].str.lower()  # converting to lower case
    df[column] = df[column].str.replace('/^#\w+$/', '')  # removing hashtags
    df[column] = df[column].str.replace('[^\w\s]', '')  # removing punctuation
    df[column] = df[column].str.replace('\s\s+', '')  # converting multiple spaces to single space
    df[column] = df[column].str.replace('\d', '')  # removing numbers

    # Removing stopwords ##
    sw = stopwords.words('turkish')
    df[column] = df[column].apply(lambda x: " ".join(x for x in str(x).split() if x not in sw))

    # Lemmetization (köklerine indirgeme)##
    df[column] = df[column].apply(lemmetization)

    ## DROPPING RARE WORDS ##
    temp_df = pd.Series(' '.join(df[column]).split()).value_counts()
    drops = temp_df[temp_df <= 1]

    df[column] = df[column].apply(lambda x: " ".join(x for x in x.split() if x not in drops))
    return df

## APPLYING PREPROCESSING FUNC ##
df = preprocess(df, 'sentence')
df['label'].value_counts()

def label_encoder(dataframe, binary_col):
    labelencoder = LabelEncoder()
    dataframe[binary_col] = labelencoder.fit_transform(dataframe[binary_col])
    return dataframe
df['label'].value_counts()
## APPLYING LABEL ENCODING ##
df = label_encoder(df, 'label')

## SEPERATING X AND Y ##
X = df['sentence']
y = df['label']

### COUNT VECTORIZER ###
vectorizer = CountVectorizer()
vector = vectorizer.fit_transform(X)
X = vector.toarray()

import pickle

filename = 'SENTIMENT_VECTOR_5MART_1620.vector'
pickle.dump(vectorizer, open(filename, 'wb'))
####

## APPLYING TD-IDF vectorizer ##

# tf_idf_word_vectorizer = TfidfVectorizer(ngram_range=(2, 3))
# tf_idf_word_vectorizer.fit(X)
# X = tf_idf_word_vectorizer.transform(X)

## MODELING ##

## LOGISTIC REGRESSION ##

solvers = ['newton-cg', 'lbfgs', 'liblinear']
penalty = ['l2']
c_values = [100, 10, 1.0, 0.1, 0.01]
log_params = dict(solver=solvers,penalty=penalty,C=c_values)

log_model = LogisticRegression()
rscv = RandomizedSearchCV(log_model ,param_distributions = log_params ,  cv = 5 , n_iter=5 , scoring = 'accuracy', verbose =10)
search = rscv.fit(X, y)

log_model = LogisticRegression().set_params(**search.best_params_)
log_model.fit(X,y)

scoring = {'accuracy' : make_scorer(accuracy_score),
       'precision' : make_scorer(precision_score, average = 'micro'),
       'recall' : make_scorer(recall_score, average = 'micro'),
       'f1_score' : make_scorer(f1_score, average = 'micro')}

## CROSS VALIDATION ##
cv_results = cross_validate(log_model,
                            X, y,
                            cv=5,
                            scoring=scoring, error_score='raise')

## PRINTING RESULTS ##
for key, value in cv_results.items():
    print(key, value.mean())

import pickle

filename = 'SENTIMENT_MODEL_5MART_1620.science'
pickle.dump(log_model, open(filename, 'wb'))

## XGBOOST ##

xgb_param = {'n_estimators':list(range(100,500)) ,
         'max_depth':list(range(1,10)) ,
         'learning_rate':[0.001,0.002,0.003,0.004,0.005,0.006,0.007,0.008,0.05,0.09] ,
         'min_child_weight':list(range(1,10))
}

xgboost_model = XGBClassifier(random_state=17)

rscv = RandomizedSearchCV(xgboost_model ,param_distributions = xgb_param ,  cv =5 , n_iter=10 , scoring = 'accuracy', verbose =10)
search = rscv.fit(X_tf, y)

xgboost_model = XGBClassifier(random_state=17).set_params(**search.best_params_)
xgboost_model.fit(X_tf, y)

scoring = {'accuracy' : make_scorer(accuracy_score),
       'precision' : make_scorer(precision_score, average = 'micro'),
       'recall' : make_scorer(recall_score, average = 'micro'),
       'f1_score' : make_scorer(f1_score, average = 'micro')}

cv_results = cross_validate(xgboost_model,
                            X_tf, y,
                            cv=5,
                            scoring=scoring, error_score='raise')

##################

## RANDOM FOREST ##

rf_params = {"max_depth": [5, 8, 10, 12],
             "max_features": [3, 5, 7, "auto"],
             "min_samples_split": [5, 8, 11, 15, 20],
             "n_estimators": [50, 80, 100, 200, 250, 300, 350]}

rf_model = RandomForestClassifier(random_state=17)

rscv_rf = RandomizedSearchCV(rf_model, rf_params, cv =5 , n_iter=10 , scoring = 'accuracy')
best_rf_params = rscv_rf.fit(X, y)

rf_model_best = RandomForestClassifier(random_state=17).set_params(**best_rf_params.best_params_)
rf_model_best.fit(X, y)

scoring = {'accuracy' : make_scorer(accuracy_score),
       'precision' : make_scorer(precision_score, average = 'micro'),
       'recall' : make_scorer(recall_score, average = 'micro'),
       'f1_score' : make_scorer(f1_score, average = 'micro')}

cv_results = cross_validate(rf_model_best,
                            X, y,
                            cv=5,
                            scoring=scoring, error_score='raise')

## PREDICT ##

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

prd_tweet = 'çok güzel'
prd_tweet = prepare_tweet(prd_tweet)

log_model.predict(prd_tweet)

prd_tweet = prepare_tweet(prd_tweet)
y_predicted = rf_model.predict(X_tf)

classification_report(y, y_predicted)
# precision    recall  f1-score   support
# 0       0.99      0.62      0.76       328
# 1       0.92      0.84      0.88       498
# 2       0.54      0.98      0.70       561
# 3       1.00      0.15      0.26       339
# accuracy                           0.71      1726
# macro avg       0.86      0.65      0.65      1726
# weighted avg       0.82      0.71      0.68      1726

rf_model.predict(prd_tweet)
df['label'].value_counts()


xgboost_final.predict(prd_tweet)
cv_results['test_accuracy'].mean()
cv_results['test_f1'].mean()
cv_results['test_roc_auc'].mean()

# 2    561
# 1    498
# 3    339
# 0    328

## RANDOM FOREST WITH 2 CLASSES(OLUMLU/OLUMSUZ) WITHOUT PREPROCESS WITH COUNT VECTORIZER##
# {'fit_time': array([7.78533101, 7.59588194, 7.56140399, 7.6185658 , 7.54167295]),
#  'score_time': array([0.17842889, 0.15572715, 0.17152095, 0.15331721, 0.15680003]),
#  'test_accuracy': array([0.8557126 , 0.87396938, 0.84746761, 0.87279152, 0.86328816]),
#  'test_precision': array([0.8557126 , 0.87396938, 0.84746761, 0.87279152, 0.86328816]),
#  'test_recall': array([0.8557126 , 0.87396938, 0.84746761, 0.87279152, 0.86328816]),
#  'test_f1_score': array([0.8557126 , 0.87396938, 0.84746761, 0.87279152, 0.86328816])}

## RANDOM FOREST WITH 2 CLASSES(OLUMLU/OLUMSUZ) WITH PREPROCESS WITH TF-IDF n_gram = (3, 2)##

# {'fit_time': array([4.39514375, 4.58415914, 4.4251349 , 4.38240314, 4.41341019]),
#  'score_time': array([0.0948863 , 0.09235501, 0.09234214, 0.0930028 , 0.09107089]),
#  'test_accuracy': array([0.75265018, 0.76678445, 0.7762073 , 0.76796231, 0.77136123]),
#  'test_precision': array([0.75265018, 0.76678445, 0.7762073 , 0.76796231, 0.77136123]),
#  'test_recall': array([0.75265018, 0.76678445, 0.7762073 , 0.76796231, 0.77136123]),
#  'test_f1_score': array([0.75265018, 0.76678445, 0.7762073 , 0.76796231, 0.77136123])}

## RANDOM FOREST WITH 2 CLASSES(OLUMLU/OLUMSUZ) WITH PREPROCESS WITH COUNT VECTORIZER##

# {'fit_time': array([7.07865071, 7.05276489, 7.09173107, 7.36896014, 7.27055907]),
#  'score_time': array([0.08669424, 0.0851562 , 0.08477497, 0.08548307, 0.08577299]),
#  'test_accuracy': array([0.86101296, 0.85924617, 0.84275618, 0.85924617, 0.85680613]),
#  'test_precision': array([0.86101296, 0.85924617, 0.84275618, 0.85924617, 0.85680613]),
#  'test_recall': array([0.86101296, 0.85924617, 0.84275618, 0.85924617, 0.85680613]),
#  'test_f1_score': array([0.86101296, 0.85924617, 0.84275618, 0.85924617, 0.85680613])}

## RANDOM FOREST WITH 3 CLASSES(OLUMLU/OLUMSUZ/TARAFSIZ) WITH PREPROCESS WITH TF-IDF n_gram = (3, 2)##
# {'fit_time': array([5.25184321, 5.4845891 , 5.33547616, 5.14899373, 5.22382903]),
#  'score_time': array([0.08366179, 0.07745004, 0.0777247 , 0.07830024, 0.07971215]),
#  'test_accuracy': array([0.56124234, 0.56761488, 0.56498906, 0.55798687, 0.56280088]),
#  'test_precision': array([0.56124234, 0.56761488, 0.56498906, 0.55798687, 0.56280088]),
#  'test_recall': array([0.56124234, 0.56761488, 0.56498906, 0.55798687, 0.56280088]),
#  'test_f1_score': array([0.56124234, 0.56761488, 0.56498906, 0.55798687, 0.56280088])}