import pandas as pd
from nltk.corpus import stopwords
from textblob import Word
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_validate, RandomizedSearchCV, train_test_split
from sklearn.metrics import make_scorer, precision_score, recall_score, f1_score, accuracy_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import numpy as np
import re
from sklearn.svm import SVC
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from PIL import Image
import seaborn as sns

df = pd.read_csv('TWEETS.csv', index_col=0)

df = df.reset_index(drop=True)

df.reset_index(inplace=True, drop=True)

df.columns = ['tweet', 'label']
df.shape
## PREPROCESSING ##

df.dropna(inplace=True)

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
    df[column] = df[column].apply(lambda x: " ".join([Word(word).lemmatize() for word in x.split()]))

    ## DROPPING RARE WORDS ##
    temp_df = pd.Series(' '.join(df[column]).split()).value_counts()
    drops = temp_df[temp_df <= 1]

    df[column] = df[column].apply(lambda x: " ".join(x for x in x.split() if x not in drops))
    return df

## APPLYING PREPROCESS FUNC ##
df = preprocess(df, 'tweet')

## ONLY HOLD UNIQUE VARIABLES ##
df.drop_duplicates(subset=['tweet'], inplace=True)
df['label'].value_counts()
## ONLY MANDOTARY FOR SUPPORT VECTOR MACHINES ##
def label_encoder(dataframe, binary_col):
    labelencoder = LabelEncoder()
    dataframe[binary_col] = labelencoder.fit_transform(dataframe[binary_col])
    return dataframe

## ONLY MANDOTARY FOR SUPPORT VECTOR MACHINES ##
## APPLYING LABEL ENCODING ##
df = label_encoder(df, 'label')

## SEPERATING X AND Y ##
X = df['tweet']
y = df['label']

### COUNT VECTORIZER ###
vectorizer = CountVectorizer()
vector = vectorizer.fit_transform(X)
X = vector.toarray()

## TD-IDF VECTORIZER ##
# tf_idf_word_vectorizer = TfidfVectorizer(ngram_range=(2, 3))
# tf_idf_word_vectorizer.fit(X)
# X = tf_idf_word_vectorizer.transform(X)

#### MODELING ####

##################### XGBOOST #####################

xgb_param = {'n_estimators':list(range(100,500)) ,
         'max_depth':list(range(1,10)) ,
         'learning_rate':[0.001,0.002,0.003,0.004,0.005,0.006,0.007,0.008,0.05,0.09] ,
         'min_child_weight':list(range(1,10))
}

xgboost_model = XGBClassifier(random_state=17)

## LOOKING FOR BEST PARAMS ##
rscv = RandomizedSearchCV(xgboost_model ,param_distributions = xgb_param ,  cv = 5 , n_iter=2 , scoring = 'accuracy', verbose =10)
search = rscv.fit(X, y)

## APPLYING BEST PARAMS TO MODEL ##
xgboost_model = XGBClassifier(random_state=17).set_params(**search.best_params_)

## SUCCESS METRICS ##
scoring = {'accuracy' : make_scorer(accuracy_score),
       'precision' : make_scorer(precision_score, average = 'micro'),
       'recall' : make_scorer(recall_score, average = 'micro'),
       'f1_score' : make_scorer(f1_score, average = 'micro')}

## CROSS VALIDATION ##
cv_results = cross_validate(xgboost_model,
                            X, y,
                            cv=5,
                            scoring=scoring, error_score='raise')

## PRINTING RESULTS ##
for key, value in cv_results.items():
    print(key, value.mean())


#################### LOGISTIC REGRESSION ####################

solvers = ['newton-cg', 'lbfgs', 'liblinear']
penalty = ['l2']
c_values = [100, 10, 1.0, 0.1, 0.01]
log_params = dict(solver=solvers,penalty=penalty,C=c_values)

log_model = LogisticRegression()
rscv = RandomizedSearchCV(log_model ,param_distributions = log_params ,  cv = 5 , n_iter=15 , scoring = 'accuracy', verbose =10)
search = rscv.fit(X, y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

## APPLYING BEST PARAMS TO MODEL ##
log_model = LogisticRegression(random_state=17).set_params(**search.best_params_)
log_model.fit(X_train, y_train)

import pickle

filename = 'main_model_vectorizer_5MART_1620.vector'
pickle.dump(vectorizer, open(filename, 'wb'))

## CROSS VALIDATION ##
cv_results = cross_validate(log_model,
                            X, y,
                            cv=5,
                            scoring=scoring, error_score='raise')

## PRINTING RESULTS ##
for key, value in cv_results.items():
    print(key, value.mean())
df['label'].value_counts()

y_pred = log_model.predict(X_test)

# Confusion Matrix
def plot_confusion_matrix(y, y_pred):
    acc = round(accuracy_score(y, y_pred), 2)
    cm = confusion_matrix(y, y_pred)
    sns.heatmap(cm, annot=True, fmt=".0f")
    plt.xlabel('y_pred')
    plt.ylabel('y')
    plt.title('Accuracy Score: {0}'.format(acc), size=10)
    plt.show()

plot_confusion_matrix(y_test, y_pred)

#################### SUPPORT VECTOR MACHINES ####################

## HYPERPARAMETERS TO TRY ##
param_grid = {'C': [0.1, 1, 10, 100, 1000],
              'gamma': [1, 0.1, 0.01, 0.001, 0.0001],
              'kernel': ['rbf']}

scoring = {'accuracy' : make_scorer(accuracy_score),
       'precision' : make_scorer(precision_score, average = 'micro'),
       'recall' : make_scorer(recall_score, average = 'micro'),
       'f1_score' : make_scorer(f1_score, average = 'micro')}

svm = SVC()

rscv = RandomizedSearchCV(svm, param_grid, cv =5 , n_iter=10 , scoring = 'accuracy')
best_rf_params = rscv.fit(X, y)

svm_best = svm.set_params(**best_rf_params.best_params_)

cv_results = cross_validate(svm,
                            X, y,
                            cv=5,
                            scoring=scoring, error_score='raise')

## PRINTING RESULTS ##
for key, value in cv_results.items():
    print(key, value.mean())

#################### RANDOM FOREST ####################

## HYPERPARAMETERS LIST TO TRY ##
rf_params = {"max_depth": [5, 8, 10, 12],
             "max_features": [3, 5, 7, "auto"],
             "min_samples_split": [5, 8, 11, 15, 20],
             "n_estimators": [50, 80, 100, 200, 250, 300, 350]}

rf_model = RandomForestClassifier(random_state=17)

rscv_rf = RandomizedSearchCV(rf_model, rf_params, cv =5 , n_iter=10 , scoring = 'accuracy')
best_rf_params = rscv_rf.fit(X, y)

rf_model_best = RandomForestClassifier(random_state=17).set_params(**best_rf_params.best_params_)

scoring = {'accuracy' : make_scorer(accuracy_score),
       'precision' : make_scorer(precision_score, average = 'micro'),
       'recall' : make_scorer(recall_score, average = 'micro'),
       'f1_score' : make_scorer(f1_score, average = 'micro')}

cv_results = cross_validate(rf_model_best,
                            X, y,
                            cv=5,
                            scoring=scoring, error_score='raise')

## PRINTING RESULTS ##
for key, value in cv_results.items():
    print(key, value.mean())

## PREDICT FOR ALL MODELS ##

## PREPARE FUNCTION TO MAKE TWEET READY TO PREDICT ##
def prepare_tweet(tweet):
    tweet = tweet.lower() # küçük harf
    tweet = re.sub(r'/^#\w+$/', '', tweet)  # numbers
    tweet = re.sub(r'[^\w\s]', '', tweet) # noktalama şeylerinden
    tweet = re.sub('\s\s+', '', tweet) # multiple spaces to single space
    tweet = re.sub('\d', '', tweet) # numbers
    sw = stopwords.words('turkish')
    tweet = " ".join(x for x in str(tweet).split() if x not in sw)
    tweet = " ".join([Word(word).lemmatize() for word in tweet.split()])
    vector = vectorizer.transform([tweet])
    tweet = vector.toarray()
    print(tweet)
    return tweet

## PREDICTION ##
tweet_ = 'deneme'
tweet_ = prepare_tweet(tweet_)

## log_model.predict(tweet_)
## svm.predict(tweet_)
## rf_model_best.predict(tweet_)
## ___xgboost__

### DO ALL THINGS ONCE ###

def label_encoder(dataframe, binary_col):
    labelencoder = LabelEncoder()
    dataframe[binary_col] = labelencoder.fit_transform(dataframe[binary_col])
    return dataframe

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
    df[column] = df[column].apply(lambda x: " ".join([Word(word).lemmatize() for word in x.split()]))

    ## DROPPING RARE WORDS ##
    temp_df = pd.Series(' '.join(df[column]).split()).value_counts()
    drops = temp_df[temp_df <= 1]

    df[column] = df[column].apply(lambda x: " ".join(x for x in x.split() if x not in drops))
    return df

def test_models(vectorized_x, y):

    test_results = []

    scoring = {'accuracy': make_scorer(accuracy_score),
               'precision': make_scorer(precision_score, average='micro'),
               'recall': make_scorer(recall_score, average='micro'),
               'f1_score': make_scorer(f1_score, average='micro')}

    ### LOGISTIC REG. PARAMS ###
    solvers = ['newton-cg', 'lbfgs', 'liblinear']
    penalty = ['l2']
    c_values = [100, 10, 1.0, 0.1, 0.01]
    log_params = dict(solver=solvers, penalty=penalty, C=c_values)
    ###

    ### SVM PARAMS. ###
    param_grid = {'C': [0.1, 1, 10, 100, 1000],
                  'gamma': [1, 0.1, 0.01, 0.001, 0.0001],
                  'kernel': ['rbf']}
    ###
    rf_params = {"max_depth": [5, 8, 10, 12],
                 "max_features": [3, 5, 7, "auto"],
                 "min_samples_split": [5, 8, 11, 15, 20],
                 "n_estimators": [50, 80, 100, 200, 250, 300, 350]}

    ###
    xgb_param = {'n_estimators': list(range(100, 500)),
                 'max_depth': list(range(1, 10)),
                 'learning_rate': [0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.05, 0.09],
                 'min_child_weight': list(range(1, 10))
                 }

    log_model = LogisticRegression()
    xgboost_model = XGBClassifier(random_state=17)
    svm = SVC()
    rf_model = RandomForestClassifier(random_state=17)


    for idx, (model, params) in enumerate([(log_model, log_params), (svm, param_grid), (rf_model, rf_params), (xgboost_model, xgb_param)]):
        model_result = []

        text = str(model) + ' HYPERPARAMETER --> TRAIN'
        print(text)
        model_result.append(text)

        n_iter = 15
        if idx == 3:
            n_iter = 5

        rscv = RandomizedSearchCV(model, param_distributions=params, cv=5, n_iter=n_iter, scoring='accuracy',
                                  verbose=10)
        search = rscv.fit(vectorized_x, y)

        text = f'BEST PARAMS FOR {str(model)}: {search.best_params_}'
        print(text)
        model_result.append(text)

        model = model.set_params(**search.best_params_)

        cv_iter = 5
        if idx == 3:
            cv_iter = 3

        cv_results = cross_validate(model,
                                    vectorized_x, y,
                                    cv=cv_iter,
                                    scoring=scoring, error_score='raise')

        for key, value in cv_results.items():
            text = str(key) + ':' + str(value.mean())
            print(text)
            model_result.append(text)

        print('###########################\n')

        test_results.append(model_result)

    return test_results


def letsgo():
    ALL_INFO = []

    df = pd.read_csv('/Users/ardaakdere/PycharmProjects/VBODEV/DF_SON_5MART_SON2.csv', index_col=0)
    df = df.reset_index(drop=True)
    df.reset_index(inplace=True, drop=True)

    df = preprocess(df, 'tweet')
    ## ONLY HOLD UNIQUE VARIABLES ##
    df.drop_duplicates(subset=['tweet'], inplace=True)

    print(f"""
    LABEL DAĞILIMLARI:
    {df['label'].value_counts()}
    ###########################
    """)

    df = label_encoder(df, 'label')

    print(f"""
    ENCODE SONRASI LABEL DAĞILIMLARI:
    {df['label'].value_counts()}
    ###########################
    """)

    X = df['tweet']
    y = df['label']


    vectorizer = CountVectorizer()
    vector = vectorizer.fit_transform(X)
    X_with_count_vectorizer = vector.toarray()

    tf_idf_word_vectorizer = TfidfVectorizer()
    tf_idf_word_vectorizer.fit(X)
    X_with_tfidf_vectorizer = tf_idf_word_vectorizer.transform(X)

    for vectorized_x in [X_with_tfidf_vectorizer, X_with_count_vectorizer]:
        model_results = test_models(vectorized_x, y)
        ALL_INFO.append(model_results)

    print("---RESULT---\n"+ALL_INFO)

letsgo()

# # Confusion Matrix
# def plot_confusion_matrix(y, y_pred):
#     acc = round(accuracy_score(y, y_pred), 2)
#     cm = confusion_matrix(y, y_pred)
#     sns.heatmap(cm, annot=True, fmt=".0f")
#     plt.xlabel('y_pred')
#     plt.ylabel('y')
#     plt.title('Accuracy Score: {0}'.format(acc), size=10)
#     plt.show()
#
# plot_confusion_matrix(y_test, y_pred)

##################################################


### VISUALIZATION ###

## FREQUENCIES OF WORDS ##
freq_df = df["tweet"].apply(lambda x: pd.value_counts(x.split(" "))).sum(axis=0).reset_index()
freq_df.columns = ["kelimeler", "frekanslar"]
freq_df.shape
frekans = freq_df[freq_df.frekanslar > 45]
frekans.shape
frekans.plot.bar(x="kelimeler", y="frekanslar")

## TWITTER LOGO ##
twitter = np.array(Image.open("/Users/ardaakdere/PycharmProjects/VBODEV/PROJE/twt.jpeg"))
text = " ".join(i for i in df.tweet)
wordcloud = WordCloud(background_color= "white", max_words=60, mask=twitter, contour_width=3, contour_color="black", width=1920, height=1080)
wordcloud.generate(text)
plt.figure(figsize=(5,5))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.tight_layout(pad=0)
plt.show()

## TWEET OKUYUCULAR TAKIMI ##
TweetOkuyucular = ["Gizem","GizemNur","Şimal","Hatice","Arda","Dilara"]
vbo = np.array(Image.open("VBO.jpg"))
text = " ".join(i for i in TweetOkuyucular)
wordcloud = WordCloud(background_color="white", mask=vbo, contour_width=3, contour_color="firebrick")
wordcloud.generate(text)
plt.figure(figsize=(10,10))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.tight_layout(pad=0)
plt.show()



