# Based on the following tutorial: https://data-dive.com/german-nlp-binary-text-classification-of-reviews-part1/

### Natural Language Processing of German texts - Using machine learning to predict ratings of suicide risk 
# Dataset: Downloaded from Reddit, Subreddit "SuicideWatch". Healthcare professionals have rated the suicide risk
# based on the posted Reddit (column: is_suicide). 

# This code has been implemented in Visual Studio Code (locally).

## 1. Set up modules

import re
import pickle
import sklearn
import pandas as pd
import numpy as np
import holoviews as hv
import hvplot
import nltk
import time
import json
#from keras.models import Sequential
#from keras.layers import SimpleRNN, Dense

# For Jupyter notebook: 
#from bokeh.io import output_notebook
#output_notebook()

# For VS Code: 
from bokeh.plotting import figure, show

from hvplot import pandas
from pathlib import Path
import os

CURR_PATH = Path(os.getcwd())
print(CURR_PATH)

#hv.extension("bokeh")

pd.options.display.max_columns = 100
pd.options.display.max_rows = 300
pd.options.display.max_colwidth = 100
np.set_printoptions(threshold=2000)

# We will use nltk to handle text related tasks and scikit library will be used to handle the machine learning part. 
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier
from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

## 2. Read raw data
# The data has the reviews in column 1 and the rating in column 2, which ranges from 1 to 6 with 6 being the worst.
FILE_REVIEWS = "/Users/samanthaweber/NLP_Suicide/combined-set-2.csv"
data = pd.read_csv(FILE_REVIEWS, sep=",", na_values=[""])
print(data.shape) #Verify that data is loaded
print(data.head())

## 3. Cleaning and pre-processing
# We will use frequency based representation methods for our text. Thus, we usually want to have a pretty thorough manipulation of the input data:

nltk.download('punkt') #tokenization of words and sentences
nltk.download('stopwords') #includes a list of stop words for English --> They are usually filtered out as they are less informative

stemmer = SnowballStemmer("english") #Stemming is the process of reducing words to their root or base form. Snowball stemmer is an algorithm implemented in the NLTK library
stop_words = set(stopwords.words("english")) #From stopwords we now retrieve those for English. 

def clean_text(text, for_embedding=False):
    # This function takes a string input and applies a bunch of manipulations to it. Removing characters and words that don't hold much meaning
    # will significantly reduce the size of our data. Thus, it can improve prediction performance when modelling with lower noise in the data. 
    """
        - remove any html tags (< /br> often found)
        - Keep only ASCII + European Chars and whitespace, no digits
        - remove single letter chars
        - convert all whitespaces (tabs etc.) to single wspace
        if not for embedding (but e.g. Term Frequency - Inverse Document Frequency (tdf-idf)):
        - all lowercase
        - remove stopwords, punctuation and stemm
    """
    # We first define the regular expressions
    RE_WSPACE = re.compile(r"\s+", re.IGNORECASE) #replaces multiple white space characters with a single whitespace
    RE_TAGS = re.compile(r"<[^>]+>") #This removes HTML tags from the text, e.g., like </br>
    RE_ASCII = re.compile(r"[^A-Za-zÀ-ž ]", re.IGNORECASE) #This removes non-alphabetic characters from text
    RE_SINGLECHAR = re.compile(r"\b[A-Za-zÀ-ž]\b", re.IGNORECASE) #This removes single letter words from the text
    if for_embedding: 
        # Keep punctuation and single letter words
        RE_ASCII = re.compile(r"[^A-Za-zÀ-ž,.!? ]", re.IGNORECASE)
        RE_SINGLECHAR = re.compile(r"\b[A-Za-zÀ-ž,.!?]\b", re.IGNORECASE)

    # Now we apply them to the text
    text = re.sub(RE_TAGS, " ", text)
    text = re.sub(RE_ASCII, " ", text)
    text = re.sub(RE_SINGLECHAR, " ", text)
    text = re.sub(RE_WSPACE, " ", text)

    word_tokens = word_tokenize(text) #initiate tokenizer function
    words_tokens_lower = [word.lower() for word in word_tokens] #Convert all words to lower case to avoid duplicaiton

    if for_embedding:
        # no stemming, lowering and punctuation / stop words removal
        words_filtered = word_tokens
    else:
        words_filtered = [
            stemmer.stem(word) for word in words_tokens_lower if word not in stop_words
        ]

    text_clean = " ".join(words_filtered)
    return text_clean

# Clean Text column
data = data.iloc[:, [6,3]]
print(data)
#data["text_clean"] = data.loc[data["selftext_clean"].str.len() > 20, "selftext_clean"] #we create new column in the dataframe "data". We include only reviews that are longer than 20 characters. 
data["text_clean"] = data.loc[data["selftext"].str.len() > 20, "selftext"] #we create new column in the dataframe "data". We include only reviews that are longer than 20 characters. 

print(data.head()) #Verify that column was created

# we now apply the clean_text function (defined above) to each row/review in the text_clean column of our data frame, but only if its value (instance) is a string. 
data["text_clean"] = data["text_clean"].map(
    lambda x: clean_text(x, for_embedding=False) if isinstance(x, str) else x # the lambda function takes x as an input, and checks if it's a string. If yes, it applies the clean_text function, else it leaaves it unchagned. 
)

print(data.head())

# Drop when any of x missing, i.e., when text_clean is empty. 
data = data[(data["text_clean"] != "") & (data["text_clean"] != "null")]

# Drops columns where label, text or text_clean columns are empty. 
data = data.dropna(
    axis="index", subset=["is_suicide", "selftext", "text_clean"]
).reset_index(drop=True)

data_clean = data.copy()
data_clean.head(3)
data_clean.shape

## 4. Descriptive analysis
# We can plot now the 40 most common words

word_freq = pd.Series(" ".join(data_clean["text_clean"]).split()).value_counts()
import matplotlib.pyplot as plt

# Word Frequency of most common word_freq = pd.Series(" ".join(data_clean["text_clean"]).split()).value_counts()
word_freq[1:40].rename("Word frequency of most common words in Reviews").plot(kind='bar', rot=45, figsize=(10, 6))
plt.xlabel('Words')
plt.ylabel('Frequency')
plt.title('Word Frequency of most common words in texts')
plt.show()

# list most uncommon words
word_freq[-10:].reset_index(name="freq").hvplot.table()

# Convert the last 10 values of word_freq to a DataFrame
table_data = word_freq[-10:].reset_index(name="freq")
print(table_data) #10 last frequent words

table_data = word_freq[:10].reset_index(name="freq")
print(table_data) #10 most frequent words

# Print DataFrame with better formatting
print(table_data.to_string(index=False))

# Identifying the most frequent or least frequent words could help to add words to our stop list. For example "Doktor" could be a good word to remove, as this has no sentiment,
# while words like "gut" have a strong semtiment and should be kept. Particularly, the least frequent words could be removed too as they might be very 
# rare (e.g., rotatorenmanschettenruptur)

# Finally, we also look at distribution of our target variable
# Calculate the distribution of ratings
distribution = data_clean["is_suicide"].value_counts(normalize=True).sort_index()

# Create a bar plot
plt.bar(distribution.index, distribution.values)

# Set plot title and labels
plt.title("Distribution of ratings")
plt.xlabel("Rating")
plt.ylabel("Proportion")

# Customize x-axis tick labels
plt.xticks([0, 1], ['0 (not at risk)', '1 (at risk)'])
# Set x-axis limits
plt.xlim(-0.5, 1.5)

# Show plot 
plt.show()

## 5. Feature creation with TF-IDF
# TF: summarizes how often a word apprears in a comment in relation to all words. 
# IDF: downscales the words prevalent in many other comments, giving higher weights to frequent and SPECIFIC words given the context. 

# Compute unique word vector with frequencies
# exclude very uncommon (<10 obsv.) and common (>=30%) words
# use pairs of two words (ngram)

# Ngram: an ngram of one means you look at each word seperately. An ngram of two means you take the preceding and following word into account as well, which adds context
# This helps, as e.g., "good" and "not good" are different. 

# Settign up the TF-IDF Veectorizer
vectorizer = TfidfVectorizer(
    # analyzer = analyse the text at word level
    # max_df = maximum document frequency. Here we exclude words with a frequency higher than 30%
    # min_df = minimum document frequency. Here the word must appear at least in 5 documents to be included. Helps to remove overly rare words. 
    # ngram_range = we use unigrams (single words) and bigrams (pairs of words), so that we can add some context. 
    # L2 normalization = ensures that each vector is normalized to 1
    analyzer="word", max_df=0.3, min_df=5, ngram_range=(1, 2), norm="l2"
)

# Now we compute the TF-IDF in the comment_clean column, which will create a vector that we can use for the ML. 
vectorizer.fit(data_clean["text_clean"])

# Vector representation of vocabulary
word_vector = pd.Series(vectorizer.vocabulary_).sample(5, random_state=1)
# This creates a numeric representation of ngrams in our corpus. It uses unigrams and bigrams
print(f"Unique word (ngram) vector extract:\n\n {word_vector}")


## Prepare the data for modeling part
# We split data into training and testing set. 

# Sample data - 25% of data to test set
train, test = train_test_split(data_clean, random_state=1, test_size=0.25, shuffle=True)
print(train)

X_train = train["text_clean"]
Y_train = train["is_suicide"]
X_test = test["text_clean"]
Y_test = test["is_suicide"]
print(X_train.shape)
print(X_test.shape)

# transform each sentence to numeric vector with tf-idf value as elements
X_train_vec = vectorizer.transform(X_train)
X_test_vec = vectorizer.transform(X_test)
X_train_vec.get_shape()
print(X_train_vec)

# We can compare the vector representation of text with the numeric representation
# Compare original comment text with its numeric vector representation
print(f"Original sentence:\n{X_train[3:4].values}\n")
# Feature Matrix
features = pd.DataFrame(
    X_train_vec[3:4].toarray(), columns=vectorizer.get_feature_names_out()
)
nonempty_feat = features.loc[:, (features != 0).any(axis=0)]
print(f"Vector representation of sentence:\n {nonempty_feat}")

## 5. Classification
# from scikit-learn package we can use different classifiers: 
# Logistic regression, support vector classifier, ensemble methods (bosting, random forest), neural networks (multi layer perceptron)

# models to test
classifiers = [
    LogisticRegression(solver="sag", random_state=1),
    LinearSVC(random_state=1),
    RandomForestClassifier(random_state=1),
    XGBClassifier(random_state=1),
    MLPClassifier(
        random_state=1,
        solver="adam", #sgd
        hidden_layer_sizes=([15,]*5), #three hidden layers, each with 12 neurons
        activation="relu", #relu
        early_stopping=True,
        n_iter_no_change=1,
    ),
    #rnn_model
]
# get names of the objects in list (too lazy for c&p...)
names = [re.match(r"[^\(]+", name.__str__())[0] for name in classifiers]
print(f"Classifiers to test: {names}")

start_time = time.time()

# test all classifiers and save pred. results on test data
results = {}
for name, clf in zip(names, classifiers):
    print(f"Training classifier: {name}")
    clf.fit(X_train_vec, Y_train)
    prediction = clf.predict(X_test_vec)
    report = sklearn.metrics.classification_report(Y_test, prediction)
    results[name] = report
    print(f"Classification with {name}: DONE!")


end_time = time.time()

#Calculate elapsed time

elapsed_time = end_time - start_time

print("Elapsed time: {:.2f} seconds".format(elapsed_time))

# Prediction results
for k, v in results.items():
    print(f"Results for {k}:")
    print(f"{v}\n")

# Explanation metrics: 
    # Accuracy: overall proportion of correctly classified instances. This might not be appropriate for an unbalanced dataset
    # F1-score: harmonic mean between precision and recall. More fine grained, and more appropriate for unbalanced dataset
    # Macro F1: average across all classes giving no weights
    # weighted F1: average across all classes giving weights based on its support (i.e., number of cases per class)


## 6. Parameter Tuning
# For simplicity, we continue with SVM and MLP
# We can apply a more guided approach to choose our parameters. In scikit learn we have the grid search  functionality that allows us to find the perfect parameters. 
# We will evaluate all parameter combinations against one another in a cross validation. 

# feature creation and modelling in a single function
pipe = Pipeline([("tfidf", TfidfVectorizer()), ("svc", LinearSVC())])

# define parameter space to test 
params = {
    "tfidf__ngram_range": [(1, 1), (1, 2), (1, 3)], # unigram, bigrams and trigrams
    "tfidf__max_df": np.arange(0.3, 0.8, 0.2), #include maximum document frequencies from 0.3 to 0.7 in steps on 0.2
    "tfidf__min_df": np.arange(5, 100, 45), #from 5 to less than 100 with stepsize of 45
}
# initialize grid search cross-validation (CV) based on define pipeline. Setting jobs = -1 will run the job on all available CPU cores. 
pipe_clf = GridSearchCV(pipe, params, n_jobs=-1, scoring="f1_macro")
pipe_clf.fit(X_train, Y_train)
pickle.dump(pipe_clf, open("./clf_pipe.pck", "wb")) #serialize Python objects, open file in write-binary mode (wb) --> structure pickle.dump(pipe, file_object)

# Check best parameters and then use them
print(pipe_clf.best_params_)

## Optimize classification with best parameter set
# feature creation and modelling in a single function
pipe = Pipeline([("tfidf", TfidfVectorizer()), ("svc", LinearSVC())])

# define parameter space to test # runtime 19min
params = {
    "tfidf__ngram_range": [(1, 3)],
    "tfidf__max_df": [0.5],
    "tfidf__min_df": [5],
    "svc__C": np.arange(0.2, 1, 0.15),
}
pipe_svc_clf = GridSearchCV(pipe, params, n_jobs=-1, scoring="f1_macro")
pipe_svc_clf.fit(X_train, Y_train)
pickle.dump(pipe_svc_clf, open("./pipe_svc_clf.pck", "wb"))

best_params = pipe_svc_clf.best_params_
print(best_params)

# Run the prediction using the best parameter fit

pipe.set_params(**best_params).fit(X_train, Y_train)
pipe_pred = pipe.predict(X_test)
report = sklearn.metrics.classification_report(Y_test, pipe_pred)
print(report)

# Get confidence score for prediction
conf_score = pipe.decision_function(X_test)

#print(conf_score)


## Optimize classification with best parameter set
# feature creation and modelling in a single function
pipe = Pipeline([("tfidf", TfidfVectorizer()), ("mlp", MLPClassifier())])

# define parameter space to test # runtime 19min
params = {
    "tfidf__ngram_range": [(1, 2)],
    "tfidf__max_df": [0.8],
    "tfidf__min_df": [2],
    "mlp__hidden_layer_sizes": [(15,)*7],  # Example of different hidden layer configurations
    "mlp__alpha": [0.0001, 0.00001],  # Regularization parameter
}
pipe_mlp_clf = GridSearchCV(pipe, params, n_jobs=-1, scoring="f1_macro")
pipe_mlp_clf.fit(X_train, Y_train)
pickle.dump(pipe_mlp_clf, open("./pipe_mlp_clf.pck", "wb"))

best_params = pipe_mlp_clf.best_params_
print(best_params)

# Run the prediction using the best parameter fit

pipe.set_params(**best_params).fit(X_train, Y_train)
pipe_pred = pipe.predict(X_test)
report = sklearn.metrics.classification_report(Y_test, pipe_pred)
print(report)
