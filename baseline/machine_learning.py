#!/usr/bin/env python

'''This script performs text classification using various machine 
   learning algorithms. It supports different text vectorization 
   methods and hyperparameter tuning'''

import argparse
import json
import nltk
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import accuracy_score, precision_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import recall_score, f1_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import FeatureUnion
from nltk import word_tokenize, pos_tag
from sklearn.svm import SVC, LinearSVC
from sklearn.pipeline import Pipeline

nltk.download('punkt')
try:
    nltk.data.find('taggers/averaged_perceptron_tagger')
except LookupError:
    print("Downloading the 'averaged_perceptron_tagger' resource...")
    nltk.download('averaged_perceptron_tagger')

def create_arg_parser():
    '''Function to create command line arguments'''
    parser = argparse.ArgumentParser()
    parser.add_argument("-tf", "--train_file", default='train.json', type=str, help="Train file to learn from")
    parser.add_argument("-df", "--dev_file", default='dev.json', type=str, help="Dev file to evaluate on")
    parser.add_argument("-s", "--sentiment", action="store_true", help="Do sentiment analysis (2-class problem)")
    parser.add_argument("-t", "--tfidf", action="store_true", help="Use the TF-IDF vectorizer instead of CountVectorizer")
    parser.add_argument("-u", "--union", action="store_true", help="Use the combined TF-IDF vectorizer and CountVectorizer")
    parser.add_argument("--algorithm", choices=["nb", "dt", "rf", "knn", "svm"], default="nb", help="Choose the classification algorithm (naive_bayes, decision_tree, random_forest, knn, svm")
    parser.add_argument("--alpha", type=float, default=1.0, help="Smoothing parameter (Laplace smoothing)")
    parser.add_argument("--fit_prior", type=bool, default=True, help="Whether to learn class prior probabilities from data")
    parser.add_argument("--class_prior", type=float, nargs="+", default=None, help="Explicit class prior probabilities")
    parser.add_argument("--n_estimators", type=int, default=100, help="The number of trees in the forest. A higher value may lead to better generalization but increased computational cost.")
    parser.add_argument("--bootstrap", action="store_true", help="Whether to bootstrap samples when building trees. If True, each tree is trained on a random subset of the data.")
    parser.add_argument("--oob_score", action="store_true", help="Whether to compute out-of-bag (OOB) score to estimate the model's accuracy without the need for a separate validation set.")
    parser.add_argument("--max_depth", type=int, default=None, help="Maximum depth of each tree in the forest. A higher value can lead to overfitting. Default is None, which means nodes are expanded until they contain less than min_samples_split samples or until they contain less than min_samples_leaf samples if min_samples_leaf is specified.")
    parser.add_argument("--random_state", type=int, default=None, help="Seed for random number generator. Ensures reproducibility of results.")
    parser.add_argument("--criterion", type=str, default="gini", help="The function to measure the quality of a split in each tree. 'gini' for Gini impurity or 'entropy' for information gain.")
    parser.add_argument("--min_samples_split", type=int, default=2, help="Minimum number of samples required to split an internal node in each tree. Higher values prevent overfitting by making splits more conservative.")
    parser.add_argument("--min_samples_leaf", type=int, default=1, help="Minimum number of samples required to be in a leaf node in each tree. Helps control tree complexity and prevents overfitting.")
    parser.add_argument("--n_neighbors", type=int, default=5, help="Number of neighbors for KNN (default: 5)")
    parser.add_argument("--svm_kernel", choices=["linear", "rbf", "poly"], default="linear", help="SVM kernel type (linear, rbf, poly, default: linear)")
    parser.add_argument("--svm_c", type=float, default=1.0, help="Regularization parameter C for SVM (default: 1.0)")
    parser.add_argument("-ngram", type=int, help="Do N-gram analysis, specify the range number 1-number")
    parser.add_argument("-char", help="Do character-level analysis")
    parser.add_argument("-pos", action="store_true", help="Do POS tagging")
    args = parser.parse_args()
    return args

def read_corpus(corpus_file, use_sentiment):
    '''Function to read and process the data'''
    documents = []
    documents_not_split = []  # Added for character-level analysis
    labels = []
    with open(corpus_file, encoding='utf-8-sig') as in_file:
        data = json.load(in_file)
        for entry in data:
            text = entry.get("text")
            label = entry.get("label")
            if text is not None:
                tokens = text.split()
                documents.append(" ".join(tokens))
                documents_not_split.append(text)
            if label == "OFF":                       # change the label accordingly
                labels.append(1)
            else:
                labels.append(0)
    return documents, documents_not_split, labels

def identity(inp):
    '''Dummy function that just returns the input'''
    return inp

def pos_tagging(X_train):
    ''' A function for doing POS analysis '''
    tagged_corpus=[]
    for sentence in X_train:
        sent=' '.join(sentence)
        tokens = word_tokenize(sent)
        tagged_tokens = pos_tag(tokens)
        tagged_corpus.append(tagged_tokens)
    return tagged_corpus

def select_vectorizer(args):
    '''A function for selecting a vectorizer'''
    if args.ngram:
        ngram_range = (1, args.ngram)
    else:
        ngram_range = (1, 3)

    # Create different vectorizers
    ngram_range = (1, args.ngram)
    count_vectorizer = CountVectorizer(preprocessor=identity, tokenizer=identity)
    tfidf_vectorizer = TfidfVectorizer(preprocessor=identity, tokenizer=identity)
    union_vectorizer = FeatureUnion([("count", count_vectorizer), ("tf", tfidf_vectorizer)])
    ngram_vectorizer = TfidfVectorizer(ngram_range=ngram_range, min_df=3)
    char_count_vec = CountVectorizer(analyzer='char_wb', min_df=3)

    if args.tfidf:
        vec = tfidf_vectorizer
        if args.ngram:
            vec = ngram_vectorizer
    elif args.union:
        vec = union_vectorizer
    elif args.char:
        vec = char_count_vec
    else:
        vec = count_vectorizer
    return vec

def classifier_create(args, vec, X_train, Y_train):
    ''' Create and configure a classifier based on the chosen algorithm and hyperparameters'''  
    if args.algorithm == "nb":
        return Pipeline([('vec', vec), ('cls', MultinomialNB(
            alpha=args.alpha,
            fit_prior=args.fit_prior,
            class_prior=args.class_prior,
        ))])
    elif args.algorithm == "dt":
        return Pipeline([('vec', vec), ('cls', DecisionTreeClassifier(
            max_depth=args.max_depth,
            min_samples_split=args.min_samples_split,
            min_samples_leaf=args.min_samples_leaf,
            criterion=args.criterion,
            random_state=args.random_state
        ))])
    elif args.algorithm == "rf":
        return Pipeline([('vec', vec), ('cls', RandomForestClassifier(
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            min_samples_split=args.min_samples_split,
            min_samples_leaf=args.min_samples_leaf,
            criterion=args.criterion,
            bootstrap=args.bootstrap,
            oob_score=args.oob_score,
            random_state=args.random_state
            ))])
    elif args.algorithm == "knn":
        return Pipeline([('vec', vec), ('cls', KNeighborsClassifier(
            n_neighbors=args.n_neighbors,
            weights=args.weights,
            p=args.p,
            leaf_size=args.leaf_size,
        ))])
    elif args.algorithm == "svm":
        if args.svm_kernel == "linear":
            return Pipeline([('vec', vec), ('cls', LinearSVC(
                C=args.svm_c,
            ))])
        elif args.svm_kernel == "rbf":
            return Pipeline([('vec', vec), ('cls', SVC(
                C=args.svm_c,
                kernel='rbf',
                gamma='scale'
            ))])
        elif args.svm_kernel == "poly":
            return Pipeline([('vec', vec), ('cls', SVC(
                C=args.svm_c,
                kernel='poly',
                degree=3,
                gamma='auto'
            ))])

if __name__ == "__main__":
    args = create_arg_parser()

    # To read the train file and the dev data set, Z is added for character level analysis
    X_train, Z_train, Y_train = read_corpus(args.train_file, args.sentiment)
    X_test, Z_test, Y_test = read_corpus(args.dev_file, args.sentiment)

    # To do POS tagging
    tagged_corpus = pos_tagging(X_train)

    # Select vectorizer based on the command line arguments
    vec = select_vectorizer(args)

    # Create and train the classifier
    classifier = classifier_create(args, vec, X_train, Y_train)

    if args.char:
        # Train the classifier with Z_train data, character level
        classifier.fit(Z_train, Y_train)
        Y_pred = classifier.predict(Z_test)
    elif args.pos:
        # Use POS tagged data for training
        tagged_corpus = pos_tagging(X_train)
        classifier.fit(tagged_corpus, Y_train)
        Y_pred = classifier.predict(pos_tagging(X_test))
    else:
        classifier.fit(X_train, Y_train)
        Y_pred = classifier.predict(X_test)

    # Calculate and print final accuracy
    acc = accuracy_score(Y_test, Y_pred)
    print(f"Final accuracy: {acc}")

    # Calculate presicion, recall, and f-score
    pres = precision_score(Y_test, Y_pred, average=None)
    rec = recall_score(Y_test, Y_pred, average=None)
    f1 = f1_score(Y_test, Y_pred, average=None)


    # Create and print a classification report
    report = classification_report(Y_test, Y_pred, target_names = ["OFF", "NOT"]) # adjust labels
    print("Classification Report:")
    print(report)

    # Calculate and print the confusion matrix
    conf_matrix = confusion_matrix(Y_test, Y_pred)
    print("Confusion Matrix:")
    print(conf_matrix)
