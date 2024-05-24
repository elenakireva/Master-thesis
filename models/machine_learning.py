# Name: Elena Kireva
# Description: This Python script takes json files as arguments, generates word and sentence embeddings,calculates 
# cosine similarity between vectors, and performs classification on the data based on the command line arguments. 
# Supported algorithms: naive bayes, svm, random forest, decision trees. The word embeddings are generated with word2vec 
# and the sentence embeddings are generated with sbert. The cosine similarity is calculated based on the used embeddings 
# and implemented as a feature to the classifiers. Additionally, different hyperparameters and vectorizers could be 
# adjusted for each classifier directly from the command line

from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sentence_transformers import SentenceTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import make_pipeline
from sklearn.pipeline import FeatureUnion
from scipy.spatial.distance import cosine
from nltk.tokenize import word_tokenize
from gensim.models import Word2Vec
from sklearn.svm import SVC
import numpy as np
import argparse
import json

def load_data(file_path):
    """ Function to load the data from json files """
    with open(file_path, 'r') as file:
        data = json.load(file)
    texts = [item['combined_text'] for item in data]
    labels = [1 if item['condition'] == 'healthy' else 0 for item in data]
    return texts, labels

def generate_word_embeddings(texts, window_size):
    """ Function to generate Word2Vec embeddings """
    tokenized_texts = [word_tokenize(text.lower()) for text in texts]
    model = Word2Vec(tokenized_texts, vector_size=200, window=window_size, min_count=2, workers=-1)
    embeddings = [np.mean([model.wv[word] for word in text if word in model.wv], axis=0) for text in tokenized_texts]
    embeddings = [embedding if not np.isnan(embedding).any() else np.zeros(100) for embedding in embeddings]
    return np.array(embeddings)

def generate_sentence_embeddings(texts):
    """ Function to generate sentence embeddings using SBERT """
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(texts)
    return np.array(embeddings)

def calculate_cosine_metrics(embeddings, max_window, embedding_type):
    """ Function to calculate cosine similarity metrics (mean and variance) """
    metrics = np.zeros((len(embeddings), 2 * max_window))
    for index in range(len(embeddings)):
        for window_size in range(1, max_window+1):
            cos_sims = []
            for i in range(max(0, index - window_size + 1), min(len(embeddings), index + window_size)):
                if i + 1 < len(embeddings):
                    sim = 1 - cosine(embeddings[i], embeddings[i + 1])
                    cos_sims.append(sim)
            if cos_sims:
                metrics[index, (window_size - 1) * 2] = np.mean(cos_sims)
                metrics[index, (window_size - 1) * 2 + 1] = np.var(cos_sims)
    if embedding_type == 'word':
        print("Word Embeddings Cosine Similarity Metrics (Mean and Variance):")
    elif embedding_type == 'sentence':
        print("Sentence Embeddings Cosine Similarity Metrics (Mean and Variance):")
    for index, participant_metrics in enumerate(metrics):
        print(f"Participant {index + 1}: Mean: {participant_metrics[::2]}, Variance: {participant_metrics[1::2]}")
    return metrics

def get_vectorizer(vectorizer_type):
    """ Function to pick different vectorizers """
    if vectorizer_type == 'tf-idf':   # tf-idf vectorizer
        return TfidfVectorizer()
    elif vectorizer_type == 'count':  # count vectorizer
        return CountVectorizer()
    elif vectorizer_type == 'union':  # combination of count + tf-idf
        return FeatureUnion([('tfidf', TfidfVectorizer()), ('count', CountVectorizer())])
    else:
        raise ValueError("Unsupported vectorizer type")

def get_classifier(classifier_type, hyperparameters):
    """ Function to return the classifier based on the input type """
    if classifier_type == 'svm':   # svm classifier
        return SVC(**hyperparameters)
    elif classifier_type == 'nb':  # naive bayes classifier
        return GaussianNB(**hyperparameters)
    elif classifier_type == 'rf':  # random forest classifier
        return RandomForestClassifier(**hyperparameters)
    elif classifier_type == 'dt':  # decision trees classifier
        return DecisionTreeClassifier(**hyperparameters)
    else:
        raise ValueError("Unsupported classifier type")

def main(args):
    train_texts, train_labels = load_data(args.train)
    dev_texts, dev_labels = load_data(args.dev)
    
    # If only the classifiers are used
    train_embeddings, dev_embeddings = np.zeros((len(train_texts), 0)), np.zeros((len(dev_texts), 0))
    train_metrics, dev_metrics = np.zeros((len(train_texts), 0)), np.zeros((len(dev_texts), 0))

    # If word embeddings are used
    if args.use_word_embeddings:
        train_embeddings = generate_word_embeddings(train_texts, args.window_size)
        dev_embeddings = generate_word_embeddings(dev_texts, args.window_size)
        train_metrics = calculate_cosine_metrics(train_embeddings, 15, 'word')
        dev_metrics = calculate_cosine_metrics(dev_embeddings, 15, 'word')
    # If sentence embeddings are used    
    elif args.use_sentence_embeddings:
        train_embeddings = generate_sentence_embeddings(train_texts)
        dev_embeddings = generate_sentence_embeddings(dev_texts)
        train_metrics = calculate_cosine_metrics(train_embeddings, 9, 'sentence')
        dev_metrics = calculate_cosine_metrics(dev_embeddings, 9, 'sentence')

    vectorizer = get_vectorizer(args.vectorizer)
    train_features = vectorizer.fit_transform(train_texts)
    dev_features = vectorizer.transform(dev_texts)

    # Debugging statements to verify feature shapes (used only during the setup to ensure correct shapes)
    print("Shape of train features from vectorizer:", train_features.shape)
    print("Shape of dev features from vectorizer:", dev_features.shape)
    print("Shape of train embeddings:", train_embeddings.shape)
    print("Shape of dev embeddings:", dev_embeddings.shape)
    print("Shape of train metrics:", train_metrics.shape)
    print("Shape of dev metrics:", dev_metrics.shape)

    # Combining embeddings and text features if selected
    train_features = np.hstack([train_features.toarray(), train_embeddings, train_metrics])
    dev_features = np.hstack([dev_features.toarray(), dev_embeddings, dev_metrics])

    # Debugging statements to verify combined feature shapes (used only during setup)
    print("Shape of combined train features:", train_features.shape)
    print("Shape of combined dev features:", dev_features.shape)

    # Print the first few samples of combined features for verification
    print("First few samples of combined train features:")
    print(train_features[:5])
    print("First few samples of combined dev features:")
    print(dev_features[:5])

    # Classifier hyperparameters based on the command line (different options are shown in the arg parse below)
    classifier_hyperparameters = {}
    if args.classifier == 'svm':  # for the svm classifier
        classifier_hyperparameters['C'] = args.svm_C
        classifier_hyperparameters['kernel'] = args.svm_kernel
        classifier_hyperparameters['gamma'] = args.svm_gamma
    elif args.classifier == 'nb':  # for the naive bayes classifier
        classifier_hyperparameters['var_smoothing'] = args.nb_var_smoothing
    elif args.classifier == 'rf':  # for the random forest classifier
        classifier_hyperparameters['n_estimators'] = args.rf_n_estimators
        classifier_hyperparameters['max_depth'] = args.rf_max_depth
        classifier_hyperparameters['min_samples_split'] = args.rf_min_samples_split
    elif args.classifier == 'dt':  # for the decision trees classifier
        classifier_hyperparameters['max_depth'] = args.dt_max_depth
        classifier_hyperparameters['min_samples_split'] = args.dt_min_samples_split
    
    classifier = get_classifier(args.classifier, classifier_hyperparameters)
    trained_classifier = classifier.fit(train_features, train_labels)
    dev_pred = trained_classifier.predict(dev_features)

    # Printing the evaluation metrics
    print("Classification Report:")
    print(classification_report(dev_labels, dev_pred))
    print("Confusion Matrix:")
    print(confusion_matrix(dev_labels, dev_pred))
    print("F1 Score:")
    print(f1_score(dev_labels, dev_pred, average='weighted'))
    print("Accuracy:")
    print(accuracy_score(dev_labels, dev_pred))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Select files, classifier, vectorizer, embeddings, and other parameters")
    parser.add_argument('--train', type=str, required=True, help='Path to the JSON train data file')
    parser.add_argument('--dev', type=str, required=True, help='Path to the JSON dev data file')
    parser.add_argument('--classifier', type=str, required=True, help='Classifier type, choices: svm, nb, rf, dt')
    parser.add_argument('--vectorizer', type=str, choices=['tf-idf', 'count', 'union'], default='tf-idf', help='Vectorizer type, choices: tf-idf, count, or union')
    parser.add_argument('--use_word_embeddings', action='store_true', help='Use Word2Vec for word embeddings')
    parser.add_argument('--use_sentence_embeddings', action='store_true', help='Use SBERT for sentence embeddings')
    parser.add_argument('--window_size', type=int, default=5, help='Window size for Word2Vec')
    parser.add_argument('--max_window', type=int, default=15, help='Max window size for cosine similarity')

    # Support Vector Machine (SVM) hyperparameters
    parser.add_argument('--svm_C', type=float, default=1.0, help='C hyperparameter for SVM classifier')
    parser.add_argument('--svm_kernel', type=str, default='rbf', choices=['linear', 'poly', 'rbf', 'sigmoid'], help='Kernel type for SVM classifier')
    parser.add_argument('--svm_gamma', type=str, default='scale', choices=['scale', 'auto'], help='Gamma value for SVM classifier')

    # Naive Bayes (NB) hyperparameters
    parser.add_argument('--nb_var_smoothing', type=float, default=1e-9, help='Variance smoothing for Naive Bayes classifier')

    # Random Forest (RF) hyperparameters
    parser.add_argument('--rf_n_estimators', type=int, default=100, help='Number of trees in Random Forest classifier')
    parser.add_argument('--rf_max_depth', type=int, default=None, help='Maximum depth of trees in Random Forest classifier')
    parser.add_argument('--rf_min_samples_split', type=int, default=2, help='Minimum samples required to split in Random Forest classifier')

    # Decision Tree (DT) hyperparameters
    parser.add_argument('--dt_max_depth', type=int, default=None, help='Maximum depth of Decision Tree classifier')
    parser.add_argument('--dt_min_samples_split', type=int, default=2, help='Minimum samples required to split in Decision Tree classifier')

    args = parser.parse_args()
    main(args)
