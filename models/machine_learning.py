# Name: Elena Kireva
# Description: This Python script takes a json file as an argument, generates word and sentence embeddings,
# calculates cosine similarity between vectors, and performs classification on the data based on the command line arguments.
# Supported algorithms: naive bayes, svm, random forest, decision trees. The word embeddings are generated with word2vec 
# and the sentence embeddings are generated with sbert. The cosine similarity is calculated based on the used embeddings 
# and implemented as a feature to the classifiers. Additionally, different hyperparameters and vectorizers could be 
# adjusted for each classifier directly from the command line.

from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sentence_transformers import SentenceTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import FeatureUnion
from scipy.spatial.distance import cosine
from nltk.tokenize import word_tokenize
from gensim.models import Word2Vec
import matplotlib.pyplot as plt
from sklearn.svm import SVC
import numpy as np
import argparse
import json

def load_data(file_path):
    """ Function to load the data from the json file """
    with open(file_path, 'r') as file:
        data = json.load(file)
    texts = [item['combined_text'] for item in data]
    pauses = [item['pauses'] for item in data]
    labels = [1 if item['condition'] == 'healthy' else 0 for item in data]
    return texts, pauses, labels

def generate_word_embeddings(texts, window_size=5):
    """ Function to generate Word2Vec embeddings """
    tokenized_texts = [word_tokenize(text.lower()) for text in texts]
    model = Word2Vec(tokenized_texts, vector_size=200, window=window_size, min_count=2, workers=-1)
    embeddings = [np.mean([model.wv[word] for word in text if word in model.wv], axis=0) for text in tokenized_texts]
    embeddings = [embedding if not np.isnan(embedding).any() else np.zeros(200) for embedding in embeddings]
    return np.array(embeddings)

def generate_sentence_embeddings(texts):
    """ Function to generate sentence embeddings using SBERT """
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(texts)
    return np.array(embeddings)

def calculate_cosine_metrics(embeddings, max_window, embedding_type='word'):
    """ Function to calculate cosine similarity metrics (mean and variance) """
    metrics = np.zeros((len(embeddings), 2 * max_window))
    for index in range(len(embeddings)):
        for window_size in range(1, max_window + 1):
            cos_sims = []
            for i in range(max(0, index - window_size + 1), min(len(embeddings), index + window_size)):
                if i + 1 < len(embeddings):
                    sim = 1 - cosine(embeddings[i], embeddings[i + 1])
                    cos_sims.append(sim)
            if cos_sims:
                metrics[index, (window_size - 1) * 2] = np.mean(cos_sims)
                metrics[index, (window_size - 1) * 2 + 1] = np.var(cos_sims)
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
    """ Function to return the classifier based on the input """
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

def plot_feature_importances(classifier, feature_names, output_file):
    """ Function to plot feature importances for RF and SVM"""
    if isinstance(classifier, RandomForestClassifier):
        importances = classifier.feature_importances_
    elif isinstance(classifier, SVC) and classifier.kernel == 'linear':
        importances = np.abs(classifier.coef_[0])
    else:
        return
    
    indices = np.argsort(importances)[-48:]
    plt.figure(figsize=(10, len(indices) // 3))
    plt.title("Feature importances")
    plt.barh(range(len(indices)), importances[indices], color="b", align="center")
    plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
    plt.xlabel("Relative importance")
    plt.show()

def main(args):
    texts, pauses, labels = load_data(args.data)
    labels = np.array(labels)

    vectorizer = get_vectorizer(args.vectorizer)
    text_features = vectorizer.fit_transform(texts).toarray()

    scaler = StandardScaler()
    pause_features = scaler.fit_transform(np.array(pauses).reshape(-1, 1))
    
    # Generate word and sentence embeddings
    word_embeddings = generate_word_embeddings(texts)
    sentence_embeddings = generate_sentence_embeddings(texts)

    # Calculate cosine similarity
    word_cosine_metrics = calculate_cosine_metrics(word_embeddings, max_window=15, embedding_type='word')
    sentence_cosine_metrics = calculate_cosine_metrics(sentence_embeddings, max_window=9, embedding_type='sentence')

    # Combine features based on the selected configuration
    feature_combinations = []
    feature_names = []
    if args.use_text:  # if only the pre-processed text is used
        feature_combinations.append(text_features)
        feature_names.extend([f"text_{i}" for i in range(text_features.shape[1])])
    if args.use_pauses:  # if only pauses are used
        feature_combinations.append(pause_features)
        feature_names.append("pause")
    if args.use_both:  # both text and pauses
        feature_combinations.append(text_features)
        feature_combinations.append(pause_features)
        feature_names.extend([f"text_{i}" for i in range(text_features.shape[1])])
        feature_names.append("pause")
    if args.use_cosine == 'word':  # word cosine
        feature_combinations.append(word_cosine_metrics)
        feature_names.extend([f"word_cosine_mean_{i}" for i in range(word_cosine_metrics.shape[1] // 2)])
        feature_names.extend([f"word_cosine_var_{i}" for i in range(word_cosine_metrics.shape[1] // 2)])
    elif args.use_cosine == 'sentence':  # sentence cosine
        feature_combinations.append(sentence_cosine_metrics)
        feature_names.extend([f"sentence_cosine_mean_{i}" for i in range(sentence_cosine_metrics.shape[1] // 2)])
        feature_names.extend([f"sentence_cosine_var_{i}" for i in range(sentence_cosine_metrics.shape[1] // 2)])
    elif args.use_cosine == 'both':  # both word and sentence
        feature_combinations.append(word_cosine_metrics)
        feature_combinations.append(sentence_cosine_metrics)
        feature_names.extend([f"word_cosine_mean_{i}" for i in range(word_cosine_metrics.shape[1] // 2)])
        feature_names.extend([f"word_cosine_var_{i}" for i in range(word_cosine_metrics.shape[1] // 2)])
        feature_names.extend([f"sentence_cosine_mean_{i}" for i in range(sentence_cosine_metrics.shape[1] // 2)])
        feature_names.extend([f"sentence_cosine_var_{i}" for i in range(sentence_cosine_metrics.shape[1] // 2)])

    features = np.hstack(feature_combinations)

    # Classifier hyperparameters
    classifier_hyperparameters = {}
    if args.classifier == 'svm':
        classifier_hyperparameters['C'] = args.svm_C
        classifier_hyperparameters['kernel'] = args.svm_kernel
        classifier_hyperparameters['gamma'] = args.svm_gamma
        classifier_hyperparameters['class_weight'] = 'balanced'
    elif args.classifier == 'nb':
        classifier_hyperparameters['var_smoothing'] = args.nb_var_smoothing
    elif args.classifier == 'rf':
        classifier_hyperparameters['n_estimators'] = args.rf_n_estimators
        classifier_hyperparameters['max_depth'] = args.rf_max_depth
        classifier_hyperparameters['min_samples_split'] = args.rf_min_samples_split
    elif args.classifier == 'dt':
        classifier_hyperparameters['max_depth'] = args.dt_max_depth
        classifier_hyperparameters['min_samples_split'] = args.dt_min_samples_split

    classifier = get_classifier(args.classifier, classifier_hyperparameters)

    # K-Fold Cross Validation
    kf = StratifiedKFold(n_splits=args.k_folds, shuffle=True, random_state=42)
    f1_scores = []
    accuracies = []
    confusion_matrices = []
    classification_reports = []

    for train_index, test_index in kf.split(features, labels):
        X_train, X_test = features[train_index], features[test_index]
        y_train, y_test = labels[train_index], labels[test_index]

        trained_classifier = classifier.fit(X_train, y_train)
        y_pred = trained_classifier.predict(X_test)

        f1_scores.append(f1_score(y_test, y_pred, average='weighted'))
        accuracies.append(accuracy_score(y_test, y_pred))
        confusion_matrices.append(confusion_matrix(y_test, y_pred))
        classification_reports.append(classification_report(y_test, y_pred, output_dict=True))

    avg_f1_score = np.mean(f1_scores)
    avg_accuracy = np.mean(accuracies)
    avg_confusion_matrix = np.mean(confusion_matrices, axis=0)
    
    # Compute average classification report
    avg_classification_report = {}
    for report in classification_reports:
        for label, metrics in report.items():
            if label not in avg_classification_report:
                avg_classification_report[label] = metrics if isinstance(metrics, dict) else {"support": metrics}
            else:
                for metric, value in (metrics.items() if isinstance(metrics, dict) else {"support": metrics}.items()):
                    if metric not in avg_classification_report[label]:
                        avg_classification_report[label][metric] = 0
                    avg_classification_report[label][metric] += value
    for label, metrics in avg_classification_report.items():
        for metric in metrics.keys():
            avg_classification_report[label][metric] /= len(classification_reports)
    
    print("\nAverage F1 Score:", avg_f1_score)
    print("Average Accuracy:", avg_accuracy)
    print("Average Confusion Matrix:\n", avg_confusion_matrix)
    print("Average Classification Report:\n", json.dumps(avg_classification_report, indent=2))
    
    # Plot feature importances
    if args.plot and args.classifier in ['rf', 'svm']:
        plot_feature_importances(trained_classifier, feature_names, args.output_file)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Select files, classifier, vectorizer, embeddings, and other parameters")
    parser.add_argument('--data', type=str, required=True, help='Path to the JSON data file')
    parser.add_argument('--classifier', type=str, required=True, help='Classifier type, choices: svm, nb, rf, dt')
    parser.add_argument('--vectorizer', type=str, choices=['tf-idf', 'count', 'union'], default='tf-idf', help='Vectorizer type, choices: tf-idf, count, or union')
    parser.add_argument('--use_text', action='store_true', help='Use pre-processed text')
    parser.add_argument('--use_pauses', action='store_true', help='Use pauses')
    parser.add_argument('--use_both', action='store_true', help='Use both combined text and pauses for classification')
    parser.add_argument('--use_cosine', type=str, choices=['word', 'sentence', 'both'], help='Use cosine similarity metrics from word, sentence, or both embeddings')
    parser.add_argument('--plot', action='store_true', help='Plot the 10 most important features for RF and SVM classifiers')
    parser.add_argument('--k_folds', type=int, default=5, help='Number of folds for cross-validation')

    # Support Vector Machine (SVM) hyperparameters
    parser.add_argument('--svm_C', type=float, default=1.0, help='C hyperparameter for SVM classifier')
    parser.add_argument('--svm_kernel', type=str, default='linear', choices=['linear', 'poly', 'rbf', 'sigmoid'], help='Kernel type for SVM classifier')
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
