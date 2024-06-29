# LLMs baseline

import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report, precision_recall_fscore_support
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, BertTokenizer, BertForSequenceClassification
from sentence_transformers import SentenceTransformer
from gensim.models import Word2Vec
import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from scipy.spatial.distance import cosine

def parse_arguments():
    parser = argparse.ArgumentParser(description='Train and evaluate models on the data')
    parser.add_argument('--file', type=str, required=True, help='Path to the JSON file')
    parser.add_argument('--embeddings', choices=['none', 'word', 'sentence', 'both'], required=True, help='Type of embeddings to use')
    parser.add_argument('--text_type', choices=['combined_text', 'pauses', 'both'], required=True, help='Type of text to use')
    parser.add_argument('--model', choices=['distilbert', 'bert'], required=True, help='Model to use for training and evaluation')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size for training and evaluation')
    parser.add_argument('--epochs', type=int, default=5, help='Number of epochs for training')
    parser.add_argument('--lr', type=float, default=2e-5, help='Learning rate for training')
    parser.add_argument('--print_features', action='store_true', help='Print cosine similarity features for each participant')
    return parser.parse_args()

def load_dataset(file_path, text_type):
    with open(file_path, 'r') as f:
        data = json.load(f)
    texts, labels = [], []
    combined_texts = []
    for item in data:
        combined_texts.append(item['combined_text'])
        if text_type == 'combined_text':
            texts.append(item['combined_text'])
        elif text_type == 'pauses':
            texts.append(str(item['pauses']))
        elif text_type == 'both':
            combined_text = item['combined_text'] + ' ' + str(item['pauses'])
            texts.append(combined_text)
        labels.append(0 if item['condition'] == 'dementia' else 1)
    return texts, labels, combined_texts

def compute_word_embeddings(texts):
    processed_texts = [str(text).split() for text in texts]
    model = Word2Vec(sentences=processed_texts, vector_size=200, window=2, min_count=2, workers=-1)
    return model

def compute_sentence_embeddings(texts):
    model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
    embeddings = model.encode(texts)
    return embeddings

def word_cosine_similarities(texts, word2vec_model, window_sizes=range(1, 16)):
    features = []
    for text in texts:
        words = str(text).split()
        text_features = []
        for window_size in window_sizes:
            cosines = []
            for i in range(len(words) - window_size + 1):
                window = words[i:i + window_size]
                if all(word in word2vec_model.wv for word in window):
                    vectors = [word2vec_model.wv[word] for word in window]
                    for j in range(len(vectors) - 1):
                        cosines.append(1 - cosine(vectors[j], vectors[j + 1]))
            if cosines:
                text_features.extend([np.mean(cosines), np.var(cosines)])
            else:
                text_features.extend([0, 0])
        features.append(text_features)
    return features

def sentence_cosine_similarities(sentences, sentence_model, window_sizes=range(1, 10)):
    features = []
    embeddings = sentence_model.encode(sentences)
    for i in range(len(sentences)):
        text_features = []
        for window_size in window_sizes:
            cosines = []
            for j in range(len(embeddings) - window_size + 1):
                window = embeddings[j:j + window_size]
                for k in range(len(window) - 1):
                    cosines.append(1 - cosine(window[k], window[k + 1]))
            if cosines:
                text_features.extend([np.mean(cosines), np.var(cosines)])
            else:
                text_features.extend([0, 0])
        features.append(text_features)
    return features

def create_dataset(encodings, labels, features=None):
    inputs = {key: torch.tensor(val) for key, val in encodings.items()}
    inputs['labels'] = torch.tensor(labels)

    if features is not None:
        features = torch.tensor(features, dtype=torch.float32)
        dataset = TensorDataset(inputs['input_ids'], inputs['attention_mask'], inputs['labels'], features)
    else:
        dataset = TensorDataset(inputs['input_ids'], inputs['attention_mask'], inputs['labels'])

    return dataset

def train_and_evaluate(model, train_loader, val_loader, epochs, lr):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)
    model.train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    for epoch in range(epochs):
        for batch in tqdm(train_loader):
            optimizer.zero_grad()
            inputs = {key: val.to(device) for key, val in zip(['input_ids', 'attention_mask', 'labels'], batch[:3])}
            outputs = model(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'], labels=inputs['labels'])
            loss = outputs.loss
            loss.backward()
            optimizer.step()

    model.eval()
    preds, true_labels = [], []

    for batch in val_loader:
        inputs = {key: val.to(device) for key, val in zip(['input_ids', 'attention_mask', 'labels'], batch[:3])}
        with torch.no_grad():
            outputs = model(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'])
        logits = outputs.logits
        preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
        true_labels.extend(inputs['labels'].cpu().numpy())

    acc = accuracy_score(true_labels, preds)
    f1 = f1_score(true_labels, preds, average='weighted')
    precision, recall, f1_score_macro, _ = precision_recall_fscore_support(true_labels, preds, average='macro')
    cm = confusion_matrix(true_labels, preds)
    cr = classification_report(true_labels, preds, output_dict=True)
    return acc, f1, precision, recall, cm, cr, model

def plot_top_features(trained_model, feature_names):
    importances = np.abs(trained_model.classifier.weight.cpu().detach().numpy()).sum(axis=0)
    indices = np.argsort(importances)[-10:]
    plt.figure(figsize=(10, 8))
    plt.title("Top Feature Importances")
    plt.barh(range(len(indices)), importances[indices], color="b", align="center")
    plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
    plt.xlabel("Relative importance")
    plt.show()

def aggregate_classification_reports(crs):
    labels = list(crs[0].keys())
    aggregated_report = {label: {} for label in labels}
    for label in labels:
        for metric in crs[0][label].keys():
            aggregated_report[label][metric] = np.mean([cr[label][metric] for cr in crs])
    return aggregated_report

def main():
    args = parse_arguments()

    texts, labels, combined_texts = load_dataset(args.file, args.text_type)
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    accuracies, f1_scores, precisions, recalls, cms, crs = [], [], [], [], [], []

    if args.embeddings in ['word', 'both']:
        word2vec_model = compute_word_embeddings(combined_texts)
        word_features = word_cosine_similarities(combined_texts, word2vec_model)
    else:
        word_features = None

    if args.embeddings in ['sentence', 'both']:
        sentence_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
        sentence_features = sentence_cosine_similarities(combined_texts, sentence_model)
    else:
        sentence_features = None

    if args.print_features:
        def print_features(features, feature_type, window_sizes):
            for idx, feature_set in enumerate(features):
                print(f'Participant {idx + 1} {feature_type} Cosine Similarities:')
                for i, window_size in enumerate(window_sizes):
                    print(f'  Window Size {window_size}: Mean = {feature_set[i * 2]}, Variance = {feature_set[i * 2 + 1]}')

        if word_features is not None:
            print_features(word_features, "Word", range(1, 16))
        if sentence_features is not None:
            print_features(sentence_features, "Sentence", range(1, 10))

    model_dict = {
        'distilbert': (DistilBertTokenizer, DistilBertForSequenceClassification, 'distilbert-base-uncased'),
        'bert': (BertTokenizer, BertForSequenceClassification, 'bert-base-uncased')
    }

    tokenizer_class, model_class, pretrained_model_name = model_dict[args.model]

    tokenizer = tokenizer_class.from_pretrained(pretrained_model_name)
    model = model_class.from_pretrained(pretrained_model_name, num_labels=2)

    for train_index, test_index in kf.split(texts):
        train_texts = [texts[i] for i in train_index]
        val_texts = [texts[i] for i in test_index]
        train_labels = [labels[i] for i in train_index]
        val_labels = [labels[i] for i in test_index]

        train_encodings = tokenizer(train_texts, truncation=True, padding=True)
        val_encodings = tokenizer(val_texts, truncation=True, padding=True)

        if word_features is not None:
            scaler = StandardScaler()
            train_word_features = scaler.fit_transform([word_features[i] for i in train_index])
            val_word_features = scaler.transform([word_features[i] for i in test_index])
        if sentence_features is not None:
            scaler = StandardScaler()
            train_sentence_features = scaler.fit_transform([sentence_features[i] for i in train_index])
            val_sentence_features = scaler.transform([sentence_features[i] for i in test_index])

        if args.embeddings == 'word':
            train_dataset = create_dataset(train_encodings, train_labels, train_word_features)
            val_dataset = create_dataset(val_encodings, val_labels, val_word_features)
        elif args.embeddings == 'sentence':
            train_dataset = create_dataset(train_encodings, train_labels, train_sentence_features)
            val_dataset = create_dataset(val_encodings, val_labels, val_sentence_features)
        elif args.embeddings == 'both':
            combined_train_features = np.hstack([train_word_features, train_sentence_features])
            combined_val_features = np.hstack([val_word_features, val_sentence_features])
            train_dataset = create_dataset(train_encodings, train_labels, combined_train_features)
            val_dataset = create_dataset(val_encodings, val_labels, combined_val_features)
        else:
            train_dataset = create_dataset(train_encodings, train_labels)
            val_dataset = create_dataset(val_encodings, val_labels)

        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

        acc, f1, precision, recall, cm, cr, trained_model = train_and_evaluate(model, train_loader, val_loader, args.epochs, args.lr)
        
        accuracies.append(acc)
        f1_scores.append(f1)
        precisions.append(precision)
        recalls.append(recall)
        cms.append(cm)
        crs.append(cr)

    mean_acc = np.mean(accuracies)
    mean_f1 = np.mean(f1_scores)
    mean_precision = np.mean(precisions)
    mean_recall = np.mean(recalls)
    mean_cm = np.mean(cms, axis=0)
    mean_cr = aggregate_classification_reports(crs)

    print(f'Mean Accuracy: {mean_acc}')
    print(f'Mean F1 Score: {mean_f1}')
    print(f'Mean Precision: {mean_precision}')
    print(f'Mean Recall: {mean_recall}')
    print(f'Mean Confusion Matrix:\n{mean_cm}')
    print(f'Mean Classification Report:\n{json.dumps(mean_cr, indent=2)}')

if __name__ == '__main__':
    main()
