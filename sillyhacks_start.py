import csv
import sys

import pandas as pd
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import numpy as np


def print_predictions(predictions, y_test):
    with open('predictions_file.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        test = list(y_test)
        test.insert(0, "Test")
        writer.writerow(test)
        # print(test)
        for classifier in predictions:
            row = list(predictions[classifier])
            row.insert(0, classifier)
            # print(row)
            writer.writerow(row)


def train(x, y):
    tfidf = TfidfVectorizer(max_df=0.8, use_idf=True)
    linear_clf = forest_clf = sgd_clf = bayes_clf = tree_clf = None  # for testing

    x = tfidf.fit_transform(x, y)  # removes stopwords based on frequency
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.15, random_state=0)

    #linear_clf = LinearSVC(class_weight='balanced')
    #forest_clf = RandomForestClassifier(class_weight='balanced', n_estimators=15, max_depth=40)
    sgd_clf = SGDClassifier(alpha=0.001, class_weight='balanced', loss='log')
    #bayes_clf = MultinomialNB()
    tree_clf = DecisionTreeClassifier(max_depth=12, class_weight='balanced')

    classifiers = {'LinearSVC': linear_clf, 'RandomForestClassifier': forest_clf, 'SGDClassifier': sgd_clf,
                   'MultinomialNB': bayes_clf,
                   'DecisionTreeClassifier': tree_clf}
    predictions = {}

    meme_label_codes = set(y.astype(int))
    meme_label_names = le.inverse_transform(list(meme_label_codes))

    for val in classifiers:
        if classifiers[val] is None:  # for testing
            continue
        print(str(val))
        classifiers[val].fit(x_train, y_train)
        y_pred = classifiers[val].predict(x_test)
        predictions[val] = [meme_label_names[int(pred[0])] for pred in y_pred]
    print_predictions(predictions, [meme_label_names[int(true[0])] for true in y_test])


def visualize(classifier, y_pred, y_test, names):
    print("\n"+str(classifier)) # name of classifier
    print("Accuracy: " + str(round(accuracy_score(y_test, y_pred), 5)))
    print(str(names))  # print labels
    cm = confusion_matrix(y_test, y_pred, names)
    # print(cm)  # full matrix
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    # print(cm.diagonal())  # Accuracy scores for each label
    print(classification_report(y_test, y_pred))


def do_visualize():
    with open('./predictions_file.csv', 'r', newline='') as f:
        reader = csv.reader(f, delimiter=',')
        y_test = next(reader) # first row is test data
        meme_names = sorted(set(y_test[1:]))
        for row in reader:
            if row[0] == "Test":
                continue
            visualize(row[0], row[1:], y_test[1:], meme_names)


def main():
    global le
    le = preprocessing.LabelEncoder()

    # get 81 image names from memes_reference_data
    labels = []
    df = pd.read_csv('memes_reference_data.tsv', sep="\t", header=0)
    labels = df['MemeLabel']
    print(labels)

    # use the samples to train the model
    sample_df = pd.read_csv('memes_data.tsv', sep="\t", header=0)
    sample_df['MemeLabel'] = le.fit_transform(sample_df.MemeLabel.values)  # encode labels
    print(len(sample_df.index))
    x = sample_df['CaptionText'].astype('U')
    y = sample_df['MemeLabel'].astype('U')
    train(x, y)
    do_visualize()


if __name__ == "__main__":
    print('running')
    main()
    sys.exit()