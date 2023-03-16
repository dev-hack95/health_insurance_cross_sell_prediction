import pickle
import json
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
from imblearn.over_sampling import SMOTE

df = pd.read_csv("./data/raw/train_processed.csv")
df = df.sample(n=10000)
x = df.iloc[:, :-1]
y = df.iloc[:, -1]

smote = SMOTE(random_state=42)
x, y = smote.fit_resample(x, y)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

clf = BaggingClassifier(estimator=DecisionTreeClassifier(), n_estimators=100, oob_score=True)


clf.fit(x_train , y_train)
y_pred = clf.predict(x_test)
accuracy = clf.score(x_test, y_test)
print(accuracy)

acc = accuracy * 100

tn, fp, fn, tp = confusion_matrix(y, y_pred).ravel()
specificity = tn / (tn + fp)
sensitivity = tp / (tp + fn)

with open('metrics.json') as outfile:
    json.dump({ "accuracy": acc,  "specificity": specificity, "sensitivity": sensitivity })

pickle.dump(clf, open('./artifacts/model.pkl', 'wb'))
