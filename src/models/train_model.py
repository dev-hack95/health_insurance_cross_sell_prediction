import pandas as pd
import yaml
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE

df = pd.read_csv("./data/raw/train_processed.csv")
df = df.sample(n=1000)
x = df.iloc[:, :-1]
y = df.iloc[:, -1]

smote = SMOTE(random_state=42)
x, y = smote.fit_resample(x, y)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.30, random_state=42)

params = [
    {'max_depth': [80, 90, 100],
    'min_samples_leaf': [3,4,5],
    'min_samples_split': [5, 10],
    'n_estimators': [800, 1000]}
]

clf = GridSearchCV(RandomForestClassifier(random_state=42, max_features='sqrt', bootstrap=True), param_grid=params, cv= 5, verbose=2, refit=True)
clf.fit(x_train , y_train)
y_pred = clf.predict(x_test)
acc = clf.score(x_test, y_test)
print(acc)

metrics = """
Accuracy: {:10.4f}
![Confusion Matrix](plot.png)
""".format(acc)
with open("metrics.txt", "w") as outfile:
    outfile.write(metrics)
conf_matric = confusion_matrix(y_test, y_pred)
disp = sns.heatmap(conf_matric, annot=True, cmap='Blues')
plt.savefig("plot.png")
