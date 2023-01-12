import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import ExtraTreesClassifier
from imblearn.over_sampling import SMOTE

df = pd.read_csv("./data/raw/train_processed.csv")
df = df.sample(n=50000)
x = df.iloc[:, :-1]
y = df.iloc[:, -1]

smote = SMOTE(random_state=42)
x, y = smote.fit_resample(x, y)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)


params = [
    {
     'n_estimators': [100, 250, 500],
     'criterion': ['gini', 'entropy', 'log_loss'],
     'max_features': ['sqrt', 'log2'],
     }
]

clf = ExtraTreesClassifier(n_estimators = 100, criterion = 'gini', random_state = 0)
clf.fit(x_train , y_train)
print(clf.best_params_)
y_pred = clf.predict(x_test)
acc = clf.score(x_test, y_test)
print(acc)

metrics = """
Accuracy: {:10.4f}
![Confusion Matrix](plot.png)
""".format(acc)
with open("metrics.txt", "w") as outfile:
    outfile.write(metrics)
conf_matrix = confusion_matrix(y_test, y_pred)
disp = sns.heatmap(conf_matrix/np.sum(conf_matrix), annot=True, cmap='Blues', fmt='.2%')
plt.savefig("plot.png")
