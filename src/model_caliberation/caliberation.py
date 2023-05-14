import pickle
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import BaggingClassifier
from sklearn.metrics import brier_score_loss
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.tree import DecisionTreeClassifier
from imblearn.over_sampling import SMOTE

df = pd.read_csv("../../data/raw/train_processed.csv")
df = df.sample(n=10000)
x = df.iloc[:, :-1]
y = df.iloc[:, -1]

smote = SMOTE(random_state=42)
x, y = smote.fit_resample(x, y)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)

print("#"*20 + "Uncaliberated model" + "#"*20)

clf = BaggingClassifier(
    estimator=DecisionTreeClassifier(criterion='gini',
      max_depth=50,
      min_samples_split=3
      ),
    n_estimators=300,
    bootstrap_features=True,
    oob_score=True,
    max_features=1.0
    )



clf.fit(x_train , y_train)
y_pred = clf.predict(x_test)
acc = clf.score(x_test, y_test)
print(acc)

print(brier_score_loss(y_test, y_pred))


plt.rcParams.update({'font.size': 10})
frac_of_positives, pred_prob = calibration_curve(y_test, y_pred, n_bins=10)
X = np.linspace(0, 1, 10)
Y = X
sns.lineplot(x=X, y=Y, linestyle='dotted')
sns.lineplot(x=pred_prob, y=frac_of_positives)
plt.grid(linestyle='-', linewidth=0.2)
plt.title("Probability vs Fraction Postives")
xlabel = plt.xlabel("Probability of positive")
ylabel = plt.ylabel("Fraction of positives")
ticks = [0, 0.2, 0.4, 0.6, 0.8, 1]
xticks = plt.xticks(ticks)
yticks = plt.yticks(ticks)
plt.show()


#balance class weights

print("#"*20 + "Caliberated model" + "#"*20)

smote = SMOTE(random_state=42)
x, y = smote.fit_resample(x, y)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)

clf = BaggingClassifier(
    estimator=DecisionTreeClassifier(criterion='gini',
      max_depth=50,
      min_samples_split=3
      ),
    n_estimators=300,
    bootstrap_features=True,
    oob_score=True,
    max_features=1.0
    )

caliberation_clf = CalibratedClassifierCV(estimator=clf, cv=3, method='isotonic')

caliberation_clf.fit(x_train , y_train)
y_pred = caliberation_clf.predict_proba(x_test)[:, 1]
acc = caliberation_clf.score(x_test, y_test)
print(acc)

print(brier_score_loss(y_test, y_pred))


plt.rcParams.update({'font.size': 10})
frac_of_positives, pred_prob = calibration_curve(y_test, y_pred, n_bins=10)
X = np.linspace(0, 1, 10)
Y = X
sns.lineplot(x=X, y=Y, linestyle='dotted')
sns.lineplot(x=pred_prob, y=frac_of_positives)
plt.grid(linestyle='-', linewidth=0.2)
plt.title("Probability vs Fraction Postives")
xlabel = plt.xlabel("Probability of positive")
ylabel = plt.ylabel("Fraction of positives")
ticks = [0, 0.2, 0.4, 0.6, 0.8, 1]
xticks = plt.xticks(ticks)
yticks = plt.yticks(ticks)
plt.show()