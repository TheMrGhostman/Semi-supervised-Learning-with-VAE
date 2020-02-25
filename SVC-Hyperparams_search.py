import numpy as np
import pandas as pd
import sklearn.svm as svm
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, f1_score, make_scorer

raise ValueError("Je ještě třeba doplnit názvy dat")
X = np.load(...)
y = np.load(...)

clf = svm.SVC()
parameters = [{'kernel':['linear'], 'C': [1, 10, 50, 100]},
            {'kernel':['rbf'], 'C': [1, 10, 50, 100],  'gamma': [1e-1, 1e-2, 1e-3,'scale']}]

metric_list = [make_scorer(accuracy_score),make_scorer(f1_score,average="macro")]

GS = GridSearchCV(
    estimator=clf,
    param_grid=parameters,
    scoring=metric_list,
    cv=5,
    return_train_score=True,
    verbose=True
)

GS.fit(X,y)

df = pd.DataFrame(GS.cv_results_)
df.to_csv("SVM_GS_results.csv")
