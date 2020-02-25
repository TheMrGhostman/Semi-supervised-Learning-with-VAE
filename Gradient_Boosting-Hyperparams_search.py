import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, f1_score, make_scorer

raise ValueError("Je ještě třeba doplnit názvy dat")
X = np.load(...)
y = np.load(...)

clf = GradientBoostingClassifier()
parameters = {'learning_rate':[1e-1, 1e-2],
            'n_estimators':[100, 400, 1000, 1500],
            'subsample': [0.7],
            'n_iter_no_change':[20]
            }

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
df.to_csv("Graident_Tree_Boosting_GS_results.csv")
