#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, f1_score, make_scorer
import joblib


# In[3]:


X_train = np.load("data/encoded_data_train_DeepDenseVAE_mark_V_[160-256-128-15]_VDO_GNLL_400ep_lr-1e-4.npy")
X_test = np.load("data/encoded_data_test_DeepDenseVAE_mark_V_[160-256-128-15]_VDO_GNLL_400ep_lr-1e-4.npy")
y_train = np.load("data/labels_train.npy")
y_test = np.load("data/labels_test.npy")


# In[4]:


clf = GradientBoostingClassifier()
parameters = {'learning_rate':[1e-1, 1e-2],
            'n_estimators':[100, 500, 1000, 1500],
            'subsample': [0.7],
            'n_iter_no_change':[20]
            }


# In[5]:


metric_list = {"accuracy": make_scorer(accuracy_score), "F1": make_scorer(f1_score,average="macro")}


# In[ ]:


GS = GridSearchCV(
    estimator=clf,
    param_grid=parameters,
    scoring=metric_list,
    cv=5,
    refit="F1",
    n_jobs=4,
    #return_train_score=True,
    verbose=True
)

GS.fit(X_train,y_train)


# In[ ]:





# In[ ]:


df = pd.DataFrame(GS.cv_results_)
df.to_csv("Graident_Tree_Boosting_GS_results.csv")
joblib.dump(GS, "GTB_GS.joblib")


# In[ ]:




