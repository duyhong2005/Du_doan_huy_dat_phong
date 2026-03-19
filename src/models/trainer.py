from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import SMOTE
import pandas as pd

class Trainer:
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def split(self):
        return train_test_split(self.X, self.y, test_size=0.2, random_state=42)

    def train_supervised(self, X_train, y_train):
        sm = SMOTE(random_state=42)
        X_train, y_train = sm.fit_resample(X_train, y_train)

        model = LogisticRegression(max_iter=1000)
        model.fit(X_train, y_train)

        return model, X_train, y_train

    def train_semi(self, model, X_train, y_train, X_test):
        X_small = X_train.sample(frac=0.3, random_state=42)
        y_small = y_train.loc[X_small.index]

        model.fit(X_small, y_small)

        y_prob = model.predict_proba(X_test)[:,1]
        idx = (y_prob > 0.9) | (y_prob < 0.1)

        X_pseudo = X_test[idx]
        y_pseudo = (y_prob[idx] > 0.5).astype(int)

        y_pseudo = pd.Series(y_pseudo, index=X_pseudo.index)

        X_new = pd.concat([X_small, X_pseudo])
        y_new = pd.concat([y_small, y_pseudo])

        model.fit(X_new, y_new)

        return model