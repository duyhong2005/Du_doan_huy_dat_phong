import pandas as pd

class FeatureBuilder:
    def __init__(self, df):
        self.df = df

    def build(self):
        X = pd.get_dummies(self.df.drop('is_canceled', axis=1))
        y = self.df['is_canceled']
        X = X.fillna(0)
        return X, y