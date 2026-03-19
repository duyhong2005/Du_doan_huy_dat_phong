import pandas as pd

class DataCleaner:
    def __init__(self, df):
        self.df = df

    def sample(self):
        self.df = self.df.sample(frac=0.3, random_state=42)
        return self

    def discretize(self):
        self.df['lead_time_group'] = pd.cut(
            self.df['lead_time'],
            bins=[0,30,90,180,365],
            labels=['short','medium','long','very_long']
        ).astype('category')
        return self

    def group_country(self):
        top_country = self.df['country'].value_counts().head(5).index
        self.df['country_group'] = self.df['country'].apply(
            lambda x: x if x in top_country else 'Other'
        )
        return self

    def remove_leakage(self):
        self.df = self.df.drop(
            columns=['reservation_status','reservation_status_date'],
            errors='ignore'
        )
        return self

    def handle_missing(self):
        for col in self.df.columns:
            if self.df[col].dtype in ['int64', 'float64']:
                self.df[col] = self.df[col].fillna(0)
            elif str(self.df[col].dtype) == 'category':
                self.df[col] = self.df[col].cat.add_categories('Unknown')
                self.df[col] = self.df[col].fillna('Unknown')
            else:
                self.df[col] = self.df[col].fillna('Unknown')
        return self

    def get_data(self):
        return self.df