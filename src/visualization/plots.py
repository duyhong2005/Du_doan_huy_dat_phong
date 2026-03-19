import matplotlib.pyplot as plt
import seaborn as sns

class Visualizer:
    def plot_eda(self, df):
        plt.figure(figsize=(6,4))
        sns.countplot(x='arrival_date_month', hue='is_canceled', data=df)
        plt.xticks(rotation=45)
        plt.title("Cancellation by Month")
        plt.tight_layout()
        plt.show()

    def plot_time_series(self, df):
        month_map = {
            'January':1,'February':2,'March':3,'April':4,
            'May':5,'June':6,'July':7,'August':8,
            'September':9,'October':10,'November':11,'December':12
        }

        ts = df.groupby('arrival_date_month')['is_canceled'].mean()
        ts.index = ts.index.map(month_map)
        ts = ts.dropna().sort_index()

        plt.figure(figsize=(6,4))
        plt.plot(ts.index, ts.values, marker='o')
        plt.title("Cancellation Trend")
        plt.show()