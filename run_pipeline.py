import os
import pandas as pd

from src.data.cleaner import DataCleaner
from src.features.builder import FeatureBuilder
from src.models.trainer import Trainer
from src.evaluation.evaluator import Evaluator
from src.visualization.plots import Visualizer

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(BASE_DIR, "data/raw/hotel_bookings.csv")

df = pd.read_csv(data_path)

# CLEAN
df = (DataCleaner(df)
      .sample()
      .discretize()
      .group_country()
      .remove_leakage()
      .handle_missing()
      .get_data())

# EDA
viz = Visualizer()
viz.plot_eda(df)

# FEATURE
X, y = FeatureBuilder(df).build()

# TRAIN
trainer = Trainer(X, y)
X_train, X_test, y_train, y_test = trainer.split()

model, X_train, y_train = trainer.train_supervised(X_train, y_train)

# EVALUATE
eval = Evaluator()
eval.evaluate(model, X_test, y_test, "SUPERVISED")

# SEMI
model = trainer.train_semi(model, X_train, y_train, X_test)
eval.evaluate(model, X_test, y_test, "SEMI-SUPERVISED")

# TIME SERIES
viz.plot_time_series(df)