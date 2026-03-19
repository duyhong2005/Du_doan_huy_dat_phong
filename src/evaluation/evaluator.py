from sklearn.metrics import f1_score, average_precision_score
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import numpy as np

class Evaluator:
    def evaluate(self, model, X_test, y_test, name="MODEL"):
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:,1]

        print(f"\n=== {name} ===")
        print("F1:", f1_score(y_test, y_pred))
        print("PR-AUC:", average_precision_score(y_test, y_prob))
        print("MAE:", mean_absolute_error(y_test, y_pred))
        print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))

        cm = confusion_matrix(y_test, y_pred)
        disp = ConfusionMatrixDisplay(cm)

        disp.plot()
        plt.title(f"Confusion Matrix - {name}")
        plt.show()