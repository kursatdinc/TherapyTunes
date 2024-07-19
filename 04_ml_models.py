import pandas as pd
import numpy as np
from catboost import CatBoostClassifier
from sklearn.metrics import cohen_kappa_score
from sklearn.model_selection import train_test_split

pd.set_option("display.max_columns", None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)

mental_after_eda = pd.read_csv("./datasets/mental_after_eda.csv")

mental_after_eda.info()

for col in ["anxiety", "depression", "insomnia"]:
    mental_after_eda[col] = mental_after_eda[col].astype(object)



y = mental_after_eda["anxiety"]
X = mental_after_eda.drop(columns=["tempo", "anxiety", "depression", "insomnia"], axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

class QWKObjective(object):
    def calc_ders_range(self, approxes, targets, weights):
        labels = np.array(targets) + a
        preds = np.array(approxes) + a
        preds = preds.clip(1, 6)
        f = 1/2 * np.sum((preds - labels)**2)
        g = 1/2 * np.sum((preds - a)**2 + b)
        df = preds - labels
        dg = preds - a
        grad = (df / g - f * dg / g**2) * len(labels)
        hess = (1 / g - (2 * df * dg) / (g**2) + (2 * f * dg**2) / (g**3)) * len(labels)
        return list(zip(grad, hess))

class QWKMetric(object):
    def get_final_error(self, error, weight):
        return error

    def is_max_optimal(self):
        return True

    def evaluate(self, approxes, targets, weight):
        approxes = approxes[0]
        targets = np.array(targets) + a
        approxes = np.array(approxes) + a
        approxes = approxes.clip(1, 6).round()
        qwk = cohen_kappa_score(targets, approxes, weights="quadratic")
        return qwk, 0
    
model = CatBoostClassifier(iterations=100,
                           learning_rate=0.05,
                           depth=6,
                           custom_metric=[QWKMetric()],
                           loss_function=QWKObjective())

model.fit(X_train, y_train)