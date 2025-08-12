from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import joblib


class BaseClassificationModel:
    def __init__(self, model):
        self.model = model

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X):
        return self.model.predict(X)

    def evaluate(self, X, y, y_pred=None, show_confusion=False):
        if y_pred is None:
            y_pred = self.predict(X)

        precision = precision_score(y, y_pred, average='macro')
        recall = recall_score(y, y_pred, average='macro')
        f1 = f1_score(y, y_pred, average='macro')
        acc = accuracy_score(y, y_pred)

        if show_confusion:
            cm = confusion_matrix(y, y_pred)
            disp = ConfusionMatrixDisplay(confusion_matrix=cm)
            disp.plot(cmap=plt.cm.Blues, xticks_rotation=45)
            plt.show()

        return {
            'accuracy': acc,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }

    def save(self, path):
        joblib.dump(self.model, path)

    def load(self, path):
        self.model = joblib.load(path)


# Example model wrappers
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression  # TODO delete this
from sklearn.ensemble import RandomForestClassifier


class KNNModel(BaseClassificationModel):
    def __init__(self, n_neighbors=5):
        super().__init__(KNeighborsClassifier(n_neighbors=n_neighbors))


class SVMModel(BaseClassificationModel):
    def __init__(self, kernel='linear', C=1.0):
        super().__init__(SVC(kernel=kernel, C=C))

# TODO delete this
"""
class LogisticModel(BaseClassificationModel): 
    def __init__(self, C=1.0, max_iter=1000):
        super().__init__(LogisticRegression(C=C, max_iter=max_iter))
"""

class RandomForestModel(BaseClassificationModel):
    def __init__(self, n_estimators=100):
        super().__init__(RandomForestClassifier(n_estimators=n_estimators))
