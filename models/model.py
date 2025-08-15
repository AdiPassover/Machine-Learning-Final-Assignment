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

    def train_and_evaluate(self, X_train, y_train, X_test, y_test, show_confusion=False):
        self.train(X_train, y_train)

        results = self.evaluate(X_test, y_test, show_confusion=show_confusion)
        for metric, value in results.items():
            print(f"{metric}: {value:.4f}")
