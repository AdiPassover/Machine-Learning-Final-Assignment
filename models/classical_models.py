from model import BaseClassificationModel

from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier


class KNNModel(BaseClassificationModel):
    def __init__(self, k=5):
        super().__init__(KNeighborsClassifier(n_neighbors=k))


class SVMModel(BaseClassificationModel):
    def __init__(self, kernel='linear', C=1.0, random_state=None):
        super().__init__(SVC(kernel=kernel, C=C, random_state=random_state))

class RandomForestModel(BaseClassificationModel):
    def __init__(self, n_estimators=100, max_depth=None, random_state=None):
        super().__init__(RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=random_state))
