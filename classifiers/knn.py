from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import matplotlib.pyplot as plt

class Knn:
    def __init__(self, X_train, X_test, y_train, y_test):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.k_range = None

    def set_krange(self, krange):
        self.k_range = krange
    
    def best_k(self):
        error = []
        for i in self.k_range:
            knn = KNeighborsClassifier(n_neighbors=i)
            knn.fit(self.X_train, self.y_train)
            y_pred = knn.predict(self.X_test)
            error.append(np.mean(y_pred != self.y_test))
        return error

    def train(self, k):
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(self.X_train, self.y_train)
        return knn

    def plot_best_k(self, k_range, error):
        plt.figure(figsize=(12, 6))
        plt.plot(k_range, error, color='red', linestyle='dashed', marker='o',
                markerfacecolor='blue', markersize=7)
        plt.title('Error Rate K Value')
        plt.xlabel('K Value')
        plt.ylabel('Mean Error')
        plt.show()

    # def plot_ROC_curve():
