from SVM import SVM
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score
import pandas as pd
import numpy as np
from pandas_ml import ConfusionMatrix




class OneVsOne:

    def __init__(self):
        self.setosa_vs_versicolor = SVM(kernel_name='polynomial', C=10, kernel_param=7)
        self.setosa_vs_virginica = SVM(kernel_name='polynomial', C=10, kernel_param=7)
        self.versicolo_vs_virginica = SVM(kernel_name='polynomial', C=10, kernel_param=7)

        return

    def revert_multilabel(self, class_vector, positive_label, negative_label):
        iris_class_vector = []

        for i in range(len(class_vector)):
            if np.all(class_vector[i] == 1):
                iris_class_vector.append(positive_label)
            else:
                iris_class_vector.append(negative_label)
        return iris_class_vector


    def train(self):
        iris_training = pd.read_csv('dataset/iris/iris_training.csv')
        class_label = 'class'

        X_training = iris_training.drop('class', axis=1)
        y_training = iris_training['class']

        iris_test = pd.read_csv('dataset/iris/iris_test.csv')
        X_test = iris_test.drop('class', axis=1)
        y_test = iris_test['class']

    def train_model(self, X: pd.DataFrame, y: pd.Series, model_reference: SVM, positive_label, negative_label, dropped_label):
        X = X[X['class'] != dropped_label]

        model_reference.fit(X, y)



        for i in range(len(class_vector)):
            if np.all(class_vector[i] == 1):
                iris_class_vector.append(p)
            else:
                iris_class_vector.append(negative_label)
        return iris_class_vector

