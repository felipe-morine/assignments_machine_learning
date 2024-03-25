from tree.Tree import Tree
from sklearn.metrics import confusion_matrix, accuracy_score
import pandas as pd


if __name__=="__main__":
    # carrega o conjunto de treinamento
    iris_training = pd.read_csv('../dataset/hepatitis/hepatitis_training.csv')
    class_label = 'class'

    # separa o vetor de classe
    X_train = iris_training.drop(class_label, axis=1)
    y_train = iris_training[class_label]

    # faz o mesmo para o conjunto de teste
    iris_test = pd.read_csv('../dataset/hepatitis/hepatitis_test.csv')
    X_test = iris_test.drop(class_label, axis=1)
    y_test = iris_test[class_label]

    # cria o modelo
    tree = Tree()
    tree.fit(X_train, y_train)

    y_predicted = tree.predict(X_test)
    acc = accuracy_score(y_test.values, y_predicted)
    cm = confusion_matrix(y_test.values, y_predicted)

    print(acc)
    print(cm)