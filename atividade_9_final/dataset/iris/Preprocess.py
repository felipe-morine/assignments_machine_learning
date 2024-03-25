import pandas as pd


def create_validation_datasets(df: pd.DataFrame, class_column):
    """
    cria o conjunto de teste
    :param df:
    :param class_column:
    :return:
    """
    class_values_list = list(df[class_column].unique())
    training_dataset = pd.DataFrame(data=None, columns=df.columns)
    test_dataset = pd.DataFrame(data=None, columns=df.columns)

    for i in range(0, 3):
        for j in range(0,  34):
            index = (i*50) + j
            training_dataset = training_dataset.append(df.iloc[index])
        for j in range(34, 50):
            index = (i * 50) + j
            test_dataset = test_dataset.append(df.iloc[index])

    return training_dataset, test_dataset



if __name__=="__main__":
    # carrega o dataset
    df = pd.read_csv('iris_training.csv')

    # cria cada dataset para o one vs one, retirando as instancias cuja classe nao eh de interesse
    X_setosa_versicolor = df[df['class'] != 'Iris-virginica']
    X_setosa_virginica = df[df['class'] != 'Iris-versicolor']
    X_versicolor_virginica = df[df['class'] != 'Iris-setosa']

    # binariza, criando as classes "positivas e negativas"
    X_setosa_versicolor['class'] = X_setosa_versicolor['class'].replace('Iris-setosa', 1.0)
    X_setosa_versicolor['class'] = X_setosa_versicolor['class'].replace('Iris-versicolor', -1.0)

    X_setosa_virginica['class'] = X_setosa_virginica['class'].replace('Iris-setosa', 1.0)
    X_setosa_virginica['class'] = X_setosa_virginica['class'].replace('Iris-virginica', -1.0)

    X_versicolor_virginica['class'] = X_versicolor_virginica['class'].replace('Iris-versicolor', 1.0)
    X_versicolor_virginica['class'] = X_versicolor_virginica['class'].replace('Iris-virginica', -1.0)

    # grava os datasets
    X_setosa_versicolor.to_csv('iris_setosa_versicolor.csv', index=False)
    X_setosa_virginica.to_csv('iris_setosa_virginica.csv', index=False)
    X_versicolor_virginica.to_csv('iris_versicolor_virginica.csv', index=False)