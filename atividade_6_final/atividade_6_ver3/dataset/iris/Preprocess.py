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
    filename = 'iris.csv'
    df = pd.read_csv(filename)
    class_column = 'class'

    training, test = create_validation_datasets(df, class_column)

    training.to_csv('iris_training.csv', index=False)
    test.to_csv('iris_test.csv', index=False)
