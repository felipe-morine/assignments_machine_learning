import pandas as pd
from tree.TreeNode import TreeNode


class Tree:
    """
    Cria um modelo de arvode de decisao. Ao contrario dos outros modelos, ele usa predominantemente a biblioteca Pandas
    """

    def __init__(self):
        # noh raiz
        self.root: TreeNode = TreeNode()

    def fit(self, X, y):
        """
        Treina o modelo
        :param X: dataset de treinamento sem o vetor de classes
        :param y: vetor de classes
        :return:
        """

        # cresce a arvore
        self.root.evaluate_node(X, y, None)
        if not self.root.is_leaf:
            self.root.grow_branch(X, y)

        return

    def predict(self, X: pd.DataFrame):
        """
        Faz a predicao de instancias desconhecidas
        :param X: dataset de teste
        :return: vetor de clases preditas
        """
        y_predict = []
        num_instances, num_features = X.shape

        for i in range(num_instances):
            x = X.iloc[i]
            y_predict.append(self.root.predict(x))

        return y_predict
