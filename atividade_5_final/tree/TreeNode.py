import pandas as pd
import math



class TreeNode:
    """Noh de uma arvore de decisao"""

    def __init__(self):

        # se for folha, a classe predita
        self.class_prediction = None
        self.is_leaf = False

        # se nao for folha

        # classe majoritaria incidente no noh durante o treinamento
        self.majority_class = None
        # nome da caracteristica que o noh separa
        self.feature_name = None
        # se a caracteristica eh numerica ou nao
        self.is_numeric_feature = False
        # nos filhos. eh um mapa com chave=valor da caracteristica e o noh correspondente
        # se for numerico, 'lower' para o noh conrrespondente a um valor <= que o split_point e 'upper' para valor maior que split_point
        self.children_nodes: dict = {}
        # valor de divisao otimo para features numericas
        self.split_point = None

        return

    def evaluate_node(self, X: pd.DataFrame, y: pd.Series, parent_majority_class):
        """
        verifica se um noh deve ser folha ou nao
        :param X: dataset de treinamento sem o vetor de classes
        :param y: vetor de classes
        :param parent_majority_class: classe majoritaria do noh pai
        :return:
        """
        if  X.empty or y.empty:
            # se nao existir mais dados no conjunto, classifica como a classe majoritaria do pai
            self.class_prediction = parent_majority_class
            self.is_leaf = True
            return True

        y_classes = y.unique()
        if len(y_classes) == 1:
            # se todas as instancias sao da mesma classe, classifica como a classe majoritaria do pai
            self.class_prediction = y_classes[0]
            self.is_leaf = True
            return True

        self.is_leaf = False
        # define classe majoritaria
        self.majority_class = y.value_counts().idxmax()

        return


    def grow_branch(self, X: pd.DataFrame, y: pd.Series):
        """
        Continua com o crescimento da arvore (ou seja, cria os filhos)
        :param X: conjunto de treinamento sem o vetor de classes
        :param y: vetor de classes
        :return:
        """

        best_feature, best_split_dict, best_split_point = self.attribute_selection_method(X, y)

        self.feature_name = best_feature

        # guarda o valor de divisao otimo se o atributo for numerico
        if best_split_point is not None:
            self.is_numeric_feature = True
            self.split_point = best_split_point

        self.generate_children_nodes(X, y)


    def generate_children_nodes(self, X: pd.DataFrame, y: pd.Series):
        """
        cria os filhos
        :param X: dataset de treinamento sem o vetor de classes
        :param y: vetor de classe
        :return:
        """
        merged_frame = pd.concat((X, y), axis=1)

        if self.is_numeric_feature:
            # cria o dataset cujos valores da feature selecionada desse noh sao <= do que o valor do split point, sem a feature selecionada
            lower_split_child_node = TreeNode()
            lower_split = merged_frame[merged_frame[self.feature_name] <= self.split_point]
            lower_X = lower_split.drop(y.name, axis=1)
            lower_y = lower_split[y.name]

            # cria o noh filho para os nos com valor da feature selecionada <= do que o valor do spĺit point
            lower_split_child_node.evaluate_node(lower_X, lower_y, self.majority_class)
            # se o filho nao for folha, continua crescimento
            if not lower_split_child_node.is_leaf:
                lower_split_child_node.grow_branch(lower_X, lower_y)

            # cria o dataset cujos valores da feature selecionada desse noh sao > do que o valor do split point, sem a feature selecionada
            upper_split_child_node = TreeNode()
            upper_split = merged_frame[merged_frame[self.feature_name] > self.split_point]
            upper_X = upper_split.drop(y.name, axis=1)
            upper_y = upper_split[y.name]

            # cria o noh filho para os nos com valor da feature selecionada <= do que o valor do spĺit point
            upper_split_child_node.evaluate_node(upper_X, upper_y, self.majority_class)
            # se o noh filho nao for folha continua crescimento
            if not upper_split_child_node.is_leaf:
                upper_split_child_node.grow_branch(upper_X, upper_y)

            # salva os filhos
            self.children_nodes['lower'] = lower_split_child_node
            self.children_nodes['upper'] = upper_split_child_node


        else:
            # feature selecionada eh categorica
            split_values_list = X[self.feature_name].unique()

            for split_value in split_values_list:
                # cria o dataset cujos valores da feature selecionada desse noh sao um dos valores possiveis, sem a feature selecionada
                split_child_node = TreeNode()
                split_frame = merged_frame[merged_frame[self.feature_name] == split_value]
                X_split = split_frame.drop(y.name, axis=1)
                y_split = split_frame[y.name]

                # cria o noh filho para os nos com valor da feature selecionada
                split_child_node.evaluate_node(X_split, y_split, self.majority_class)
                # se o noh filho nao eh folha continua crescimento
                if not split_child_node.is_leaf:
                    split_child_node.grow_branch(X_split, y_split)

                # salva o filho
                self.children_nodes[split_value] = split_child_node

        return

    def attribute_selection_method(self, X:pd.DataFrame, y: pd.Series):
        """
        seleciona a melhor caracteristica (que melhor divide o dataset)
        :param X: dataset de treinamento sem o vetor de classes
        :param y:vetor de classes
        :return: - a melhor feature
          - o dicionario com:
            - se a feature eh categorica, os possiveis valores dessa feature com o dataset correspondente a essa divisao
            - se for numerico, os datasets divididos de acordo com o split point
          - o melhor ponto de visiao (nulo se melhor atributo eh categorico
        """
        features_names_list = list(X.columns)

        best_feature = None
        best_information_gain = -math.inf
        best_split_dict = None
        best_split_point = None

        # todas as caracteristicas disponiveis
        for feature_name in features_names_list:
            feature_data = X[feature_name]

            information_gain = -math.inf
            split_dict = None
            split_point = None

            if pd.api.types.is_numeric_dtype(feature_data):
                # se atributo eh numerico
                information_gain, split_dict, split_point = self.calculate_numeric_feature_information_gain(feature_data, y)
            else:
                # se atributo eh categorico
                information_gain, split_dict = self.calculate_categoric_feature_information_gain(feature_data, y)

            if information_gain > best_information_gain:
                # se ganho de informacao da feature avaliada melhor do que as outras features, salva os valores correspondentes
                best_feature = feature_name
                best_information_gain = information_gain
                best_split_dict = split_dict
                best_split_point = split_point

        return best_feature, best_split_dict, best_split_point

    def calculate_categoric_feature_information_gain(self, feature_data: pd.Series, y: pd.Series):
        """
        calcula o ganho de informacao para atributos categoricos
        :param feature_data: vetor da caracteristica
        :param y: vetor de classes
        :return: ganho de informacao da feature e o dicionario cujas chaves sao os possiveis valores da caracteristica e os
        datasets com a divisao correspondente
        """
        # calcula entropia total
        current_information = self.calculate_entropy(y)
        n_instances = len(feature_data)

        # encontra os valores unicos da caracteristica e os coloca como chave de um dicionario
        split_point_values = feature_data.unique()
        split_dict = {}

        merged_frame = pd.concat((feature_data, y), axis=1)
        entropy_sum = 0

        for split_value in split_point_values:
            # divide o vetor de caracteristica
            split_frame = merged_frame[merged_frame[feature_data.name] == split_value]
            split = split_frame[y.name]

            # calcula entropia proporcional ao numero de instancias para a divisao
            split_entropy = self.calculate_entropy(split)
            split_proportion = len(split) / n_instances

            entropy_sum += split_proportion * split_entropy

            # guarda a divisao no dicionario
            split_dict[split_value] = split

        # calcula o ganho de informacao
        information_gain = current_information - entropy_sum

        return information_gain, split_dict

    def calculate_numeric_feature_information_gain(self, feature_data: pd.Series, y: pd.Series):
        """
        calcula o ganho de informacao para atributos categoricos
        :param feature_data: vetor da caracteristica
        :param y: vetor de classes
        :return: ganho de informacao da feature,
        dicionario cujas chaves sao 'lower' ou 'upper', correspondente para os datasets
        cujo valor do atributo avaliado sao <= ou > do que o split point otimo
        split point otimo
        """

        # calcula entropia total
        current_information = self.calculate_entropy(y)
        n_instances = len(feature_data)

        # encontra todos os split points possiveis
        split_point_candidates = feature_data.unique()
        best_information_gain = -math.inf
        best_split_point = -math.inf
        best_split = None

        # para cada split point possivel
        for split_point in split_point_candidates:
            split_dict = self.get_numeric_y_splits(feature_data, y, split_point)

            # entropia para o dataset com valores <= ao split_point
            lower_split_entropy = self.calculate_entropy(split_dict['lower'])
            # entropia para o dataset com valores > ao split_point
            upper_split_entropy = self.calculate_entropy(split_dict['upper'])

            lower_split_proportion = len(split_dict['lower'])/n_instances
            upper_split_proportion = len(split_dict['upper'])/n_instances

            # calcula o ganho de informacao
            information_gain = current_information - (
                (lower_split_proportion * lower_split_entropy) + (upper_split_proportion * upper_split_entropy)
            )

            # se ganho de informacao foi maior para esse split point (melhor split point) guarda
            if information_gain > best_information_gain:
                best_information_gain = information_gain
                best_split_point = split_point
                best_split = split_dict

        return best_information_gain, best_split, best_split_point

    def get_numeric_y_splits(self, feature_data: pd.Series, y: pd.Series, split_point):
        """
        divide o dataset de acordo com o split point (se o valor da feature da instancia eh <= ou > que o split point
        :param feature_data:
        :param y:
        :param split_point:
        :return: dicionario com os datasets divididos
        """

        merged_frame = pd.concat((feature_data, y), axis=1)

        lower_split = merged_frame[merged_frame[feature_data.name] <= split_point]
        upper_split = merged_frame[merged_frame[feature_data.name] > split_point]

        split_dict = {
            'lower': lower_split[y.name],
            'upper': upper_split[y.name]
        }

        return split_dict

    def calculate_entropy(self, vector: pd.Series):
        """
        calcula entropia
        :param vector: vetor de classes correspondente
        :return:
        """
        entropy = 0

        n_instances = len(vector)
        counter = vector.value_counts()
        num_values = len(counter)

        if len(counter) <= 1:
            return 0

        for i in counter:
            proportion = i / n_instances
            entropy -= proportion * (math.log(proportion, num_values))

        return entropy

    def predict(self, x):
        """
        retorna a instancia da classe a ser predita
        :param x: instancia desconhecida
        :return: classe predita
        """

        # se noh eh folha traz a classe correspondente
        if self.is_leaf:
            return self.class_prediction

        x_value = x[self.feature_name]

        # se atributo eh numerico
        if self.is_numeric_feature:
            # verifica qual filho ir de acordo com o split point
            if x_value <= self.split_point:
                return self.children_nodes['lower'].predict(x)
            else:
                return self.children_nodes['upper'].predict(x)

        else:
            # atributo eh categorico. Se nao existir ramo correspondente para o valor, retorna classe majoritaria
            if x_value in self.children_nodes:
                return self.children_nodes[x_value]
            else:
                return self.majority_class


