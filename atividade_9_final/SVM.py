import numpy as np
import cvxopt
import cvxopt.solvers
import pandas as pd
import KernelFunctions


class SVM(object):

    """
    Classe para o modelo SVM
    """

    def __init__(self, kernel_name='linear', C=None, kernel_param = None):
        """
        :param kernel_name: nome do kernel. possiveis: linear, polynomial, gaussian
        :param C: parametro de custo
        :param kernel_param: parametro do kernel
        """
        self.kernel = KernelFunctions.get_kernel_function(kernel_name)
        self.kernel_param = kernel_param
        self.C = C

        # garantia para transformar C em float
        if self.C is not None: self.C = float(self.C)

        # vetores suporte
        self.support_vectors: np.array = None
        # lagrangianos suporte
        self.support_alphas: np.array = None
        # classes dos vetores suporte
        self.support_y: list = None
        # bias (intercept)
        self.bias = 0

        return


    def fit(self, X, y):
        """
        Treina o modelo
        :param X: conjunto de treinamento sem o vetor de classes
        :param y: vetor de classes
        :return:
        """
        alpha_vector = self.dual_solver(X, y)
        self.get_support_values(X, y, alpha_vector)

    def dual_solver(self, X, y):
        """
        Resolve o dual. CVXOPT resolve o problema da forma

        min (1/2) x'Px + q'x
            subj Gx <= h
                 Ax = b
        apostofres sao a transposta

        :param X: conjunto de treinamento sem o vetor de classes
        :param y: vetor de classes
        :return:
        """
        n_instances, n_features = X.shape
        P = self.__get_P_matrix(X, y)

        G = self.__get_G_matrix(n_instances)
        h = self.__get_h_vector(n_instances)

        # transforma os numpy.array em matrizes cvxopt. Necessario
        P = cvxopt.matrix(P)
        q = cvxopt.matrix(np.ones(n_instances) * -1)
        G = cvxopt.matrix(G)
        h = cvxopt.matrix(h)
        A = cvxopt.matrix(y, (1, n_instances)) # eh preciso manter a dimensao
        b = cvxopt.matrix(0.0)

        # evita print da execucao do cvxopt.
        cvxopt.solvers.options['show_progress'] = False

        # chama cvxopt para resolver o dual
        solution = cvxopt.solvers.qp(P, q, G, h, A, b)
        # lagrangianos. retorna eles em um n.array de dimensao 1, nao em uma matriz
        solution = np.array(solution['x']).reshape((n_instances,))

        return solution

    def __get_P_matrix(self, X, y):
        """
        calcula a matriz P
        :param X: dataset de treinamento sem o vetor de classes
        :param y: vetor de classes
        :return:
        """
        num_instances, num_features = X.shape
        X_copy = np.copy(X)
        P = np.zeros((num_instances, num_instances))

        # nao parece ser possivel fazer np.outer(X, X), por causa da funcao kernel. para y OK, mas foi no loop direto
        for i in range(num_instances):
            for j in range(num_instances):
                P[i, j] = self.kernel(X_copy[i], X_copy[j], self.kernel_param) * (y[i] * y[j])
        return P

    def __get_G_matrix(self, n_instances):
        """
        calcula matriz G
        :param n_instances:
        :return:
        """
        G = np.diag(np.ones(n_instances) * -1)
        # se soft-margin
        if self.C:
            C_constraints = np.identity(n_instances)
            G = np.vstack((G, C_constraints))
        return G

    def __get_h_vector(self, n_instances):
        """
        calcula vetor h
        :param n_instances:
        :return:
        """
        h = np.zeros(n_instances)
        # se soft-margin
        if self.C:
            C_constraints = np.ones(n_instances) * self.C
            h = np.hstack((h, C_constraints))
        return h

    def get_support_values(self, X, y, alpha_vector, epsilon=1e-5):
        """
        guarda os vetores suporte, bem como seus alfas e classes correspondentes
        :param X: dataset de treinamento
        :param y: vetor de classes
        :param alpha_vector: resolucao do dual
        :param epsilon: vetores nao-suporte nao sao iguais a 0. Se menores que epsilon, considerados vetores nao-suporte.
        EM ALGUNS POUCOS CASOS, ACABA RETIRANDO TODOS OS VETORES; UTILIZAR epsilon APROPRIADO
        :return:
        """

        # indices usados durante o desenvolvimento do trabalho. Nao sao utilizados
        support_vector_index_list = []
        support_alphas_list = []
        support_vectors_list = []
        support_y_list = []

        for i in range(0, len(alpha_vector)):
            if alpha_vector[i] > epsilon:
                support_alphas_list.append(alpha_vector[i])
                support_vectors_list.append(X[i])
                support_y_list.append(y[i])
                support_vector_index_list.append(i)

        self.support_vectors = np.array(support_vectors_list)
        self.support_alphas = np.array(support_alphas_list)
        self.support_y = np.array(support_y_list)
        self.calculate_bias()

        return

    def calculate_bias(self):
        """
        calcula o bias (intercept do hiperplano)
        :return:
        """

        bias = 0
        num_sv = self.support_vectors.shape[0]

        for m in range(num_sv):
            n_sum = 0
            for n in range(num_sv):
                n_sum += (
                        self.support_alphas[n] * self.support_y[n] * self.kernel(self.support_vectors[n], self.support_vectors[m], self.kernel_param)
                )
            bias += (self.support_y[m] - n_sum)
        bias /= num_sv

        self.bias = bias

        return


    def calculate_output(self, X):
        """
        prediz as classes para instancias desconhecidas
        :param X: dataset de teste
        :return:
        """

        n_instances, n_features = X.shape
        y_predict = np.zeros(n_instances)
        for i in range(0, n_instances):
            instance_projection = 0
            # calcula projecao
            for alpha, support_y, support_vector in zip(self.support_alphas, self.support_y, self.support_vectors):
                instance_projection += alpha * support_y * self.kernel(X[i], support_vector, self.kernel_param)
            instance_projection += self.bias
            y_predict[i] = instance_projection
        return y_predict

    def predict(self, X):
        """
        se maior que 0, retorna 1, caso contrario, retorna -1
        :param X:
        :return:
        """
        return np.sign(self.calculate_output(X))

class fullprint:
    """utilizado para printar np.array sem cortar. nao utilizado"""

    def __init__(self, **kwargs):
        if 'threshold' not in kwargs:
            kwargs['threshold'] = np.nan
        self.opt = kwargs

    def __enter__(self):
        self._opt = np.get_printoptions()
        np.set_printoptions(**self.opt)

    def __exit__(self, type, value, traceback):
        np.set_printoptions(**self._opt)
