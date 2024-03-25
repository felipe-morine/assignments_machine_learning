from mlp.MLP import MLP
import pandas as pd
import numpy as np
from mlp import plot_confusion_matrix


def get_dataset_training_filename(is_undersample, is_PCA):

    csv_extension = '.csv'
    dataset_base_name = 'dataset_training'

    dataset_full_suffix = '_full'
    dataset_undersample_suffix = '_undersample'
    dataset_PCA_suffix = '_PCA'

    if is_undersample:
        dataset_base_name+=dataset_undersample_suffix
    else:
        dataset_base_name+=dataset_full_suffix
    if is_PCA:
        dataset_base_name+=dataset_PCA_suffix

    dataset_base_name+=csv_extension
    return dataset_base_name

def get_dataset_test_filename(is_PCA):

    csv_extension = '.csv'
    dataset_base_name = 'dataset_test'
    dataset_PCA_suffix = '_PCA'
    if is_PCA:
        dataset_base_name+=dataset_PCA_suffix

    dataset_base_name+=csv_extension
    return dataset_base_name

def save_model(mlp: MLP, k, is_undersample, is_PCA):
    """
    Salva as matrizes de peso da camada escondida e da camada de saida
    :param mlp:
    :param k:
    :param is_undersample:
    :param is_PCA:
    :return:
    """

    path = '../mlp/saved_models/{}/'.format(k)
    full_path = 'full'
    undersample_path = 'under'
    PCA_suffix = '_PCA'
    if is_undersample:
        path+=undersample_path
    else:
        path+=full_path

    if is_PCA:
        path+=PCA_suffix
    path+='/'


    a_w_file = path+'a_w'
    b_w_file = path+'b_w'

    np.save(a_w_file, mlp.a_weight_matrix)
    np.save(b_w_file, mlp.b_weight_matrix)

    return

def load_model(k, is_undersample, is_PCA):
    """
    carrega as matrizes de peso da camada escondida e da camada de saida
    :param k:
    :param is_undersample:
    :param is_PCA:
    :return:
    """

    numpy_array_extension = '.npy'

    path = 'saved_models/{}/'.format(k)
    full_path = 'full'
    undersample_path = 'under'
    PCA_suffix = '_PCA'
    if is_undersample:
        path+=undersample_path
    else:
        path+=full_path

    if is_PCA:
        path+=PCA_suffix
    path+='/'


    a_w_file = path+'a_w'+numpy_array_extension
    b_w_file = path+'b_w'+numpy_array_extension

    mlp = MLP()

    mlp.a_weight_matrix = np.load(a_w_file)
    mlp.b_weight_matrix = np.load(b_w_file)

    return mlp

def save_results(acc, y_test, y_pred, labels, k, is_undersample, is_PCA):
    """
    salva a acuracia e a matriz de confusao
    :param acc:
    :param y_test:
    :param y_pred:
    :param labels:
    :param k:
    :param is_undersample:
    :param is_PCA:
    :return:
    """
    path = '../mlp/results/'
    full_path = 'full'
    filename_suffix = ''
    undersample_path = 'under'
    PCA_suffix = '_PCA'
    if is_undersample:
        filename_suffix+=undersample_path
        path+=undersample_path
    else:
        path+=full_path
        filename_suffix += full_path

    if is_PCA:
        path+=PCA_suffix
        filename_suffix+=PCA_suffix
    path+='/{}/'.format(k)
    filename_suffix+='_{}'.format(k)

    acc_filename = '{}mlp_acc_{}.txt'.format(path, filename_suffix)
    cm_filename = '{}mlp_cm_{}'.format(path, filename_suffix)

    with open(acc_filename, 'w') as f:
        f.write(acc)

    plot_confusion_matrix.cm_analysis(y_test, y_pred, labels, filename=cm_filename, figsize=(15, 10))

    return

def revert_multilabel(class_array, columns_names):
    """
    reverte as classes dummificadas. Ex: (0, 1) vira true e (1, 0) vira false
    :param class_array:
    :param columns_names:
    :return:
    """

    dummies = pd.DataFrame(class_array, columns=columns_names)
    reverted = dummies.idxmax(axis=1)

    return reverted
