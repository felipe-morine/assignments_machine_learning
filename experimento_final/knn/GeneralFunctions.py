from knn.KNN import KNN
import pandas as pd
import numpy as np
from knn import plot_confusion_matrix

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

def save_model(knn: KNN, is_undersample, is_PCA):
    """
    Salva a matriz de instancias suporte. Ver a classe KNN para mais detalhes
    :param knn:
    :param is_undersample:
    :param is_PCA:
    :return:
    """

    path = '../knn/support_instances_matrix/'
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

    distance_matrix_file = path+'matrix'
    np.save(distance_matrix_file, knn.support_instances_matrix)

    return

def load_model(knn: KNN, is_undersample, is_PCA):
    """
    Carrega a matriz de instancias suporte. Para mais detalhes, ver a classe KNN
    :param knn:
    :param is_undersample:
    :param is_PCA:
    :return:
    """

    path = '../knn/support_instances_matrix/'
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

    distance_matrix_file = path+'matrix.npy'
    knn.support_instances_matrix = np.load(distance_matrix_file)
    return knn

def save_results(acc, y_test, y_pred, labels, k, is_undersample, is_PCA):
    """
    Salva a acuracia e a matriz de confusao
    :param acc:
    :param y_test:
    :param y_pred:
    :param labels:
    :param k:
    :param is_undersample:
    :param is_PCA:
    :return:
    """

    path = '../knn/results/'
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

    acc_filename = '{}knn_acc_{}.txt'.format(path, filename_suffix)
    cm_filename = '{}knn_cm_{}'.format(path, filename_suffix)

    with open(acc_filename, 'w') as f:
        f.write(acc)

    plot_confusion_matrix.cm_analysis(y_test, y_pred, labels, filename=cm_filename, figsize=(15, 8))

    return

def revert_multilabel(class_array, columns_names):
    # reverte as classes dummificadas
    dummies = pd.DataFrame(class_array, columns=columns_names)
    reverted = dummies.idxmax(axis=1)

    return reverted