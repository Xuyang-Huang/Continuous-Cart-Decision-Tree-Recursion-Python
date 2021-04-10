#-- coding: utf-8 --
#@Time : 2021/4/9 14:20
#@Author : HUANG XUYANG
#@Email : xhuang032@e.ntu.edu.sg
#@File : Cart_Decision_Tree_on_Continuous_Value_by_Recursion.py
#@Software: PyCharm

import numpy as np
import sklearn.datasets as sk_dataset

minimum_leaf = 5


class TreeNode:
    """A decision tree node

    Attributes:
        feature_index: An integer of feature index, specify the decision feature.
        thr: A floating number of threshold to split the data.
        left: Left node.
        right: Right node.
    """
    def __init__(self, feature_index, thr):
        self.feature_index = feature_index
        self.thr = thr
        self.left = None
        self.right = None
        self.left_class = None
        self.right_class = None

    def split(self, data, label=None):
        """Split the input data.

        :param data: A 2-D Numpy array.
        :param label: A 1-D Numpy array.
        :return: [left split data, right split data], [left split data ground truth, right split data ground truth]
        """
        left_mask = data[:, self.feature_index] < self.thr
        right_mask = ~left_mask

        left_leaf_data = data[left_mask]
        right_leaf_data = data[right_mask]

        if label is not None:
            left_leaf_label = label[left_mask]
            right_leaf_label = label[right_mask]
            return [left_leaf_data, right_leaf_data], [left_leaf_label, right_leaf_label]
        else:
            return [left_leaf_data, right_leaf_data]

    def split_predict(self, data, index, label=None):
        """

        :param data: A 2-D numpy array.
        :return: [left split data, right split data],
                 [left split data index, right split data index]
                 [left split data ground truth, right split data ground truth]
        """
        left_mask = data[:, self.feature_index] < self.thr
        right_mask = ~left_mask

        left_leaf_data = data[left_mask]
        right_leaf_data = data[right_mask]

        left_leaf_data_index = index[left_mask]
        right_leaf_data_index = index[right_mask]
        if label is not None:
            left_leaf_label = label[left_mask]
            right_leaf_label = label[right_mask]
            return [left_leaf_data, right_leaf_data], [left_leaf_data_index, right_leaf_data_index], [left_leaf_label, right_leaf_label]
        else:
            return [left_leaf_data, right_leaf_data], [left_leaf_data_index, right_leaf_data_index]


class CartDecisionTree:
    """
    Attributes:
        root: A decision tree node class.
        min_leaf: An integer of minimum leaf number.
        n_class: An integer of class number.
        split_method: String, choose 'gini' or 'entropy'.
    """
    def __init__(self, min_leaf, split_method='gini'):
        self.__root = None
        self.__min_leaf = min_leaf
        self.__n_class = None
        c = Criterion()
        self.__train_node = getattr(c, split_method)

    def train(self, data, label, n_class):
        """Train a decision tree.

        Using recursion to train a Cart DT.

        :param data: A 2-D Numpy array.
        :param label: A 1-D Numpy array.
        :param n_class: An integer of class number.
        :return: No return.
        """
        self.n_class = n_class

        def grow(_data, _label):
            # Train single node.
            _feature_index, _thr = self.__train_node(_data, _label)
            _node = TreeNode(_feature_index, _thr)
            _split_data, _split_label = _node.split(_data, _label)

            if (len(_split_label[0]) == 0) | (len(_split_label[1]) == 0):
                return None
            # Regard most class as the class at the leaf.
            _node.left_class = np.argmax(np.bincount(_split_label[0]))
            _node.right_class = np.argmax(np.bincount(_split_label[1]))

            if len(_split_label[0]) < self.__min_leaf:
                return _node
            if (_split_label[0] == _split_label[0][0]).all():
                return _node

            _node.left = grow(_split_data[0], _split_label[0])

            if len(_split_label[1]) < self.__min_leaf:
                return _node
            if (_split_label[1] == _split_label[1][0]).all():
                return _node

            _node.right = grow(_split_data[1], _split_label[1])

            return _node

        self.__root = grow(data, label)

    def predict(self, data):
        """Traverse DT get a result.

        :param data: A 2-D Numpy array.
        :return: Prediction.
        """
        result = [[] for _ in range(self.n_class)]
        index = np.arange(len(data))

        def __inference(root, _data, _index):
            _split_data, _split_data_index = root.split_predict(_data, _index)
            if root.left is None:
                if len(_split_data[0]) != 0:
                    result[root.left_class].extend(_split_data_index[0])
                return None

            __inference(root.left, _split_data[0], _split_data_index[0])

            if root.right is None:
                if len(_split_data[1]) != 0:
                    result[root.left_class].extend(_split_data_index[1])
                return None

            __inference(root.right, _split_data[1], _split_data_index[1])

            return None
        __inference(self.__root, data, index)
        return result

    def eval(self, data, label):
        """

        :param data: A 2-D Numpy array.
        :param label: A 1-D Numpy array.
        :return: Prediction, Prediction with label, Accuracy.
        """
        result = [[] for _ in range(self.n_class)]
        result_label = [[] for _ in range(self.n_class)]
        index = np.arange(len(data))

        def __inference(root, _data, _index, _label):
            _split_data, _split_data_index, _split_label = root.split_predict(_data, _index, _label)
            if root.left is None:
                if len(_split_data[0]) != 0:
                    result[root.left_class].extend(_split_data_index[0])
                    result_label[root.left_class].extend(_split_label[0])
                return None

            __inference(root.left, _split_data[0], _split_data_index[0], _split_label[0])

            if root.right is None:
                if len(_split_data[1]) != 0:
                    result[root.left_class].extend(_split_data_index[1])
                    result_label[root.left_class].extend(_split_label[1])
                return None

            __inference(root.right, _split_data[1], _split_data_index[1], _split_label[1])

            return None
        __inference(self.__root, data, index, label)
        acc = []
        for i in range(self.n_class):
            if len(result_label[i]) == 0:
                acc.append(0)
            else:
                acc.append(np.mean(np.array(result_label[i]) == i))
        acc = np.mean(acc)
        return result, result_label, acc


class Criterion:
    def gini(self, data, label):
        """Traverse all features and value, find the best split feature and threshold.

        :param data: A 2-D Numpy array.
        :param label: A 1-D Numpy array.
        :return: Best feature to split, best threshold to split.
        """
        best_gini = np.inf
        for i in range(data.shape[1]):
            for j in range(1, data.shape[0]):
                tmp_gini_value = j / len(data) * self.__gini(label[:j]) + \
                                 (len(data) - j) / len(data) * self.__gini(label[j:])
                if tmp_gini_value < best_gini:
                    best_gini = tmp_gini_value
                    best_thr = np.mean([data[j-1, i], data[j, i]])
                    best_feature = i
        return best_feature, best_thr

    def entropy(self, data, label):
        """Traverse all features and value, find the best split feature and threshold.

        Find the gain higher than average, pick the highest gain ratio one.

        :param data: A 2-D Numpy array.
        :param label: A 1-D Numpy array.
        :return: Best feature to split, best threshold to split.
        """
        gain = []
        gain_ratio = []
        ent_before = self.__ent(label)
        for i in range(data.shape[1]):
            for j in range(1, data.shape[0]):
                tmp_gain = ent_before - \
                           (j / len(data) * self.__ent(label[:j]) + (len(data) - j) / len(data) * self.__ent(label[j:]))
                tmp_gain_ratio = tmp_gain / (- j / len(data) * np.log2(j / len(data)) -
                                             (len(data) - j) / len(data) * np.log2((len(data) - j) / len(data)))
                gain.append(tmp_gain)
                gain_ratio.append(tmp_gain_ratio)
        gain = np.array(gain)
        gain_ratio = np.array(gain_ratio)
        gain_ratio[gain < np.mean(gain)] = -np.inf
        best_index = np.argmax(gain_ratio)
        mat_index = np.unravel_index(best_index, [data.shape[1], data.shape[0] - 1])
        best_feature = mat_index[0]
        best_thr = np.mean([data[mat_index[1], best_feature], data[mat_index[1] + 1, best_feature]])
        return best_feature, best_thr

    def __gini(self, label):
        _label_class = list(set(label))
        gini_value = 1 - np.sum([(np.sum(label == i) / len(label)) ** 2 for i in _label_class])
        return gini_value

    def __gain(self, label, _n_class):
        gini_value = 1 - np.sum([(np.sum(label == i) / len(label)) ** 2 for i in range(_n_class)])
        return gini_value

    def __ent(self, label):
        _label_class = list(set(label))
        ent_value = - np.sum([np.sum(label == i) / len(label) * np.log2(np.sum(label == i) / len(label)) for i in _label_class])
        return ent_value


def prepare_data(proportion):
    dataset = sk_dataset.load_breast_cancer()
    label = dataset['target']
    data = dataset['data']
    n_class = len(dataset['target_names'])

    shuffle_index = np.arange(len(label))
    np.random.shuffle(shuffle_index)

    train_number = int(proportion * len(label))
    train_index = shuffle_index[:train_number]
    val_index = shuffle_index[train_number:]
    data_train = data[train_index]
    label_train = label[train_index]
    data_val = data[val_index]
    label_val = label[val_index]
    return (data_train, label_train), (data_val, label_val), n_class


if __name__ == '__main__':
    train, val, num_class = prepare_data(0.8)
    cart_dt = CartDecisionTree(minimum_leaf, 'entropy')
    cart_dt.train(train[0], train[1], num_class)
    _, _, train_acc = cart_dt.eval(train[0], train[1])
    pred, pred_gt, val_acc = cart_dt.eval(val[0], val[1])
    print('train_acc', train_acc)
    print('val_acc', val_acc)