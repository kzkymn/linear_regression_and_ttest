import numpy as np
import pandas as pd
import scipy
from sklearn.linear_model._base import LinearModel


def get_coef_t_values_linear_model(X,
                                   y,
                                   model: LinearModel,
                                   zero_coef_threshold=10**-10):

    # 本関数は基本的に下記を参考に実装
    # https://betashort-lab.com/%E3%83%87%E3%83%BC%E3%82%BF%E3%82%B5%E3%82%A4%E3%82%A8%E3%83%B3%E3%82%B9/%E7%B5%B1%E8%A8%88%E5%AD%A6/%E5%9B%9E%E5%B8%B0%E5%88%86%E6%9E%90%E3%81%AEt%E5%80%A4/
    # 切片の標準誤差の計算は下記を参考に実装
    # http://sayalabo.com/labo4.html

    if isinstance(X, pd.DataFrame):
        X_arr = X.values
    elif isinstance(X, np.ndarray):
        X_arr = X
    else:
        raise ValueError()
    if isinstance(y, pd.Series):
        y_arr = y.values
    elif isinstance(y, np.ndarray):
        y_arr = y
    else:
        raise ValueError()

    # 残差平方和SSeの算出
    y_pred = model.predict(X)
    sse = np.sum((y_pred - y_arr)**2)
    # NOTE: nとpはそれぞれ、サンプルサイズと説明変数の件数。
    # プログラミングの命名規則からは外れているが、数学的にはこのように
    # アルファベット1字で表すことが多いので、今回はそれに倣った。
    n = X_arr.shape[0]
    is_not_zero_coefs = [False if np.abs(
        coef) <= zero_coef_threshold else True for coef in model.coef_]
    p = sum(is_not_zero_coefs)
    deg_of_freedom = n - p - 1
    # 誤差分散Veの算出
    ve = sse / deg_of_freedom
    # 中心化した(= 平均値で各要素を引いた)説明変数について、
    # 精度行列(分散共分散行列の逆行列)を算出
    X_means = X_arr.mean(axis=0)
    centerized_X_arr = X_arr - X_means
    precision_matrix = np.linalg.inv(
        np.dot(centerized_X_arr.T, centerized_X_arr))

    # 切片および各係数のt値の算出
    X_means_for_intercept_t = X_means[is_not_zero_coefs]
    _tmp = [X_means_for_intercept_t for _ in range(
        len(X_means_for_intercept_t))]
    precision_matrix_for_intercept_t = precision_matrix[np.ix_(is_not_zero_coefs,
                                                        is_not_zero_coefs)]
    mean_mat_1 = np.vstack(_tmp)
    mean_mat_2 = mean_mat_1.T
    intercept_variance = 1/n + \
        np.sum(mean_mat_1 * mean_mat_2 * precision_matrix_for_intercept_t)
    t_value_of_intercept = model.intercept_ / np.sqrt(intercept_variance * ve)
    t_values = model.coef_ / np.sqrt(np.diag(precision_matrix) * ve)
    t_values = np.insert(t_values, 0, t_value_of_intercept)

    p_values = scipy.stats.t.sf(np.abs(t_values), deg_of_freedom)*2

    return t_values, p_values
