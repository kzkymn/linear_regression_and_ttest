import numpy as np
import pandas as pd
import scipy
from sklearn.linear_model._base import LinearModel


def get_coef_t_values_linear_model(X,
                                   y,
                                   model: LinearModel):

    # 本関数は、下記を参考に実装
    # https://betashort-lab.com/%E3%83%87%E3%83%BC%E3%82%BF%E3%82%B5%E3%82%A4%E3%82%A8%E3%83%B3%E3%82%B9/%E7%B5%B1%E8%A8%88%E5%AD%A6/%E5%9B%9E%E5%B8%B0%E5%88%86%E6%9E%90%E3%81%AEt%E5%80%A4/

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
    deg_of_freedom = X_arr.shape[0] - X_arr.shape[1] - 1
    # 誤差分散Veの算出
    ve = sse / deg_of_freedom
    # 中心化した(= 平均値で各要素を引いた)説明変数について、
    # 精度行列(分散共分散行列の逆行列)を算出
    centerized_X_arr = X_arr - X_arr.mean(axis=0)
    precision_matrix = np.linalg.inv(
        np.dot(centerized_X_arr.T, centerized_X_arr))

    # 各係数のt値を格納した配列を返す
    t_values = model.coef_ / np.sqrt(np.diag(precision_matrix) * ve)

    p_values = scipy.stats.t.sf(np.abs(t_values), deg_of_freedom)*2

    return t_values, p_values
