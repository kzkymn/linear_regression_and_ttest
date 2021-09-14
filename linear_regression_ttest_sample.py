# NOTE: You should run this code with VSCode Jupyter extension.

# %%
from sklearn.datasets import load_iris, load_boston
from sklearn.linear_model import LinearRegression, LassoLarsCV, LassoLars

from skutils import get_coef_t_values_linear_regression

# %%
LOAD_BOSTON = False

# %%
if LOAD_BOSTON:
    X, y = load_boston(return_X_y=True)
else:
    X, y = load_iris(return_X_y=True)
    y = X[:, 0]
    X = X[:, 1:4]

# %%
model = LinearRegression()
model.fit(X, y)

# %%
# NOTE: sklearnの線形モデルの切片および説明変数のt値とp値を
# 算出する。LinearRegressionでなくとも、例えばLassoLarsCVなどの
# 線形モデルもmodelに渡せると思われる。(今はLassoLarsCVのみ確認済)
t_values, p_values = get_coef_t_values_linear_regression(X, y, model)

# %%
model.intercept_

# %%
model.coef_

# %%
# NOTE: t_valuesには切片、1番目の説明変数の偏回帰係数、
# 2番目の説明変数の偏回帰係数…にそれぞれ対応したt値が
# 入っている。
t_values

# %%
# NOTE: p_valuesには、t_valueの各要素に対応したp値が
# 入っている。本コードのt値とp値は、Rのlmで上と同じ
# irisデータの回帰をして求まるものと同じことを確認している。
p_values

# %%
