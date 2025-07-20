import numpy as np
import pandas as pd
import nndt_lib.MockTree as nndt

def calculate_ECM(y, y_hat):
    residuos = y - y_hat
    if residuos.shape[0] <= 1:
        print("Warning: no hay suficientes datos en el área.")
        return np.NaN
    print(residuos)
    ECM = np.sum(np.power(residuos, 2), axis = 0)/(residuos.shape[0]-1)
    print(ECM.shape)
    if ECM == 0:
        print("Warning: no hay error en el área.")
        return np.NaN
    return ECM

def calculate_SE(x, y):
    ECM = calculate_ECM(y.iloc[:, 0], y.iloc[:, 1])
    numerador = y.shape[0]*ECM
    denominador = x.shape[0]*np.sum(np.power(x ,2))-np.power(np.sum(x), 2)
    if np.any(np.sum(denominador == 0) > 0):
        # print("Warning: no hay variación en los datos en este área.")
        denominador[denominador == 0] = np.NaN
    sse = (numerador/denominador)
    se = np.sqrt(sse)
    se.rename(index = {f"X_{i+1}" : f"SE_{i+1}" for i in range(x.shape[1])}, inplace = True)
    return se

def aggregation_function(df):
    betas = df.filter(regex = "B\_[0-9]+").drop_duplicates().iloc[0]
    x = df.filter(regex = "X\_[0-9]+")
    betas.drop("B_0", inplace = True, axis = 0)
    x.drop("X_0", inplace = True, axis = 1)
    y = df.filter(regex = r"Y_[0-9]+")
    y_hat = df.filter(regex = r"Yhat_[0-9]+")
    # se = calculate_SE(x, y)
    # print(x)
    # print(betas)
    # print(y)
    # print(y_hat)
    se = calculate_regression_covariance(x, y, y_hat)
    final = pd.Series(dict(betas, **se))
    return final

def calculate_regression_covariance(X, y, y_hat):
    ecm = calculate_ECM(y.values, y_hat.values)
    if ecm == 0:
        print("Warning: no hay error en el área.")
        return pd.Series(np.NaN, index = [f"SE_{i+1}" for i in range(X.shape[1])])
    # Nos quedamos con las variables que tienen variabilidad en el dataset. El resto no podemos calcularlos.
    X = X.loc[:, X.nunique() > 1]
    mid = X.T @ X
    if not np.linalg.det(mid):
        print("Warning: singular matrix") 
        return pd.Series(np.NaN, index = [f"SE_{i+1}" for i in range(X.shape[1])])
    mid = np.linalg.inv(mid)
    print(mid)
    sigma = ecm * mid
    denominador = np.sqrt(np.diagonal(sigma))
    print(np.diagonal(sigma))
    return pd.Series(denominador, index = [f"SE_{i+1}" for i in range(X.shape[1])])

def calculate_classification_covariance(X, y_hat):
    # y_hat tiene las probabilidades de pertenecer a cada clase j
    W = np.diag(np.multiply(y_hat, 1 - y_hat))
    sigma = np.linalg.inv(X.T @ W @ X)
    denominador = np.sqrt(np.diagonal(sigma))
    return denominador

def calculate_t_values(results, expected, task):
    # if task == "categorization":
    #     results.rename({"category":"Y_Hat"}, inplace = True, axis = 1)
    # if task == "regression":
    #     results.rename({"value":"Y_Hat"}, inplace = True, axis = 1)
    # results[expected.columns] = expected

    t_num_den = results.groupby("node_id").apply(aggregation_function)
    t_num_den = t_num_den.dropna()

    betas = t_num_den.filter(regex = "B\_[0-9]+")
    se = t_num_den.filter(regex = "SE\_[0-9]+")
    se = se.rename({f"SE_{i+1}":f"B_{i+1}" for i in range(results.shape[1])}, axis = 1)

    t_values = betas.abs().div(se, axis = 0)
    t_values = t_values.rename({f"B_{i+1}":f"T_{i+1}" for i in range(t_values.shape[1])}, axis = 1)
    # t_values = t_values.rename({t_values.columns[i]:results.columns[i] for i in range(results.shape[1])}, axis = 1)
    return t_values

def calculate_global_t_values(results, expected, task):
    t_values = calculate_t_values(results, expected, task).sum(axis = 0, skipna = True)
    return t_values