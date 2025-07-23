import numpy as np
from copy import deepcopy
from nndt_lib.binary_tree import Tree, TNode

def get_samples_for_node(tree, node, X):
    indices = []
    for i, x in enumerate(X):
        current = tree.root
        x_aug = np.insert(x, 0, 1)
        while True:
            if current is node:
                indices.append(i)
                break
            if len(current.child_nodes) == 0:
                break
            eval_ = current.eval_node(x_aug)
            if eval_ <= 0:
                next_idx = 0
            else:
                next_idx = 1
            if next_idx >= len(current.child_nodes):
                break
            current = current.child_nodes[next_idx]
    return indices

def prune_tree_for_speed(tree: Tree, X_val, y_val, task='classification', metric='accuracy', tolerance=0.001, relative_to_initial=False, metric_initial=None):
    """
    Poda el árbol para reducir su tamaño y tiempo de inferencia, manteniendo el rendimiento en validación lo más parecido posible.
    Args:
        tree (Tree): Árbol a podar (modifica in-place)
        X_val, y_val: Conjunto de validación
        task: 'classification' o 'regression'.
        metric: métrica a mantener
        tolerance: tolerancia máxima de pérdida de rendimiento
        relative_to_initial: si True, la tolerancia se aplica respecto al árbol original (no localmente en cada nodo)
        metric_initial: valor de la métrica del árbol original (opcional, para tolerancia global)
    """
    # Calcular métrica del árbol original si se usa tolerancia global
    if relative_to_initial:
        if metric_initial is None:
            metric_initial_val = tree.evaluate_model(X_val, y_val, task=task, metrics=[metric])[metric]
        else:
            metric_initial_val = metric_initial
    def post_prune(node):
        if len(node.child_nodes) == 0:
            return
        for child in node.child_nodes:
            post_prune(child)
        # Intentar podar este nodo
        original_children = deepcopy(node.child_nodes)
        original_is_leaf = getattr(node, '_is_leaf', False)
        node._is_leaf = True
        node._child_nodes = []
        # Asignar predicción hoja
        indices = get_samples_for_node(tree, node, X_val)
        if len(indices) == 0:
            node._leaf_value = 0 if task == 'classification' else np.mean(y_val)
            if task == 'classification':
                node._cat_vector = '0'
        else:
            if task == 'classification':
                vals, counts = np.unique(y_val.iloc[indices], return_counts=True)
                majority_class = vals[np.argmax(counts)]
                node._leaf_value = majority_class
                node._cat_vector = bin(int(majority_class))[2:]
            else:
                node._leaf_value = np.mean(y_val.iloc[indices])
        # Evaluar métrica tras la poda
        metric_after = tree.evaluate_model(X_val, y_val, task=task, metrics=[metric])[metric]
        metric_before = tree.evaluate_model(X_val, y_val, task=task, metrics=[metric])[metric]
        accept_prune = False
        if relative_to_initial:
            if task == 'classification':
                if metric_after >= metric_initial_val - tolerance:
                    accept_prune = True
            else:
                if metric_after <= metric_initial_val + tolerance:
                    accept_prune = True
        else:
            if task == 'classification':
                if metric_after >= metric_before - tolerance:
                    accept_prune = True
            else:
                if metric_after <= metric_before + tolerance:
                    accept_prune = True
        if not accept_prune:
            # Restaurar si la poda no es aceptada
            node._is_leaf = original_is_leaf
            node._child_nodes = original_children
            if hasattr(node, '_leaf_value'):
                del node._leaf_value
    post_prune(tree.root)

# Ejemplo de uso
if __name__ == '__main__':
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from tensorflow import keras
    # Cargar datos
    df = pd.read_csv('cdc_diabetes_health_indicators.csv')
    if df.isnull().values.any():
        df = df.dropna()
    # --- Ejemplo de clasificación ---
    target_col = 'Diabetes_binary'
    X = df.drop(columns=[target_col])
    y = df[target_col]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_train, X_temp, y_train, y_temp = train_test_split(X_scaled, y, test_size=0.3, random_state=42, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)
    # Red pequeña
    model = keras.Sequential([
        keras.layers.Input(shape=(X_train.shape[1],)),
        keras.layers.Dense(3, activation='relu'),
        keras.layers.Dense(2, activation='relu'),
        keras.layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_val, y_val))
    # Árbol
    from nndt_lib.binary_tree import Tree
    tree = Tree(model)
    tree.create_DT(auto_prune=False, verbose=False)
    # Medir accuracy y tiempo antes
    import time
    acc_before = tree.evaluate_model(X_val, y_val, task='classification', metrics=['accuracy'])['accuracy']
    start = time.time()
    _ = tree.predict(X_val, task='classification')
    t_before = time.time() - start
    print(f'Antes de la poda: accuracy={acc_before:.4f}, tiempo={t_before:.4f}s')
    # Poda para velocidad (clasificación)
    prune_tree_for_speed(tree, X_val, y_val, task='classification', metric='accuracy', tolerance=0.001)
    # Medir accuracy y tiempo después
    acc_after = tree.evaluate_model(X_val, y_val, task='classification', metrics=['accuracy'])['accuracy']
    start = time.time()
    _ = tree.predict(X_val, task='classification')
    t_after = time.time() - start
    print(f'Después de la poda: accuracy={acc_after:.4f}, tiempo={t_after:.4f}s')

    # --- Ejemplo de regresión ---
    # Simular un target continuo para regresión
    y_reg = df['BMI'] if 'BMI' in df.columns else df.iloc[:, 0]  # Usar BMI o la primera columna numérica
    Xr_train, Xr_temp, yr_train, yr_temp = train_test_split(X_scaled, y_reg, test_size=0.3, random_state=42)
    Xr_val, Xr_test, yr_val, yr_test = train_test_split(Xr_temp, yr_temp, test_size=0.5, random_state=42)
    # Red para regresión
    model_reg = keras.Sequential([
        keras.layers.Input(shape=(Xr_train.shape[1],)),
        keras.layers.Dense(4, activation='relu'),
        keras.layers.Dense(2, activation='relu'),
        keras.layers.Dense(1, activation='linear')
    ])
    model_reg.compile(optimizer='adam', loss='mean_squared_error', metrics=['mean_squared_error'])
    model_reg.fit(Xr_train, yr_train, epochs=10, batch_size=32, validation_data=(Xr_val, yr_val))
    tree_reg = Tree(model_reg)
    tree_reg.create_DT(auto_prune=False, verbose=False)
    mse_before = tree_reg.evaluate_model(Xr_val, yr_val, task='regression', metrics=['mean_squared_error'])['mean_squared_error']
    print(f'Antes de la poda (regresión): MSE={mse_before:.4f}')
    prune_tree_for_speed(tree_reg, Xr_val, yr_val, task='regression', metric='mean_squared_error', tolerance=0.1)
    mse_after = tree_reg.evaluate_model(Xr_val, yr_val, task='regression', metrics=['mean_squared_error'])['mean_squared_error']
    print(f'Después de la poda (regresión): MSE={mse_after:.4f}') 