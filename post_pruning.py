import numpy as np
from .binary_tree import Tree, TNode
from copy import deepcopy


def get_samples_for_node(tree, node, X):
    """
    Devuelve los índices de las muestras de X que llegarían a 'node'.
    """
    indices = []
    for i, x in enumerate(X):
        current = tree.root
        x_aug = np.insert(x, 0, 1)  # El árbol espera el bias al inicio
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


def post_prune_tree(tree: Tree, X_val, y_val, task='classification', metric='accuracy'):
    """
    Realiza post-poda sobre un árbol de decisión generado a partir de una red neuronal Keras.
    Args:
        tree (Tree): Árbol a podar.
        X_val (np.ndarray o pd.DataFrame): Conjunto de validación (entradas).
        y_val (np.ndarray o pd.Series): Etiquetas verdaderas del conjunto de validación.
        task (str): 'classification' o 'regression'.
        metric (str): Métrica a optimizar (por defecto 'accuracy' para clasificación, 'mean_squared_error' para regresión).
    """
    X_val_np = X_val.values if hasattr(X_val, 'values') else X_val
    y_val_np = y_val.values if hasattr(y_val, 'values') else y_val

    def prune_node(node):
        if len(node.child_nodes) == 0:
            return
        # Guardar hijos originales
        original_children = node.child_nodes.copy()
        original_is_leaf = node._is_leaf
        # Para regresión, guardar el valor original de predicción si existe
        original_leaf_value = getattr(node, '_leaf_value', None)
        # Poda: convertir en hoja
        node.child_nodes.clear()
        node._is_leaf = True
        # Para regresión: asignar valor medio de y_val de las muestras que llegan a este nodo
        if task == 'regression':
            idxs = get_samples_for_node(tree, node, X_val_np)
            if len(idxs) > 0:
                node._leaf_value = float(np.mean(y_val_np[idxs]))
            else:
                node._leaf_value = float(np.mean(y_val_np))  # fallback
        elif task == 'classification':
            idxs = get_samples_for_node(tree, node, X_val_np)
            if len(idxs) > 0:
                vals, counts = np.unique(y_val_np[idxs], return_counts=True)
                majority_class = vals[np.argmax(counts)]
                node._leaf_value = int(majority_class)
            else:
                node._leaf_value = 0  # fallback
        # Evaluar error tras poda
        pruned_score = tree.evaluate_model(X_val, y_val, task, metrics=[metric])[metric]
        # Restaurar hijos y estado
        node.child_nodes.extend(original_children)
        node._is_leaf = original_is_leaf
        if task == 'regression':
            if original_leaf_value is not None:
                node._leaf_value = original_leaf_value
            elif hasattr(node, '_leaf_value'):
                delattr(node, '_leaf_value')
        # Evaluar error sin podar
        unpruned_score = tree.evaluate_model(X_val, y_val, task, metrics=[metric])[metric]
        # Para accuracy, mayor es mejor; para error, menor es mejor
        if (task == 'classification' and metric == 'accuracy' and pruned_score >= unpruned_score) or \
           (task == 'classification' and metric != 'accuracy' and pruned_score <= unpruned_score) or \
           (task == 'regression' and pruned_score <= unpruned_score):
            # Si la poda no empeora, dejar podado
            node.child_nodes.clear()
            node._is_leaf = True
            if task == 'regression':
                idxs = get_samples_for_node(tree, node, X_val_np)
                if len(idxs) > 0:
                    node._leaf_value = float(np.mean(y_val_np[idxs]))
                else:
                    node._leaf_value = float(np.mean(y_val_np))

    # Recorrido postorden
    def postorder(node):
        for child in node.child_nodes:
            postorder(child)
        prune_node(node)

    postorder(tree.root) 