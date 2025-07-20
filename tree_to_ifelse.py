import numpy as np

def export_tree_to_ifelse(tree, feature_names=None, function_name='tree_predict'):
    """
    Exporta el árbol de decisión a una función Python en formato if-else para clasificación binaria.
    Devuelve el código como string.
    """
    if feature_names is None:
        # Por defecto, X[0], X[1], ...
        n_features = tree.root._matrix.shape[1] - 1  # -1 por el bias
        feature_names = [f'X[{i}]' for i in range(n_features)]

    def node_to_code(node, depth=0):
        indent = '    ' * depth
        if node._is_leaf or len(node._child_nodes) == 0:
            # Nodo hoja: devolver la clase
            clase = int(getattr(node, '_leaf_value', 0))
            return f'{indent}return {clase}'
        # Nodo interno: construir condición
        filtro = node.get_filter()
        # La condición es: filtro[0] + sum(filtro[1:] * X) <= 0 ?
        cond_terms = [f'{filtro[i+1]:.10f}*{feature_names[i]}' for i in range(len(feature_names))]
        cond = f'{filtro[0]:.10f} + ' + ' + '.join(cond_terms)
        code = f'{indent}if {cond} <= 0:'
        code += '\n' + node_to_code(node._child_nodes[0], depth+1)
        code += f'\n{indent}else:'
        code += '\n' + node_to_code(node._child_nodes[1], depth+1)
        return code

    code = f'def {function_name}(X):\n'
    code += node_to_code(tree.root, 1)
    return code

# Ejemplo de uso:
# from nndt_lib.binary_tree import Tree
# tree = ...
# code = export_tree_to_ifelse(tree)
# with open('tree_ifelse.py', 'w') as f:
#     f.write(code)
# exec(code)
# y_pred = [tree_predict(x) for x in X] 