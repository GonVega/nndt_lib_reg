import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
import time
from nndt_lib.binary_tree import Tree
from nndt_lib.post_pruning import post_prune_tree
from nndt_lib.prune_for_speed import prune_tree_for_speed, patch_evaluate_input_all_nodes
from sklearn.metrics import confusion_matrix

def count_nodes(tree):
    node_dict = tree.to_dict()
    return sum(len(nodes) for nodes in node_dict.values())

def print_confusion(title, tree, X, y):
    y_pred = tree.predict(X, task='classification')
    cm = confusion_matrix(y, y_pred)
    print(f"{title}\n{cm}\n")

# 1. Cargar el dataset
df = pd.read_csv('cdc_diabetes_health_indicators.csv')
if df.isnull().values.any():
    df = df.dropna()
target_col = 'Diabetes_binary'
X = df.drop(columns=[target_col])
y = df[target_col]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_temp, y_train, y_temp = train_test_split(X_scaled, y, test_size=0.3, random_state=42, stratify=y)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

# 2. Entrenar una red neuronal para clasificación binaria
model = keras.Sequential([
    keras.layers.Input(shape=(X_train.shape[1],)),
    keras.layers.Dense(5, activation='relu'),
    #keras.layers.Dense(3, activation='relu'),
    keras.layers.Dense(1, activation='sigmoid')
])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_val, y_val))

# 3. Convertir la red neuronal a árbol de decisión
print('Construyendo árbol de decisión a partir de la red neuronal...')
tree = Tree(model)
tree.create_DT(auto_prune=False, verbose=False)

# Chequeo: asegurar que todos los _leaf_value son 0 o 1
def check_leaf_values(node):
    if getattr(node, '_is_leaf', False) or len(getattr(node, '_child_nodes', [])) == 0:
        if hasattr(node, '_leaf_value'):
            if getattr(node, '_leaf_value') not in [0, 1]:
                print(f"[ADVERTENCIA] Nodo hoja con _leaf_value inesperado: {getattr(node, '_leaf_value')}")
    for child in getattr(node, '_child_nodes', []):
        check_leaf_values(child)

# 4. Evaluar antes de la poda
acc_before = tree.evaluate_model(X_val, y_val, task='classification', metrics=['accuracy'])['accuracy']
n_before = count_nodes(tree)
start = time.time()
_ = tree.predict(X_val, task='classification')
t_before = time.time() - start
# Profundidad media recorrida antes de la poda (igual que en postprune_binary)
def path_length(x):
    current = tree.root
    x_aug = np.insert(x, 0, 1)
    depth = 0
    while len(current.child_nodes) > 0:
        eval_ = current.eval_node(x_aug)
        if eval_ <= 0:
            next_idx = 0
        else:
            next_idx = 1
        if next_idx >= len(current.child_nodes):
            break
        current = current.child_nodes[next_idx]
        depth += 1
    return depth
profundidad_media_before = np.mean([path_length(x) for x in X_scaled])
print(f'Antes de la poda: accuracy={acc_before:.4f}, nodos={n_before}, tiempo={t_before:.4f}s')
print(f'Profundidad media recorrida antes de la poda: {profundidad_media_before:.2f}')
print_confusion('Matriz de confusión ANTES de la poda', tree, X_val, y_val)
check_leaf_values(tree.root)
print('Valores únicos en las predicciones antes de la poda:', np.unique(tree.predict(X_val, task='classification')))

# 5. Post-poda clásica
print('Aplicando post-poda clásica...')
post_prune_tree(tree, X_val, y_val, task='classification', metric='accuracy')
original_evaluate_input = tree.root.__class__.__dict__['evaluate_input']
patch_evaluate_input_all_nodes(tree.root, original_evaluate_input, task='classification')
check_leaf_values(tree.root)
print('Valores únicos en las predicciones tras post-poda:', np.unique(tree.predict(X_val, task='classification')))
acc_post = tree.evaluate_model(X_val, y_val, task='classification', metrics=['accuracy'])['accuracy']
n_post = count_nodes(tree)
start = time.time()
_ = tree.predict(X_val, task='classification')
t_post = time.time() - start
profundidad_media_post = np.mean([path_length(x) for x in X_scaled])
print(f'Después de post-poda: accuracy={acc_post:.4f}, nodos={n_post}, tiempo={t_post:.4f}s')
print(f'Profundidad media recorrida después de post-poda: {profundidad_media_post:.2f}')
print_confusion('Matriz de confusión DESPUÉS de post-poda', tree, X_val, y_val)

# 6. Poda greedy para velocidad
print('Aplicando poda greedy para velocidad...')
tolerance = 0.001  # 0.1% de tolerancia en accuracy
prune_tree_for_speed(tree, X_val, y_val, task='classification', metric='accuracy', tolerance=tolerance)
original_evaluate_input = tree.root.__class__.__dict__['evaluate_input']
patch_evaluate_input_all_nodes(tree.root, original_evaluate_input, task='classification')
check_leaf_values(tree.root)
print('Valores únicos en las predicciones tras greedy:', np.unique(tree.predict(X_val, task='classification')))
acc_greedy = tree.evaluate_model(X_val, y_val, task='classification', metrics=['accuracy'])['accuracy']
n_greedy = count_nodes(tree)
start = time.time()
_ = tree.predict(X_val, task='classification')
t_greedy = time.time() - start
profundidad_media_greedy = np.mean([path_length(x) for x in X_scaled])
print(f'Después de poda greedy: accuracy={acc_greedy:.4f}, nodos={n_greedy}, tiempo={t_greedy:.4f}s')
print(f'Profundidad media recorrida después de greedy: {profundidad_media_greedy:.2f}')
print_confusion('Matriz de confusión DESPUÉS de poda greedy', tree, X_val, y_val)

# 7. Tabla resumen de resultados

def infer_time(tree):
    start = time.time()
    _ = tree.predict(X_scaled, task='classification')
    return time.time() - start

data = []
data.append({
    'Etapa': 'Original',
    'Tiempo_inferencia': infer_time(tree),
    'Nodos': n_before,
    'Profundidad_media': profundidad_media_before
})
data.append({
    'Etapa': 'Post-poda',
    'Tiempo_inferencia': infer_time(tree),
    'Nodos': n_post,
    'Profundidad_media': profundidad_media_post
})
data.append({
    'Etapa': 'Greedy',
    'Tiempo_inferencia': infer_time(tree),
    'Nodos': n_greedy,
    'Profundidad_media': profundidad_media_greedy
})

summary = pd.DataFrame(data)
print('\nResumen de resultados:')
print(summary.to_string(index=False)) 