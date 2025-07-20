import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
from nndt_lib.binary_tree import Tree
from nndt_lib.post_pruning import post_prune_tree
import time
from sklearn.tree import DecisionTreeClassifier

# 1. Cargar el dataset
csv_path = 'cdc_diabetes_health_indicators.csv'  # Ruta correcta desde la raíz

df = pd.read_csv(csv_path)

# 2. Preprocesamiento
# Eliminar filas con valores NaN si existen
if df.isnull().values.any():
    df = df.dropna()

target_col = 'Diabetes_binary'
X = df.drop(columns=[target_col])
y = df[target_col]

# Normalizar features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3. Separar en train/val/test
X_train, X_temp, y_train, y_temp = train_test_split(X_scaled, y, test_size=0.3, random_state=42, stratify=y)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

# 4. Entrenar una red neuronal Keras para clasificación binaria
model = keras.Sequential([
    keras.layers.Input(shape=(X_train.shape[1],)),
    keras.layers.Dense(5, activation='relu'),
    keras.layers.Dense(3, activation='relu'),
    keras.layers.Dense(1, activation='sigmoid')
])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_val, y_val))

# 5. Convertir la red neuronal a árbol de decisión
print('Construyendo árbol de decisión a partir de la red neuronal...')
tree = Tree(model)
tree.create_DT(auto_prune=True, verbose=False)

def count_nodes_and_avg_depth(tree, X):
    # Cuenta nodos y calcula profundidad media recorrida por las instancias de X
    node_dict = tree.to_dict()
    total_nodes = sum(len(nodes) for nodes in node_dict.values())
    # Profundidad media: para cada x, cuenta los saltos hasta la hoja
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
    avg_depth = np.mean([path_length(x) for x in X])
    return total_nodes, avg_depth

# 6. Evaluar el árbol antes de la poda (validación)
start = time.time()
acc_before = tree.evaluate_model(X_val, y_val, task='classification', metrics=['accuracy'])['accuracy']
time_before = time.time() - start
print(f'Accuracy antes de la post-poda (validación): {acc_before:.4f}')
print(f'Tiempo de inferencia antes de la post-poda (validación): {time_before:.4f} segundos')

total_nodes_before, avg_depth_before = count_nodes_and_avg_depth(tree, X_scaled)
print(f'Nodos totales antes de la post-poda: {total_nodes_before}')
print(f'Profundidad media recorrida antes de la post-poda: {avg_depth_before:.2f}')

# Medir tiempo de inferencia sobre TODO el dataset antes de la poda
start = time.time()
_ = tree.evaluate_model(X_scaled, y, task='classification', metrics=['accuracy'])
time_before_all = time.time() - start

# Medir tiempo de inferencia usando predict antes de la poda
start = time.time()
_ = tree.predict(X_scaled, task='classification')
time_predict_before = time.time() - start

# 7. Aplicar post-poda
print('Aplicando post-poda...')
post_prune_tree(tree, X_val, y_val, task='classification', metric='accuracy')

# 8. Evaluar el árbol después de la poda (validación)
start = time.time()
acc_after = tree.evaluate_model(X_val, y_val, task='classification', metrics=['accuracy'])['accuracy']
time_after = time.time() - start
print(f'Accuracy después de la post-poda (validación): {acc_after:.4f}')
print(f'Tiempo de inferencia después de la post-poda (validación): {time_after:.4f} segundos')

total_nodes_after, avg_depth_after = count_nodes_and_avg_depth(tree, X_scaled)
print(f'Nodos totales después de la post-poda: {total_nodes_after}')
print(f'Profundidad media recorrida después de la post-poda: {avg_depth_after:.2f}')

# Medir tiempo de inferencia sobre TODO el dataset después de la poda
start = time.time()
_ = tree.evaluate_model(X_scaled, y, task='classification', metrics=['accuracy'])
time_after_all = time.time() - start

# Medir tiempo de inferencia usando predict después de la poda
start = time.time()
_ = tree.predict(X_scaled, task='classification')
time_predict_after = time.time() - start

# 9. (Opcional) Evaluar en test
acc_test = tree.evaluate_model(X_test, y_test, task='classification', metrics=['accuracy'])['accuracy']
print(f'Accuracy en test: {acc_test:.4f}')

# Mostrar tiempos de inferencia sobre TODO el dataset
print(f'Tiempo de inferencia sobre TODO el dataset antes de la post-poda: {time_before_all:.4f} segundos')
print(f'Tiempo de inferencia sobre TODO el dataset después de la post-poda: {time_after_all:.4f} segundos')
print(f'Tiempo de inferencia usando predict antes de la post-poda: {time_predict_before:.4f} segundos')
print(f'Tiempo de inferencia usando predict después de la post-poda: {time_predict_after:.4f} segundos')

# Entrenar un árbol de decisión de sklearn para comparar tiempos
print('Entrenando árbol de decisión de sklearn...')
sk_tree = DecisionTreeClassifier(random_state=42)
sk_tree.fit(X_train, y_train)

# Medir tiempo de inferencia del árbol sklearn sobre todo el dataset
start = time.time()
_ = sk_tree.predict(X_scaled)
time_sklearn = time.time() - start

# Obtener número de nodos del árbol sklearn
n_nodes_sklearn = sk_tree.tree_.node_count

print(f'Tiempo de inferencia sklearn DecisionTree sobre TODO el dataset: {time_sklearn:.4f} segundos')
print(f'Nodos totales sklearn DecisionTree: {n_nodes_sklearn}') 