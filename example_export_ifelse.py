import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
from nndt_lib.binary_tree import Tree
from nndt_lib.post_pruning import post_prune_tree
from nndt_lib.tree_to_ifelse import export_tree_to_ifelse

# 1. Cargar y preparar datos
csv_path = 'cdc_diabetes_health_indicators.csv'
df = pd.read_csv(csv_path)
if df.isnull().values.any():
    df = df.dropna()
target_col = 'Diabetes_binary'
X = df.drop(columns=[target_col])
y = df[target_col]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_temp, y_train, y_temp = train_test_split(X_scaled, y, test_size=0.3, random_state=42, stratify=y)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

# 2. Entrenar red y convertir a árbol
model = keras.Sequential([
    keras.layers.Input(shape=(X_train.shape[1],)),
    keras.layers.Dense(5, activation='relu'),
    keras.layers.Dense(3, activation='relu'),
    keras.layers.Dense(1, activation='sigmoid')
])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=3, batch_size=32, validation_data=(X_val, y_val))
tree = Tree(model)
tree.create_DT(auto_prune=False, verbose=False)

# 3. Post-poda
post_prune_tree(tree, X_val, y_val, task='classification', metric='accuracy')

# 4. Exportar a if-else
code = export_tree_to_ifelse(tree)
with open('tree_ifelse_exported.py', 'w') as f:
    f.write(code)

# 5. Ejecutar la función generada y evaluar
local_vars = {}
exec(code, globals(), local_vars)
tree_predict = local_vars['tree_predict']

# Evaluar sobre el conjunto de validación
preds = np.array([tree_predict(x) for x in X_val])
acc = np.mean(preds == y_val.values)
print(f'Accuracy de la función if-else exportada en validación: {acc:.4f}') 