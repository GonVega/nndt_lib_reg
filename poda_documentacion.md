# Documentación de Técnicas de Poda de Árboles
Documentos utiles

- post_pruning.py
- prune_for_speed.py
- example_postprune_binary.py
- example_prune_hybrid_manual.py

El resto no sive para mucho

## Descripción de las Técnicas

### Post-poda clásica (Cost-Complexity Pruning)
La **post-poda clásica** es una técnica inspirada en la poda de árboles de decisión tradicionales como CART. Consiste en construir primero el árbol completo (o casi completo) y luego recorrerlo de abajo hacia arriba (postorden), probando en cada nodo interno si es posible convertirlo en hoja sin empeorar el rendimiento en un conjunto de validación. Si la poda no reduce la métrica de interés (por ejemplo, accuracy en clasificación o MSE en regresión), el subárbol se reemplaza por una hoja que predice la clase mayoritaria o el valor medio de las muestras que llegarían a ese nodo. Esta técnica busca evitar el sobreajuste y simplificar el modelo, manteniendo la capacidad predictiva.

### Poda greedy con tolerancia (Pruning for Speed)
La **poda greedy con tolerancia** es una técnica orientada a la eficiencia, cuyo objetivo es reducir el tamaño y la profundidad del árbol para acelerar la inferencia, permitiendo una pequeña pérdida controlada de rendimiento. Tras recorrer el árbol en postorden, en cada nodo interno se prueba convertirlo en hoja y se evalúa el impacto en la métrica de validación. Si la pérdida de accuracy (u otra métrica) no supera un umbral de tolerancia definido por el usuario, la poda se acepta. Así, se eliminan ramas poco relevantes, logrando árboles más compactos y rápidos, pero manteniendo el rendimiento lo más parecido posible al original.

---

# Documentación de Métodos

## post_pruning.py

### `get_samples_for_node(tree, node, X)`
Devuelve los índices de las muestras de `X` que llegarían a un nodo específico del árbol.
- **tree**: instancia de `Tree`.
- **node**: nodo objetivo (`TNode`).
- **X**: matriz de entrada (numpy o DataFrame).

### `post_prune_tree(tree, X_val, y_val, task='classification', metric='accuracy')`
Realiza **post-poda clásica** (cost-complexity pruning) sobre un árbol de decisión generado a partir de una red neuronal.
- **tree**: árbol a podar (modifica in-place).
- **X_val, y_val**: conjunto de validación.
- **task**: `'classification'` o `'regression'`.
- **metric**: métrica a optimizar (por defecto `'accuracy'` para clasificación).
- **Funcionamiento**:
  1. Recorre el árbol en postorden.
  2. Para cada nodo interno, prueba convertirlo en hoja y evalúa el rendimiento en validación.
  3. Si la poda no empeora la métrica, deja el nodo como hoja.
  4. Para regresión, la hoja predice la media de los valores de las muestras que llegan a ese nodo.
  5. Parchea el método `evaluate_input` de los nodos hoja para devolver el valor correcto tras la poda.

---

## prune_for_speed.py

### `get_samples_for_node(tree, node, X)`
(Idéntica a la de `post_pruning.py`.)

### `prune_tree_for_speed(tree, X_val, y_val, task='classification', metric='accuracy', tolerance=0.001)`
Realiza una **poda greedy** para reducir el tamaño y el tiempo de inferencia del árbol, manteniendo el accuracy lo más parecido posible.
- **tree**: árbol a podar (modifica in-place).
- **X_val, y_val**: conjunto de validación.
- **task**: `'classification'` o `'regression'`.
- **metric**: métrica a mantener.
- **tolerance**: tolerancia máxima de pérdida de accuracy (por ejemplo, 0.001 = 0.1%).
- **Funcionamiento**:
  1. Recorre el árbol en postorden.
  2. Para cada nodo interno, prueba convertirlo en hoja y evalúa el rendimiento en validación.
  3. Si el accuracy no baja más que la tolerancia, deja el nodo como hoja.
  4. Asigna correctamente el valor de predicción de la hoja (clase mayoritaria o media).
  5. Parchea el método `evaluate_input` para que los nodos hoja devuelvan el valor correcto.

### `patch_evaluate_input_all_nodes(node, original_evaluate_input, task='classification')`
Parchea recursivamente el método `evaluate_input` de todos los nodos del árbol para que los nodos hoja devuelvan el valor correcto tras la poda.
- **node**: nodo raíz desde el que empezar.
- **original_evaluate_input**: método original a parchear.
- **task**: `'classification'` o `'regression'`.

---

## prune_hybrid.py

### `prune_tree_hybrid(tree, X_val, y_val, task='classification', metric='accuracy', tolerance=0.001)`
Aplica una **estrategia híbrida** de poda:
1. Primero realiza la post-poda clásica (`post_prune_tree`).
2. Después aplica la poda greedy con tolerancia (`prune_tree_for_speed`).
- **tree**: árbol a podar (modifica in-place).
- **X_val, y_val**: conjunto de validación.
- **task**: `'classification'` o `'regression'`.
- **metric**: métrica a mantener.
- **tolerance**: tolerancia máxima de pérdida de accuracy.

### `compact_tree(tree, task='classification')`
Genera un **nuevo árbol compacto** (`Tree`) solo con los nodos y hojas necesarios tras la poda.
- El árbol resultante no depende de lógica de parcheo ni atributos temporales.
- **tree**: árbol podado del que partir.
- **task**: `'classification'` o `'regression'`.
- **Funcionamiento**:
  1. Recorre recursivamente el árbol podado.
  2. Crea nuevos nodos (`TNode`) copiando solo los atributos relevantes para la inferencia.
  3. El árbol resultante es eficiente y limpio para inferencia y serialización.

### `patch_evaluate_input_all_nodes(node, original_evaluate_input, task='classification')`
(Idéntica a la de `prune_for_speed.py`.)

---

# Ejemplo de uso

```python
from nndt_lib.binary_tree import Tree
from nndt_lib.post_pruning import post_prune_tree
from nndt_lib.prune_for_speed import prune_tree_for_speed
from nndt_lib.prune_hybrid import prune_tree_hybrid, compact_tree

# Supón que ya tienes un árbol entrenado y un conjunto de validación X_val, y_val
post_prune_tree(tree, X_val, y_val, task='classification', metric='accuracy')
prune_tree_for_speed(tree, X_val, y_val, task='classification', metric='accuracy', tolerance=0.001)
prune_tree_hybrid(tree, X_val, y_val, task='classification', metric='accuracy', tolerance=0.001)
tree_compact = compact_tree(tree, task='classification')
``` 