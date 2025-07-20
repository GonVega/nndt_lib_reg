'''
Implements TNode, Tree classes and create_DT function for transforming
neural network into decision tree.
'''
import numpy as np
import pandas as pd
import cvxpy as cp

class TNode:
    '''Represents a Node in a Decision Tree'''
    node_id = 0
    def __init__(self, depth, tag="", capa=0, filtro=0, matrix=None, cat_vector="", constraints = []):
        '''
        Creates TNode object representing a node in the Neural Network decision tree.

        Parameters
        ----------
        depth (int):
            Profundidad del nodo en el árbol.
        tag (int):
            Etiqueta del nodo que representa el camino de categorización en el árbol.
        capa (int):
            Indice de la capa en el modelo secuencial.
        filtro (int):
            Numero del filtro en la capa (fila de la matriz de pesos)
        matrix (numpy.matrix):
            Matriz de pesos de la capa actual.
        cat_vector (list):
            Vector de categorización en la capa actual.

        Returns
        -------
        None
        '''
        self._depth = depth
        self._tag = tag
        self._left, self._right = None, None
        self._child_nodes = []
        self._capa, self._filtro, self._matrix, self._cat_vector = capa, filtro, matrix, cat_vector
        self._is_leaf = False
        self._node_id = TNode.node_id
        self._input_constraints = constraints
        TNode.node_id += 1

    def branch(self, modelo, prune, verbose = False, max_branch_depth = float('+inf'), **kwargs):
        '''
        Updates current TNode to the effective layers and filter,
        and branches the node.

        Parameters
        ----------
        modelo : keras.Sequential
            Object representing the Neural Network to transform.
        verbose : boolean, optional
            If set to True, TNode will print information about the current
            layer and filter.

        Returns
        -------
        None
        '''
        if self.depth == max_branch_depth:
            return [self]

        next_cat_vector = self._cat_vector
        # El metodo branch del nodo lo llama el nodo padre.
        # El nodo ya tiene la capa y el filtro.
        # Deberá determinar si actualizar sus capas y parámetros y hacer branching en dos nodos mas.
        if self._filtro == self._matrix.shape[0]:
            #Hemos terminado todos los filtros de una capa.
            #Reseteamos los filtros y aumentamos la capa en 1.
            self._filtro = 0
            self._capa += 1
            if self._capa == len(modelo.layers):
                #Hemos terminado todas las capas de la red. No hacemos branch.
                self._is_leaf = True
                return
            #Si estamos en una capa nueva, calculamos la matriz efectiva hasta esa capa.
            #Será la capa de pesos unida al array de bias, traspuesta W^ = [W,b].T
            new_m = np.vstack((modelo.layers[self._capa].get_weights()[1],
                            modelo.layers[self._capa].get_weights()[0])).T
            # La matriz efectiva hasta esa capa será la matriz efeciva anterior
            # por la clasificación del nodo.
            new_input = np.multiply(self.get_cat_vector(), self._matrix.T).T
            # Añadimos un array de 1 para incluir el término independiente.
            array_1 = np.zeros(new_input.shape[1])
            array_1[0] = 1
            #new_input_1 = np.vstack((new_input, np.ones(new_input.shape[1])))
            new_input_1 = np.vstack((array_1 ,new_input))
            # La nueva matriz efectiva será el producto matricial de la capa actual
            # y la matriz efectiva.
            self.matrix = new_m @ new_input_1
            #El vector de clasificación a se resetea.
            next_cat_vector = ""
        if verbose:
            if (TNode.node_id % 100 == 0):
                print(f"Realizado {TNode.node_id/2**(sum([layer.get_weights()[0].shape[1] for layer in modelo.layers]))*100}%", flush = True)
                print(f"CAPA {self._capa} FILTRO {self._filtro} CATVECTOR {self._cat_vector}", flush=True)
        for new_cat in ['0', '1']:
            child = TNode(self._depth+1,self._tag+new_cat, self._capa, self._filtro+1,
                        self._matrix, next_cat_vector+new_cat, self._input_constraints[:])
            if prune:
                if child.check_consistency(**kwargs) < 0:
                    print(f"----- Nodo {self.node_id}, {self._tag + new_cat} inconsistente -----")
                    TNode.node_id -= 1
                    del child
                    continue
            self._child_nodes.append(child)
            (self._child_nodes[-1]).add_constraint([
                *(self.get_filter()[1:]),
                self.generate_constraint_operator(new_cat),
                -self.get_filter()[0]
            ])
        # child = TNode(self._depth+1,self._tag+'1', self._capa, self._filtro+1,
        #                 self._matrix, next_cat_vector+'1', self._input_constraints[:])
        # if child.check_consistency() < 0:
        #     #print(f"----- Nodo {self.node_id}, {self._tag} inconsistente -----")
        #     #print(self._input_constraints)
        #     TNode.node_id -= 1
        #     del child
        # else:
        #     self._child_nodes.append(child)
        #     self._child_nodes[-1].add_constraint([
        #         *(self.get_filter()[1:]),
        #         ">=",
        #         -self.get_filter()[0]
        #     ])
        unbranched_nodes = []
        for node in self._child_nodes:
            node.branch(modelo, prune = prune, verbose = verbose, max_branch_depth = max_branch_depth, **kwargs)
            #unbranched_nodes = unbranched_nodes + node.branch(modelo, prune = prune, verbose = verbose, max_branch_depth = max_branch_depth, **kwargs)
            #unbranched_nodes.append(node.branch(modelo, prune = prune, verbose = verbose, max_depth = max_depth, **kwargs))
            # unbranched_nodes += node.branch(modelo, prune = prune, verbose = verbose, max_depth = max_depth, **kwargs)
        return unbranched_nodes
    
    def check_consistency(self, **kwargs):
        """
        Checks the consistency of the constraints in input_constraints by finding a solution via solver.
        If no feasible solution exists, returns -1, 0 otherwise.
        """
        # La x será el número de columnas de la matriz efectiva menos uno (por el término indepediente añadido)
        x = cp.Variable(self._matrix.shape[1]-1)
        constraints = self.get_cp_constraints(x, **kwargs)
        problem = cp.Problem(cp.Maximize(cp.sum(x)), constraints)
        result = problem.solve()
        if problem.status == "infeasible":
            return -1
        # if result == float('-inf'):
        #     return -1
        return 0

    def get_cp_constraints(self, x: cp.Variable, lower_limits = -10000, upper_limits = 10000):
        cp_constraints = []
        for constraint in self._input_constraints:
            m = constraint[:-2]
            op = constraint[-2]
            n = constraint[-1]
            if op == "<=":
                cp_constraint = np.array(m) @ x <= n
            elif op == ">=":
                cp_constraint = np.array(m) @ x >= n
            else:
                cp_constraint = []
            cp_constraints.append(cp_constraint)
        if isinstance(lower_limits, np.ndarray):
            for i in range(lower_limits.shape[0]):
                cp_constraints.append(x[i] >= lower_limits[i])
        else:
            cp_constraints.append(x >= lower_limits)
        if isinstance(upper_limits, np.ndarray):
            for i in range(upper_limits.shape[0]):
                cp_constraints.append(x[i] <= upper_limits[i])
        else:
            cp_constraints.append(x <= upper_limits)
        return cp_constraints

    def generate_constraint_operator(self, cat):
        if cat == "0":
            return "<="
        if cat == "1":
            return ">="

    def get_cat_vector(self):
        '''
        Calculates the categorization vector of the node

        Parameters
        ----------
        None

        Returns
        -------
        categorization_vector : numpy.array
            A vector of 0s and 1s representing the categorization result
            represented by the current node.
        '''
        return np.array([int(i) for i in list(self._cat_vector)])

    def get_filter(self):
        '''
        Calculates the effective filter for current matrix and filter.

        Parameters
        ----------
        None

        Returns
        -------
        filter : numpy.array
            A vector of len(input) elements which are the coefficients
            of the effective filter.
        '''
        return self._matrix[self._filtro]

    def eval_node(self, x):
        '''
        Evaluates the input.

        Parameters
        ----------
        value : numpy.array
            Array of floats
        
        Returns
        -------
        evaluation : numpy.array
            Float resulting from doing the inner product between the
            effective filter and the input.
        '''
        evaluation = self.get_filter() @ x.T
        return evaluation

    def evaluate_input(self, x, max_depth, verbose = False, task=None):
        '''
        Calcula la evaluación de la entrada en el árbol de decisión
        bajo el nodo con una profundidad máxima de max_level.
        Si el nodo es hoja y tiene _leaf_value, devuelve directamente el valor.
        '''
        # Si el nodo es hoja y tiene _leaf_value, devolvemos directamente
        if self._is_leaf or len(self._child_nodes) == 0:
            if hasattr(self, '_leaf_value'):
                # Si se especifica la tarea, devolvemos el campo adecuado
                if task == 'classification':
                    return {'category': int(self._leaf_value), 'leaf': True}
                else:
                    return {'value': self._leaf_value, 'leaf': True}
            # Fallback: si no hay _leaf_value, usar el vector de categorización
            if not getattr(self, '_cat_vector', None):
                return {'category': 0, 'leaf': True}
        # Si no es hoja, seguimos el recorrido normal
        evaluation = self.eval_node(x)
        if evaluation <= 0:
            child_node_id = 0
        else:
            child_node_id = 1
        if len(self._child_nodes) > 0 and max_depth > 0:
            return self._child_nodes[child_node_id].evaluate_input(
                x, max_depth - 1, verbose, task=task
            )
        else:
            if verbose:
                print(f"Nodo hoja {self._tag} {self._cat_vector}: {self}  => evaluation = {evaluation}")
            category = int(self._cat_vector, 2)
            filtro = self.get_filter()
            betas = {f"B_{i}":filtro[i] for i in range(len(filtro))}
            output_dict = {
                **{f"X_{i}":x[i] for i in range(len(x))},
                'value':evaluation,
                'category':category,
                'node_id':self._node_id,
                'tag':self._tag,
                'leaf':self._is_leaf,
                **betas
            }
            return output_dict

    @property
    def depth(self):
        return self._depth

    @property
    def matrix(self):
        return self._matrix
    @matrix.setter
    def matrix(self, matrix):
        self._matrix = matrix

    @property
    def child_nodes(self):
        return self._child_nodes

    @property
    def height(self):
        if len(self.child_nodes) == 0:
            return 0
        return  max([node.height for node in self.child_nodes]) +1
    
    def add_constraint(self, constraint):
        self._input_constraints.append(constraint)

    def __str__(self):
        #TODO: Actualizar la funciond de impresión del nodo
        #para que tenga en cuenta los limites diferentes
        return f"{self.get_filter()[0]} + {self.get_filter()[1:]} * X > 0"

class Tree:
    '''
    Implements a Tree of TNodes.
    '''
    def __init__(self, modelo):
        TNode.node_id = 0
        self.root = TNode(0)
        self.node_dict = None
        self.modelo = modelo

    def create_DT(self, auto_prune = True, verbose: bool = False, **kwargs):
        if not auto_prune and ('lower_limits' in kwargs or 'upper_limits' in kwargs):
            print("Warning: upper or lower limits set for pruning, but pruning is set off.\
                Limits parameters will have no effect in the tree construction.")
        m = self.modelo.layers[0].get_weights()[0]
        b = self.modelo.layers[0].get_weights()[1]
        self.root.matrix = np.vstack((b, m)).T
        # Quitar esto y el return NO ME GUSTA
        return self.root.branch(self.modelo, prune = auto_prune, verbose = verbose, **kwargs)

    def evaluate_input(self, x: np.array, returns = 'all', max_depth = float('+inf'), verbose = False, task=None):
        '''
        Evaluates given input and returns output dict.

        Parameters
        ----------
        x : numpy.array
            Input to evaluate.
        returns: optional, str, list
            IIDSKNDAK
        max_depth : optional, float
            Max depth in the tree to evaluate.
        verbose : boolean
            If set to True, each node in the evauation will print its information.

        Returns
        -------
        output: dict
            Dictionary with evaluation results.
        '''
        x = np.insert(x, 0, 1)
        output = self.root.evaluate_input(x , max_depth=max_depth, verbose = verbose, task=task)
        if returns == 'all':
            return output
        if type(returns) == str:
            return output[returns]
        if type(returns) == list and len(returns) == 1:
            return output[returns [0]]
        else:
            return  {key:output[key] for key in returns}
    
    def evaluate_dataset(self, X, returns = 'all', task=None, **kwargs):
        '''
        Evaluates set of inputs.

        Parameters
        ----------
        values : numpy.matrix, pandas.DataFrame
            Array of inputs to evaluate
        verbose : boolean
            If set to True, each node in the evauation will print its information.

        Returns
        -------
        results : dict
            Set of results for given inputs.
        '''
        index = None
        if type(X) == pd.DataFrame:
            index = X.index
            X = X.values
        results = [self.evaluate_input(x, returns = returns, task=task, **kwargs) for _, x in enumerate(X)]
        if returns == 'all' or (type(returns) == list and len(returns) > 1):
            return pd.DataFrame(results, index = index)
        else:
            return np.array(results)

    
    def predict(self, X, task, **kwargs):
        '''
        Evaluates set of inputs to return the given result for each one.

        Parameters
        ----------
        values : numpy.array
            Array of inputs to evaluate
        task : str
            Regression or classification
        verbose : boolean
            If set to True, each node in the evauation will print its information.

        Returns
        -------
        results : numpy.array
            Set of predictions for given inputs.
        '''
        task_params = {"regression":"value", "classification":"category"}
        predictions = self.evaluate_dataset(X, returns = task_params[task], task=task, **kwargs)
        return predictions

    def evaluate_model(self, X, Y, task, metrics = ["accuracy"], **kwargs):
        '''
        Evaluates model on given dataset accoding to metrics.

        Parameters
        ----------
        X : numpy.array
            Array of inputs.
        Y : numpy.array
            Array of expected outputs
        task : str
            Regression or classification
        metrics : list
            List of metrics to use.

        Returns
        -------
        metrics : dict
            Dictionary with the metrics and their values.
        '''
        if type(Y)== pd.Series or type(Y) == pd.DataFrame:
            Y = Y.values
        metric_results = {}
        predictions = self.predict(X, task = task, **kwargs)
        n_correct = sum(predictions == Y)
        for metric in metrics:
            if metric == "accuracy":
                metric_results["accuracy"] = n_correct/Y.shape[0]
            if metric == "mean_squared_error" or metric.upper() == "MSE":
                metric_results["mean_squared_error"] = sum((predictions - Y)**2)/Y.shape[0]
            if metric == "root_mean_squared_error" or metric.upper() == "RMSE":
                metric_results["root_mean_squared_error"] = (sum((predictions - Y)**2)/Y.shape[0])**0.5
            if metric == "mean_absolute_error" or metric.upper() == "MAE":
                metric_results["mean_absolute_error"] = sum(abs(predictions - Y))/Y.shape[0]
        return metric_results
    
    def to_dict(self):
        '''
        Returns a dictionary with the node at each level.

        Parameters
        ----------
        None

        Returns
        -------
        dictionary : dict
            The dictionary representing the tree.
        '''
        node_dict = {}
        Tree.preorder(self.root, self.add_to_dict, target = node_dict)
        return node_dict

    def add_to_dict(self, node, target):
        '''
        Adds TNode to nodes dictionary

        Parameters
        ----------
        node : TNode
            The node to be added.

        Returns
        -------
        None
        '''
        if node.depth not in target:
            target[node.depth] = []
        target[node.depth].append(node)

    @property
    def height(self):
        if self.root is None:
            return -1
        return self.root.height

    @property
    def shape(self):
        return [len(x) for _, x in self.to_dict().items()]

    @staticmethod
    def preorder(node: TNode, function = None, **kwargs):
        result = []
        if node is None:
            return
        result.append(function(node, **kwargs))
        for node in node.child_nodes:
            result += Tree.preorder(node, function, **kwargs)
        return result

    @staticmethod
    def postorder(node: TNode, function = None, **kwargs):
        if node is None:
            return
        for node in node.child_nodes:
            Tree.preorder(node, function, **kwargs)
        function(node, **kwargs)

    def __sizeof__(self):
        import sys
        return sum(Tree.preorder(self.root, sys.getsizeof))
