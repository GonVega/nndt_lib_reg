'''
Implements TNode, Tree classes and create_DT function for transforming
neural network into decision tree.
'''
import numpy as np
import pandas as pd
from tensorflow import keras
from torch import nn 
import re
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score, recall_score, precision_score, f1_score

class Tree:
    '''
    Implements an abstraction of DTree which can evalute inputs and give the outputs,
    but does not build the whole structure.
    '''
    def __init__(self, modelo):
        self.modelo = normalize_model(modelo)

    def evaluate_input(self, x: np.array, returns = 'all', activation = 'linear'):
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
        m = self.modelo[0]['weights']
        b = self.modelo[0]['bias']
        eff_matrix = np.vstack((b, m)).T
        params = {
            'layer':0,
            'filter':0,
            'effective_matrix':eff_matrix,
            'trail':'',
            'categorization_vector':'',
            'output_layer_values':[]
        }
        output = self.recursive_evaluate_input(x, params)

        if activation == 'softmax':
            layer = keras.layers.Softmax()
            output_keys = list(filter(lambda x: re.match("Yhat_[0-9]+", x), output.keys()))
            output_values = np.array([output[key] for key in output_keys])
            output_values = layer(output_values).numpy()
            for i in range(len(output_keys)):
                output[output_keys[i]] = output_values[i]

        if returns == 'all':
            return output
        if type(returns) == str:
            return output[returns]
        if type(returns) == list and len(returns) == 1:
            return output[returns [0]]
        else:
            return  {key:output[key] for key in returns}

    def recursive_evaluate_input(self, x: np.array, params):
        capa = params.get('layer')
        filtro = params.get('filter')
        categorization_vector = params.get('categorization_vector')
        effective_matrix = params.get('effective_matrix')
        trail = params.get('trail')

        next_cat_vector = params.get('categorization_vector')
        # Deberá determinar si actualizar sus capas y parámetros y hacer branching en dos nodos mas.
        if filtro == effective_matrix.shape[0]:
            # Hemos terminado todos los filtros de una capa.
            # Reseteamos los filtros y aumentamos la capa en 1.
            filtro = 0
            capa += 1
            if capa == len(self.modelo):
                return self.generate_output(x, params)
            #Si estamos en una capa nueva, calculamos la matriz efectiva hasta esa capa.
            #Será la capa de pesos unida al array de bias, traspuesta W^ = [W,b].T
            new_m = np.vstack((self.modelo[capa]['bias'],
                            self.modelo[capa]['weights'])).T
            # La matriz efectiva hasta esa capa será la matriz efeciva anterior
            # por la clasificación del nodo.
            new_input = np.multiply(
                np.array([int(i) for i in list(categorization_vector)]),
                effective_matrix.T).T
            # Añadimos un array de 1 para incluir el término independiente.
            array_1 = np.zeros(new_input.shape[1])
            array_1[0] = 1
            #new_input_1 = np.vstack((new_input, np.ones(new_input.shape[1])))
            new_input_1 = np.vstack((array_1 ,new_input))
            # La nueva matriz efectiva será el producto matricial de la capa actual
            # y la matriz efectiva.
            effective_matrix = new_m @ new_input_1
            #El vector de clasificación a se resetea.
            next_cat_vector = ""
    
        evaluation = effective_matrix[filtro] @ x.T
        params = {
            'layer':capa,
            'filter':filtro+1,
            'effective_matrix':effective_matrix,
            'output_layer_values':params['output_layer_values']
        }
        if capa == len(self.modelo)-1:
            # Al estar en la última capa, guardamos los valores de 
            # evaluar la entrada a cada neurona de salida.
            params['output_layer_values'].append(evaluation)
        if evaluation >= 0:
            params['trail'] = trail + '1'
            params['categorization_vector'] = next_cat_vector + '1'
        else:
            params['trail'] = trail + '0'
            params['categorization_vector'] = next_cat_vector + '0'
        return self.recursive_evaluate_input(x, params)
    
    def evaluate_dataset(self, X, returns = 'all', **kwargs):
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
        results = [self.evaluate_input(x, returns = returns, **kwargs) for _, x in enumerate(X)]
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
        predictions = self.evaluate_dataset(X, returns = task_params[task], **kwargs)
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
                # metric_results["accuracy"] = n_correct/Y.shape[0]
                metric_results["accuracy"] = accuracy_score(Y, predictions)
            if metric == "mean_squared_error" or metric.upper() == "MSE":
               # metric_results["mean_squared_error"] = sum((predictions - Y)**2)/Y.shape[0]
               metric_results["mean_squared_error"] = mean_squared_error(Y, predictions)
            if metric == "root_mean_squared_error" or metric.upper() == "RMSE":
                metric_results["root_mean_squared_error"] = (sum((predictions - Y)**2)/Y.shape[0])**0.5
            if metric == "mean_absolute_error" or metric.upper() == "MAE":
                # metric_results["mean_absolute_error"] = sum(abs(predictions - Y))/Y.shape[0]
                metric_results["mean_absolute_error"] = mean_absolute_error(Y, predictions)
            if metric == "r_squared" or metric.upper() == "R2":
                metric_results["r_squared"] = r2_score(y_true = Y, y_pred = predictions)
            if metric == 'recall':
                metric_results['recall'] = recall_score(Y, predictions)
            if metric == 'precision':
                metric_results['precision'] = precision_score(Y, predictions)
            if metric == 'f1_score':
                metric_results['f1_score'] = f1_score(Y, predictions)

        return metric_results
    
    def generate_output(self, x, params):
        capa = params.get('layer')
        filtro = params.get('filter')-1
        categorization_vector = params.get('categorization_vector')
        effective_matrix = params.get('effective_matrix')
        trail = params.get('trail')
        output_layer_values = params.get('output_layer_values')

        evaluation = effective_matrix[filtro] @ x.T
        category = int(categorization_vector, 2)
        ecuacion = effective_matrix[filtro]
        betas = {f"B_{i}":ecuacion[i] for i in range(len(ecuacion))}

        output_dict = {
            **{f"X_{i}":x[i] for i in range(len(x))},
            **betas,
            **{f"Yhat_{i}":output_layer_values[i] for i in range(len(output_layer_values))},
            'value':evaluation,
            'category':category,
            'trail':trail,
            'node_id': int(trail, 2),
        }
        return output_dict

def normalize_model(modelo):
    norm_model = {}
    if isinstance(modelo, nn.Module):
        for module in modelo.modules():
            if isinstance(module, nn.ModuleList):
                for ilayer, layer in enumerate(module):
                    norm_model[ilayer] = {
                        'weights' : layer.weight.detach().numpy().T,
                        'bias' : layer.bias.detach().numpy()
                    }
    if isinstance(modelo, keras.Sequential):
        for ilayer, layer in enumerate(modelo.layers):
            norm_model[ilayer] = {
                'weights' : layer.get_weights()[0],
                'bias' : layer.get_weights()[1]
            }
    return norm_model
