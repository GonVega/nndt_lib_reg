from .binary_tree import Tree
import numpy as np
import multiprocessing as mp
import time

class ParalelTree(Tree):
    def __init__(self, modelo):
        super().__init__(modelo)
        self.PROCESSES = mp.cpu_count()-1
        self.unbranched_nodes_queue = mp.Queue()

    def create_DT(self, auto_prune = True, verbose: bool = False, max_branch_depth = 5, **kwargs):
        if not auto_prune and ('lower_limits' in kwargs or 'upper_limits' in kwargs):
            print("Warning: upper or lower limits set for pruning, but pruning is set off.\
                Limits parameters will have no effect in the tree construction.")
        m = self.modelo.layers[0].get_weights()[0]
        b = self.modelo.layers[0].get_weights()[1]
        self.root.matrix = np.vstack((b, m)).T
        self.unbranched_nodes_queue.put(self.root)
        self.run_parallel(
            auto_prune = True, verbose = verbose, max_branch_depth = max_branch_depth, **kwargs
            )
        # Quitar esto y el return NO ME GUSTA
        # return self.root.branch(self.modelo, prune = auto_prune,
        #                         verbose = verbose, max_branch_depth=max_branch_depth,
        #                         **kwargs)
        
    def process_task(self, **kwargs):
        print(f"processing_task: {self.unbranched_nodes_queue.}")
        while not self.unbranched_nodes_queue.empty:
            child = self.unbranched_nodes_queue.get()
            unbranched = child.branch(kwargs)
            self.add_task(unbranched)
            time.sleep(0.1)
        return True

    def add_task(self, unbranched_nodes):
        for child in unbranched_nodes:
            self.unbranched_nodes_queue.put(child)

    def run_parallel(self, **kwargs):
        processes = []
        start = time.time()
        for _ in range(self.PROCESSES):
            p = mp.Process(target = self.process_task, kwargs=kwargs)
            processes.append(p)
            p.start()
        for p in processes:
            p.join()
        print(f"Time taken {time.time() - start:.10f}")