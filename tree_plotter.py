import matplotlib.pyplot as plt
import binary_tree as bt

def plot_tree(tree: bt.Tree, WIDTH_DIST, DEPTH_DIST):
    

def show( WIDTH_DIST, DEPTH_DIST, levels, line_segments):

def plot_tree(tree: bt.Tree, WIDTH_DIST, DEPTH_DIST, levels, line_segments):
    _, ax = plt.subplots()
    ax.set_xlim(-1, levels * DEPTH_DIST + 1)
    ax.set_ylim(-1.1*WIDTH_DIST, 1.1*WIDTH_DIST)
    ax.add_collection(line_segments)
    plt.show()


