import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

'''
import sys
sys.path.append("../myUtils")  # add parent directory to path to import myGraph

import myGraph
import text
import ...
'''


def createData(nPerClust, locations, labels=None, blur=1, xyLabels=('x', 'y'), title='Data', plot=False):
    ''' 
    nPerClust: number of points per cluster
    locations: list of locations of clusters, eg.: [[1,1],[5,1],...]
    labels: list of labels per cluster, eg.: [0,1,...] (default: [0,1,...])
    blur: standard deviation of gaussian noise (default: 1)
    
    returns: data, labels, clusters
    data: tensor of size (2n x 2) (2n points, 2 dimensions)
    labels: tensor of size (2n x 1) (2n points, 1 dimension)
    clusters: list of clusters, each cluster is a list of 2 lists (x and y coordinates)
    
    plot using:
    for i,cluster in enumerate(clusters):
        plt.scatter(cluster[0], cluster[1], label=i)
    
    '''
    def noise():
        return np.random.randn(nPerClust)*blur

    # generate data
    if labels is None:
        labels = range(len(locations))  # [0,1,...]
    
    data = []
    for i,loc in enumerate(locations):
        data.append([loc[0]+noise(), loc[1]+noise()])
        
    # labels (0 for a, 1 for b) (2n x 1)
    labels_np = np.vstack([np.ones((nPerClust,1))*label for label in labels])
    
    # concatanate into a matrix
    data_np = np.hstack(data).T
     
    # convert to a pytorch tensor
    data = torch.tensor(data_np).float()

    # convert to a pytorch tensor
    labels = torch.tensor(labels_np).float()
    
    # data for plotting
    clusters = []
    for i,loc in enumerate(locations):
        clusters.append([data[np.where(labels_np==i)[0],0], data[np.where(labels_np==i)[0],1]])
        
    if plot:
        # figsize 5,5
        plt.figure(figsize=(5,5))
        for i,cluster in enumerate(clusters):
            plt.scatter(cluster[0], cluster[1], label=i)
        plt.legend()
        plt.grid(color='lightgray', linestyle='--', linewidth=0.5)
        # set same scale for x and y axes
        plt.gca().set_aspect('equal', adjustable='box')
        plt.xlabel(xyLabels[0])
        plt.ylabel(xyLabels[1])
        plt.title(title)
        plt.show()
        plt.close()
    
    return data, labels, clusters



