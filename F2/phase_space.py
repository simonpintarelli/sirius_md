import json
import numpy as np
import matplotlib.pyplot as plt

def diff(dd):
    vc = np.array(dd['vc'])
    x = np.array(dd['x'])
    vdiff = np.linalg.norm(vc[:,0]-vc[:,1])
    xdiff = np.linalg.norm(x[:,0]-x[:,1])
    return xdiff, vdiff

data = json.load(open('logger.out', 'r'))
xv = np.array([diff(dd) for dd in data])
plt.scatter(xv[:,0], xv[:,1], marker='x')
plt.xlabel('x')
plt.ylabel('v')
