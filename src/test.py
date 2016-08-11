#  For surfaces it's a bit different than a list of 3-tuples, 
#  you should pass in a grid for the domain in 2d arrays.
#  
#  If all you have is a list of 3d points, rather than some
#  function f(x, y) -> z, then you will have a problem because 
#  there are multiple ways to triangulate that 3d point cloud 
#  into a surface.
#  
#  Here's a smooth surface example:

import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

def fun(x, y):
    return x**2 + y

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
x = y = np.arange(-3.0, 3.0, 0.05)
X, Y = np.meshgrid(x, y)
zs = np.array([fun(x,y) for x,y in zip(np.ravel(X), np.ravel(Y))])
Z = zs.reshape(X.shape)

ax.plot_surface(X, Y, Z)

ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')

plt.savefig('../images/tmp.png')