import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

# a: y = -   x + 2      => x + y - 2 = 0
# b: y =  0.1x + 1.5    => 0.1x - y + 1.5 = 0
# c: y =     x - 1      => x - y - 1 = 0
# A: y = a + 2b + 3c

lin = lambda x: x
sig = lambda x: 1 / ( 1 + np.e**(-x) )
relu = lambda x: np.clip(x, 0, float('inf'))

a = lambda x, y: x + y - 2
b = lambda x, y: 0.1 * x - y + 1.5
c = lambda x, y: x - y - 1

A = lambda a, b, c: 1 * a + 2 * b + 3 * c - 1

def net(x, y, f):
    return f(A(f(a(x, y)), f(b(x, y)), f(c(x, y))))

x = np.arange(-5, 5, 0.1)
y = np.arange(-5, 5, 0.1)
x, y = np.meshgrid(x, y)

z1 = net(x, y, lin)
z2 = net(x, y, sig)
z3 = net(x, y, relu)


fig = plt.figure("Linear")
ax = fig.gca(projection='3d')
surf = ax.plot_surface(x, y, z1, rstride=1, cstride=1, cmap=cm.coolwarm,
linewidth=0, antialiased=False)
fig.colorbar(surf, shrink=0.5, aspect=5)

fig = plt.figure("sigmoid")
ax = fig.gca(projection='3d')
surf = ax.plot_surface(x, y, z2, rstride=1, cstride=1, cmap=cm.coolwarm,
linewidth=0, antialiased=False)
fig.colorbar(surf, shrink=0.5, aspect=5)

fig = plt.figure("relu")
ax = fig.gca(projection='3d')
surf = ax.plot_surface(x, y, z3, rstride=1, cstride=1, cmap=cm.coolwarm,
linewidth=0, antialiased=False)
fig.colorbar(surf, shrink=0.5, aspect=5)

plt.show()
