import numpy as np
import matplotlib.pyplot as plt
from tree_ext import QuadTree
from time import time

from discretize import TreeMesh


call2 = lambda fun, xyz: fun(xyz[:, 0], xyz[:, 1])
cart_row2 = lambda g, xfun, yfun: np.c_[call2(xfun, g), call2(yfun, g)]
cartE2 = lambda M, ex, ey: np.vstack(
      (cart_row2(M.gridEx, ex, ey), cart_row2(M.gridEy, ex, ey)))
cartF2 = lambda M, fx, fy: np.vstack((cart_row2(M.gridFx, fx, fy), cart_row2(M.gridFy, fx, fy)))

def go():
    nc = 256
    level = int(np.log2(nc))
    print(level)
    h = [nc, nc]

    def func(x, y):
        x = x-0.5
        y = y-0.5
        dist = np.sqrt(x*x+y*y)
        if dist < 0.2:
            return level
        return level-1

    def func2(cell):
        r = cell.center - np.array([0.5]*len(cell.center))
        dist = np.sqrt(r.dot(r))
        if dist < 0.2:
            return level
        return level-1


    t1 = time()
    # tree = QuadTree(h, func, max_level=level)
    tree = QuadTree(h, max_level=level)
    tree.build_tree(func2)
    tree.number()
    t2 = time()
    print('QuadTree Construction time:', t2-t1)
    print(tree.nC)
    print(tree.ntN)
    print(tree.ntEx)
    print(tree.ntEy)
    print(tree.nhN)
    print(tree.nhEx)
    print(tree.nhEy)
    print(tree.nN)
    print(tree.nEx)
    print(tree.nEy)

    t1 = time()
    dTree = TreeMesh(h, levels=level)
    dTree.refine(func2, balance=True)
    dTree.number(balance=True)
    t2 = time()
    print('TreeMesh Construction time', t2-t1)
    print(dTree.nC)
    print(dTree.ntN)
    print(dTree.ntEx)
    print(dTree.ntEy)
    print(dTree.nhN)
    print(dTree.nhEx)
    print(dTree.nhEy)
    print(dTree.nN)
    print(dTree.nEx)
    print(dTree.nEy)

    plt.figure()
    tree.plotGrid(nodes=True)
    plt.figure()
    dTree.plotGrid(nodes=True)
    plt.show()

    # Face Divergence test
    fx = lambda x, y: np.sin(2*np.pi*x)
    fy = lambda x, y: np.sin(2*np.pi*y)
    sol = lambda x, y: 2*np.pi*(np.cos(2*np.pi*x)+np.cos(2*np.pi*y))

    divF_ana = call2(sol, tree.gridCC)
    Fc1 = cartF2(tree, fx, fy)
    F1 = tree.projectFaceVector(Fc1)

    divF = tree.faceDiv.dot(F1)
    print('QuadTree Dnorm:', np.linalg.norm(divF-divF_ana))

    Fc2 = cartF2(dTree, fx, fy)
    F2 = dTree.projectFaceVector(Fc2)
    divF2 = dTree.faceDiv.dot(F2)
    divF_ana2 = call2(sol, dTree.gridCC)

    print('TreeMesh Dnorm:', np.linalg.norm(divF2-divF_ana2))

    # Nodal gradient test
    fun = lambda x, y: (np.cos(x)+np.cos(y))
    solX = lambda x, y: -np.sin(x)
    solY = lambda x, y: -np.sin(y)

    G = tree.nodalGrad
    gradE_ana = tree.projectEdgeVector(cartE2(tree, solX, solY))
    fn = call2(fun, tree.gridN)
    gradE = G*fn
    print('QuadTree Gnorm:', np.linalg.norm(gradE-gradE_ana))

    dG = dTree.nodalGrad
    fn2 = call2(fun, dTree.gridN)
    gradE_ana2 = dTree.projectEdgeVector(cartE2(dTree, solX, solY))
    gradE2 = dG.dot(fn2)

    print('TreeMesh Gnorm:', np.linalg.norm(gradE2-gradE_ana2))


if __name__=='__main__':
    go()
