import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sp
from TreeMesh import TreeMesh as Tree
from time import time

from discretize import TreeMesh


call2 = lambda fun, xyz: fun(xyz[:, 0], xyz[:, 1])
cart_row2 = lambda g, xfun, yfun: np.c_[call2(xfun, g), call2(yfun, g)]
cartE2 = lambda M, ex, ey: np.vstack(
      (cart_row2(M.gridEx, ex, ey), cart_row2(M.gridEy, ex, ey)))
cartF2 = lambda M, fx, fy: np.vstack((cart_row2(M.gridFx, fx, fy), cart_row2(M.gridFy, fx, fy)))

def go():
    nc = 8
    level = int(np.log2(nc))
    print(level)
    h = [nc, nc]

    def func(cell):
        r = cell.center - np.array([0.5]*len(cell.center))
        dist = np.sqrt(r.dot(r))
        if dist < 0.25:
            return level
        return level-1

    t1 = time()
    # tree = QuadTree(h, func, max_level=level)
    tree = Tree(h, levels=level)
    tree.refine(func)
    t2 = time()
    print('Tree Construction time:', t2-t1)
    print("nC", tree.nC)
    print("ntN", tree.ntN)
    print("nhN", tree.nhN)
    print("ntE", tree.ntEx, tree.ntEy)
    print("nhE", tree.nhEx, tree.nhEy)

    print("nN", tree.nN)
    print("nE", tree.nEx, tree.nEy)

    t1 = time()
    dTree = TreeMesh(h, levels=level)
    dTree.refine(func, balance=True)
    dTree.number(balance=True)
    t2 = time()
    print('TreeMesh Construction time', t2-t1)
    print("nC", dTree.nC)
    print("ntN", dTree.ntN)
    print("nhN", dTree.nhN)
    print("ntE", dTree.ntEx, dTree.ntEy)
    print("nhE", dTree.nhEx, dTree.nhEy)

    print("nN", dTree.nN)
    print("nE", dTree.nEx, dTree.nEy)

    print("Same gridCC", np.allclose(tree.gridCC, dTree.gridCC))

    orderN1 = np.lexsort((tree.gridN[:, 1], tree.gridN[:, 0]))
    orderN2 = np.lexsort((dTree.gridN[:, 1], dTree.gridN[:, 0]))
    print("Same gridN", np.allclose(tree.gridN[orderN1], dTree.gridN[orderN2]))

    orderEx1 = np.lexsort((tree.gridEx[:, 1], tree.gridEx[:, 0]))
    orderEx2 = np.lexsort((dTree.gridEx[:, 1], dTree.gridEx[:, 0]))
    print("Same gridEx", np.allclose(tree.gridEx[orderEx1], dTree.gridEx[orderEx2]))

    orderEy1 = np.lexsort((tree.gridEy[:, 1], tree.gridEy[:, 0]))
    orderEy2 = np.lexsort((dTree.gridEy[:, 1], dTree.gridEy[:, 0]))
    print("Same gridEy", np.allclose(tree.gridEy[orderEy1], dTree.gridEy[orderEy2]))

    # Face Divergence test
    fx = lambda x, y: np.sin(2*np.pi*x)
    fy = lambda x, y: np.sin(2*np.pi*y)
    sol = lambda x, y: 2*np.pi*(np.cos(2*np.pi*x)+np.cos(2*np.pi*y))

    divF_ana = call2(sol, tree.gridCC)
    Fc1 = cartF2(tree, fx, fy)
    F1 = tree.projectFaceVector(Fc1)

    divF = tree.faceDiv.dot(F1)
    print('cppTree Dnorm:', np.linalg.norm(divF-divF_ana))

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
    print('cppTree Gnorm:', np.linalg.norm(gradE-gradE_ana))

    dG = dTree.nodalGrad
    fn2 = call2(fun, dTree.gridN)
    gradE_ana2 = dTree.projectEdgeVector(cartE2(dTree, solX, solY))
    gradE2 = dG.dot(fn2)

    print('TreeMesh Gnorm:', np.linalg.norm(gradE2-gradE_ana2))

    tree.plotGrid(showIt=True, edgesY=True)

if __name__=='__main__':
    go()
