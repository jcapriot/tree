import numpy as np
import matplotlib.pyplot as plt
from tree_ext import QuadTree as Tree
from time import time

from discretize import TreeMesh


call3 = lambda fun, xyz: fun(xyz[:, 0], xyz[:, 1], xyz[:, 2])
cart_row3 = lambda g, xfun, yfun, zfun: np.c_[call3(xfun, g), call3(yfun, g), call3(zfun, g)]
cartF3 = lambda M, fx, fy, fz: np.vstack((cart_row3(M.gridFx, fx, fy, fz), cart_row3(M.gridFy, fx, fy, fz), cart_row3(M.gridFz, fx, fy, fz)))
cartE3 = lambda M, ex, ey, ez: np.vstack((cart_row3(M.gridEx, ex, ey, ez), cart_row3(M.gridEy, ex, ey, ez), cart_row3(M.gridEz, ex, ey, ez)))

def go():
    nc = 16
    level = int(np.log2(nc))
    h = [nc, nc, nc]

    def func(cell):
        r = cell.center - np.array([0.5]*len(cell.center))
        dist = np.sqrt(r.dot(r))
        if dist < 0.2:
            return level
        return level-1

    #"""
    t1 = time()
    # tree = QuadTree(h, func, max_level=level)
    tree = Tree(h, max_level=level)
    tree.build_tree(func)
    tree.number()
    t2 = time()
    print('Tree Construction time:', t2-t1)
    print("nC", tree.nC)
    print("ntN", tree.ntN)
    print("nhN", tree.nhN)
    print("ntE", tree.ntEx, tree.ntEy, tree.ntEz)
    print("nhE", tree.nhEx, tree.nhEy, tree.nhEz)

    print("ntF", tree.ntFx, tree.ntFy, tree.ntFz)
    print("nhF", tree.nhFx, tree.nhFy, tree.nhFz)

    print("nN", tree.nN)
    print("nE", tree.nEx, tree.nEy, tree.nEz)
    print("nF", tree.nFx, tree.nFy, tree.nFz)
    #"""
    t1 = time()
    dTree = TreeMesh(h, levels=level)
    dTree.refine(func, balance=True)
    dTree.number(balance=True)
    t2 = time()
    print('TreeMesh Construction time', t2-t1)
    print("nC", dTree.nC)
    print("ntN", dTree.ntN)
    print("nhN", dTree.nhN)
    print("ntE", dTree.ntEx, dTree.ntEy, dTree.ntEz)
    print("nhE", dTree.nhEx, dTree.nhEy, dTree.nhEz)

    print("ntF", dTree.ntFx, dTree.ntFy, dTree.ntFz)
    print("nhF", dTree.nhFx, dTree.nhFy, dTree.nhFz)

    print("nN", dTree.nN)
    print("nE", dTree.nEx, dTree.nEy, dTree.nEz)
    print("nF", dTree.nFx, dTree.nFy, dTree.nFz)

    print(np.allclose(tree.gridCC, dTree.gridCC))
    print(np.allclose(np.sort(tree.gridN), np.sort(tree.gridN)))
    print(np.allclose(np.sort(tree.gridFx), np.sort(tree.gridFx)))
    print(np.allclose(np.sort(tree.gridFy), np.sort(tree.gridFy)))
    print(np.allclose(np.sort(tree.gridFz), np.sort(tree.gridFz)))
    print(np.allclose(np.sort(tree.gridEx), np.sort(tree.gridEx)))
    print(np.allclose(np.sort(tree.gridEy), np.sort(tree.gridEy)))
    print(np.allclose(np.sort(tree.gridEz), np.sort(tree.gridEz)))

    """
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
    """

if __name__=='__main__':
    go()
