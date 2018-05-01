import numpy as np
import matplotlib.pyplot as plt
from TreeMesh import TreeMesh as Tree
from time import time
from vtk_writer import writeVTK, write_points

from discretize import TreeMesh


call3 = lambda fun, xyz: fun(xyz[:, 0], xyz[:, 1], xyz[:, 2])
cart_row3 = lambda g, xfun, yfun, zfun: np.c_[call3(xfun, g), call3(yfun, g), call3(zfun, g)]
cartF3 = lambda M, fx, fy, fz: np.vstack((cart_row3(M.gridFx, fx, fy, fz), cart_row3(M.gridFy, fx, fy, fz), cart_row3(M.gridFz, fx, fy, fz)))
cartE3 = lambda M, ex, ey, ez: np.vstack((cart_row3(M.gridEx, ex, ey, ez), cart_row3(M.gridEy, ex, ey, ez), cart_row3(M.gridEz, ex, ey, ez)))

def go():
    nc = 8
    level = int(np.log2(nc))
    h = [nc, nc, nc]

    def func(cell):
        r = cell.center - np.array([0.5]*len(cell.center))
        dist = np.sqrt(r.dot(r))
        if dist < 0.25:
            return level
        return level-1

    #"""
    t1 = time()
    # tree = QuadTree(h, func, max_level=level)
    tree = Tree(h, levels=level)
    tree.refine(func)
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
    dTree.balance()
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

    grid1, grid2 = tree.gridCC, dTree.gridCC
    print("Same gridCC", np.allclose(grid1, grid2))

    grid1, grid2 = tree.gridN, dTree.gridN
    order1 = np.lexsort((grid1[:, 2], grid1[:, 1], grid1[:, 0]))
    order2 = np.lexsort((grid2[:, 2], grid2[:, 1], grid2[:, 0]))
    print("Same gridN", np.allclose(grid1[order1], grid2[order2]))

    grid1, grid2 = tree.gridEx, dTree.gridEx
    order1 = np.lexsort((grid1[:, 2], grid1[:, 1], grid1[:, 0]))
    order2 = np.lexsort((grid2[:, 2], grid2[:, 1], grid2[:, 0]))
    print("Same gridEx", np.allclose(grid1[order1], grid2[order2]))

    grid1, grid2 = tree.gridEy, dTree.gridEy
    order1 = np.lexsort((grid1[:, 2], grid1[:, 1], grid1[:, 0]))
    order2 = np.lexsort((grid2[:, 2], grid2[:, 1], grid2[:, 0]))
    print("Same gridEy", np.allclose(grid1[order1], grid2[order2]))

    grid1, grid2 = tree.gridEz, dTree.gridEz
    order1 = np.lexsort((grid1[:, 2], grid1[:, 1], grid1[:, 0]))
    order2 = np.lexsort((grid2[:, 2], grid2[:, 1], grid2[:, 0]))
    print("Same gridEz", np.allclose(grid1[order1], grid2[order2]))

    grid1, grid2 = tree.gridFx, dTree.gridFx
    order1 = np.lexsort((grid1[:, 2], grid1[:, 1], grid1[:, 0]))
    order2 = np.lexsort((grid2[:, 2], grid2[:, 1], grid2[:, 0]))
    print("Same gridFx", np.allclose(grid1[order1], grid2[order2]))

    grid1, grid2 = tree.gridFy, dTree.gridFy
    order1 = np.lexsort((grid1[:, 2], grid1[:, 1], grid1[:, 0]))
    order2 = np.lexsort((grid2[:, 2], grid2[:, 1], grid2[:, 0]))
    print("Same gridFy", np.allclose(grid1[order1], grid2[order2]))

    grid1, grid2 = tree.gridFz, dTree.gridFz
    order1 = np.lexsort((grid1[:, 2], grid1[:, 1], grid1[:, 0]))
    order2 = np.lexsort((grid2[:, 2], grid2[:, 1], grid2[:, 0]))
    print("Same gridFz", np.allclose(grid1[order1], grid2[order2]))

    # Face Divergence test
    fx = lambda x, y, z: np.sin(2*np.pi*x)
    fy = lambda x, y, z: np.sin(2*np.pi*y)
    fz = lambda x, y, z: np.sin(2*np.pi*z)
    sol = lambda x, y, z: (2*np.pi*np.cos(2*np.pi*x)+2*np.pi*np.cos(2*np.pi*y)+2*np.pi*np.cos(2*np.pi*z))

    Fc = cartF3(tree, fx, fy, fz)
    F = tree.projectFaceVector(Fc)

    divF = tree.faceDiv.dot(F)
    divF_ana = call3(sol, tree.gridCC)
    print('Tree Dnorm:', np.linalg.norm(divF-divF_ana))

    Fc = cartF3(dTree, fx, fy, fz)
    F = dTree.projectFaceVector(Fc)

    divF = dTree.faceDiv.dot(F)
    divF_ana = call3(sol, dTree.gridCC)

    print('TreeMesh Dnorm:', np.linalg.norm(divF-divF_ana))

    # Nodal gradient test
    fun = lambda x, y, z: (np.cos(x)+np.cos(y)+np.cos(z))
    # i (sin(x)) + j (sin(y)) + k (sin(z))
    solX = lambda x, y, z: -np.sin(x)
    solY = lambda x, y, z: -np.sin(y)
    solZ = lambda x, y, z: -np.sin(z)

    phi = call3(fun, tree.gridN)
    gradE = tree.nodalGrad.dot(phi)
    Ec = cartE3(tree, solX, solY, solZ)
    gradE_ana = tree.projectEdgeVector(Ec)
    print('Tree Gnorm:', np.linalg.norm(gradE-gradE_ana))

    phi = call3(fun, dTree.gridN)
    gradE = dTree.nodalGrad.dot(phi)
    Ec = cartE3(dTree, solX, solY, solZ)
    gradE_ana = dTree.projectEdgeVector(Ec)
    print('TreeMesh Gnorm:', np.linalg.norm(gradE-gradE_ana))
    # """

    funX = lambda x, y, z: np.cos(2*np.pi*y)
    funY = lambda x, y, z: np.cos(2*np.pi*z)
    funZ = lambda x, y, z: np.cos(2*np.pi*x)

    solX = lambda x, y, z: 2*np.pi*np.sin(2*np.pi*z)
    solY = lambda x, y, z: 2*np.pi*np.sin(2*np.pi*x)
    solZ = lambda x, y, z: 2*np.pi*np.sin(2*np.pi*y)

    Ec = cartE3(tree, funX, funY, funZ)
    E = tree.projectEdgeVector(Ec)
    Fc = cartF3(tree, solX, solY, solZ)
    curlE_ana = tree.projectFaceVector(Fc)
    C = tree.edgeCurl
    curlE = C.dot(E)
    print('Tree Cnorm:', np.linalg.norm((curlE - curlE_ana)))

    Ec = cartE3(dTree, funX, funY, funZ)
    E = dTree.projectEdgeVector(Ec)
    Fc = cartF3(dTree, solX, solY, solZ)
    curlE_ana = dTree.projectFaceVector(Fc)
    C = dTree.edgeCurl
    curlE = C.dot(E)
    print('TreeMesh Cnorm:', np.linalg.norm((curlE - curlE_ana)))

    tree.plotGrid(showIt=True, facesX=True)

def test_io():
    #meshFile = 'Horseshoe_SingleBlk7m_Center_Octree_Core6_4_2_500mPad_Reg.msh'

    meshFile = 'octree_mesh.txt'

    dTree = TreeMesh.readUBC(meshFile)
    print(dTree)

    tree = Tree.readUBC(meshFile)

    tree.writeUBC('test.txt')

    print(tree)

    print(tree.fill, dTree.fill)

    """
    t1 = time()
    C = dTree.edgeCurl
    t2 = time()
    print(t2-t1)
    #writeVTK(tree, 'MyTree.vtu')
    #dTree.writeVTK('Tree.vtu')
    #writeVTK(dTree, 'TreeMesh.vtu')
    """

if __name__=='__main__':
    # go()
    test_io()
