import numpy as np
import matplotlib.pyplot as plt
from TreeMesh import TreeMesh as Tree
from time import time
from vtk_writer import writeVTK, write_points


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
    go()
    #test_io()
