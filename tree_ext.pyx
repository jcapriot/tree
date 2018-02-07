cimport cython
cimport numpy as np
from libc.math cimport sqrt,abs

from tree cimport int_t, QuadTree as c_QuadTree, PyWrapper, Node, Edge, QuadCell

import scipy.sparse as sp
import numpy as np

cdef class Cell:
    cdef double _x,_y,_x0,_y0,_wx,_wy
    def __cinit__(self, double x, double y, double x0, double y0):
        self._x = x
        self._y = y
        self._x0 = x0
        self._y0 = y0
        self._wx = 2*(x-x0)
        self._wy = 2*(y-y0)

    @property
    def center(self):
        return tuple((self._x,self._y))

    @property
    def x0(self):
        return tuple((self._x0,self._y0))

    @property
    def width(self):
        return tuple((self._wx,self._wy))

cdef int _evaluate_func(void* obj, void* function, QuadCell* cell) with gil:
    self = <object> obj
    func = <object> function
    cdef double scale_x,scale_y
    scale_x = self.scale_x
    scale_y = self.scale_y
    cdef double shift_x, shift_y
    shift_x = self.x0[0]
    shift_y = self.x0[1]

    cdef double x = cell.center[0]*scale_x+shift_x
    cdef double y = cell.center[1]*scale_y+shift_y
    cdef double x0 = cell.points[0].location[0]*scale_x + shift_x
    cdef double y0 = cell.points[0].location[1]*scale_y + shift_y
    pycell = Cell(x,y,x0,y0)
    return <int> func(pycell)

cdef struct double4:
    double w0,w1,w2,w3

cdef void get_weights(double4 weights, double x, double y, QuadCell *cells[4], int_t n_cells):
    cdef int_t p0x,p1x,p2x,p3x
    cdef int_t p0y,p1y,p2y,p3y
    cdef double a[4]
    cdef double b[4]
    cdef double aa,bb,cc,l,m

    weights.w0 = 0
    weights.w1 = 0
    weights.w2 = 0
    weights.w3 = 0
    #triangle
    if n_cells == 3:
        p0x = cells[0].center[0]
        p1x = cells[1].center[0]
        p2x = cells[2].center[0]
        p0y = cells[0].center[1]
        p1y = cells[1].center[1]
        p2y = cells[2].center[1]
        aa = (-p1y*p2x+p0y*(p2x-p1x)+p0x*(p1y-p2y)+p1x*p2y)
        l = (p0y*p2x-p0x*p2y+(p2y-p0y)*x+(p0x-p2x)*y)/aa
        m = (p0x*p1y-p0y*p1x+(p0y-p1y)*x+(p1x-p0x)*y)/aa
        weights.w0 = l
        weights.w1 = m
        weights.w2 = 1-l-m

    #Quadrilateral
    elif cells[1]==NULL or cells[2]==NULL or cells[3]==NULL:
    #point was outside
        pass
    else:
        #n_cells == 4:
        p0x = cells[0].center[0]
        p1x = cells[1].center[0]
        p2x = cells[2].center[0]
        p3x = cells[3].center[0]
        p0y = cells[0].center[1]
        p1y = cells[1].center[0]
        p2y = cells[2].center[0]
        p3y = cells[3].center[0]
        a[0] = p0x
        a[1] = p1x-p0x
        a[2] = p3x-p0x
        a[3] = p0x-p1x+p2x-p3x
        b[0] = p0y
        b[1] = p1y-p0y
        b[2] = p3y-p0y
        b[3] = p0y-p1y+p2y-p3y
        aa = a[3]*b[2]-a[2]*b[3]
        bb = a[3]*b[0]-a[0]*b[3]+a[1]*b[2]-a[2]*b[1]+x*b[3]-y*a[3]
        cc = a[1]*b[0]-a[0]*b[1]+x*b[1]-y*a[1]

        m = (-bb+sqrt(bb*bb - 4*aa*cc))/(2*aa)
        l = (x-a[0]-a[2]*m)/(a[1]+a[3]*m)

        weights.w0=(1-l)*(1-m)
        weights.w1=l*(1-m)
        weights.w2=l*m
        weights.w3=(1-l)*m

cdef inline int sign(double val):
    return (0<val)-(val<0)

cdef class _QuadTree:
    cdef c_QuadTree *tree
    cdef PyWrapper *wrapper
    cdef int_t _nx, _ny, max_level
    cdef double _scale_x, _scale_y

    cdef object _gridCC, _gridN, _gridEx, _gridEy
    cdef object _gridhN, _gridhEx, _gridhEy
    cdef object _aveFx2CC, _aveFy2CC, _aveF2CC, _aveF2CCV,_aveN2CC,_aveE2CC,_aveE2CCV

    def __cinit__(self):
        self.wrapper = new PyWrapper()
        self.tree = new c_QuadTree()

    def __init__(self, max_level, x0, w):
        self.max_level = max_level
        self._nx = 2<<max_level
        self._ny = 2<<max_level

        self._scale_x = w[0]/self._nx
        self._scale_y = w[1]/self._ny

        self.tree.set_level(self.max_level)

        self._gridCC = None
        self._gridN = None
        self._gridhN = None
        self._gridEx = None
        self._gridEy = None
        self._gridhEx = None
        self._gridhEy = None
        self._aveFx2CC = None
        self._aveFy2CC = None
        self._aveF2CC = None
        self._aveF2CCV = None
        self._aveE2CC = None
        self._aveE2CCV = None
        self._aveN2CC = None

    def build_tree(self,test_function):
        #Wrapping test_function so it can be call in c++
        cdef void * self_ptr
        cdef void * func_ptr;
        self_ptr = <void *> self;
        func_ptr = <void *> test_function;
        self.wrapper.set(self_ptr, func_ptr, _evaluate_func)

        #Then tell c++ to build the tree
        self.tree.build_tree(self.wrapper)

    def number(self):
        self.tree.number()

    @property
    def dim(self):
        return 2

    @property
    def nx(self):
        return self._nx

    @property
    def ny(self):
        return self._ny

    @property
    def scale_x(self):
        return self._scale_x

    @property
    def scale_y(self):
        return self._scale_y

    @property
    def nC(self):
        return self.tree.cells.size()

    @property
    def nN(self):
        return self.ntN-self.nhN

    @property
    def ntN(self):
        return self.tree.nodes.size()

    @property
    def nhN(self):
        return self.tree.hanging_nodes.size()

    @property
    def nE(self):
        return self.nEx+self.nEy

    @property
    def nhE(self):
        return self.nhEx+self.nhEy

    @property
    def ntE(self):
        return self.ntEx+self.ntEy

    @property
    def nEx(self):
        return self.ntEx-self.nhEx

    @property
    def nEy(self):
        return self.ntEy-self.nhEy

    @property
    def ntEx(self):
        return self.tree.edges_x.size()

    @property
    def ntEy(self):
        return self.tree.edges_y.size()

    @property
    def nhEx(self):
        return self.tree.hanging_edges_x.size()

    @property
    def nhEy(self):
        return self.tree.hanging_edges_y.size()

    @property
    def nF(self):
        return self.nE

    @property
    def nhF(self):
        return self.nhE

    @property
    def ntF(self):
        return self.ntE

    @property
    def nFx(self):
        return self.nEy

    @property
    def nFy(self):
        return self.nEx

    @property
    def ntFx(self):
        return self.ntEy

    @property
    def ntFy(self):
        return self.ntEx

    @property
    def nhFx(self):
        return self.nhEy

    @property
    def nhFy(self):
        return self.nhEx

    @property
    def gridN(self):
        cdef np.float64_t[:,:] gridN
        cdef Node *node
        cdef np.int64_t ii;
        cdef double shift_x,shift_y
        shift_x = self.x0[0]
        shift_y = self.x0[1]
        if self._gridN is None:
            self._gridN = np.empty((self.nN,2),dtype=np.float64)
            gridN = self._gridN
            for it in self.tree.nodes:
                node = it.second
                if not node.hanging:
                    ii = node.index
                    gridN[ii,0] = node.location[0]*self._scale_x + shift_x
                    gridN[ii,1] = node.location[1]*self._scale_y + shift_y
        return self._gridN

    @property
    def gridhN(self):
        cdef np.float64_t[:,:] gridN
        cdef Node *node
        cdef np.int64_t ii;
        cdef double shift_x, shift_y
        shift_x = self.x0[0]
        shift_y = self.x0[1]
        if self._gridhN is None:
            self._gridhN = np.empty((self.nhN,2),dtype=np.float64)
            gridhN = self._gridhN
            for it in self.tree.nodes:
                node = it.second
                if node.hanging:
                    ii = node.index-self.nN
                    gridhN[ii,0] = node.location[0]*self._scale_x + shift_x
                    gridhN[ii,1] = node.location[1]*self._scale_y + shift_y
        return self._gridhN

    @property
    def gridCC(self):
        cdef np.float64_t[:,:] gridCC
        cdef np.int64_t ii;
        cdef double shift_x,shift_y
        shift_x = self.x0[0]
        shift_y = self.x0[1]
        if self._gridCC is None:
            self._gridCC = np.empty((self.nC,2),dtype=np.float64)
            gridCC = self._gridCC
            for cell in self.tree.cells:
                ii = cell.index
                gridCC[ii,0] = cell.center[0]*self._scale_x + shift_x
                gridCC[ii,1] = cell.center[1]*self._scale_y + shift_y
        return self._gridCC

    @property
    def gridEx(self):
        cdef np.float64_t[:,:] gridEx
        cdef Edge *edge
        cdef np.int64_t ii;
        cdef double shift_x,shift_y
        shift_x = self.x0[0]
        shift_y = self.x0[1]
        if self._gridEx is None:
            self._gridEx = np.empty((self.nEx,2),dtype=np.float64)
            gridEx = self._gridEx
            for it in self.tree.edges_x:
                edge = it.second
                if not edge.hanging:
                    ii = edge.index
                    gridEx[ii,0] = edge.location[0]*self._scale_x + shift_x
                    gridEx[ii,1] = edge.location[1]*self._scale_y + shift_y
        return self._gridEx

    @property
    def gridhEx(self):
        cdef np.float64_t[:,:] gridhEx
        cdef Edge *edge
        cdef np.int64_t ii;
        cdef double shift_x,shift_y
        shift_x = self.x0[0]
        shift_y = self.x0[1]
        if self._gridhEx is None:
            self._gridhEx = np.empty((self.nhEx,2),dtype=np.float64)
            gridhEx = self._gridhEx
            for it in self.tree.edges_x:
                edge = it.second
                if edge.hanging:
                    ii = edge.index-self.nEx
                    gridhEx[ii,0] = edge.location[0]*self._scale_x + shift_x
                    gridhEx[ii,1] = edge.location[1]*self._scale_y + shift_y
        return self._gridhEx

    @property
    def gridEy(self):
        cdef np.float64_t[:,:] gridEy
        cdef Edge *edge
        cdef np.int64_t ii;
        cdef double shift_x,shift_y
        shift_x = self.x0[0]
        shift_y = self.x0[1]
        if self._gridEy is None:
            self._gridEy = np.empty((self.nEy,2),dtype=np.float64)
            gridEy = self._gridEy
            for it in self.tree.edges_y:
                edge = it.second
                if not edge.hanging:
                    ii = edge.index
                    gridEy[ii,0] = edge.location[0]*self._scale_x + shift_x
                    gridEy[ii,1] = edge.location[1]*self._scale_y + shift_y
        return self._gridEy

    @property
    def gridhEy(self):
        cdef np.float64_t[:,:] gridhEy
        cdef Edge *edge
        cdef np.int64_t ii;
        cdef double shift_x,shift_y
        shift_x = self.x0[0]
        shift_y = self.x0[1]
        if self._gridhEy is None:
            self._gridhEy = np.empty((self.nhEy,2),dtype=np.float64)
            gridhEy = self._gridhEy
            for it in self.tree.edges_y:
                edge = it.second
                if edge.hanging:
                    ii = edge.index-self.nEy
                    gridhEy[ii,0] = edge.location[0]*self._scale_x + shift_x
                    gridhEy[ii,1] = edge.location[1]*self._scale_y + shift_y
        return self._gridhEy

    @property
    def gridFx(self):
        return self.gridEy

    @property
    def gridFy(self):
        return self.gridEx

    @property
    def gridhFx(self):
        return self.gridhEy

    @property
    def gridhFy(self):
        return self.gridhEx

    @property
    def faceDiv(self):
        cdef np.int64_t[:] I = np.empty(self.nC*4,dtype=np.int64)
        cdef np.int64_t[:] J = np.empty(self.nC*4,dtype=np.int64)
        cdef np.float64_t[:] V = np.empty(self.nC*4,dtype=np.float64)

        cdef np.int64_t i = 0
        cdef Edge *edges[4]
        cdef np.int64_t offset = self.tree.edges_y.size()
        cdef double volume

        for cell in self.tree.cells:
            edges = cell.edges
            I[i*4:i*4+4] = cell.index
            J[i*4  ] = edges[0].index #y edge, x face
            J[i*4+1] = edges[1].index+offset #x face, y face (add offset)
            J[i*4+2] = edges[2].index #y edge, x face
            J[i*4+3] = edges[3].index+offset #x edge, y face (add offset)

            volume = (edges[0].length*self._scale_y)*(edges[1].length*self._scale_x)
            V[i*4  ] = -(edges[0].length*self._scale_y/volume)
            V[i*4+1] = edges[1].length*self._scale_x/volume
            V[i*4+2] = edges[2].length*self._scale_y/volume
            V[i*4+3] = -(edges[3].length*self._scale_x/volume)
            i += 1

        R = self.deflate_faces()
        return sp.csr_matrix((V,(I,J)))*R

    @property
    def cellGrad(self):
        cdef np.int64_t[:] I = np.empty(self.nC*4,dtype=np.int64)
        cdef np.int64_t[:] J = np.empty(self.nC*4,dtype=np.int64)
        cdef np.float64_t[:] V = np.empty(self.nC*4,dtype=np.float64)

        cdef np.int64_t i = 0
        cdef Edge *edges[4]
        cdef np.int64_t offset = self.tree.edges_y.size()
        cdef double volume

        for cell in self.tree.cells:
            edges = cell.edges
            I[i*4:i*4+4] = cell.index
            J[i*4  ] = edges[0].index #y edge, x face
            J[i*4+1] = edges[1].index+offset #x face, y face (add offset)
            J[i*4+2] = edges[2].index #y edge, x face
            J[i*4+3] = edges[3].index+offset #x edge, y face (add offset)

            V[i*4  ] = edges[0].length*self._scale_y
            V[i*4+1] = -(edges[1].length*self._scale_x)
            V[i*4+2] = -(edges[2].length*self._scale_y)
            V[i*4+3] = edges[3].length*self._scale_x
            i += 1

        R = self.deflate_faces()

        G = R.T*sp.csr_matrix((V,(J,I)))
        G = G.tocsr()
        cdef np.int64_t[:] indptr = G.indptr
        cdef np.float64_t[:] data = G.data
        cdef double dx
        for i in range(indptr.shape[0]-1):
            if(indptr[i+1]-indptr[i]!=2):
                dx = abs(data[indptr[i]])
            else:
                dx = abs(data[indptr[i]])+abs(data[indptr[i+1]])/2
                data[indptr[i]] = sign(data[indptr[i]])/dx
                data[indptr[i+1]] = sign(data[indptr[i+1]])/dx
        return G

    def deflate_faces(self):
        #I is output index (with hanging)
        #J is input index (without hanging)
        cdef np.int64_t[:] I = np.empty(self.ntE,dtype=np.int64)
        cdef np.int64_t[:] J = np.empty(self.ntE,dtype=np.int64)
        cdef np.float64_t[:] V = np.empty(self.ntE,dtype=np.float64)
        cdef Edge *edge
        cdef np.int64_t ii;
        # x faces = y edges
        for it in self.tree.edges_y:
            edge = it.second
            ii = edge.index
            I[ii] = ii
            if edge.hanging:
                J[ii] = edge.parent.index
            else:
                J[ii] = ii
            V[ii] = 1.0

        cdef np.int64_t offset1 = self.ntEy
        cdef np.int64_t offset2 = self.nEy
        # y faces = x edges
        for it in self.tree.edges_x:
            edge = it.second
            ii = edge.index+offset1
            I[ii] = ii
            if edge.hanging:
                J[ii] = edge.parent.index+offset2
            else:
                J[ii] = edge.index+offset2
            V[ii] = 1.0

        return sp.csr_matrix((V,(I,J)))

    def deflate_faces_x(self):
        #I is output index (with hanging)
        #J is input index (without hanging)
        cdef np.int64_t[:] I = np.empty(self.ntEx,dtype=np.int64)
        cdef np.int64_t[:] J = np.empty(self.ntEx,dtype=np.int64)
        cdef np.float64_t[:] V = np.empty(self.ntEx,dtype=np.float64)
        cdef Edge *edge
        cdef np.int64_t ii;
        # x faces = y edges
        for it in self.tree.edges_y:
            edge = it.second
            ii = edge.index
            I[ii] = ii
            if edge.hanging:
                J[ii] = edge.parent.index
            else:
                J[ii] = ii
            V[ii] = 1.0
        return sp.csr_matrix((V,(I,J)))

    def deflate_faces_y(self):
        #I is output index (with hanging)
        #J is input index (without hanging)
        cdef np.int64_t[:] I = np.empty(self.ntEx,dtype=np.int64)
        cdef np.int64_t[:] J = np.empty(self.ntEx,dtype=np.int64)
        cdef np.float64_t[:] V = np.empty(self.ntEx,dtype=np.float64)
        cdef Edge *edge
        cdef np.int64_t ii;
        # y faces = x edges
        for it in self.tree.edges_x:
            edge = it.second
            ii = edge.index
            I[ii] = ii
            if edge.hanging:
                J[ii] = edge.parent.index
            else:
                J[ii] = ii
            V[ii] = 1.0
        return sp.csr_matrix((V,(I,J)))

    @property
    def nodalGrad(self):
        cdef np.int64_t[:] I = np.empty(self.nE*2,dtype=np.int64)
        cdef np.int64_t[:] J = np.empty(self.nE*2,dtype=np.int64)
        cdef np.float64_t[:] V = np.empty(self.nE*2,dtype=np.float64)

        cdef Edge *edge
        cdef np.int64_t i
        cdef double length
        cdef int_t ii

        for it in self.tree.edges_x:
            edge = it.second
            if edge.hanging:
                continue
            ii = edge.index
            I[ii*2:ii*2+2] = ii
            J[ii*2] = edge.points[0].index
            J[ii*2+1] = edge.points[1].index

            length = edge.length*self._scale_x
            V[ii*2] = -1.0/length
            V[ii*2+1] = 1.0/length

        cdef np.int64_t offset1 = self.nEx
        for it in self.tree.edges_y:
            edge = it.second
            if edge.hanging:
                continue
            ii = edge.index+offset1
            I[ii*2:ii*2+2] = ii
            J[ii*2] = edge.points[0].index
            J[ii*2+1] = edge.points[1].index

            length = edge.length*self._scale_y
            V[ii*2] = -1.0/length
            V[ii*2+1] = 1.0/length

        Rn = self.deflate_nodes()
        G = sp.csr_matrix((V,(I,J)))
        return G*Rn

    def deflate_nodes(self):
        cdef np.int64_t[:] I = np.empty(self.nN+2*self.nhN,dtype=np.int64)
        cdef np.int64_t[:] J = np.empty(self.nN+2*self.nhN,dtype=np.int64)
        cdef np.float64_t[:] V = np.empty(self.nN+2*self.nhN,dtype=np.float64)

        # I is output index
        # J is input index
        cdef Node *node
        cdef np.int64_t ii
        cdef np.int64_t offset = self.nN

        for it in self.tree.nodes:
            node = it.second
            if node.hanging:
                ii = 2*node.index-offset

                I[ii:ii+2] = node.index
                J[ii] = node.parents[0].index
                J[ii+1] = node.parents[1].index
                V[ii:ii+2] = 0.5
            else:
                ii = node.index
                I[ii] = ii
                J[ii] = ii
                V[ii] = 1.0

        return sp.csr_matrix((V,(I,J)))


    @property
    def aveFx2CC(self):
        cdef np.int64_t[:] I,J
        cdef np.float64_t[:] V
        cdef Edge *edge1
        cdef Edge *edge2
        cdef np.int64_t ii
        if self._aveFx2CC is None:
            I = np.empty(self.nC*2,dtype=np.int64)
            J = np.empty(self.nC*2,dtype=np.int64)
            V = np.empty(self.nC*2,dtype=np.float64)
            for cell in self.tree.cells:
                edge1 = cell.edges[0] #y edge/ x face
                edge2 = cell.edges[2] #y edge/ x face
                ii = cell.index
                I[ii*2:ii*2+2] = ii
                J[ii*2] = edge1.index
                J[ii*2+1] = edge2.index
                V[ii*2:ii*2+2] = 0.5
            Rfx = self.deflate_faces_x()
            self._aveFx2CC = sp.csr_matrix((V,(I,J)))*Rfx
        return self._aveFx2CC

    @property
    def aveFy2CC(self):
        cdef np.int64_t[:] I,J
        cdef np.float64_t[:] V
        cdef Edge *edge1
        cdef Edge *edge2
        cdef np.int64_t ii
        if self._aveFy2CC is None:
            I = np.empty(self.nC*2,dtype=np.int64)
            J = np.empty(self.nC*2,dtype=np.int64)
            V = np.empty(self.nC*2,dtype=np.float64)
            for cell in self.tree.cells:
                edge1 = cell.edges[1] #x edge/ y face
                edge2 = cell.edges[3] #x edge/ y face
                ii = cell.index
                I[ii*2:ii*2+2] = ii
                J[ii*2] = edge1.index
                J[ii*2+1] = edge2.index
                V[ii*2:ii*2+2] = 0.5
            Rfy = self.deflate_faces_y()
            self._aveFy2CC = sp.csr_matrix((V,(I,J)))*Rfy
        return self._aveFy2CC

    @property
    def aveF2CC(self):
        if self._aveF2CC is None:
            self._aveF2CC = 1.0/2*sp.hstack([self.aveFx2CC,self.aveFy2CC]).tocsr()
        return self._aveF2CC

    @property
    def aveF2CCV(self):
        if self._aveF2CCV is None:
            self._aveF2CCV = sp.block_diag([self.aveFx2CC,self.aveFy2CC]).tocsr()
        return self._aveF2CCV

    @property
    def aveEx2CC(self):
        return self.aveFy2CC

    @property
    def aveEy2CC(self):
        return self.aveFx2CC

    @property
    def aveE2CC(self):
        if self._aveE2CC is None:
            self._aveE2CC = 1.0/2*sp.hstack([self.aveEx2CC,self.aveEy2CC]).tocsr()
        return self._aveE2CC

    @property
    def aveE2CCV(self):
        if self._aveE2CCV is None:
            self._aveE2CCV = sp.block_diag([self.aveEx2CC,self.aveEy2CC]).tocsr()
        return self._aveE2CCV

    @property
    def aveN2CC(self):
        cdef np.int64_t[:] I,J
        cdef np.float64_t[:] V
        cdef np.int64_t ii
        if self._aveN2CC is None:
            I = np.empty(self.nC*4,dtype=np.int64)
            J = np.empty(self.nC*4,dtype=np.int64)
            V = np.empty(self.nC*4,dtype=np.float64)
            for cell in self.tree.cells:
                ii = cell.index
                I[ii*4:ii*4+4] = ii
                J[ii*4  ] = cell.points[0].index
                J[ii*4+1] = cell.points[1].index
                J[ii*4+2] = cell.points[2].index
                J[ii*4+3] = cell.points[3].index
                V[ii*4:ii*4+4] = 0.25
            Rn = self.deflate_nodes()
            self._aveN2CC = sp.csr_matrix((V,(I,J)))*Rn
        return self._aveN2CC

    def getInterpolationMat(self, locs, locType, zerosOutside = False):
        pass

    def _getNodeIntMat(self,locs):
        cdef:
            double[:,:] locations = locs
            int_t n_loc = locs.shape[0]
            np.int64_t[:] I = np.empty(n_loc*4,dtype=np.int64)
            np.int64_t[:] J = np.empty(n_loc*4,dtype=np.int64)
            np.float64_t[:] V = np.empty(n_loc*4,dtype=np.float64)

            int_t ii,i
            QuadCell *cell
            double x,y, scale_x, scale_y, shift_x, shift_y
            double wx,wy

        scale_x = self._scale_x
        scale_y = self._scale_y
        shift_x = self.x0[0]
        shift_y = self.x0[1]
        for i in range(n_loc):
            x = (locations[i,0]-shift_x)/scale_x
            y = (locations[i,1]-shift_y)/scale_y
            #get containing (or closest) cell
            cell = self.tree.containing_cell(x,y)
            #calculate weights
            wx = ((cell.points[3].location[0]-x)/
                  (cell.points[3].location[0]-cell.points[0].location[0]))
            wy = ((cell.points[3].location[1]-y)/
                  (cell.points[3].location[1]-cell.points[0].location[1]))
            I[4*i:4*i+4] = i

            J[4*i] = cell.points[0].index
            J[4*i+1] = cell.points[1].index
            J[4*i+2] = cell.points[2].index
            J[4*i+2] = cell.points[3].index

            V[4*i] = wx*wy
            V[4*i+1] = (1-wx)*wy
            V[4*i+2] = wx*(1-wy)
            V[4*i+2] = (1-wx)*(1-wy)

        Rn = self.deflate_nodes()
        return sp.csr((V,(I,J)))*Rn

    def _getCellCenterIntMat(self,locs):
        cdef:
            double[:,:] locations = locs
            int_t n_loc = locs.shape[0]
            np.int64_t[:] I = np.empty(n_loc*4,dtype=np.int64)
            np.int64_t[:] J = np.empty(n_loc*4,dtype=np.int64)
            np.float64_t[:] V = np.empty(n_loc*4,dtype=np.float64)

            int_t ii,i,n_cells
            QuadCell *cell
            QuadCell *test_cell
            double x,y, scale_x, scale_y, shift_x, shift_y
            double4 weights
        scale_x = self._scale_x
        scale_y = self._scale_y
        shift_x = self.x0[0]
        shift_y = self.x0[1]

        cdef QuadCell *cells[4] #closest cells to a point

        for i in range(n_loc):
            x = (locations[i,0]-shift_x)/scale_x
            y = (locations[i,1]-shift_y)/scale_y
            #get containing (or closest) cell
            cell = self.tree.containing_cell(x,y)
            cells[0] = cell
            cells[3] = NULL
            n_cells = 0
            #first check if it is in any triangle...
            if cell.neighbors[0] != NULL and x<cell.center[0]:
                if cell.neighbors[0].is_leaf() and cell.inside_triangle(x,y,0):
                    cells[1] = cell.neighbors[0].children[1]
                    cells[2] = cell.neighbors[0].children[3]
                    n_cells=3
            elif cell.neighbors[1] != NULL and x>cell.center[0]:
                if cell.neighbors[1].is_leaf() and cell.inside_triangle(x,y,1):
                    cells[1] = cell.neighbors[1].children[0]
                    cells[2] = cell.neighbors[1].children[2]
                    n_cells=3
            elif cell.neighbors[2] != NULL and y<cell.center[1]:
                if cell.neighbors[2].is_leaf() and cell.inside_triangle(x,y,2):
                    cells[1] = cell.neighbors[2].children[2]
                    cells[2] = cell.neighbors[2].children[3]
                    n_cells=3
            elif cell.neighbors[3] != NULL and y>cell.center[1]:
                if cell.neighbors[3].is_leaf() and cell.inside_triangle(x,y,3):
                    cells[1] = cell.neighbors[3].children[0]
                    cells[2] = cell.neighbors[3].children[1]
                    n_cells=3
            if n_cells != 0:
                n_cells = 4
                #then it is in a quadrilateral
                if x<cell.center[0]:
                    if y<cell.center[1]: #lower left
                        cells[1] = cell.neighbors[2]
                        if cells[1] != NULL and not cells[1].is_leaf():
                            cells[1] = cells[1].children[2]

                        cells[2] = cells[1].neighbors[0]
                        if cells[2] != NULL and not cells[2].is_leaf():
                            cells[2] = cells[1].neighbors[0].children[3]

                        cells[3] = cell.neighbors[0]
                        if cells[3] != NULL and not cells[3].is_leaf():
                            cells[3] = cells[3].children[1]
                    else: #upper left
                        cells[1] = cell.neighbors[0]
                        if cells[1] != NULL and not cells[1].is_leaf():
                            cells[1] = cells[1].children[3]

                        cells[2] = cells[1].neighbors[3]
                        if cells[2] != NULL and not cells[2].is_leaf():
                            cells[2] = cells[1].neighbors[3].children[1]

                        cells[3] = cell.neighbors[3]
                        if cells[3] != NULL and not cells[3].is_leaf():
                            cells[3] = cells[3].children[0]
                else:
                    if y<cell.center[1]: #lower right
                        cells[1] = cell.neighbors[1]
                        if cells[1] != NULL and not cells[1].is_leaf():
                            cells[1] = cells[1].children[0]

                        cells[2] = cells[1].neighbors[2]
                        if cells[2] != NULL and not cells[2].is_leaf():
                            cells[2] = cells[1].neighbors[2].children[2]

                        cells[3] = cell.neighbors[2]
                        if cells[3] != NULL and not cells[3].is_leaf():
                            cells[3] = cells[3].children[3]
                    else: #upper right
                        cells[1] = cell.neighbors[3]
                        if cells[1] != NULL and not cells[1].is_leaf():
                            cells[1] = cells[1].children[1]

                        cells[2] = cells[1].neighbors[1]
                        if cells[2] != NULL and not cells[2].is_leaf():
                            cells[2] = cells[1].neighbors[1].children[0]

                        cells[3] = cell.neighbors[1]
                        if cells[3] != NULL and not cells[3].is_leaf():
                            cells[3] = cells[3].children[2]
            #now cells contains a list of the 3 or 4
            #bounding cells to interpolate from
            #just need to figure out the weights
            get_weights(weights, x, y, cells, n_cells)
            I[4*i:4*i+4] = i

            if cells[0] != NULL: J[4*i] = cells[0].index
            if cells[1] != NULL: J[4*i+1] = cells[1].index
            if cells[2] != NULL: J[4*i+2] = cells[2].index
            if cells[3] != NULL: J[4*i+2] = cells[3].index

            V[4*i] = weights.w0
            V[4*i+1] = weights.w1
            V[4*i+2] = weights.w2
            V[4*i+2] = weights.w3

        return sp.csr_matrix((V,(I,J)))

    def _getFaceXIntMat(self,locs):
        cdef double[:,:] locations = locs
        cdef n_loc = locs.shape[0]
        cdef np.int64_t[:] I = np.empty(n_loc*2,dtype=np.int64)
        cdef np.int64_t[:] J = np.empty(n_loc*2,dtype=np.int64)
        cdef np.float64_t[:] V = np.empty(n_loc*2,dtype=np.float64)

        cdef int_t ii,i
        cdef QuadCell *cell
        cdef double x,y
        for i in range(n_loc):
            x = locations[i,0]
            y = locations[i,1]
            cell = self.tree.containing_cell(x,y)

    def plotGrid(self, ax=None, showIt=False,
        grid=True,
        cells=False, cellLine=False,
        nodes = False,
        facesX = False, facesY = False,
        edgesX = False, edgesY = False):

        if ax is None:
            import matplotlib.pyplot as plt
            ax = plt.subplot(111)
        else:
            fig = ax.figure

        cdef:
            double scale_x = self._scale_x
            double scale_y = self._scale_y
            double shift_x = self.x0[0]
            double shift_y = self.x0[1]
            Node *p1
            Node *p2
            Node *p3
            Node *p4

        if grid:
            X,Y = [],[]
            for cell in self.tree.cells:
                p1 = cell.points[0]
                p2 = cell.points[1]
                p3 = cell.points[2]
                p4 = cell.points[3]
                X.extend([p1.location[0]*scale_x+shift_x,
                          p3.location[0]*scale_x+shift_x,
                          p4.location[0]*scale_x+shift_x,
                          p2.location[0]*scale_x+shift_x,
                          p1.location[0]*scale_x+shift_x,
                          np.nan])
                Y.extend([p1.location[1]*scale_y+shift_y,
                          p3.location[1]*scale_y+shift_y,
                          p4.location[1]*scale_y+shift_y,
                          p2.location[1]*scale_y+shift_y,
                          p1.location[1]*scale_y+shift_y,
                          np.nan])
            ax.plot(X,Y,'b-')

        if cells:
            ax.plot(self.gridCC[:,0], self.gridCC[:,1], 'r.')
        if cellLine:
            ax.plot(self.gridCC[:,0], self.gridCC[:,1], 'r:')
            ax.plot(self.gridCC[[0,-1],0], self.gridCC[[0,-1],1], 'ro')
        if nodes:
            ax.plot(self.gridN[:,0], self.gridN[:,1], 'ms')
            # Hanging Nodes
            ax.plot(self.gridhN[:,0], self.gridhN[:,1], 'ms')
            ax.plot(self.gridhN[:,0], self.gridhN[:,1], 'ms', ms=10, mfc='none', mec='m')
        if facesX:
            ax.plot(self.gridFx[:,0], self.gridFx[:,1], 'g>')
            # Hanging Faces x
            ax.plot(self.gridhFx[:,0], self.gridhFx[:,1], 'g>')
            ax.plot(self.gridhFx[:,0], self.gridhFx[:,1], 'gs', ms=10, mfc='none', mec='g')
        if facesY:
            ax.plot(self.gridFy[:,0], self.gridFy[:,1], 'g^')
            # Hanging Faces y
            ax.plot(self.gridhFy[:,0], self.gridhFy[:,1], 'g^')
            ax.plot(self.gridhFy[:,0], self.gridhFy[:,1], 'gs', ms=10, mfc='none', mec='g')
        ax.set_xlabel('x1')
        ax.set_ylabel('x2')

        ax.grid(True)

    def __dealloc__(self):
        del self.tree
        del self.wrapper

from discretize.TensorMesh import BaseTensorMesh
class QuadTree(_QuadTree, BaseTensorMesh):
    #inheriting stuff from BaseTensorMesh that isn't defined in _QuadTree
    def __init__(self, h_in, x0=None, max_level=None):
        BaseTensorMesh.__init__(self, h_in, x0)

        if max_level is None:
            max_level = int(np.log2(len(self._h[0])))

        xF = np.array([self.vectorNx[-1], self.vectorNy[-1]])
        ws = xF-self.x0

        # Now can initialize quadtree parent
        _QuadTree.__init__(self, max_level, self.x0, ws)
