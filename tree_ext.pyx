cimport cython
cimport numpy as np
from libc.math cimport sqrt,abs

from tree cimport int_t, Tree as c_Tree, PyWrapper, Node, Edge, Face, Cell as c_Cell

import scipy.sparse as sp
import numpy as np

cdef class Cell:
    cdef double _x, _y, _z, _x0, _y0, _z0, _wx, _wy, _wz
    cdef int_t _dim
    cdef c_Cell* _cell
    cdef void _set(self, c_Cell* cell, double* scale, double* shift):
        self._dim = cell.n_dim
        self._cell = cell
        self._x = cell.center[0]*scale[0]+shift[0]
        self._y = cell.center[1]*scale[1]+shift[1]
        self._x0 = cell.points[0].location[0]*scale[0]+shift[0]
        self._y0 = cell.points[0].location[1]*scale[1]+shift[1]
        self._wx = 2*(self._x-self._x0)
        self._wy = 2*(self._y-self._y0)
        if(self._dim>2):
            self._z0 = cell.points[0].location[2]*scale[2]+shift[2]
            self._z = cell.center[2]*scale[2]+shift[2]
            self._wz = 2*(self._z-self._z0)

    @property
    def nodes(self):
        cdef c_Cell* cell = self._cell
        if self._dim>2:
            return tuple((cell.points[0].index, cell.points[1].index,
                          cell.points[2].index, cell.points[3].index,
                          cell.points[4].index, cell.points[5].index,
                          cell.points[6].index, cell.points[7].index))
        return tuple((cell.points[0].index, cell.points[1].index,
                      cell.points[2].index, cell.points[3].index))

    @property
    def center(self):
        if self._dim>2: return tuple((self._x, self._y, self._z))
        return tuple((self._x, self._y))

    @property
    def x0(self):
        if self._dim>2: return tuple((self._x0, self._y0, self._z0))
        return tuple((self._x0, self._y0))

    @property
    def h(self):
        if self._dim>2: return tuple((self._wx, self._wy, self._wz))
        return tuple((self._wx, self._wy))

    @property
    def dim(self):
        return self._dim

cdef int_t _evaluate_func(void* obj, void* function, c_Cell* cell) with gil:
    self = <object> obj
    func = <object> function
    cdef double[3] scale, shift
    cdef int_t i
    for i in range(cell.n_dim):
        scale[i] = self.scale[i]
        shift[i] = self.x0[i]
    pycell = Cell()
    pycell._set(cell, &scale[0], &shift[0])
    return <int_t> func(pycell)

cdef struct double4:
    double w0, w1, w2, w3

cdef void get_weights(double4 weights, double x, double y, c_Cell *cells[4], int_t n_cells):
    cdef int_t p0x, p1x, p2x, p3x
    cdef int_t p0y, p1y, p2y, p3y
    cdef double a[4]
    cdef double b[4]
    cdef double aa, bb, cc, l, m

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
    cdef c_Tree *tree
    cdef PyWrapper *wrapper
    cdef int_t _nx, _ny, _nz, max_level
    cdef double[3] _scale, _shift

    cdef object _gridCC, _gridN, _gridEx, _gridEy
    cdef object _gridhN, _gridhEx, _gridhEy
    cdef object _aveFx2CC, _aveFy2CC, _aveF2CC, _aveF2CCV,_aveN2CC,_aveE2CC,_aveE2CCV

    def __cinit__(self, *args, **kwargs):
        self.wrapper = new PyWrapper()
        self.tree = new c_Tree()

    def __init__(self, max_level, x0, w):
        self.max_level = max_level
        self._nx = 2<<max_level
        self._ny = 2<<max_level

        self._shift[0] = x0[0]
        self._scale[0] = w[0]/self._nx

        self._shift[1] = x0[1]
        self._scale[1] = w[1]/self._ny

        if self.dim>2:
            self._nz = 2<<max_level
            self._scale[2] = w[2]/self._ny
            self._shift[2] = x0[2]

        self.tree.set_dimension(self.dim)
        self.tree.set_level(self.max_level)

        self._gridCC = None
        self._gridN = None
        self._gridhN = None

        self._gridEx = None
        self._gridEy = None
        self._gridEz = None
        self._gridhEx = None
        self._gridhEy = None
        self._gridhEz = None

        self._gridFx = None
        self._gridFy = None
        self._gridFz = None
        self._gridhFx = None
        self._gridhFy = None
        self._gridhFz = None

        self._aveFx2CC = None
        self._aveFy2CC = None
        self._aveF2CC = None
        self._aveF2CCV = None
        self._aveE2CC = None
        self._aveE2CCV = None
        self._aveN2CC = None

    def build_tree(self, test_function):
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
    def scale(self):
        return self._scale

    @property
    def nx(self):
        return self._nx

    @property
    def ny(self):
        return self._ny

    @property
    def nz(self):
        return self._nz

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
        return self.nEx+self.nEy+self.nEz

    @property
    def nhE(self):
        return self.nhEx+self.nhEy+self.nhEz

    @property
    def ntE(self):
        return self.nE+self.nhE

    @property
    def nEx(self):
        return self.ntEx-self.nhEx

    @property
    def nEy(self):
        return self.ntEy-self.nhEy

    @property
    def nEz(self):
        return self.ntEz-self.nhEz

    @property
    def ntEx(self):
        return self.tree.edges_x.size()

    @property
    def ntEy(self):
        return self.tree.edges_y.size()

    @property
    def ntEz(self):
        return self.tree.edges_z.size()

    @property
    def nhEx(self):
        return self.tree.hanging_edges_x.size()

    @property
    def nhEy(self):
        return self.tree.hanging_edges_y.size()

    @property
    def nhEz(self):
        return self.tree.hanging_edges_z.size()

    @property
    def nF(self):
        return self.nFx+self.nFy+self.nFz

    @property
    def nhF(self):
        return self.nhFx+self.nhFy+self.nhFz

    @property
    def ntF(self):
        return self.nF+self.nhF

    @property
    def nFx(self):
        return self.ntFx-self.nhFx

    @property
    def nFy(self):
        return self.ntFy-self.nhFy

    @property
    def nFz(self):
        return self.ntFz-self.nhFz

    @property
    def ntFx(self):
        if(self.dim==2): return self.ntEy
        return self.tree.faces_x.size()

    @property
    def ntFy(self):
        if(self.dim==2): return self.ntEx
        return self.tree.faces_y.size()

    @property
    def ntFz(self):
        if(self.dim==2): return 0
        return self.tree.faces_z.size()

    @property
    def nhFx(self):
        if(self.dim==2): return self.nhEy
        return self.tree.hanging_faces_x.size()

    @property
    def nhFy(self):
        if(self.dim==2): return self.nhEx
        return self.tree.hanging_faces_y.size()

    @property
    def nhFz(self):
        if(self.dim==2): return 0
        return self.tree.hanging_faces_z.size()

    @property
    def gridN(self):
        cdef np.float64_t[:, :] gridN
        cdef Node *node
        cdef np.int64_t ii, ind, dim
        if self._gridN is None:
            dim = self.dim
            self._gridN = np.empty((self.nN, dim) ,dtype=np.float64)
            gridN = self._gridN
            scale = self._scale
            shift = self._shift
            for it in self.tree.nodes:
                node = it.second
                if not node.hanging:
                    ind = node.index
                    for ii in range(dim):
                        gridN[ind, ii] = node.location[ii]*scale[ii]+shift[ii]
        return self._gridN

    @property
    def gridhN(self):
        cdef np.float64_t[:, :] gridN
        cdef Node *node
        cdef np.int64_t ii, ind, dim
        if self._gridhN is None:
            dim = self.dim
            self._gridhN = np.empty((self.nhN, dim), dtype=np.float64)
            gridhN = self._gridhN
            scale = self._scale
            shift = self._shift
            for node in self.tree.hanging_nodes:
                ind = node.index-self.nN
                for ii in range(dim):
                    gridhN[ind, ii] = node.location[ii]*scale[ii]+shift[ii]
        return self._gridhN

    @property
    def gridCC(self):
        cdef np.float64_t[:, :] gridCC
        cdef np.int64_t ii, ind, dim
        if self._gridCC is None:
            dim = self.dim
            self._gridCC = np.empty((self.nC, dim), dtype=np.float64)
            gridCC = self._gridCC
            scale = self._scale
            shift = self._shift
            for cell in self.tree.cells:
                ind = cell.index
                for ii in range(dim):
                    gridCC[ind, ii] = cell.center[ii]*scale[ii]+shift[ii]
        return self._gridCC

    @property
    def gridEx(self):
        cdef np.float64_t[:, :] gridEx
        cdef Edge *edge
        cdef np.int64_t ii, ind, dim
        if self._gridEx is None:
            dim = self.dim
            self._gridEx = np.empty((self.nEx, dim), dtype=np.float64)
            gridEx = self._gridEx
            scale = self._scale
            shift = self._shift
            for it in self.tree.edges_x:
                edge = it.second
                if not edge.hanging:
                    ind = edge.index
                    for ii in range(dim):
                        gridEx[ind, ii] = edge.location[ii]*scale[ii]+shift[ii]
        return self._gridEx

    @property
    def gridhEx(self):
        cdef np.float64_t[:, :] gridhEx
        cdef Edge *edge
        cdef np.int64_t ii, ind, dim
        if self._gridhEx is None:
            dim = self.dim
            self._gridhEx = np.empty((self.nhEx, dim), dtype=np.float64)
            gridhEx = self._gridhEx
            scale = self._scale
            shift = self._shift
            for edge in self.tree.hanging_edges_x:
                ind = edge.index-self.nEx
                for ii in range(dim):
                    gridhEx[ind,ii] = edge.location[ii]*scale[ii]+shift[ii]
        return self._gridhEx

    @property
    def gridEy(self):
        cdef np.float64_t[:, :] gridEy
        cdef Edge *edge
        cdef np.int64_t ii, ind, dim
        if self._gridEy is None:
            dim = self.dim
            self._gridEy = np.empty((self.nEy, dim), dtype=np.float64)
            gridEy = self._gridEy
            scale = self._scale
            shift = self._shift
            for it in self.tree.edges_y:
                edge = it.second
                if not edge.hanging:
                    ind = edge.index
                    for ii in range(dim):
                        gridEy[ind,ii] = edge.location[ii]*scale[ii]+shift[ii]
        return self._gridEy

    @property
    def gridhEy(self):
        cdef np.float64_t[:,:] gridhEy
        cdef Edge *edge
        cdef np.int64_t ii, ind, dim
        if self._gridhEy is None:
            dim = self.dim
            self._gridhEy = np.empty((self.nhEy, dim), dtype=np.float64)
            gridhEy = self._gridhEy
            scale = self._scale
            shift = self._shift
            for edge in self.tree.hanging_edges_y:
                ind = edge.index-self.nEy
                for ii in range(dim):
                    gridhEy[ind, ii] = edge.location[ii]*scale[ii]+shift[ii]
        return self._gridhEy

    @property
    def gridEz(self):
        cdef np.float64_t[:, :] gridEz
        cdef Edge *edge
        cdef np.int64_t ii, ind, dim
        if self._gridEz is None:
            dim = self.dim
            self._gridEz = np.empty((self.nEz, dim), dtype=np.float64)
            gridEz = self._gridEz
            scale = self._scale
            shift = self._shift
            for it in self.tree.edges_z:
                edge = it.second
                if not edge.hanging:
                    ind = edge.index
                    for ii in range(dim):
                        gridEz[ind,ii] = edge.location[ii]*scale[ii]+shift[ii]
        return self._gridEz

    @property
    def gridhEz(self):
        cdef np.float64_t[:,:] gridhEz
        cdef Edge *edge
        cdef np.int64_t ii, ind, dim
        if self._gridhEz is None:
            dim = self.dim
            self._gridhEz = np.empty((self.nhEz, dim), dtype=np.float64)
            gridhEz = self._gridhEz
            scale = self._scale
            shift = self._shift
            for edge in self.tree.hanging_edges_z:
                ind = edge.index-self.nEz
                for ii in range(dim):
                    gridhEz[ind, ii] = edge.location[ii]*scale[ii]+shift[ii]
        return self._gridhEz

    @property
    def gridFx(self):
        if(self.dim==2): return self.gridEy

        cdef np.float64_t[:,:] gridFx
        cdef Face *face
        cdef np.int64_t ii, ind, dim
        if self._gridFx is None:
            dim = self.dim
            self._gridFx = np.empty((self.nFx, dim), dtype=np.float64)
            gridFx = self._gridFx
            scale = self._scale
            shift = self._shift
            for it in self.tree.faces_x:
                face = it.second
                if not face.hanging:
                    ind = face.index
                    for ii in range(dim):
                        gridFx[ind, ii] = face.location[ii]*scale[ii]+shift[ii]
        return self._gridFx

    @property
    def gridFy(self):
        if(self.dim==2): return self.gridEx

        cdef np.float64_t[:,:] gridFy
        cdef Face *face
        cdef np.int64_t ii, ind, dim
        if self._gridFy is None:
            dim = self.dim
            self._gridFy = np.empty((self.nFy, dim), dtype=np.float64)
            gridFy = self._gridFy
            scale = self._scale
            shift = self._shift
            for it in self.tree.faces_y:
                face = it.second
                if not face.hanging:
                    ind = face.index
                    for ii in range(dim):
                        gridFy[ind, ii] = face.location[ii]*scale[ii]+shift[ii]
        return self._gridFy

    @property
    def gridFz(self):
        if(self.dim==2): return self.gridCC

        cdef np.float64_t[:,:] gridFz
        cdef Face *face
        cdef np.int64_t ii, ind, dim
        if self._gridFz is None:
            dim = self.dim
            self._gridFz = np.empty((self.nFz, dim), dtype=np.float64)
            gridFz = self._gridFz
            scale = self._scale
            shift = self._shift
            for it in self.tree.faces_z:
                face = it.second
                if not face.hanging:
                    ind = face.index
                    for ii in range(dim):
                        gridFz[ind, ii] = face.location[ii]*scale[ii]+shift[ii]
        return self._gridFz

    @property
    def gridhFx(self):
        if(self.dim==2): return self.gridhEy

        cdef np.float64_t[:,:] gridFx
        cdef Face *face
        cdef np.int64_t ii, ind, dim
        if self._gridhFx is None:
            dim = self.dim
            self._gridhFx = np.empty((self.nhFx, dim), dtype=np.float64)
            gridhFx = self._gridhFx
            scale = self._scale
            shift = self._shift
            for face in self.tree.hanging_faces_x:
                ind = face.index-self.nFx
                for ii in range(dim):
                    gridhFx[ind, ii] = face.location[ii]*scale[ii]+shift[ii]
        return self._gridFx

    @property
    def gridhFy(self):
        if(self.dim==2): return self.gridhEx

        cdef np.float64_t[:,:] gridhFy
        cdef Face *face
        cdef np.int64_t ii, ind, dim
        if self._gridhFy is None:
            dim = self.dim
            self._gridhFy = np.empty((self.nhFy, dim), dtype=np.float64)
            gridhFy = self._gridhFy
            scale = self._scale
            shift = self._shift
            for face in self.tree.hanging_faces_y:
                ind = face.index-self.nFy
                for ii in range(dim):
                    gridhFy[ind, ii] = face.location[ii]*scale[ii]+shift[ii]
        return self._gridFy

    @property
    def gridhFz(self):
        if(self.dim==2): return np.array([])

        cdef np.float64_t[:,:] gridhFz
        cdef Face *face
        cdef np.int64_t ii, ind, dim
        if self._gridhFz is None:
            dim = self.dim
            self._gridhFz = np.empty((self.nhFz, dim), dtype=np.float64)
            gridhFz = self._gridhFz
            scale = self._scale
            shift = self._shift
            for face in self.tree.hanging_faces_z:
                ind = face.index-self.nFz
                for ii in range(dim):
                    gridhFz[ind, ii] = face.location[ii]*scale[ii]+shift[ii]
        return self._gridFz

    @property
    def faceDiv(self):
        if(self.dim==2):
            D = self._faceDiv2D() # Because it uses edges instead of faces
        else:
            D = self._faceDiv3D()
        R = self._deflate_faces()
        return D*R

    def _faceDiv2D(self):
        cdef np.int64_t[:] I = np.empty(self.nC*4, dtype=np.int64)
        cdef np.int64_t[:] J = np.empty(self.nC*4, dtype=np.int64)
        cdef np.float64_t[:] V = np.empty(self.nC*4, dtype=np.float64)

        cdef np.int64_t i = 0
        cdef Edge *edges[4]
        cdef np.int64_t offset = self.tree.edges_y.size()
        cdef double volume

        for cell in self.tree.cells:
            edges = cell.edges
            i = cell.index
            I[i*4:i*4+4] = i
            J[i*4  ] = edges[0].index #y edge, x face
            J[i*4+1] = edges[1].index+offset #x edge, y face (add offset)
            J[i*4+2] = edges[2].index #y edge, x face
            J[i*4+3] = edges[3].index+offset #x edge, y face (add offset)

            volume = cell.volume*self._scale[0]*self._scale[1]
            V[i*4  ] = -(edges[0].length*self._scale[1]/volume)
            V[i*4+1] = edges[1].length*self._scale[0]/volume
            V[i*4+2] = edges[2].length*self._scale[1]/volume
            V[i*4+3] = -(edges[3].length*self._scale[0]/volume)
        return sp.csr_matrix((V, (I, J)))

    def _faceDiv3D(self):
        cdef:
            np.int64_t[:] I = np.empty(self.nC*6, dtype=np.int64)
            np.int64_t[:] J = np.empty(self.nC*6, dtype=np.int64)
            np.float64_t[:] V = np.empty(self.nC*6, dtype=np.float64)

            np.int64_t i = 0
            Face *faces[6]
            np.int64_t offset1 = self.tree.faces_x.size()
            np.int64_t offset2 = offset1+self.tree.faces_y.size()
            double volume, fx_area, fy_area, fz_area

        for cell in self.tree.cells:
            faces = cell.faces
            i = cell.index
            I[i*6:i*6+6] = i
            J[i*6  ] = faces[0].index #x1 face
            J[i*6+1] = faces[1].index #x2 face
            J[i*6+2] = faces[2].index+offset1 #y face (add offset1)
            J[i*6+3] = faces[3].index+offset1 #y face (add offset1)
            J[i*6+4] = faces[4].index+offset2 #z face (add offset2)
            J[i*6+5] = faces[5].index+offset2 #z face (add offset2)

            volume = cell.volume*self._scale[0]*self._scale[1]*self._scale[2]
            fx_area = faces[0].area*self._scale[1]*self.scale[2]
            fy_area = faces[2].area*self._scale[0]*self.scale[2]
            fz_area = faces[4].area*self._scale[0]*self.scale[1]
            V[i*6  ] = -(fx_area)/volume
            V[i*6+1] =  (fx_area)/volume
            V[i*6+2] = -(fy_area)/volume
            V[i*6+3] =  (fy_area)/volume
            V[i*6+4] = -(fz_area)/volume
            V[i*6+5] =  (fz_area)/volume
        return sp.csr_matrix((V, (I, J)))

    @property
    def cellGrad(self):
        cdef np.int64_t[:] I = np.empty(self.nC*4, dtype=np.int64)
        cdef np.int64_t[:] J = np.empty(self.nC*4, dtype=np.int64)
        cdef np.float64_t[:] V = np.empty(self.nC*4, dtype=np.float64)

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

            V[i*4  ] = edges[0].length*self._scale[1]
            V[i*4+1] = -(edges[1].length*self._scale[0])
            V[i*4+2] = -(edges[2].length*self._scale[1])
            V[i*4+3] = edges[3].length*self._scale[0]
            i += 1

        R = self.deflate_faces()

        G = R.T*sp.csr_matrix((V, (J, I)))
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

    def _deflate_edges_x(self):
        #I is output index (with hanging)
        #J is input index (without hanging)
        cdef np.int64_t[:] I = np.empty(2*self.ntEx, dtype=np.int64)
        cdef np.int64_t[:] J = np.empty(2*self.ntEx, dtype=np.int64)
        cdef np.float64_t[:] V = np.empty(2*self.ntEx, dtype=np.float64)
        cdef Edge *edge
        cdef np.int64_t ii
        #x edges:
        for it in self.tree.edges_x:
            edge = it.second
            ii = edge.index
            I[2*ii  ] = ii
            I[2*ii+1] = ii
            if edge.hanging:
                J[2*ii  ] = edge.parents[0].index
                J[2*ii+1] = edge.parents[1].index
            else:
                J[2*ii  ] = ii
                J[2*ii+1] = ii
            V[2*ii  ] = 0.5
            V[2*ii+1] = 0.5
        Rh = sp.csr_matrix((V, (I, J)), shape=(self.ntEx, self.ntEx))
        # Test if it needs to be deflated again, (if any parents were also hanging)
        last_ind = max(np.nonzero(Rh.getnnz(0)>0)[0][-1], self.nEx)
        while(last_ind > self.nEx):
            Rh = Rh*Rh
            last_ind = max(np.nonzero(Rh.getnnz(0)>0)[0][-1], self.nEx)
        Rh = Rh[:,:last_ind]
        return Rh

    def _deflate_edges_y(self):
        #I is output index (with hanging)
        #J is input index (without hanging)
        cdef int_t dim = self.dim
        cdef np.int64_t[:] I = np.empty(2*self.ntEy, dtype=np.int64)
        cdef np.int64_t[:] J = np.empty(2*self.ntEy, dtype=np.int64)
        cdef np.float64_t[:] V = np.empty(2*self.ntEy, dtype=np.float64)
        cdef Edge *edge
        cdef np.int64_t ii
        #x edges:
        for it in self.tree.edges_y:
            edge = it.second
            ii = edge.index
            I[2*ii  ] = ii
            I[2*ii+1] = ii
            if edge.hanging:
                J[2*ii  ] = edge.parents[0].index
                J[2*ii+1] = edge.parents[1].index
            else:
                J[2*ii  ] = ii
                J[2*ii+1] = ii
            V[2*ii  ] = 0.5
            V[2*ii+1] = 0.5
        Rh = sp.csr_matrix((V, (I, J)), shape=(self.ntEy, self.ntEy))
        # Test if it needs to be deflated again, (if any parents were also hanging)
        last_ind = max(np.nonzero(Rh.getnnz(0)>0)[0][-1], self.nEy)
        while(last_ind > self.nEy):
            Rh = Rh*Rh
            last_ind = max(np.nonzero(Rh.getnnz(0)>0)[0][-1], self.nEy)
        Rh = Rh[:,:last_ind]
        return Rh

    def _deflate_edges_z(self):
        #I is output index (with hanging)
        #J is input index (without hanging)
        cdef int_t dim = self.dim
        cdef np.int64_t[:] I = np.empty(2*self.ntEz, dtype=np.int64)
        cdef np.int64_t[:] J = np.empty(2*self.ntEz, dtype=np.int64)
        cdef np.float64_t[:] V = np.empty(2*self.ntEz, dtype=np.float64)
        cdef Edge *edge
        cdef np.int64_t ii
        #x edges:
        for it in self.tree.edges_z:
            edge = it.second
            ii = edge.index
            I[2*ii  ] = ii
            I[2*ii+1] = ii
            if edge.hanging:
                J[2*ii  ] = edge.parents[0].index
                J[2*ii+1] = edge.parents[1].index
            else:
                J[2*ii  ] = ii
                J[2*ii+1] = ii
            V[2*ii  ] = 0.5
            V[2*ii+1] = 0.5
        Rh = sp.csr_matrix((V, (I, J)), shape=(self.ntEz, self.ntEz))
        # Test if it needs to be deflated again, (if any parents were also hanging)
        last_ind = max(np.nonzero(Rh.getnnz(0)>0)[0][-1], self.nEz)
        while(last_ind > self.nEz):
            Rh = Rh*Rh
            last_ind = max(np.nonzero(Rh.getnnz(0)>0)[0][-1], self.nEz)
        Rh = Rh[:,:last_ind]
        return Rh

    def _deflate_edges(self):
        Rx = self._deflate_edges_x()
        Ry = self._deflate_edges_y()
        Rz = self._deflate_edges_z()
        return sp.block_diag((Rx, Ry, Rz))

    def _deflate_faces(self):
        if(self.dim==2):
            Rx = self._deflate_edges_x()
            Ry = self._deflate_edges_y()
            return sp.block_diag((Ry, Rx))
        else:
            Rx = self._deflate_faces_x()
            Ry = self._deflate_faces_y()
            Rz = self._deflate_faces_z()
            return sp.block_diag((Rx, Ry, Rz))

    def _deflate_faces_x(self):
        #I is output index (with hanging)
        #J is input index (without hanging)
        cdef np.int64_t[:] I = np.empty(self.ntFx, dtype=np.int64)
        cdef np.int64_t[:] J = np.empty(self.ntFx, dtype=np.int64)
        cdef np.float64_t[:] V = np.empty(self.ntFx, dtype=np.float64)
        cdef Face *face
        cdef np.int64_t ii;

        for it in self.tree.faces_x:
            face = it.second
            ii = face.index
            I[ii] = ii
            if face.hanging:
                J[ii] = face.parent.index
            else:
                J[ii] = ii
            V[ii] = 1.0
        return sp.csr_matrix((V, (I, J)))

    def _deflate_faces_y(self):
        #I is output index (with hanging)
        #J is input index (without hanging)
        cdef np.int64_t[:] I = np.empty(self.ntFy, dtype=np.int64)
        cdef np.int64_t[:] J = np.empty(self.ntFy, dtype=np.int64)
        cdef np.float64_t[:] V = np.empty(self.ntFy, dtype=np.float64)
        cdef Face *face
        cdef np.int64_t ii;

        for it in self.tree.faces_y:
            face = it.second
            ii = face.index
            I[ii] = ii
            if face.hanging:
                J[ii] = face.parent.index
            else:
                J[ii] = ii
            V[ii] = 1.0
        return sp.csr_matrix((V, (I, J)))

    def _deflate_faces_z(self):
        #I is output index (with hanging)
        #J is input index (without hanging)
        cdef np.int64_t[:] I = np.empty(self.ntFz, dtype=np.int64)
        cdef np.int64_t[:] J = np.empty(self.ntFz, dtype=np.int64)
        cdef np.float64_t[:] V = np.empty(self.ntFz, dtype=np.float64)
        cdef Face *face
        cdef np.int64_t ii;

        for it in self.tree.faces_z:
            face = it.second
            ii = face.index
            I[ii] = ii
            if face.hanging:
                J[ii] = face.parent.index
            else:
                J[ii] = ii
            V[ii] = 1.0
        return sp.csr_matrix((V, (I, J)))

    @property
    def nodalGrad(self):
        cdef int_t dim = self.dim
        cdef np.int64_t[:] I = np.empty(2*self.nE, dtype=np.int64)
        cdef np.int64_t[:] J = np.empty(2*self.nE, dtype=np.int64)
        cdef np.float64_t[:] V = np.empty(2*self.nE, dtype=np.float64)

        cdef Edge *edge
        cdef double length
        cdef int_t ii
        cdef np.int64_t offset1 = self.nEx
        cdef np.int64_t offset2 = offset1+self.nEy

        for it in self.tree.edges_x:
            edge = it.second
            if edge.hanging:
                continue
            ii = edge.index
            I[ii*2:ii*2+2] = ii
            J[ii*2  ] = edge.points[0].index
            J[ii*2+1] = edge.points[1].index

            length = edge.length*self._scale[0]
            V[ii*2  ] = -1.0/length
            V[ii*2+1] = 1.0/length

        for it in self.tree.edges_y:
            edge = it.second
            if edge.hanging:
                continue
            ii = edge.index+offset1
            I[ii*2:ii*2+2] = ii
            J[ii*2] = edge.points[0].index
            J[ii*2+1] = edge.points[1].index

            length = edge.length*self._scale[1]
            V[ii*2  ] = -1.0/length
            V[ii*2+1] = 1.0/length

        if(dim>2):
            for it in self.tree.edges_z:
                edge = it.second
                if edge.hanging:
                    continue
                ii = edge.index+offset2
                I[ii*2:ii*2+2] = ii
                J[ii*2  ] = edge.points[0].index
                J[ii*2+1] = edge.points[1].index

                length = edge.length*self._scale[1]
                V[ii*2  ] = -1.0/length
                V[ii*2+1] = 1.0/length


        Rn = self._deflate_nodes()
        G = sp.csr_matrix((V, (I, J)), shape=(self.nE, self.ntN))
        return G*Rn

    def _deflate_nodes(self):
        cdef np.int64_t[:] I = np.empty(4*self.ntN, dtype=np.int64)
        cdef np.int64_t[:] J = np.empty(4*self.ntN, dtype=np.int64)
        cdef np.float64_t[:] V = np.empty(4*self.ntN, dtype=np.float64)

        # I is output index
        # J is input index
        cdef Node *node
        cdef np.int64_t ii, i, offset
        offset = self.nN
        cdef double[4] weights

        for it in self.tree.nodes:
            node = it.second
            ii = node.index
            I[4*ii:4*ii+4] = ii
            if node.hanging:
                J[4*ii  ] = node.parents[0].index
                J[4*ii+1] = node.parents[1].index
                J[4*ii+2] = node.parents[2].index
                J[4*ii+3] = node.parents[3].index
            else:
                J[4*ii:4*ii+4] = ii
            V[4*ii:4*ii+4] = 0.25;

        Rh = sp.csr_matrix((V, (I, J)), shape=(self.ntN, self.ntN))
        # Test if it needs to be deflated again, (if any parents were also hanging)
        last_ind = max(np.nonzero(Rh.getnnz(0)>0)[0][-1], self.nN)
        while(last_ind > self.nN):
            Rh = Rh*Rh;
            last_ind = max(np.nonzero(Rh.getnnz(0)>0)[0][-1], self.nN)
        Rh = Rh[:,:last_ind]
        return Rh;

    @property
    def edgeCurl(self):
        cdef:
            int_t dim = self.dim
            np.int64_t[:] I = np.empty(4*self.nF, dtype=np.int64)
            np.int64_t[:] J = np.empty(4*self.nF, dtype=np.int64)
            np.float64_t[:] V = np.empty(4*self.nF, dtype=np.float64)
            Face *face
            int_t ii
            int_t face_offset_y = self.nFx
            int_t face_offset_z = self.nFx+self.nFy
            int_t edge_offset_y = self.ntEx
            int_t edge_offset_z = self.ntEx+self.ntEy
            double area

        for it in self.tree.faces_x:
            face = it.second
            if face.hanging:
                continue
            ii = face.index
            I[4*ii:4*ii+4] = ii
            J[4*ii  ] = face.edges[0].index+edge_offset_z
            J[4*ii+1] = face.edges[1].index+edge_offset_y
            J[4*ii+2] = face.edges[2].index+edge_offset_z
            J[4*ii+3] = face.edges[3].index+edge_offset_y

            area = face.area*self._scale[1]*self._scale[2]
            V[4*ii  ] = (face.edges[0].length*self._scale[2]/area)
            V[4*ii+1] = -(face.edges[1].length*self._scale[1]/area)
            V[4*ii+2] = -(face.edges[2].length*self._scale[2]/area)
            V[4*ii+3] = (face.edges[3].length*self._scale[1]/area)

        for it in self.tree.faces_y:
            face = it.second
            if face.hanging:
                continue
            ii = face.index+face_offset_y
            I[4*ii:4*ii+4] = ii
            J[4*ii  ] = face.edges[0].index+edge_offset_z
            J[4*ii+1] = face.edges[1].index
            J[4*ii+2] = face.edges[2].index+edge_offset_z
            J[4*ii+3] = face.edges[3].index

            area = face.area*self._scale[0]*self._scale[2]
            V[4*ii  ] = (face.edges[0].length*self._scale[2]/area)
            V[4*ii+1] = -(face.edges[1].length*self._scale[0]/area)
            V[4*ii+2] = -(face.edges[2].length*self._scale[2]/area)
            V[4*ii+3] = (face.edges[3].length*self._scale[0]/area)

        for it in self.tree.faces_z:
            face = it.second
            if face.hanging:
                continue
            ii = face.index+face_offset_z
            I[4*ii:4*ii+4] = ii
            J[4*ii  ] = face.edges[0].index+edge_offset_y
            J[4*ii+1] = face.edges[1].index
            J[4*ii+2] = face.edges[2].index+edge_offset_y
            J[4*ii+3] = face.edges[3].index

            area = face.area*self._scale[0]*self._scale[1]
            V[4*ii  ] = (face.edges[0].length*self._scale[1]/area)
            V[4*ii+1] = -(face.edges[1].length*self._scale[0]/area)
            V[4*ii+2] = -(face.edges[2].length*self._scale[1]/area)
            V[4*ii+3] = (face.edges[3].length*self._scale[0]/area)

        C = sp.csr_matrix((V, (I, J)),shape=(self.nF, self.ntE))
        R = self._deflate_edges()
        return C*R

    @property
    def aveFx2CC(self):
        cdef np.int64_t[:] I,J
        cdef np.float64_t[:] V
        cdef Edge *edge1
        cdef Edge *edge2
        cdef np.int64_t ii
        if self._aveFx2CC is None:
            I = np.empty(self.nC*2, dtype=np.int64)
            J = np.empty(self.nC*2, dtype=np.int64)
            V = np.empty(self.nC*2, dtype=np.float64)
            for cell in self.tree.cells:
                edge1 = cell.edges[0] #y edge/ x face
                edge2 = cell.edges[2] #y edge/ x face
                ii = cell.index
                I[ii*2:ii*2+2] = ii
                J[ii*2] = edge1.index
                J[ii*2+1] = edge2.index
                V[ii*2:ii*2+2] = 0.5
            Rfx = self.deflate_faces_x()
            self._aveFx2CC = sp.csr_matrix((V, (I, J)))*Rfx
        return self._aveFx2CC

    @property
    def aveFy2CC(self):
        cdef np.int64_t[:] I,J
        cdef np.float64_t[:] V
        cdef Edge *edge1
        cdef Edge *edge2
        cdef np.int64_t ii
        if self._aveFy2CC is None:
            I = np.empty(self.nC*2, dtype=np.int64)
            J = np.empty(self.nC*2, dtype=np.int64)
            V = np.empty(self.nC*2, dtype=np.float64)
            for cell in self.tree.cells:
                edge1 = cell.edges[1] #x edge/ y face
                edge2 = cell.edges[3] #x edge/ y face
                ii = cell.index
                I[ii*2:ii*2+2] = ii
                J[ii*2] = edge1.index
                J[ii*2+1] = edge2.index
                V[ii*2:ii*2+2] = 0.5
            Rfy = self.deflate_faces_y()
            self._aveFy2CC = sp.csr_matrix((V, (I, J)))*Rfy
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
        cdef np.int64_t[:] I, J
        cdef np.float64_t[:] V
        cdef np.int64_t ii
        if self._aveN2CC is None:
            I = np.empty(self.nC*4, dtype=np.int64)
            J = np.empty(self.nC*4, dtype=np.int64)
            V = np.empty(self.nC*4, dtype=np.float64)
            for cell in self.tree.cells:
                ii = cell.index
                I[ii*4:ii*4+4] = ii
                J[ii*4  ] = cell.points[0].index
                J[ii*4+1] = cell.points[1].index
                J[ii*4+2] = cell.points[2].index
                J[ii*4+3] = cell.points[3].index
                V[ii*4:ii*4+4] = 0.25
            Rn = self.deflate_nodes()
            self._aveN2CC = sp.csr_matrix((V, (I, J)))*Rn
        return self._aveN2CC

    def getInterpolationMat(self, locs, locType, zerosOutside = False):
        pass

    def _getNodeIntMat(self,locs):
        cdef:
            double[:, :] locations = locs
            int_t n_loc = locs.shape[0]
            np.int64_t[:] I = np.empty(n_loc*4, dtype=np.int64)
            np.int64_t[:] J = np.empty(n_loc*4, dtype=np.int64)
            np.float64_t[:] V = np.empty(n_loc*4, dtype=np.float64)

            int_t ii,i
            c_Cell *cell
            double x,y, scale_x, scale_y, shift_x, shift_y
            double wx,wy

        scale_x = self._scale_x
        scale_y = self._scale_y
        shift_x = self.x0[0]
        shift_y = self.x0[1]
        for i in range(n_loc):
            x = (locations[i, 0]-shift_x)/scale_x
            y = (locations[i, 1]-shift_y)/scale_y
            #get containing (or closest) cell
            cell = self.tree.containing_cell(x,y,0)
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
        return sp.csr((V, (I, J)))*Rn

    def _getCellCenterIntMat(self,locs):
        cdef:
            double[:,:] locations = locs
            int_t n_loc = locs.shape[0]
            np.int64_t[:] I = np.empty(n_loc*4, dtype=np.int64)
            np.int64_t[:] J = np.empty(n_loc*4, dtype=np.int64)
            np.float64_t[:] V = np.empty(n_loc*4, dtype=np.float64)

            int_t ii,i,n_cells
            c_Cell *cell
            c_Cell *test_cell
            double x,y, scale_x, scale_y, shift_x, shift_y
            double4 weights
        scale_x = self._scale_x
        scale_y = self._scale_y
        shift_x = self.x0[0]
        shift_y = self.x0[1]

        cdef c_Cell *cells[4] #closest cells to a point

        for i in range(n_loc):
            x = (locations[i, 0]-shift_x)/scale_x
            y = (locations[i, 1]-shift_y)/scale_y
            #get containing (or closest) cell
            cell = self.tree.containing_cell(x, y,0)
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
                # then it is in a quadrilateral
                if x<cell.center[0]:
                    if y<cell.center[1]: # lower left
                        cells[1] = cell.neighbors[2]
                        if cells[1] != NULL and not cells[1].is_leaf():
                            cells[1] = cells[1].children[2]

                        cells[2] = cells[1].neighbors[0]
                        if cells[2] != NULL and not cells[2].is_leaf():
                            cells[2] = cells[1].neighbors[0].children[3]

                        cells[3] = cell.neighbors[0]
                        if cells[3] != NULL and not cells[3].is_leaf():
                            cells[3] = cells[3].children[1]
                    else: # upper left
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
                    if y<cell.center[1]: # lower right
                        cells[1] = cell.neighbors[1]
                        if cells[1] != NULL and not cells[1].is_leaf():
                            cells[1] = cells[1].children[0]

                        cells[2] = cells[1].neighbors[2]
                        if cells[2] != NULL and not cells[2].is_leaf():
                            cells[2] = cells[1].neighbors[2].children[2]

                        cells[3] = cell.neighbors[2]
                        if cells[3] != NULL and not cells[3].is_leaf():
                            cells[3] = cells[3].children[3]
                    else: # upper right
                        cells[1] = cell.neighbors[3]
                        if cells[1] != NULL and not cells[1].is_leaf():
                            cells[1] = cells[1].children[1]

                        cells[2] = cells[1].neighbors[1]
                        if cells[2] != NULL and not cells[2].is_leaf():
                            cells[2] = cells[1].neighbors[1].children[0]

                        cells[3] = cell.neighbors[1]
                        if cells[3] != NULL and not cells[3].is_leaf():
                            cells[3] = cells[3].children[2]
            # now cells contains a list of the 3 or 4
            # bounding cells to interpolate from
            # just need to figure out the weights
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
        cdef double[:, :] locations = locs
        cdef n_loc = locs.shape[0]
        cdef np.int64_t[:] I = np.empty(n_loc*2, dtype=np.int64)
        cdef np.int64_t[:] J = np.empty(n_loc*2, dtype=np.int64)
        cdef np.float64_t[:] V = np.empty(n_loc*2, dtype=np.float64)

        cdef int_t ii, i
        cdef c_Cell *cell
        cdef double x, y
        for i in range(n_loc):
            x = locations[i, 0]
            y = locations[i, 1]
            cell = self.tree.containing_cell(x, y, 0)

    def plotGrid(self, ax=None, showIt=False,
        grid=True,
        cells=False, cellLine=False,
        nodes = False,
        facesX = False, facesY = False,
        edgesX = False, edgesY = False):

        import matplotlib
        if ax is None:
            import matplotlib.pyplot as plt
            import matplotlib.colors as colors
            import matplotlib.cm as cmx
            if(self.dim==2):
                ax = plt.subplot(111)
            else:
                from mpl_toolkits.mplot3d import Axes3D
                ax = plt.subplot(111, projection='3d')
        else:
            assert isinstance(ax,matplotlib.axes.Axes), "ax must be an Axes!"
            fig = ax.figure

        cdef:
            double scale_x = self._scale[0]
            double scale_y = self._scale[1]
            double scale_z = self._scale[2]
            double shift_x = self._shift[0]
            double shift_y = self._shift[1]
            double shift_z = self._shift[2]
            int_t i, offset
            Node *p1
            Node *p2
            Edge *edge

        if grid:
            if(self.dim)==2:
                X = np.empty((self.nE*3,))
                Y = np.empty((self.nE*3,))
                for it in self.tree.edges_x:
                    edge = it.second
                    if(edge.hanging): continue
                    i = edge.index*3
                    p1 = edge.points[0]
                    p2 = edge.points[1]
                    X[i:i+3] = [p1.location[0],p2.location[0],np.nan]
                    Y[i:i+3] = [p1.location[1],p2.location[1],np.nan]

                offset = self.nEx
                for it in self.tree.edges_y:
                    edge = it.second
                    if(edge.hanging): continue
                    i = (edge.index+offset)*3
                    p1 = edge.points[0]
                    p2 = edge.points[1]
                    X[i:i+3] = [p1.location[0],p2.location[0],np.nan]
                    Y[i:i+3] = [p1.location[1],p2.location[1],np.nan]

                X *= scale_x
                X += shift_x
                Y *= scale_y
                Y += shift_y
                ax.plot(X, Y, 'b-')
            else:
                X = np.empty((self.nE*3,))
                Y = np.empty((self.nE*3,))
                Z = np.empty((self.nE*3,))
                for it in self.tree.edges_x:
                    edge = it.second
                    if(edge.hanging): continue
                    i = edge.index*3
                    p1 = edge.points[0]
                    p2 = edge.points[1]
                    X[i:i+3] = [p1.location[0], p2.location[0], np.nan]
                    Y[i:i+3] = [p1.location[1], p2.location[1], np.nan]
                    Z[i:i+3] = [p1.location[2], p2.location[2], np.nan]

                offset = self.nEx
                for it in self.tree.edges_y:
                    edge = it.second
                    if(edge.hanging): continue
                    i = (edge.index+offset)*3
                    p1 = edge.points[0]
                    p2 = edge.points[1]
                    X[i:i+3] = [p1.location[0], p2.location[0], np.nan]
                    Y[i:i+3] = [p1.location[1], p2.location[1], np.nan]
                    Z[i:i+3] = [p1.location[2], p2.location[2], np.nan]

                offset += self.nEy
                for it in self.tree.edges_z:
                    edge = it.second
                    if(edge.hanging): continue
                    i = (edge.index+offset)*3
                    p1 = edge.points[0]
                    p2 = edge.points[1]
                    X[i:i+3] = [p1.location[0], p2.location[0], np.nan]
                    Y[i:i+3] = [p1.location[1], p2.location[1], np.nan]
                    Z[i:i+3] = [p1.location[2], p2.location[2], np.nan]

                X *= scale_x
                X += shift_x
                Y *= scale_y
                Y += shift_y
                Z *= scale_z
                Z += shift_z
                ax.plot(X, Y, 'b-', zs=Z)

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

    def __len__(self):
        return self.nC

    def __getitem__(self, key):
        if isinstance(key, slice):
            # Get the start, stop, and step from the slice
            return [self[ii] for ii in range(*key.indices(len(self)))]
        elif isinstance(key, int):
            if key < 0:  # Handle negative indices
                key += len(self)
            if key >= len(self):
                raise IndexError(
                    "The index ({0:d}) is out of range.".format(key)
                )
            pycell = Cell()
            pycell._set(self.tree.cells[key], &self._scale[0], &self._shift[0])
            return pycell
        else:
            raise TypeError("Invalid argument type.")

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

        if(self.dim==2):
            xF = np.array([self.vectorNx[-1], self.vectorNy[-1]])
        else:
            xF = np.array([self.vectorNx[-1], self.vectorNy[-1], self.vectorNz[-1]])
        ws = xF-self.x0

        # Now can initialize quadtree parent
        _QuadTree.__init__(self, max_level, self.x0, ws)
