from libcpp cimport bool
from libcpp.vector cimport vector
from libcpp.map cimport map

cdef extern from "tree.h":
    ctypedef int int_t

    cdef cppclass Node:
        int_t location[2]
        int_t key
        int_t reference
        int_t index
        bool hanging
        Node *parents[2]
        Node()
        Node(int_t, int_t)
        int_t operator[](int_t)

    cdef cppclass Edge:
        int_t location[2]
        int_t key
        int_t reference
        int_t index
        int_t length
        bool hanging
        Node *points[2]
        Edge *parent
        Edge()
        Edge(Node& p1, Node& p2)

    ctypedef map[int_t,Node *] node_map_t
    ctypedef map[int_t,Edge *] edge_map_t

    cdef cppclass QuadCell:
        QuadCell *parent
        QuadCell *children[4]
        QuadCell *neighbors[4]
        Node *points[4]
        Edge *edges[4]
        int_t center[2]
        int_t index,key,level,max_level
        inline bool is_leaf()
        bool inside_triangle(double x, double y, int_t direction)

    cdef cppclass PyWrapper:
        PyWrapper()
        void set(void*, void*, int(*)(void*, void*, QuadCell*))

    cdef cppclass QuadTree:
        QuadCell *root
        int_t max_level, nx, ny

        vector[QuadCell *] cells
        node_map_t nodes
        edge_map_t edges_x,edges_y
        vector[Node *] hanging_nodes
        vector[Edge *] hanging_edges_x, hanging_edges_y

        QuadTree()

        void set_level(int_t)
        void build_tree(PyWrapper *)
        void number()
        QuadCell * containing_cell(double, double)
