#ifndef __TREE_H
#define __TREE_H

#include <vector>
#include <map>
#include <iostream>

typedef std::size_t int_t;

inline int_t key_func(int_t x, int_t y){
//Cantor pairing
    return ((x+y)*(x+y+1))/2+y;
}
class Node;
class Edge;
class QuadCell;
class QuadTree;
class PyWrapper;
typedef PyWrapper* function;

typedef std::map<int_t, Node *> node_map_t;
typedef std::map<int_t, Edge *> edge_map_t;
typedef node_map_t::iterator node_it_type;
typedef edge_map_t::iterator edge_it_type;
typedef std::vector<QuadCell *> quad_vec_t;

class PyWrapper{
  public:
    void *py_obj;
    void *py_func;
    int (*eval)(void*, void *, QuadCell*);
  PyWrapper(){
    py_func = NULL;
  };

  void set(void* obj, void* func, int (*wrapper)(void*, void*, QuadCell*)){
    py_obj = obj;
    py_func = func;
    eval = wrapper;
  };

  int operator()(QuadCell * cell){
    return eval(py_obj, py_func, cell);
  };

};

class Node{
  public:
    int_t location[2];
    int_t key;
    int_t reference;
    int_t index;
    bool hanging;
    Node *parents[2];
    Node();
    Node(int_t, int_t);
    int_t operator[](int_t index){
      return location[index];
    };
};

class Edge{
  public:
    int_t location[2];
    int_t key;
    int_t reference;
    int_t index;
    int_t length;
    bool hanging;
    Node *points[2];
    Edge *parent;
    Edge();
    Edge(Node& p1, Node&p2);
};


class QuadCell{
  public:
    QuadCell *parent, *children[4], *neighbors[4];
    Node *points[4];
    Edge *edges[4];

    int_t center[2],index,key,level,max_level;
    function test_func;

    QuadCell();
    QuadCell(Node *pts[4], int_t maxlevel, function func);
    QuadCell(Node *pts[4], QuadCell *parent);
    ~QuadCell();

    bool inline is_leaf(){ return children[0]==NULL;};
    void spawn(node_map_t& nodes, QuadCell *kids[4]);
    void divide(node_map_t& nodes, bool force);
    void set_neighbor(QuadCell* other, int_t direction);
    void build_cell_vector(quad_vec_t& cells);
    bool inside_triangle(double x, double y, int_t direction);

    QuadCell* containing_cell(double, double);
};

class QuadTree{
  public:
    QuadCell *root;
    function test_func;
    int_t max_level, nx, ny;

    std::vector<QuadCell *> cells;
    node_map_t nodes;
    edge_map_t edges_x, edges_y;
    std::vector<Node *> hanging_nodes;
    std::vector<Edge *> hanging_edges_x, hanging_edges_y;

    QuadTree();
    ~QuadTree();

    void set_level(int_t max_level);
    void build_tree(function test_func);
    void number();

    QuadCell* containing_cell(double, double);
};
#endif
