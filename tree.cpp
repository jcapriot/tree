#include <vector>
#include <map>
#include "tree.h"
#include <iostream>

Node::Node(){
    location[0] = 0;
    location[1] = 0;
    key = 0;
    reference = 0;
    hanging = false;
    parents[0] = NULL;
    parents[1] = NULL;
};

Node::Node(int_t x, int_t y){
    location[0] = x;
    location[1] = y;
    key = key_func(x,y);
    reference = 0;
    hanging = false;
    parents[0] = NULL;
    parents[1] = NULL;
}

Edge::Edge(Node& p1, Node& p2){
      points[0] = &p1;
      points[1] = &p2;
      int_t x,y;
      x = (p1[0]+p2[0])/2;
      y = (p1[1]+p2[1])/2;
      key = key_func(x,y);
      location[0] = x;
      location[1] = y;
      length = (p2[0]-p1[0])+(p2[1]-p1[1]);
      reference = 0;
      hanging = false;
}

Node * set_default_node(node_map_t& nodes, int_t x, int_t y){
  int_t key = key_func(x,y);
  Node * point;
  if(nodes.count(key)==0){
    point = new Node(x,y);
    nodes[key] = point;
  }
  else{
    point = nodes[key];
  }
  return point;
}

Edge * set_default_edge(edge_map_t& edges, Node& p1, Node& p2){
  int_t xC = (p1[0]+p2[0])/2;
  int_t yC = (p1[1]+p2[1])/2;
  int_t key = key_func(xC,yC);
  Edge * edge;
  if(edges.count(key)==0){
    edge = new Edge(p1,p2);
    edges[key] = edge;
  }
  else{
    edge = edges[key];
  }
  return edge;
};

QuadCell::QuadCell(Node *pts[4], int_t maxlevel, function func){
    points[0] = pts[0];
    points[1] = pts[1];
    points[2] = pts[2];
    points[3] = pts[3];
    level = 0;
    max_level = maxlevel;
    parent = NULL;
    test_func = func;
    Node p1 = *pts[0];
    Node p4 = *pts[3];
    center[0] = (p1[0]+p4[0])/2;
    center[1] = (p1[1]+p4[1])/2;
    key = key_func(center[0],center[1]);
    children[0] = NULL;
    children[1] = NULL;
    children[2] = NULL;
    children[3] = NULL;
    neighbors[0] = NULL;
    neighbors[1] = NULL;
    neighbors[2] = NULL;
    neighbors[3] = NULL;
};

QuadCell::QuadCell(Node *pts[4], QuadCell *parent){
    points[0] = pts[0];
    points[1] = pts[1];
    points[2] = pts[2];
    points[3] = pts[3];
    level = parent->level+1;
    max_level = parent->max_level;
    test_func = parent->test_func;
    Node p1 = *pts[0];
    Node p4 = *pts[3];
    center[0] = (p1[0]+p4[0])/2;
    center[1] = (p1[1]+p4[1])/2;
    key = key_func(center[0],center[1]);
    children[0] = NULL;
    children[1] = NULL;
    children[2] = NULL;
    children[3] = NULL;
    neighbors[0] = NULL;
    neighbors[1] = NULL;
    neighbors[2] = NULL;
    neighbors[3] = NULL;
};

void QuadCell::spawn(node_map_t& nodes, QuadCell *kids[4]){
    Node *p1 = points[0];
    Node *p2 = points[1];
    Node *p3 = points[2];
    Node *p4 = points[3];
    int_t x0,y0,xC,yC,xF,yF;
    x0 = p1->location[0];
    y0 = p1->location[1];
    xF = p4->location[0];
    yF = p4->location[1];
    xC = center[0];
    yC = center[1];
    /*
        p3--p9--p4
        |    |   |
        p6--p7--p8
        |    |   |
        p1--p5--p2
    */

    Node *p5,*p6,*p7,*p8,*p9;
    p5 = set_default_node(nodes,xC,y0);
    p6 = set_default_node(nodes,x0,yC);
    p7 = set_default_node(nodes,xC,yC);
    p8 = set_default_node(nodes,xF,yC);
    p9 = set_default_node(nodes,xC,yF);

    Node * pQC1[4] = {p1,p5,p6,p7};
    Node * pQC2[4] = {p5,p2,p7,p8};
    Node * pQC3[4] = {p6,p7,p3,p9};
    Node * pQC4[4] = {p7,p8,p9,p4};

    kids[0] = new QuadCell(pQC1,this);
    kids[1] = new QuadCell(pQC2,this);
    kids[2] = new QuadCell(pQC3,this);
    kids[3] = new QuadCell(pQC4,this);
};

void QuadCell::set_neighbor(QuadCell * other, int_t position){
    if(other==NULL){
        //std::cout<<"Other cell was NULL"<<std::endl;
        return;
    }
    //std::cout<<"Making cells "<<center[0]<<","<<center[1];
    //std::cout<<" and "<< other->center[0]<<","<<other->center[1];
    //std::cout<<" neighbors"<<std::endl;
    if(level != other->level){
        neighbors[position] = other;
    }else{
        neighbors[position] = other;
        other->neighbors[position^1] = this;
    }//else{
     //   neighbors[position] = other;
     //   other->neighbors[position^1] = parent;
    //}
};

void QuadCell::divide(node_map_t& nodes, bool force=false){

    bool do_splitting = false;
    if(level==max_level){
        do_splitting = false;
    }else if(force){
        do_splitting = true;
    }else{
        int test_level = (*test_func)(this);
        if(test_level > level){
            do_splitting = true;
        }
    }
    if(!do_splitting){
        return;
    }
    //If i haven't already been split...
    if(children[0]==NULL){
        spawn(nodes, children);

        //If I need to be split, and my neighbor is below my level
        //Then it needs to be split
        //left
        if(neighbors[0]!= NULL && neighbors[0]->level < level){
            neighbors[0]->divide(nodes, true);
        }
        //right
        if(neighbors[1] != NULL && neighbors[1]->level < level){
            neighbors[1]->divide(nodes, true);
        }
        //down
        if(neighbors[2] != NULL && neighbors[2]->level < level){
            neighbors[2]->divide(nodes, true);;
        }
        //up
        if(neighbors[3] != NULL && neighbors[3]->level < level){
            neighbors[3]->divide(nodes, true);
        }

        //Set children's neighbors (first do the easy ones)
        // all of the children live next to each other
        children[0]->set_neighbor(children[1],1);
        children[0]->set_neighbor(children[2],3);
        children[1]->set_neighbor(children[3],3);
        children[2]->set_neighbor(children[3],1);

        //std::cout<<children[0]->neighbors[1]->key<<","<<children[1]->key<<std::endl;
        //std::cout<<children[1]->neighbors[0]->key<<","<<children[0]->key<<std::endl;

        if(neighbors[0] != NULL && !(neighbors[0]->is_leaf())){
            children[0]->set_neighbor(neighbors[0]->children[1],0);
            children[2]->set_neighbor(neighbors[0]->children[3],0);
        }
        else{
            children[0]->set_neighbor(neighbors[0],0);
            children[2]->set_neighbor(neighbors[0],0);
        }

        if(neighbors[1] != NULL && !neighbors[1]->is_leaf()){
            children[1]->set_neighbor(neighbors[1]->children[0],1);
            children[3]->set_neighbor(neighbors[1]->children[2],1);
        }else{
            children[1]->set_neighbor(neighbors[1],1);
            children[3]->set_neighbor(neighbors[1],1);
        }
        if(neighbors[2] != NULL && !neighbors[2]->is_leaf()){
            children[0]->set_neighbor(neighbors[2]->children[2],2);
            children[1]->set_neighbor(neighbors[2]->children[3],2);
        }else{
            children[0]->set_neighbor(neighbors[2],2);
            children[1]->set_neighbor(neighbors[2],2);
        }
        if(neighbors[3] != NULL && !neighbors[3]->is_leaf()){
            children[2]->set_neighbor(neighbors[3]->children[0],3);
            children[3]->set_neighbor(neighbors[3]->children[1],3);
        }else{
            children[2]->set_neighbor(neighbors[3],3);
            children[3]->set_neighbor(neighbors[3],3);
        }
    }
    if(!force){
        children[0]->divide(nodes);
        children[1]->divide(nodes);
        children[2]->divide(nodes);
        children[3]->divide(nodes);
    }
};

void QuadCell::build_cell_vector(quad_vec_t& cells){
    if(this->is_leaf()){
        cells.push_back(this);
        return;
    }
    children[0]->build_cell_vector(cells);
    children[1]->build_cell_vector(cells);
    children[2]->build_cell_vector(cells);
    children[3]->build_cell_vector(cells);
}

bool QuadCell::inside_triangle(double x, double y, int_t direction){
    int p0x,p0y,p1x,p1y,p2x,p2y;

    p0x = center[0];
    p0y = center[1];
    if(direction==0){
      p1x = neighbors[0]->children[1]->center[0];
      p1y = neighbors[0]->children[1]->center[1];
      p2x = neighbors[0]->children[3]->center[0];
      p2y = neighbors[0]->children[3]->center[1];
    }else if(direction==1){
      p1x = neighbors[1]->children[0]->center[0];
      p1y = neighbors[1]->children[0]->center[1];
      p2x = neighbors[1]->children[2]->center[0];
      p2y = neighbors[1]->children[2]->center[1];
    }else if(direction==2){
      p1x = neighbors[2]->children[2]->center[0];
      p1y = neighbors[2]->children[2]->center[1];
      p2x = neighbors[2]->children[3]->center[0];
      p2y = neighbors[2]->children[3]->center[1];
    }else{
      p1x = neighbors[3]->children[0]->center[0];
      p1y = neighbors[3]->children[0]->center[1];
      p2x = neighbors[3]->children[1]->center[0];
      p2y = neighbors[3]->children[1]->center[1];
    }

    int A = (-p1y*p2x+p0y*(p2x-p1x)+p0x*(p1y-p2y)+p1x*p2y);
    int sign = (A<0)?-1:1;
    double s = (p0y*p2x-p0x*p2y+(p2y-p0y)*x+(p0x-p2x)*y)*sign;
    double t = (p0x*p1y-p0y*p1x+(p0y-p1y)*x+(p1x-p0x)*y)*sign;
    return s>0 && t>0 && (s+t)<A*sign;
}

QuadCell* QuadCell::containing_cell(double x, double y){
    if(is_leaf()){
      return this;
    }
    if(y<center[1]){
      if(x<center[0]){
        return children[0]->containing_cell(x,y);
      }else{
        return children[1]->containing_cell(x,y);
      }
    }else{
      if(x<center[0]){
        return children[2]->containing_cell(x,y);
      }else{
        return children[3]->containing_cell(x,y);
      }
    }
};

QuadCell::~QuadCell(){
        if(is_leaf()){
            return;
        }
        delete children[0];
        delete children[1];
        delete children[2];
        delete children[3];
};

QuadTree::QuadTree(){
    nx = 0;
    ny = 0;
    max_level = 0;
    root = NULL;
};

void QuadTree::set_level(int_t levels){
    max_level = levels;
    nx = 2<<max_level;
    ny = 2<<max_level;
};

void QuadTree::build_tree(function test_func){
    Node *p1 = new Node(0 ,0 );
    Node *p2 = new Node(nx,0 );
    Node *p3 = new Node(0 ,ny);
    Node *p4 = new Node(nx,ny);
    nodes[p1->key]=p1;
    nodes[p2->key]=p2;
    nodes[p3->key]=p3;
    nodes[p4->key]=p4;

    Node* points[4] = {p1,p2,p3,p4};
    root = new QuadCell(points, max_level, test_func);
    root->divide(nodes);
    root->build_cell_vector(cells);

    //Generate Edges?
    for(std::vector<QuadCell *>::size_type i = 0; i != cells.size(); i++){
        QuadCell *cell = cells[i];
        Node *p1 = cell->points[0];
        Node *p2 = cell->points[1];
        Node *p3 = cell->points[2];
        Node *p4 = cell->points[3];
        Edge *e1,*e2,*e3,*e4;
        e1 = set_default_edge(edges_y,*p1,*p3);
        e2 = set_default_edge(edges_x,*p3,*p4);
        e3 = set_default_edge(edges_y,*p2,*p4);
        e4 = set_default_edge(edges_x,*p1,*p2);
        cell->edges[0] = e1;
        cell->edges[1] = e2;
        cell->edges[2] = e3;
        cell->edges[3] = e4;
        cell->index = i;

        p1->reference++;
        p2->reference++;
        p3->reference++;
        p4->reference++;

        e1->reference++;
        e2->reference++;
        e3->reference++;
        e4->reference++;
    }

    //Find hanging nodes
    for(node_it_type it = nodes.begin(); it != nodes.end(); ++it){
        Node *node = it->second;
        if(node->reference < 4){
            int_t x,y;
            x = node->location[0];
            y = node->location[1];
            if(!(x==0 || x==nx)){
                if( !(y==0 || y==ny)){
                    node->hanging = true;
                    hanging_nodes.push_back(node);
                }
            }
        }
    }

    //Process Edges?
    for(edge_it_type it = edges_x.begin(); it != edges_x.end(); ++it){
        Edge *edge = it->second;
        if(edge->reference<2){
            int_t x,y;
            x = edge->location[0];
            y = edge->location[1];
            if( !(x==0 || x==nx)){
                if( !(y==0 || y==ny)){
                    Edge *parent;
                    Node *node;
                    if(edge->points[0]->hanging){
                        node = edge->points[0];
                        parent = edges_x.at(node->key);
                    }else if(edge->points[1]->hanging){
                        node = edge->points[1];
                        parent = edges_x.at(node->key);
                    }else{
                        continue;
                    }
                    edge->hanging = true;
                    edge->parent = parent;
                    hanging_edges_x.push_back(edge);
                    node->parents[0] = parent->points[0];
                    node->parents[1] = parent->points[1];
                }
            }
        }
    }
    for(edge_it_type it = edges_y.begin(); it != edges_y.end(); ++it){
        Edge *edge = it->second;
        if(edge->reference<2){
            int_t x,y;
            x = edge->location[0];
            y = edge->location[1];
            if( !(x==0 || x==nx)){
                if( !(y==0 || y==ny)){
                    Edge *parent;
                    Node *node;
                    if(edge->points[0]->hanging){
                        node = edge->points[0];
                        parent = edges_y.at(node->key);
                    }else if(edge->points[1]->hanging){
                        node = edge->points[1];
                        parent = edges_y.at(node->key);
                    }else{
                        continue;
                    }
                    edge->hanging = true;
                    edge->parent = parent;
                    hanging_edges_y.push_back(edge);
                    node->parents[0] = parent->points[0];
                    node->parents[1] = parent->points[1];
                }
            }
        }
    }
};

void QuadTree::number(){
    //Number Nodes
    int_t ii,ih;
    ii = 0;
    ih = nodes.size()-hanging_nodes.size();
    for(node_it_type it = nodes.begin(); it != nodes.end(); ++it){
        Node *node = it->second;
        if(node->hanging){
            node->index = ih;
            ++ih;
        }else{
            node->index = ii;
            ++ii;
        }
    }
    //Number edges_x
    ii = 0;
    ih = edges_x.size()-hanging_edges_x.size();
    for(edge_it_type it = edges_x.begin(); it != edges_x.end(); ++it){
        Edge *edge = it->second;
        if(edge->hanging){
          edge->index = ih;
          ++ih;
        }else{
          edge->index = ii;
          ++ii;
        }
    }
    //Number edges_y
    ii = 0;
    ih = edges_y.size()-hanging_edges_y.size();
    for(edge_it_type it = edges_y.begin(); it != edges_y.end(); ++it){
        Edge *edge = it->second;
        if(edge->hanging){
          edge->index = ih;
          ++ih;
        }else{
          edge->index = ii;
          ++ii;
        }
    }
};

QuadTree::~QuadTree(){
    if (root==NULL){
        return;
    }
    delete root;
    for(node_it_type it = nodes.begin(); it != nodes.end(); it++){
        delete it->second;
    }
    for(edge_it_type it = edges_x.begin(); it != edges_x.end(); it++){
        delete it->second;
    }
    for(edge_it_type it = edges_y.begin(); it != edges_y.end(); it++){
        delete it->second;
    }
    cells.clear();
    nodes.clear();
    edges_x.clear();
    edges_y.clear();
};

QuadCell* QuadTree::containing_cell(double x, double y){
    return root->containing_cell(x,y);
}
