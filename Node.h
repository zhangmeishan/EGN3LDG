#ifndef BasicNode
#define BasicNode

#include "Eigen/Dense"

using namespace Eigen;

// one Node means a vector
// the col should be 1, because we aimed for NLP only
struct Node {
public:
	Mat val;
	Mat loss;
	int dim;	
	int lock;  //node can backward only when lock = 0;


public:
	Node(){
		val.setZero();
		loss.setZero();
		dim = 0;
		lock = 0;
	}	

public: 
	virtual inline void clearValue(){
		val.setZero();
		loss.setZero();
		lock = 0;
	}

	virtual inline void clear(){
		val.setZero();
		loss.setZero();
		dim = 0;
		lock = 0;
	}

	virtual inline void backward(){
	}

	void check(){
		if (val.size() > 0){
			assert(val.cols() == 1);
			assert(val.rows() == dim);
		}
	}

	//virtual inline void forward(Graph *cg, ...) = 0

};

typedef  Node* PNode;


// The nodes created by 
class NodeBuilder{
public:
	// require definition of output nodes here, not fixed

public:
	virtual inline void resize(int maxsize) = 0;
	virtual inline void clear() = 0;

	//Require forward to build node connections as well, but the parameter is not fixed.
	//virtual inline void forward(...) = 0;

};


#endif
