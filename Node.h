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

public:
	NRVec<dtype> sval;  //only in APCNodes, SparseCNodes;
	NRVec<dtype> sloss; //only in APCNodes, SparseCNodes;
	bool smode;

public:
	int dim;
	int lock;  //node can backward only when lock = 0;

	int sid;
	bool lossed;


public:
	Node(){
		val.setZero();
		loss.setZero();
		dim = 0;
		lock = 0;
		sid = rand();

		smode = false;
		sval.dealloc();
		sloss.dealloc();
		lossed = false;
	}

public:
	virtual inline void clearValue(){
		if (!smode){
			val.setZero();
			loss.setZero();
		}
		else{
			sval = 0;
			sloss = 0;
		}
		lock = 0;
		lossed = false;

	}

	virtual inline void clear(){
		val.setZero();
		loss.setZero();
		dim = 0;
		lock = 0;
		lossed = false;

		sval.dealloc();
		sloss.dealloc();
	}

	virtual inline void backward(){
	}

	virtual inline void unlock(){
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
