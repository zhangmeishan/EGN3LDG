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

//for dropout only
public:
	Mat mask;
	bool usedrop;
	dtype dropvalue;
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

		usedrop = false;
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

	inline void setDropout(const dtype& dropOut){
		if (dropOut > 0 && dropOut < 1){
			dropvalue = dropOut;
			usedrop = true;
		}
		else{
			dropvalue = -1;
			usedrop = false;
		}
	}
	inline void applydrop_forward(bool train){
		if (usedrop)
		{
			mask = Mat::Ones(val.rows(), val.cols());
			if (train){
				std::vector<int> indexes;
				for (int i = 0; i < val.rows(); ++i)
					indexes.push_back(i);

				int dropNum = (int)(val.rows() * dropvalue);

				for (int j = 0; j < val.cols(); j++){
					random_shuffle(indexes.begin(), indexes.end());
					for (int i = 0; i < dropNum; i++){
						mask(indexes[i], j) = 0.0;
					}
				}
			}
			else{
				mask = mask * (1.0 - dropvalue);
			}

			val = val.array() * mask.array();
		}
	}

	inline void applydrop_backward(bool train){
		if (usedrop)
		{
			if (loss.size() == 0) {
				loss = Mat::Zero(val.rows(), val.cols());
			}
			loss = loss.array() * mask.array();
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
