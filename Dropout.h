#ifndef DROPOUT
#define DROPOUT

#include "Eigen/Dense"
#include "MyLib.h"
#include "Node.h"
#include "Graph.h"

using namespace Eigen;

struct DropNode : Node {
public:
	PNode in;
	Mat mask;
	dtype p;
	
public:
	DropNode(){
		clear();
	}

	inline void clear(){
		Node::clear();
		mask.setOnes();
		p = 0.0;
		in = NULL;
	}

	inline void clearValue(){
		Node::clearValue();
		mask.setOnes();
		in = NULL;
	}

	inline void setDropValue(dtype v){
		p = v;
	}
		
	
public:
	//Be careful that the row is the dim of input vector, and the col is the number of input vectors
	void forward(Graph *cg, PNode x) {
		in = x;
		mask = Mat::Ones(in->val.rows(), in->val.cols());
		if (cg->train){
			std::vector<int> indexes;
			for (int i = 0; i < in->val.rows(); ++i)
				indexes.push_back(i);

			int dropNum = (int)(in->val.rows() * p);

			for (int j = 0; j < in->val.cols(); j++){
				random_shuffle(indexes.begin(), indexes.end());
				for (int i = 0; i < dropNum; i++){
					mask(indexes[i], j) = 0.0;
				}
			}
		}
		else{
			mask = mask * (1.0 - p);
		}
		
		val = in->val.array() * mask.array();

		in->lock++;
		cg->addNode(this);
	}
	
	
	inline void backward(){
		if (in->loss.size() == 0) {
			in->loss = Mat::Zero(in->val.rows(), in->val.cols());
		}
		in->loss = loss.array() * mask.array();		
	}

	inline void unlock(){
		in->lock--;
		if(!validLoss(loss))return;
		in->lossed = true;
	}
	
};


#endif
