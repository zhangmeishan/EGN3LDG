#ifndef POOLING
#define POOLING

#include "MyLib.h"
#include "Node.h"
#include "Graph.h"

struct PoolNode : Node {
public:
	vector<Mat> masks;
	vector<PNode> ins;

public:
	PoolNode(){
		clear();
	}

	inline void clear(){
		Node::clear();
		masks.clear();
		ins.clear();
	}

	inline void clearValue(){
		Node::clearValue();
		masks.clear();
		ins.clear();
	}

public:

	virtual void forward(Graph *cg, const vector<PNode>& x) = 0;

	void backward(){
		for (int i = 0; i < ins.size(); i++){
			if (ins[i]->loss.size() == 0){
				ins[i]->loss = Mat::Zero(ins[i]->val.rows(), ins[i]->val.cols());
			}
			ins[i]->loss = ins[i]->loss.array() + loss.array() * masks[i].array();
			ins[i]->lock--;
		}
	}
};

struct MaxPoolNode : PoolNode {
public:
	MaxPoolNode(){
	}

public:
	//Be careful that the row is the dim of input vector, and the col is the number of input vectors
	//Another point is that we change the input vectors directly.
	void forward(Graph *cg, const vector<PNode>& x) {
		if (x.size() == 0 ){
			std::cout << "empty inputs for max pooling" << std::endl;
			return;
		}

		ins.clear();
		for (int i = 0; i < x.size(); i++){
			ins.push_back(x[i]);
		}

		dim = ins[0]->val.rows();
		masks.resize(ins.size());
		for (int i = 0; i < ins.size(); ++i){
			if (ins[i]->val.rows() != dim){
				std::cout << "input matrixes are not matched" << std::endl;
				clearValue();
				return;
			}
			masks[i] = Mat::Zero(dim, 1);
		}


		for (int idx = 0; idx < dim; idx++){
			int maxIndex = -1;
			for (int i = 0; i < ins.size(); ++i){
				if (maxIndex == -1 || ins[i]->val(idx, 0) > ins[maxIndex]->val(idx, 0)){
					maxIndex = i;
				}
			}
			masks[maxIndex](idx, 0) = 1.0;
		}

		val = Mat::Zero(dim, 1);
		for (int i = 0; i < ins.size(); ++i){
			val = val.array() + masks[i].array() *ins[i]->val.array();
		}

		for (int i = 0; i < ins.size(); ++i){
			ins[i]->lock++;
		}

		cg->addNode(this);
	}

};


struct SumPoolNode : PoolNode {
public:
	SumPoolNode(){
	}

public:
	//Be careful that the row is the dim of input vector, and the col is the number of input vectors
	//Another point is that we change the input vectors directly.
	void forward(Graph *cg, const vector<PNode>& x) {
		if (x.size() == 0){
			std::cout << "empty inputs for max pooling" << std::endl;
			return;
		}

		ins.clear();
		for (int i = 0; i < x.size(); i++){
			ins.push_back(x[i]);
		}

		dim = ins[0]->val.rows();
		masks.resize(ins.size());
		for (int i = 0; i < ins.size(); ++i){
			if (ins[i]->val.rows() != dim){
				std::cout << "input matrixes are not matched" << std::endl;
				clearValue();
				return;
			}
			masks[i] = Mat::Ones(dim, 1);
		}

		val = Mat::Zero(dim, 1);
		for (int i = 0; i < ins.size(); ++i){
			val = val.array() + masks[i].array() *ins[i]->val.array();
		}

		for (int i = 0; i < ins.size(); ++i){
			ins[i]->lock++;
		}

		cg->addNode(this);
	}

};


struct MinPoolNode : PoolNode {
public:
	MinPoolNode(){
	}

public:
	//Be careful that the row is the dim of input vector, and the col is the number of input vectors
	//Another point is that we change the input vectors directly.
	void forward(Graph *cg, const vector<PNode>& x) {
		if (x.size() == 0){
			std::cout << "empty inputs for max pooling" << std::endl;
			return;
		}

		ins.clear();
		for (int i = 0; i < x.size(); i++){
			ins.push_back(x[i]);
		}

		dim = ins[0]->val.rows();
		masks.resize(ins.size());
		for (int i = 0; i < ins.size(); ++i){
			if (ins[i]->val.rows() != dim){
				std::cout << "input matrixes are not matched" << std::endl;
				clearValue();
				return;
			}
			masks[i] = Mat::Zero(dim, 1);
		}


		for (int idx = 0; idx < dim; idx++){
			int minIndex = -1;
			for (int i = 0; i < ins.size(); ++i){
				if (minIndex == -1 || ins[i]->val(idx, 0) < ins[minIndex]->val(idx, 0)){
					minIndex = i;
				}
			}
			masks[minIndex](idx, 0) = 1.0;
		}

		val = Mat::Zero(dim, 1);
		for (int i = 0; i < ins.size(); ++i){
			val = val.array() + masks[i].array() *ins[i]->val.array();
		}

		for (int i = 0; i < ins.size(); ++i){
			ins[i]->lock++;
		}

		cg->addNode(this);
	}

};


#endif
