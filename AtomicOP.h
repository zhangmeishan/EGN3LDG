#ifndef ATOMICOP
#define ATOMICOP

#include "Eigen/Dense"
#include "MyLib.h"
#include "Node.h"
#include "Graph.h"

using namespace Eigen;

//Offen two nodes, sometimes may be more, but better by using two-two composition to achieve as back-propagation reasons.
struct PMultNode : Node {
	PNode in1, in2;
public:
	PMultNode(){
		clear();
	}
public:
	virtual inline void clearValue(){
		Node::clearValue();
		in1 = NULL;
		in2 = NULL;
	}

	virtual inline void clear(){
		Node::clear();
		in1 = NULL;
		in2 = NULL;
	}

public:
	void forward(Graph *cg, PNode x1, PNode x2) {
		in1 = x1;
		in2 = x2;
		val = x1->val.array() * x2->val.array();
		in1->lock++;
		in2->lock++;
		cg->addNode(this);
	}

	void backward(){
		if (in1->loss.size() == 0) {
			in1->loss = Mat::Zero(in1->val.rows(), in1->val.cols());
		}
		in1->loss = in1->loss.array() + loss.array() * in2->val.array();

		if (in2->loss.size() == 0) {
			in2->loss = Mat::Zero(in2->val.rows(), in2->val.cols());
		}
		in2->loss = in2->loss.array() + loss.array() * in1->val.array();
	}

	inline void unlock(){
		in1->lock--;
		in2->lock--;
		if(!lossed)return;
		in1->lossed = true;
		in2->lossed = true;
	}

};


struct PAddNode : Node {
	vector<PNode> ins;
	int nSize;
public:
	PAddNode(){
		clear();
	}
public:
	virtual inline void clearValue(){
		Node::clearValue();
		ins.clear();
		nSize = 0;
	}

	virtual inline void clear(){
		Node::clear();
		ins.clear();
		nSize = 0;
	}

public:
	// please better restrict col to 1
	void forward(Graph *cg, const vector<PNode>& x) {
		nSize = x.size();
		if (nSize < 2){
			std::cout << "at least two nodes are required" << std::endl;
			return;
		}

		ins.clear();
		for (int i = 0; i < nSize; i++){
			ins.push_back(x[i]);
		}

		forward();
		cg->addNode(this);
	}

	void forward(Graph *cg, PNode x1, PNode x2){
		ins.clear();
		ins.push_back(x1);
		ins.push_back(x2);
		nSize = 2;

		forward();
		cg->addNode(this);
	}

	void forward(Graph *cg, PNode x1, PNode x2, PNode x3){
		ins.clear();
		ins.push_back(x1);
		ins.push_back(x2);
		ins.push_back(x3);
		nSize = 3;

		forward();
		cg->addNode(this);
	}

	void forward(Graph *cg, PNode x1, PNode x2, PNode x3, PNode x4){
		ins.clear();
		ins.push_back(x1);
		ins.push_back(x2);
		ins.push_back(x3);
		ins.push_back(x4);
		nSize = 4;

		forward();
		cg->addNode(this);
	}

	void forward(Graph *cg, PNode x1, PNode x2, PNode x3, PNode x4, PNode x5){
		ins.clear();
		ins.push_back(x1);
		ins.push_back(x2);
		ins.push_back(x3);
		ins.push_back(x4);
		ins.push_back(x5);
		nSize = 5;

		forward();
		cg->addNode(this);
	}

	void forward(Graph *cg, PNode x1, PNode x2, PNode x3, PNode x4, PNode x5, PNode x6){
		ins.clear();
		ins.push_back(x1);
		ins.push_back(x2);
		ins.push_back(x3);
		ins.push_back(x4);
		ins.push_back(x5);
		ins.push_back(x6);
		nSize = 6;

		forward();
		cg->addNode(this);
	}


	void backward(){
		for (int i = 0; i < nSize; i++){
			if (ins[i]->loss.size() == 0){
				ins[i]->loss = Mat::Zero(dim, 1);
			}
			for (int idx = 0; idx < dim; idx++){
				ins[i]->loss.coeffRef(idx) += loss.coeffRef(idx);
			}
		}
	}

	inline void unlock(){
		for (int i = 0; i < nSize; i++){
			ins[i]->lock--;
		}
		if(!lossed)return;
		for (int i = 0; i < nSize; i++){
			ins[i]->lossed = true;
		}		
	}

protected:
	void forward() {
		dim = ins[0]->val.rows();
		if (val.size() == 0){
			val = Mat::Zero(dim, 1);
		}
		for (int idx = 0; idx < nSize; idx++){
			for (int idy = 0; idy < dim; idy++){
				val.coeffRef(idy) += ins[idx]->val.coeffRef(idy);
			}			
		}
		for (int idx = 0; idx < nSize; idx++){
			ins[idx]->lock++;
		}
	}

};


struct ActivateNode : Node {
	PNode in;

	Mat(*activate)(const Mat&);   // activation function
	Mat(*derivate)(const Mat&, const Mat&);  // derivation function of activation function
public:
	ActivateNode(){
		clear();
	}
public:
	virtual inline void clearValue(){
		Node::clearValue();
		in = NULL;
	}

	virtual inline void clear(){
		Node::clear();
		activate = tanh;
		derivate = tanh_deri;
		in = NULL;
	}

	// define the activation function and its derivation form
	inline void setFunctions(Mat(*f)(const Mat&), Mat(*f_deri)(const Mat&, const Mat&)) {
		activate = f;
		derivate = f_deri;
	}

public:

	inline void forward(Graph *cg, PNode x){
		in = x;
		val = activate(in->val);
		in->lock++;
		cg->addNode(this);
	}

	inline void backward(){
		if (in->loss.size() == 0) {
			in->loss = Mat::Zero(in->val.rows(), in->val.cols());
		}
		in->loss = in->loss.array() + loss.array() * derivate(in->val, val).array();
	}

	inline void unlock(){
		in->lock--;
		if(!lossed)return;
		in->lossed = true;
	}

};


//special case of ActivateNode
struct TanhNode : Node {
	PNode in;

public:
	TanhNode(){
		clear();
	}
public:
	virtual inline void clearValue(){
		Node::clearValue();
		in = NULL;
	}

	virtual inline void clear(){
		Node::clear();
		in = NULL;
	}


public:

	inline void forward(Graph *cg, PNode x){
		in = x;
		val = tanh(in->val);
		in->lock++;
		cg->addNode(this);
	}

	inline void backward(){
		if (in->loss.size() == 0) {
			in->loss = Mat::Zero(in->val.rows(), in->val.cols());
		}
		in->loss = in->loss.array() + loss.array() * tanh_deri(in->val, val).array();
	}

	inline void unlock(){
		in->lock--;
		if(!lossed)return;
		in->lossed = true;
	}

};


//special case of ActivateNode
struct SigmoidNode : Node {
	PNode in;

public:
	SigmoidNode(){
		clear();
	}
public:
	virtual inline void clearValue(){
		Node::clearValue();
		in = NULL;
	}

	virtual inline void clear(){
		Node::clear();
		in = NULL;
	}


public:

	inline void forward(Graph *cg, PNode x){
		in = x;
		val = sigmoid(in->val);
		in->lock++;
		cg->addNode(this);
	}

	inline void backward(){
		if (in->loss.size() == 0) {
			in->loss = Mat::Zero(in->val.rows(), in->val.cols());
		}
		in->loss = in->loss.array() + loss.array() * sigmoid_deri(in->val, val).array();
	}

	inline void unlock(){
		in->lock--;
		if(!lossed)return;
		in->lossed = true;
	}

};

//special case of ActivateNode
struct RELUNode : Node {
	PNode in;

public:
	RELUNode(){
		clear();
	}
public:
	virtual inline void clearValue(){
		Node::clearValue();
		in = NULL;
	}

	virtual inline void clear(){
		Node::clear();
		in = NULL;
	}


public:

	inline void forward(Graph *cg, PNode x){
		in = x;
		val = relu(in->val);
		in->lock++;
		cg->addNode(this);
	}

	inline void backward(){
		if (in->loss.size() == 0) {
			in->loss = Mat::Zero(in->val.rows(), in->val.cols());
		}
		in->loss = in->loss.array() + loss.array() * relu_deri(in->val, val).array();
	}

	inline void unlock(){
		in->lock--;
		if(!lossed)return;
		in->lossed = true;
	}

};


#endif
