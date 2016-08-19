#ifndef ATOMICOP
#define ATOMICOP

#include "Eigen/Dense"
#include "MyLib.h"
#include "Node.h"

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
	void forward(PNode x1, PNode x2) {
		in1 = x1;
		in2 = x2;
		val = x1->val.array() * x2->val.array();
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

};


struct PAddNode : Node {
	vector<PNode> ins;
public:
	PAddNode(){
		clear();
	}
public:
	virtual inline void clearValue(){
		Node::clearValue();
		ins.clear();
	}

	virtual inline void clear(){
		Node::clear();
		ins.clear();
	}

public:
	// please better restrict col to 1
	void forward(const vector<PNode>& x) {
		if (x.size() < 2){
			std::cout << "at least two nodes are required" << std::endl;
			return;
		}

		ins.clear();
		for (int i = 0; i < x.size(); i++){
			ins.push_back(x[i]);
		}

		forward();
	}

	void forward(PNode x1, PNode x2){
		ins.clear();
		ins.push_back(x1);
		ins.push_back(x2);

		forward();
	}

	void forward(PNode x1, PNode x2, PNode x3){
		ins.clear();
		ins.push_back(x1);
		ins.push_back(x2);
		ins.push_back(x3);

		forward();
	}

	void forward(PNode x1, PNode x2, PNode x3, PNode x4){
		ins.clear();
		ins.push_back(x1);
		ins.push_back(x2);
		ins.push_back(x3);
		ins.push_back(x4);

		forward();
	}

	void forward(PNode x1, PNode x2, PNode x3, PNode x4, PNode x5){
		ins.clear();
		ins.push_back(x1);
		ins.push_back(x2);
		ins.push_back(x3);
		ins.push_back(x4);
		ins.push_back(x5);

		forward();
	}

	void forward(PNode x1, PNode x2, PNode x3, PNode x4, PNode x5, PNode x6){
		ins.clear();
		ins.push_back(x1);
		ins.push_back(x2);
		ins.push_back(x3);
		ins.push_back(x4);
		ins.push_back(x5);
		ins.push_back(x6);

		forward();
	}


	void backward(){
		for (int i = 0; i < ins.size(); i++){
			if (ins[i]->loss.size() == 0){
				ins[i]->loss = Mat::Zero(ins[i]->val.rows(), ins[i]->val.cols());
			}
			ins[i]->loss += loss;
		}

	}

protected:
	void forward() {
		dim = ins[0]->val.rows();
		val = Mat::Zero(dim, 1);
		for (int idx = 0; idx < ins.size(); idx++){
			val += ins[idx]->val;
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

	inline void forward(PNode x){
		in = x;
		val = activate(in->val);
	}

	inline void backward(){
		if (in->loss.size() == 0) {
			in->loss = Mat::Zero(in->val.rows(), in->val.cols());
		}
		in->loss = in->loss.array() + loss.array() * derivate(in->val, val).array();
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

	inline void forward(PNode x){
		in = x;
		val = tanh(in->val);
	}

	inline void backward(){
		if (in->loss.size() == 0) {
			in->loss = Mat::Zero(in->val.rows(), in->val.cols());
		}
		in->loss = in->loss.array() + loss.array() * tanh_deri(in->val, val).array();
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

	inline void forward(PNode x){
		in = x;
		val = sigmoid(in->val);
	}

	inline void backward(){
		if (in->loss.size() == 0) {
			in->loss = Mat::Zero(in->val.rows(), in->val.cols());
		}
		in->loss = in->loss.array() + loss.array() * sigmoid_deri(in->val, val).array();
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

	inline void forward(PNode x){
		in = x;
		val = relu(in->val);
	}

	inline void backward(){
		if (in->loss.size() == 0) {
			in->loss = Mat::Zero(in->val.rows(), in->val.cols());
		}
		in->loss = in->loss.array() + loss.array() * relu_deri(in->val, val).array();
	}

};


#endif
