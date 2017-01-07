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
	PMultNode() : Node(){
		in1 = NULL;
		in2 = NULL;
	}
public:
	virtual inline void clearValue(){
		Node::clearValue();
		in1 = NULL;
		in2 = NULL;
	}

public:
	void forward(Graph *cg, PNode x1, PNode x2) {
		in1 = x1;
		in2 = x2;
		val.vec() = x1->val.vec() * x2->val.vec();
		in1->increase_loc();
		in2->increase_loc();
		cg->addNode(this);
	}

	void backward(){
		in1->loss.vec() += loss.vec() * in2->val.vec();
		in2->loss.vec() += loss.vec() * in1->val.vec();
	}

	inline void unlock(){
		in1->decrease_loc();
		in2->decrease_loc();
		if(!lossed)return;
		in1->lossed = true;
		in2->lossed = true;
	}

};


struct CHeadNode : Node {
	PNode in;
public:
	CHeadNode() : Node() {
		in = NULL;
	}
public:
	virtual inline void clearValue() {
		Node::clearValue();
		in = NULL;
	}

public:
	void forward(Graph *cg, PNode x) {
		in = x;
		for (int idx = 0; idx < dim && idx < in->dim; idx++) {
			val[idx] = in->val[idx];
		}
		in->increase_loc();
		cg->addNode(this);
	}

	void backward() {
		for (int idx = 0; idx < dim && idx < in->dim; idx++) {
			in->loss[idx] += loss[idx];
		}
	}

	inline void unlock() {
		in->decrease_loc();
		if (!lossed)return;
		in->lossed = true;
	}

};

struct HalfMergeNode : Node {
	PNode in;
public:
	HalfMergeNode() : Node() {
		in = NULL;
	}
public:
	virtual inline void clearValue() {
		Node::clearValue();
		in = NULL;
	}

public:
	void forward(Graph *cg, PNode x) {
		in = x;
		if (x->dim != 2 * dim) {
			std::cout << "error during half merging" << std::endl;
		}
		for (int idx = 0; idx < dim; idx++) {
			val[idx] = in->val[2 * idx] + in->val[2 * idx + 1];
		}
		in->increase_loc();
		cg->addNode(this);
	}

	void backward() {
		for (int idx = 0; idx < dim; idx++) {
			in->loss[2 * idx] += loss[idx];
			in->loss[2 * idx + 1] += loss[idx];
		}
	}

	inline void unlock() {
		in->decrease_loc();
		if (!lossed)return;
		in->lossed = true;
	}

};

//select a continus space from input
struct SelectionNode : Node {
	PNode in;
	int start_pos;
public:
	SelectionNode() : Node() {
		in = NULL;
		start_pos = 0;
	}
public:
	virtual inline void clearValue() {
		Node::clearValue();
		in = NULL;
		start_pos = 0;
	}

public:
	void forward(Graph *cg, PNode x, const int& start) {
		in = x;
		start_pos = start;
		if (start_pos < 0 || start_pos + dim > in->dim) {
			std::cout << "error: position overflow!" << std::endl;
			return;
		}
		int end_pos = start_pos + dim;
		int offset = 0;
		for (int idx = start_pos; idx < end_pos; idx++) {
			val[offset] = in->val[idx];
			offset++;
		}
		in->increase_loc();
		cg->addNode(this);
	}

	void backward() {
		int end_pos = start_pos + dim;
		int offset = 0;
		for (int idx = start_pos; idx < end_pos; idx++) {
			in->loss[idx] += loss[offset];
			offset++;
		}
	}

	inline void unlock() {
		in->decrease_loc();
		if (!lossed)return;
		in->lossed = true;
	}
};



struct PSubNode : Node {
	PNode in1, in2;
public:
	PSubNode() : Node(){
		in1 = NULL;
		in2 = NULL;
	}
public:
	virtual inline void clearValue(){
		Node::clearValue();
		in1 = NULL;
		in2 = NULL;
	}

public:
	void forward(Graph *cg, PNode x1, PNode x2) {
		in1 = x1;
		in2 = x2;
		val.vec() = x1->val.vec() - x2->val.vec();
		in1->increase_loc();
		in2->increase_loc();
		cg->addNode(this);
	}

	void backward(){
		in1->loss.vec() += loss.vec();
		in2->loss.vec() -= loss.vec();
	}

	inline void unlock(){
		in1->decrease_loc();
		in2->decrease_loc();
		if(!lossed)return;
		in1->lossed = true;
		in2->lossed = true;
	}
};



struct PAddNode : Node {
	vector<PNode> ins;
	int nSize;
public:
	PAddNode() : Node(){
		ins.clear();
		nSize = 0;
	}
public:
	virtual inline void clearValue(){
		Node::clearValue();
		ins.clear();
		nSize = 0;
	}

public:
	// please better restrict col to 1
	void forward(Graph *cg, const vector<PNode>& x) {
		nSize = x.size();
		/*
		if (nSize < 2){
			std::cout << "at least two nodes are required" << std::endl;
			return;
		}
		*/

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
			ins[i]->loss.vec() += loss.vec();
		}
	}

	inline void unlock(){
		for (int i = 0; i < nSize; i++){
			ins[i]->decrease_loc();
		}
		if(!lossed)return;
		for (int i = 0; i < nSize; i++){
			ins[i]->lossed = true;
		}		
	}

protected:
	void forward() {
		for (int idx = 0; idx < nSize; idx++){
			val.vec() = val.vec() + ins[idx]->val.vec();		
		}
		for (int idx = 0; idx < nSize; idx++){
			ins[idx]->increase_loc();
		}
	}

};


struct ActivateNode : Node {
	PNode in;
	dtype(*activate)(const dtype&);   // activation function
	dtype(*derivate)(const dtype&, const dtype&);  // derivation function of activation function
public:
	ActivateNode() : Node(){
		activate = ftanh;
		derivate = dtanh;
		in = NULL;
	}
public:
	virtual inline void clearValue(){
		Node::clearValue();
		in = NULL;
	}
	// define the activate function and its derivation form
	inline void setFunctions(dtype(*f)(const dtype&), dtype(*f_deri)(const dtype&, const dtype&)) {
		activate = f;
		derivate = f_deri;
	}

public:

	inline void forward(Graph *cg, PNode x){
		in = x;
		val.vec() = in->val.vec().unaryExpr(ptr_fun(activate));
		in->increase_loc();
		cg->addNode(this);
	}

	inline void backward(){
		in->loss.vec() += loss.vec() * in->val.vec().binaryExpr(val.vec(), ptr_fun(derivate));
	}

	inline void unlock(){
		in->decrease_loc();
		if(!lossed)return;
		in->lossed = true;
	}

};


//special case of ActivateNode
struct TanhNode : Node {
	PNode in;

public:
	TanhNode() : Node(){
		in = NULL;
	}
public:
	virtual inline void clearValue(){
		Node::clearValue();
		in = NULL;
	}

public:

	inline void forward(Graph *cg, PNode x){
		in = x;
		val.vec() = in->val.vec().unaryExpr(ptr_fun(ftanh));
		in->increase_loc();
		cg->addNode(this);
	}

	inline void backward(){
		in->loss.vec() += loss.vec() * in->val.vec().binaryExpr(val.vec(), ptr_fun(dtanh));
	}

	inline void unlock(){
		in->decrease_loc();
		if(!lossed)return;
		in->lossed = true;
	}

};


//special case of ActivateNode
struct SigmoidNode : Node {
	PNode in;

public:
	SigmoidNode() : Node(){
		in = NULL;
	}
public:
	virtual inline void clearValue(){
		Node::clearValue();
		in = NULL;
	}

public:

	inline void forward(Graph *cg, PNode x){
		in = x;
		val.vec() = in->val.vec().unaryExpr(ptr_fun(fsigmoid));
		in->increase_loc();
		cg->addNode(this);
	}

	inline void backward(){
		in->loss.vec() += loss.vec() * in->val.vec().binaryExpr(val.vec(), ptr_fun(dsigmoid));
	}

	inline void unlock(){
		in->decrease_loc();
		if(!lossed)return;
		in->lossed = true;
	}

};

//special case of ActivateNode
struct RELUNode : Node {
	PNode in;

public:
	RELUNode() : Node(){
		in = NULL;
	}
public:
	virtual inline void clearValue(){
		Node::clearValue();
		in = NULL;
	}

public:
	inline void forward(Graph *cg, PNode x){
		in = x;
		val.vec() = in->val.vec().unaryExpr(ptr_fun(frelu));
		in->increase_loc();
		cg->addNode(this);
	}

	inline void backward(){
		in->loss.vec() += loss.vec() * in->val.vec().binaryExpr(val.vec(), ptr_fun(drelu));
	}

	inline void unlock(){
		in->decrease_loc();
		if(!lossed)return;
		in->lossed = true;
	}

};


struct PDotNode : Node {
	PNode in1, in2;
public:
	PDotNode() : Node() {
		in1 = NULL;
		in2 = NULL;
	}
public:
	virtual inline void clearValue() {
		Node::clearValue();
		in1 = NULL;
		in2 = NULL;
	}

public:
	void forward(Graph *cg, PNode x1, PNode x2) {
		in1 = x1;
		in2 = x2;
		//assert(dim == 1 && in1->dim == in2->dim);
		if(dim != 1 || in1->dim != in2->dim){
			std::cout << "warning: input dims of PDotNode do not match" << std::endl;
		}
		val[0] = 0.0;
		for (int idx = 0; idx < in1->dim; idx++) {
			val[0] += x1->val[idx] * x2->val[idx];
		}
		in1->increase_loc();
		in2->increase_loc();
		cg->addNode(this);
	}

	void backward() {
		for (int idx = 0; idx < in1->dim; idx++) {
			in1->loss[idx] += loss[0] * in2->val[idx];
			in2->loss[idx] += loss[0] * in1->val[idx];
		}
	}

	inline void unlock() {
		in1->decrease_loc();
		in2->decrease_loc();
		if (!lossed)return;
		in1->lossed = true;
		in2->lossed = true;
	}

};


#endif
