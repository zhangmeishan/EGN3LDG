/*
 * TriOP.h
 *
 *  Created on: Jul 20, 2016
 *      Author: mason
 */

#ifndef TRIOP_H_
#define TRIOP_H_

#include "Param.h"
#include "MyLib.h"
#include "Node.h"
#include "Graph.h"

struct TriParams {
public:
	Param W1;
	Param W2;
	Param W3;
	Param b;

	bool bUseB;

public:
	TriParams() {
		bUseB = true;
	}

	inline void exportAdaParams(ModelUpdate& ada) {
		ada.addParam(&W1);
		ada.addParam(&W2);
		ada.addParam(&W3);
		if (bUseB) {
			ada.addParam(&b);
		}
	}

	inline void initial(int nOSize, int nISize1, int nISize2, int nISize3, bool useB = true, AlignedMemoryPool* mem = NULL) {
		W1.initial(nOSize, nISize1, mem);
		W2.initial(nOSize, nISize2, mem);
		W3.initial(nOSize, nISize3, mem);

		bUseB = useB;		
		if(bUseB){
			b.initial(nOSize, 1, mem);
		}
	}

	inline void save(std::ofstream &os) const {
		os << bUseB << std::endl;
		W1.save(os);
		W2.save(os);
		W3.save(os);
		if (bUseB) {
			b.save(os);
		}
	}

	inline void load(std::ifstream &is, AlignedMemoryPool* mem = NULL) {
		is >> bUseB;
		W1.load(is, mem);
		W2.load(is, mem);
		W3.load(is, mem);
		if (bUseB) {
			b.load(is, mem);
		}
	}

};

// non-linear feed-forward node
// input nodes should be specified by forward function
// for input variables, we exploit column vector,
// which means a concrete input vector x_i is represented by x(0, i), x(1, i), ..., x(n, i)
struct TriNode : Node {
public:
	PNode in1, in2, in3;
	Tensor1D ty, lty;  // t means temp, ty is to save temp vector before activation

	int inDim1, inDim2, inDim3;

	TriParams* param;

	dtype (*activate)(const dtype&);   // activation function
	dtype (*derivate)(const dtype&, const dtype&);  // derivation function of activation function

public:
	TriNode() : Node(){
		in1 = NULL;
		in2 = NULL;
		in3 = NULL;

		activate = ftanh;
		derivate = dtanh;
		param = NULL;
		
		inDim1 = 0;
		inDim2 = 0;
		inDim3 = 0;		
	}

	inline void setParam(TriParams* paramInit) {
		param = paramInit;
		inDim1 = param->W1.inDim();
		inDim2 = param->W2.inDim();
		inDim3 = param->W3.inDim();
	}

	inline void clearValue(){
		Node::clearValue();
		in1 = NULL;
		in2 = NULL;
		in3 = NULL;
		ty.zero();
		lty.zero();
	}

	// define the activation function and its derivation form
	inline void setFunctions(dtype (*f)(const dtype&), dtype (*f_deri)(const dtype&, const dtype&)) {
		activate = f;
		derivate = f_deri;
	}
	
	inline void init(int dim, dtype dropOut, AlignedMemoryPool* mem = NULL){
		Node::init(dim, dropOut, mem);
		ty.init(dim, mem);
		lty.init(dim, mem);
	}		

public:
	void forward(Graph *cg, PNode x1, PNode x2, PNode x3) {
		in1 = x1;
		in2 = x2;
		in3 = x3;

		ty.mat() = param->W1.val.mat() * in1->val.mat() + param->W2.val.mat() * in2->val.mat() + param->W3.val.mat() * in3->val.mat();
		
		if(param->bUseB){
			ty.vec() += param->b.val.vec();
		}
		
		val.vec() = ty.vec().unaryExpr(ptr_fun(activate));

		in1->lock++;
		in2->lock++;
		in3->lock++;

		cg->addNode(this);
	}

	void backward() {
		lty.vec() = loss.vec() * ty.vec().binaryExpr(val.vec(), ptr_fun(derivate));

		param->W1.grad.mat() += lty.mat() * in1->val.tmat();
		param->W2.grad.mat() += lty.mat() * in2->val.tmat();
		param->W3.grad.mat() += lty.mat() * in3->val.tmat();

		if(param->bUseB){
			param->b.grad.vec() += lty.vec();
		}

		in1->loss.mat() += param->W1.val.mat().transpose() * lty.mat();
		in2->loss.mat() += param->W2.val.mat().transpose() * lty.mat();
		in3->loss.mat() += param->W3.val.mat().transpose() * lty.mat();
	}

	inline void unlock(){
		in1->lock--;
		in2->lock--;
		in3->lock--;
		if(!lossed)return;
		in1->lossed = true;
		in2->lossed = true;
		in3->lossed = true;		
	}

};

struct LinearTriNode : Node {
public:
	PNode in1, in2, in3;

	int inDim1, inDim2, inDim3;

	TriParams* param;

public:
	LinearTriNode() : Node(){
		in1 = NULL;
		in2 = NULL;
		in3 = NULL;

		param = NULL;
		
		inDim1 = 0;
		inDim2 = 0;
		inDim3 = 0;		
	}

	inline void setParam(TriParams* paramInit) {
		param = paramInit;
		inDim1 = param->W1.inDim();
		inDim2 = param->W2.inDim();
		inDim3 = param->W3.inDim();
	}

	inline void clearValue(){
		Node::clearValue();
		in1 = NULL;
		in2 = NULL;
		in3 = NULL;
	}	

public:
	void forward(Graph *cg, PNode x1, PNode x2, PNode x3) {
		in1 = x1;
		in2 = x2;
		in3 = x3;

		val.mat() = param->W1.val.mat() * in1->val.mat() + param->W2.val.mat() * in2->val.mat() + param->W3.val.mat() * in3->val.mat();
		
		if(param->bUseB){
			val.vec() += param->b.val.vec();
		}
		
		in1->lock++;
		in2->lock++;
		in3->lock++;

		cg->addNode(this);
	}

	void backward() {
		param->W1.grad.mat() += loss.mat() * in1->val.tmat();
		param->W2.grad.mat() += loss.mat() * in2->val.tmat();
		param->W3.grad.mat() += loss.mat() * in3->val.tmat();

		if(param->bUseB){
			param->b.grad.vec() += loss.vec();
		}

		in1->loss.mat() += param->W1.val.mat().transpose() * loss.mat();
		in2->loss.mat() += param->W2.val.mat().transpose() * loss.mat();
		in3->loss.mat() += param->W3.val.mat().transpose() * loss.mat();
	}

	inline void unlock(){
		in1->lock--;
		in2->lock--;
		in3->lock--;
		if(!lossed)return;
		in1->lossed = true;
		in2->lossed = true;
		in3->lossed = true;		
	}

};

#endif /* TRIOP_H_ */
