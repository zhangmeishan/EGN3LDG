/*
 * BiOP.h
 *
 *  Created on: Jul 20, 2016
 *      Author: mason
 */

#ifndef BIOP_H_
#define BIOP_H_

#include "Param.h"
#include "MyLib.h"
#include "Node.h"
#include "Graph.h"

struct BiParams {
public:
	Param W1;
	Param W2;
	Param b;

	bool bUseB;

public:
	BiParams() {
		bUseB = true;
	}

	inline void exportAdaParams(ModelUpdate& ada) {
		ada.addParam(&W1);
		ada.addParam(&W2);
		if (bUseB) {
			ada.addParam(&b);
		}
	}

	inline void initial(int nOSize, int nISize1, int nISize2, bool useB = true, AlignedMemoryPool* mem = NULL) {
		W1.initial(nOSize, nISize1, mem);
		W2.initial(nOSize, nISize2, mem);
		bUseB = useB;
		if(bUseB){
			b.initial(nOSize, 1, mem);
		}
	}

	inline void save(std::ofstream &os) const {
		os << bUseB << std::endl;
		W1.save(os);
		W2.save(os);
		if (bUseB) {
			b.save(os);
		}
	}

	inline void load(std::ifstream &is, AlignedMemoryPool* mem = NULL) {
		is >> bUseB;
		W1.load(is, mem);
		W2.load(is, mem);
		if (bUseB) {
			b.load(is, mem);
		}
	}

};

// non-linear feed-forward node
// input nodes should be specified by forward function
// for input variables, we exploit column vector,
// which means a concrete input vector x_i is represented by x(0, i), x(1, i), ..., x(n, i)
struct BiNode : Node {
public:
	PNode in1, in2;
	Tensor1D ty, lty; 

	int inDim1, inDim2;

	BiParams* param;

	dtype (*activate)(const dtype&);   // activation function
	dtype (*derivate)(const dtype&, const dtype&);  // derivation function of activation function

public:
	BiNode() : Node() {
		in1 = NULL;
		in2 = NULL;
		
		activate = ftanh;
		derivate = dtanh;
		
		param = NULL;
		inDim1 = 0;
		inDim2 = 0;		
	}

	inline void setParam(BiParams* paramInit) {
		param = paramInit;
		inDim1 = param->W1.inDim();
		inDim2 = param->W2.inDim();
	}


	inline void clearValue(){
		Node::clearValue();
		in1 = NULL;
		in2 = NULL;
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
	void forward(Graph* cg, PNode x1, PNode x2) {
		in1 = x1;
		in2 = x2;
		ty.mat() = param->W1.val.mat() * in1->val.mat() + param->W2.val.mat() * in2->val.mat();

		if(param->bUseB){
			ty.vec() += param->b.val.vec();
		}

		val.vec() = ty.vec().unaryExpr(ptr_fun(activate));

		in1->lock++;
		in2->lock++;

		cg->addNode(this);
	}

	void backward() {
		lty.vec() = loss.vec() * ty.vec().binaryExpr(val.vec(), ptr_fun(derivate));

		param->W1.grad.mat() += lty.mat() * in1->val.tmat();
		param->W2.grad.mat() += lty.mat() * in2->val.tmat();

		if(param->bUseB){
			param->b.grad.vec() += lty.vec();
		}

		in1->loss.mat() += param->W1.val.mat().transpose() * lty.mat();
		in2->loss.mat() += param->W2.val.mat().transpose() * lty.mat();

	}

	inline void unlock(){
		in1->lock--;
		in2->lock--;
		if(!lossed)return;
		in1->lossed = true;
		in2->lossed = true;
	}

};


struct LinearBiNode : Node {
public:
	PNode in1, in2;
	int inDim1, inDim2;
	BiParams* param;

public:
	LinearBiNode() : Node() {
		in1 = NULL;
		in2 = NULL;
		
		param = NULL;
		
		inDim1 = 0;
		inDim2 = 0;		
	}

	inline void setParam(BiParams* paramInit) {
		param = paramInit;
		inDim1 = param->W1.inDim();
		inDim2 = param->W2.inDim();
	}


	inline void clearValue(){
		Node::clearValue();
		in1 = NULL;
		in2 = NULL;
	}


public:
	void forward(Graph* cg, PNode x1, PNode x2) {
		in1 = x1;
		in2 = x2;
		val.mat() = param->W1.val.mat() * in1->val.mat() + param->W2.val.mat() * in2->val.mat();

		if(param->bUseB){
			val.vec() += param->b.val.vec();
		}

		in1->lock++;
		in2->lock++;

		cg->addNode(this);
	}

	void backward() {
		param->W1.grad.mat() += loss.mat() * in1->val.tmat();
		param->W2.grad.mat() += loss.mat() * in2->val.tmat();

		if(param->bUseB){
			param->b.grad.vec() += loss.vec();
		}

		in1->loss.mat() += param->W1.val.mat().transpose() * loss.mat();
		in2->loss.mat() += param->W2.val.mat().transpose() * loss.mat();

	}

	inline void unlock(){
		in1->lock--;
		in2->lock--;
		if(!lossed)return;
		in1->lossed = true;
		in2->lossed = true;
	}

};

#endif /* BIOP_H_ */
