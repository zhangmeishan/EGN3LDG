/*
 * UniOP.h
 *
 *  Created on: Jul 20, 2016
 *      Author: mason
 */

#ifndef UNIOP_H_
#define UNIOP_H_

#include "Param.h"
#include "MyLib.h"
#include "Node.h"
#include "Graph.h"

struct UniParams {
public:
	Param W;
	Param b;
	bool bUseB;

public:
	UniParams() {
		bUseB = true;
	}

	inline void exportAdaParams(ModelUpdate& ada) {
		ada.addParam(&W);
		if (bUseB) {
			ada.addParam(&b);
		}
	}

	inline void initial(int nOSize, int nISize, bool useB = true, AlignedMemoryPool* mem = NULL) {
		W.initial(nOSize, nISize, mem);

		bUseB = useB;		
		if(bUseB){
			b.initial(nOSize, 1, mem);
		}
	}

	inline void save(std::ofstream &os) const {
		os << bUseB << std::endl;
		W.save(os);
		if (bUseB) {
			b.save(os);
		}
	}

	inline void load(std::ifstream &is, AlignedMemoryPool* mem = NULL) {
		is >> bUseB;
		W.load(is, mem);
		if (bUseB) {
			b.load(is, mem);
		}
	}

};

// non-linear feed-forward node
// input nodes should be specified by forward function
// for input variables, we exploit column vector,
// which means a concrete input vector x_i is represented by x(0, i), x(1, i), ..., x(n, i)
struct UniNode : Node{
public:
	PNode in;
	Tensor1D ty, lty; // t means temp, ty is to save temp vector before activate
	int inDim;

	UniParams* param;

	dtype (*activate)(const dtype&);   // activation function
	dtype (*derivate)(const dtype&, const dtype&);  // derivation function of activation function


public:
	UniNode()  : Node(){
		in = NULL;		
		activate = ftanh;
		derivate = dtanh;
		param = NULL;
		
		inDim = 0;
	}


	inline void setParam(UniParams* paramInit) {
		param = paramInit;
		inDim = param->W.inDim();
	}
	
	inline void clearValue(){
		Node::clearValue();
		in = NULL;
		ty.zero();
		lty.zero();
	}	

	// define the activate function and its derivation form
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
	void forward(Graph *cg, PNode x) {
		in = x;

		ty.mat() = param->W.val.mat() * in->val.mat();

		if(param->bUseB){
			ty.vec() += param->b.val.vec();
		}
		
		val.vec() = ty.vec().unaryExpr(ptr_fun(activate));

		in->lock++;
		cg->addNode(this);
	}

	void backward() {
		lty.vec() = loss.vec() * ty.vec().binaryExpr(val.vec(), ptr_fun(derivate));

		param->W.grad.mat() += lty.mat() * in->val.tmat();

		if(param->bUseB){
			param->b.grad.vec() += lty.vec();
		}

		in->loss.mat() += param->W.val.mat().transpose() * lty.mat();
		
	}


	inline void unlock(){
		in->lock--;
		if(!lossed)return;
		in->lossed = true;
	}

};

struct LinearUniNode : Node{
public:
	PNode in;
	int inDim;

	UniParams* param;


public:
	LinearUniNode()  : Node(){
		in = NULL;

		param = NULL;
		
		inDim = 0;
	}


	inline void setParam(UniParams* paramInit) {
		param = paramInit;
		inDim = param->W.inDim();
	}
	
	inline void clearValue(){
		Node::clearValue();
		in = NULL;
	}	


public:
	void forward(Graph *cg, PNode x) {
		in = x;
		
		val.mat() = param->W.val.mat() * in->val.mat();

		if(param->bUseB){
			val.vec() += param->b.val.vec();
		}
		

		in->lock++;
		cg->addNode(this);
	}

	void backward() {
		param->W.grad.mat() += loss.mat() * in->val.tmat();

		if(param->bUseB){
			param->b.grad.vec() += loss.vec();
		}

		in->loss.mat() += param->W.val.mat().transpose() * loss.mat();
	}


	inline void unlock(){
		in->lock--;
		if(!lossed)return;
		in->lossed = true;
	}

};

// Linear Node, ofen used for computing output
// input nodes should be specified by forward function
// for input variables, we exploit column vector,
// which means a concrete input vector x_i is represented by x(0, i), x(1, i), ..., x(n, i)
struct LinearNode : Node{
public:
	PNode in;
	int inDim;

	UniParams* param;


public:
	LinearNode() : Node(){
		in = NULL;

		param = NULL;
		
		inDim = 0;
	}


	inline void setParam(UniParams* paramInit) {
		param = paramInit;
		if(param->bUseB){
			std::cout << "Please check bUseB of the param" << std::endl;
		}
		inDim = param->W.inDim();
	}
	
	inline void clearValue(){
		Node::clearValue();
		in = NULL;
	}	


public:
	void forward(Graph *cg, PNode x) {
		in = x;		
		val.mat() = param->W.val.mat() * in->val.mat();

		in->lock++;
		cg->addNode(this);
	}

	void backward() {
		param->W.grad.mat() += loss.mat() * in->val.tmat();
		in->loss.mat() += param->W.val.mat().transpose() * loss.mat();
	}


	inline void unlock(){
		in->lock--;
		if(!lossed)return;
		in->lossed = true;
	}

};

#endif /* UNIOP_H_ */
