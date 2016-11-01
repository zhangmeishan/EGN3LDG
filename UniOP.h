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

	inline void initial(int nOSize, int nISize, bool useB = true) {
		W.initial(nOSize, nISize);
		b.initial(nOSize, 1);

		bUseB = useB;
	}

};

// non-linear feed-forward node
// input nodes should be specified by forward function
// for input variables, we exploit column vector,
// which means a concrete input vector x_i is represented by x(0, i), x(1, i), ..., x(n, i)
struct UniNode : Node{
public:
	PNode in;
	Mat ty, lty; // t means temp, ty is to save temp vector before activate
	int inDim;

	UniParams* param;

	Mat (*activate)(const Mat&);   // activate function
	Mat (*derivate)(const Mat&, const Mat&); // derivation function of activate function


public:
	UniNode() {
		clear();
	}


	inline void clear(){
		Node::clear();
		in = NULL;
		activate = tanh;
		derivate = tanh_deri;
		ty.setZero();
		lty.setZero();
		param = NULL;
		inDim = 0;
	}


	inline void setParam(UniParams* paramInit) {
		param = paramInit;
		inDim = param->W.inDim();
		dim = param->W.outDim();	
		if (!param->bUseB) {
			cout
					<< "please check whether bUseB is true, usually this should be true for non-linear layer"
					<< endl;
		}
	}

	// define the activate function and its derivation form
	inline void setFunctions(Mat (*f)(const Mat&),
			Mat (*f_deri)(const Mat&, const Mat&)) {
		activate = f;
		derivate = f_deri;
	}

	inline void clearValue(){
		Node::clearValue();
		in = NULL;
		ty.setZero();
		lty.setZero();
	}

public:
	void forward(Graph *cg, PNode x) {
		assert(param != NULL);

		in = x;
		assert(inDim == in->val.rows());

		ty = param->W.val * in->val;

		if(param->bUseB){
			for (int idx = 0; idx < ty.cols(); idx++) {
				ty.col(idx) += param->b.val.col(0);
			}
		}

		val = activate(ty);

		in->lock++;
		cg->addNode(this);
	}

	void backward() {
		assert(param != NULL);

		lty = loss.array() * derivate(ty, val).array();

		param->W.grad += lty * in->val.transpose();

		if(param->bUseB){
			for (int idx = 0; idx < val.cols(); idx++) {
				param->b.grad.col(0) += lty.col(idx);
			}
		}

		if (in->loss.size() == 0) {
			in->loss = Mat::Zero(in->val.rows(), in->val.cols());
		}

		in->loss += param->W.val.transpose() * lty;
		
	}


	inline void unlock(){
		in->lock--;
		if(!validLoss(loss))return;
		in->lossed = true;
	}

};

// Linear Node, ofen used for computing output
// input nodes should be specified by forward function
// for input variables, we exploit column vector,
// which means a concrete input vector x_i is represented by x(0, i), x(1, i), ..., x(n, i)
struct LinearNode : Node {
public:
	PNode in;
	int inDim;
	UniParams* param;

public:
	LinearNode() {
		clear();
	}

	inline void clear(){
		Node::clear();
		in = NULL;
		param = NULL;
		inDim = 0;
	}

	inline void setParam(UniParams* paramInit) {
		param = paramInit;
		inDim = param->W.inDim();
		dim = param->W.outDim();
		if (param->bUseB) {
			cout << "please check whether bUseB is false, usually this should be false for linear layer"
					<< endl;
		}
	}

	inline void clearValue(){
		Node::clearValue();
		in = NULL;
	}

public:
	void forward(Graph *cg, PNode x) {
		assert(param != NULL);

		in = x;
		assert(inDim == in->val.rows());

		val = param->W.val * (in->val);	

		in->lock++;
		cg->addNode(this);
	}

	void backward() {
		assert(param != NULL);
		param->W.grad += loss * in->val.transpose();
		if (in->loss.size() == 0) {
			in->loss = Mat::Zero(in->val.rows(), in->val.cols());
		}

		in->loss += param->W.val.transpose() * loss;
	}

	inline void unlock(){
		in->lock--;
		if(!validLoss(loss))return;
		in->lossed = true;
	}

};

#endif /* UNIOP_H_ */
