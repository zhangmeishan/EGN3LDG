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

	inline void initial(int nOSize, int nISize1, int nISize2, bool useB = true) {
		W1.initial(nOSize, nISize1);
		W2.initial(nOSize, nISize2);
		b.initial(nOSize, 1);

		bUseB = useB;
	}

};

// non-linear feed-forward node
// input nodes should be specified by forward function
// for input variables, we exploit column vector,
// which means a concrete input vector x_i is represented by x(0, i), x(1, i), ..., x(n, i)
struct BiNode : Node {
public:
	PNode in1, in2;
	Mat ty, lty;  // t means temp, ty is to save temp vector before activation

	int inDim1, inDim2;

	BiParams* param;

	Mat (*activate)(const Mat&);   // activation function
	Mat (*derivate)(const Mat&, const Mat&);  // derivation function of activation function

public:
	BiNode() {
		clear();
	}

	inline void setParam(BiParams* paramInit) {
		param = paramInit;
		inDim1 = param->W1.inDim();
		inDim2 = param->W2.inDim();
		dim = param->W1.outDim();
		if (!param->bUseB) {
			cout << "please check whether bUseB is true, usually this should be true for non-linear layer" << endl;
		}
	}

	inline void clear(){
		Node::clear();
		in1 = NULL;
		in2 = NULL;
		activate = tanh;
		derivate = tanh_deri;
		ty.setZero();
		lty.setZero();
		param = NULL;
		inDim1 = 0;
		inDim2 = 0;
	}

	inline void clearValue(){
		Node::clearValue();
		in1 = NULL;
		in2 = NULL;
		ty.setZero();
		lty.setZero();
	}

	// define the activation function and its derivation form
	inline void setFunctions(Mat (*f)(const Mat&), Mat (*f_deri)(const Mat&, const Mat&)) {
		activate = f;
		derivate = f_deri;
	}

public:
	void forward(Graph* cg, PNode x1, PNode x2) {
		assert(param != NULL);

		in1 = x1;
		in2 = x2;
		assert(inDim1 == in1->val.rows() && inDim2 == in2->val.rows());

		ty = param->W1.val * (in1->val) + param->W2.val * (in2->val);

		if(param->bUseB){
			for (int idx = 0; idx < ty.cols(); idx++) {
				ty.col(idx) += param->b.val.col(0);
			}
		}

		val = activate(ty);

		in1->lock++;
		in2->lock++;

		cg->addNode(this);
	}

	void backward() {
		assert(param != NULL);

		lty = loss.array() * derivate(ty, val).array();

		param->W1.grad += lty * in1->val.transpose();
		param->W2.grad += lty * in2->val.transpose();

		if(param->bUseB){
			for (int idx = 0; idx < val.cols(); idx++) {
				param->b.grad.col(0) += lty.col(idx);
			}
		}

		if (in1->loss.size() == 0) {
			in1->loss = Mat::Zero(in1->val.rows(), in1->val.cols());
		}
		in1->loss += param->W1.val.transpose() * lty;

		if (in2->loss.size() == 0) {
			in2->loss = Mat::Zero(in2->val.rows(), in2->val.cols());
		}		
		in2->loss += param->W2.val.transpose() * lty;

	}

	inline void unlock(){
		in1->lock--;
		in2->lock--;
		if(!validLoss(loss))return;
		in1->lossed = true;
		in2->lossed = true;
	}

};

#endif /* BIOP_H_ */
