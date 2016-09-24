/*
 * FourOP.h
 *
 *  Created on: Jul 20, 2016
 *      Author: mason
 */

#ifndef FOUROP_H_
#define FOUROP_H_

#include "Param.h"
#include "MyLib.h"
#include "Node.h"
#include "Graph.h"

struct FourParams {
public:
	Param W1;
	Param W2;
	Param W3;
	Param W4;
	Param b;

	bool bUseB;

public:
	FourParams() {
		bUseB = true;
	}

	inline void exportAdaParams(ModelUpdate& ada) {
		ada.addParam(&W1);
		ada.addParam(&W2);
		ada.addParam(&W3);
		ada.addParam(&W4);
		if (bUseB) {
			ada.addParam(&b);
		}
	}

	inline void initial(int nOSize, int nISize1, int nISize2, int nISize3, int nISize4, bool useB = true) {
		W1.initial(nOSize, nISize1);
		W2.initial(nOSize, nISize2);
		W3.initial(nOSize, nISize3);
		W4.initial(nOSize, nISize4);
		b.initial(nOSize, 1);

		bUseB = useB;
	}

};

// non-linear feed-forward node
// input nodes should be specified by forward function
// for input variables, we exploit column vector,
// which means a concrete input vector x_i is represented by x(0, i), x(1, i), ..., x(n, i)
struct FourNode : Node {
public:
	PNode in1, in2, in3, in4;
	Mat ty, lty;  // t means temp, ty is to save temp vector before activation

	int inDim1, inDim2, inDim3, inDim4;

	FourParams* param;

	Mat (*activate)(const Mat&);   // activation function
	Mat (*derivate)(const Mat&, const Mat&);  // derivation function of activation function

public:
	FourNode() {
	}

	inline void setParam(FourParams* paramInit) {
		param = paramInit;
		inDim1 = param->W1.inDim();
		inDim2 = param->W2.inDim();
		inDim3 = param->W3.inDim();
		inDim4 = param->W4.inDim();
		dim = param->W1.outDim();
		if (!param->bUseB) {
			cout << "please check whether bUseB is true, usually this should be true for non-linear layer" << endl;
		}
	}

	inline void clear(){
		Node::clear();
		in1 = NULL;
		in2 = NULL;
		in3 = NULL;
		in4 = NULL;
		activate = tanh;
		derivate = tanh_deri;
		ty.setZero();
		lty.setZero();
		param = NULL;
		inDim1 = 0;
		inDim2 = 0;
		inDim3 = 0;
		inDim4 = 0;
	}

	inline void clearValue(){
		Node::clearValue();
		in1 = NULL;
		in2 = NULL;
		in3 = NULL;
		in4 = NULL;
		ty.setZero();
		lty.setZero();
	}

	// define the activation function and its derivation form
	inline void setFunctions(Mat (*f)(const Mat&), Mat (*f_deri)(const Mat&, const Mat&)) {
		activate = f;
		derivate = f_deri;
	}

public:
	void forward(Graph *cg, PNode x1, PNode x2, PNode x3, PNode x4) {
		assert(param != NULL);

		in1 = x1;
		in2 = x2;
		in3 = x3;
		in4 = x4;
		assert(inDim1 == in1->val.rows() && inDim2 == in2->val.rows() && inDim3 == in3->val.rows() && inDim4 == in4->val.rows());

		ty = param->W1.val * (in1->val) + param->W2.val * (in2->val) + param->W3.val * (in3->val) + param->W4.val * (in4->val);
		if(param->bUseB){
			for (int idx = 0; idx < ty.cols(); idx++) {
				ty.col(idx) += param->b.val.col(0);
			}
		}

		val = activate(ty);

		in1->lock++;
		in2->lock++;
		in3->lock++;
		in4->lock++;

		cg->addNode(this);
	}

	void backward() {
		assert(param != NULL);

		lty = loss.array() * derivate(ty, val).array();

		param->W1.grad += lty * in1->val.transpose();
		param->W2.grad += lty * in2->val.transpose();
		param->W3.grad += lty * in3->val.transpose();
		param->W4.grad += lty * in4->val.transpose();

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

		if (in3->loss.size() == 0) {
			in3->loss = Mat::Zero(in3->val.rows(), in3->val.cols());
		}
		in3->loss += param->W3.val.transpose() * lty;
		
		if (in4->loss.size() == 0) {
			in4->loss = Mat::Zero(in4->val.rows(), in4->val.cols());
		}
		in4->loss += param->W4.val.transpose() * lty;	

	}

	inline void unlock(){
		in1->lock--;
		in2->lock--;
		in3->lock--;
		in4->lock--;
	}

};

#endif /* FOUROP_H_ */
