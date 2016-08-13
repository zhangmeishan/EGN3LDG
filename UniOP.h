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

struct UniParams {
public:
	Param _W;
	Param _b;

	bool _bUseB;

public:
	UniParams() {
		_bUseB = true;
	}

	inline void exportAdaParams(ModelUpdate& ada) {
		ada.addParam(&_W);
		if (_bUseB) {
			ada.addParam(&_b);
		}
	}

	inline void initial(int nOSize, int nISize, bool bUseB = true,
			int seed = 0) {
		srand(seed);
		_W.initial(nOSize, nISize);
		_b.initial(nOSize, 1);

		_bUseB = bUseB;
	}

};

// non-linear feed-forward node
// input nodes should be specified by forward function
// for input variables, we exploit column vector,
// which means a concrete input vector x_i is represented by x(0, i), x(1, i), ..., x(n, i)
struct UniNode {
public:
	PMat _px;
	Mat _y, _ly;
	Mat _ty, _lty; // t means temp, _ty is to save temp vector before activation

	int _inDim, _outDim;

	UniParams* _param;

	Mat (*_f)(const Mat&);   // activation function
	Mat (*_f_deri)(const Mat&, const Mat&); // derivation function of activation function


public:
	UniNode() {
		clear();
	}

	UniNode(UniParams* param) {
		_px = NULL;
		_f = tanh;
		_f_deri = tanh_deri;
		_y.setZero();
		_ly.setZero();
		_ty.setZero();
		_lty.setZero();
		setParam(param);
	}

	UniNode(Mat (*f)(const Mat&),
			Mat (*f_deri)(const Mat&, const Mat&)) {
		_px = NULL;
		setFunctions(f, f_deri);
		_y.setZero();
		_ly.setZero();
		_ty.setZero();
		_lty.setZero();
		_param = NULL;
		_inDim = _outDim = 0;
	}

	UniNode(UniParams* param, Mat (*f)(const Mat&),
			Mat (*f_deri)(const Mat&, const Mat&)) {
		_px = NULL;
		setFunctions(f, f_deri);
		_y.setZero();
		_ly.setZero();
		_ty.setZero();
		_lty.setZero();
		setParam(param);
	}

	inline void clear(){
		_px = NULL;
		_f = tanh;
		_f_deri = tanh_deri;
		_y.setZero();
		_ly.setZero();
		_ty.setZero();
		_lty.setZero();
		_param = NULL;
		_inDim = _outDim = 0;
	}


	inline void setParam(UniParams* param) {
		_param = param;
		_inDim = _param->_W.inDim();
		_outDim = _param->_W.outDim();
		if (!_param->_bUseB) {
			cout
					<< "please check whether _bUseB is true, usually this should be true for non-linear layer"
					<< endl;
		}
	}

	// define the activation function and its derivation form
	inline void setFunctions(Mat (*f)(const Mat&),
			Mat (*f_deri)(const Mat&, const Mat&)) {
		_f = f;
		_f_deri = f_deri;
	}

	inline void clearValue(){
		_px = NULL;
		_y.setZero();
		_ly.setZero();
		_ty.setZero();
		_lty.setZero();
	}

public:
	void forward(PMat px) {
		assert(_param != NULL);

		_ty = _param->_W.val * (*px);
		if(_param->_bUseB){
			for (int idx = 0; idx < _ty.cols(); idx++) {
				_ty.col(idx) += _param->_b.val.col(0);
			}
		}

		_y = _f(_ty);
		_px = px;
	}

	void backward(PMat plx) {
		assert(_param != NULL);

		_lty = _ly.array() * _f_deri(_ty, _y).array();

		_param->_W.grad += _lty * _px->transpose();

		if(_param->_bUseB){
			for (int idx = 0; idx < _y.cols(); idx++) {
				_param->_b.grad.col(0) += _lty.col(idx);
			}
		}

		if (plx->size() == 0) {
			*plx = Mat::Zero(_px->rows(), _px->cols());
		}

		*plx += _param->_W.val.transpose() * _lty;

	}

};

// Linear Node, ofen used for computing output
// input nodes should be specified by forward function
// for input variables, we exploit column vector,
// which means a concrete input vector x_i is represented by x(0, i), x(1, i), ..., x(n, i)
struct LinearNode {
public:
	PMat _px;
	Mat _y, _ly;

	int _inDim, _outDim;

	UniParams* _param;

public:
	LinearNode() {
		clear();
	}

	LinearNode(UniParams* param) {
		_px = NULL;
		_y.setZero();
		_ly.setZero();
		setParam(param);
	}

	inline void clear(){
		_px = NULL;
		_y.setZero();
		_ly.setZero();
		_param = NULL;
		_inDim = _outDim = 0;
	}

	inline void setParam(UniParams* param) {
		_param = param;
		_inDim = _param->_W.inDim();
		_outDim = _param->_W.outDim();
		if (param->_bUseB) {
			cout << "please check whether _bUseB is false, usually this should be false for linear layer"
					<< endl;
		}
	}

	inline void clearValue(){
		_px = NULL;
		_y.setZero();
		_ly.setZero();
	}

public:
	void forward(PMat px) {
		assert(_param != NULL);
		_y = _param->_W.val * (*px);
		_px = px;
	}

	void backward(PMat plx) {
		assert(_param != NULL);
		_param->_W.grad += _ly * _px->transpose();
		if (plx->size() == 0) {
			*plx = Mat::Zero(_px->rows(), _px->cols());
		}

		*plx += _param->_W.val.transpose() * _ly;

	}

};

#endif /* UNIOP_H_ */
