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

class TriParam {
public:
	Param _W1;
	Param _W2;
	Param _W3;
	Param _b;

	bool _bUseB;

public:
	TriParam() {
		_bUseB = true;
	}

	inline void exportAdaParams(AdaUpdate& ada) {
		ada.addParam(&_W1);
		ada.addParam(&_W2);
		ada.addParam(&_W3);
		if (_bUseB) {
			ada.addParam(&_b);
		}
	}

	inline void initial(int nOSize, int nISize1, int nISize2, int nISize3, bool bUseB = true, int seed = 0) {
		srand(seed);
		_W1.initial(nOSize, nISize1);
		_W2.initial(nOSize, nISize2);
		_W3.initial(nOSize, nISize3);
		_b.initial(nOSize, 1);

		_bUseB = bUseB;
	}

};

// non-linear feed-forward node
// input nodes should be specified by forward function
// for input variables, we exploit column vector,
// which means a concrete input vector x_i is represented by x(0, i), x(1, i), ..., x(n, i)
class TriNode {
public:
	MatrixXd *_x1, *_x2, *_x3;
	MatrixXd _y, _ly;
	MatrixXd _ty, _lty;  // t means temp, _ty is to save temp vector before activation

	int _inDim1, _inDim2, _inDim3, _outDim;

	TriParam* _param;

	MatrixXd (*_f)(const MatrixXd&);   // activation function
	MatrixXd (*_f_deri)(const MatrixXd&, const MatrixXd&);  // derivation function of activation function

public:
	TriNode() {
		_x1 = NULL;
		_x2 = NULL;
		_x3 = NULL;
		_f = tanh;
		_f_deri = tanh_deri;
		_y.setZero();
		_ly.setZero();
		_ty.setZero();
		_lty.setZero();
		_param = NULL;
		_inDim1 = 0;
		_inDim2 = 0;
		_inDim3 = 0;
		_outDim = 0;
	}

	TriNode(TriParam* param) {
		_x1 = NULL;
		_x2 = NULL;
		_x3 = NULL;
		_f = tanh;
		_f_deri = tanh_deri;
		_y.setZero();
		_ly.setZero();
		_ty.setZero();
		_lty.setZero();
		setParam(param);
	}

	TriNode(MatrixXd (*f)(const MatrixXd&), MatrixXd (*f_deri)(const MatrixXd&, const MatrixXd&)) {
		_x1 = NULL;
		_x2 = NULL;
		_x3 = NULL;
		setFunctions(f, f_deri);
		_y.setZero();
		_ly.setZero();
		_ty.setZero();
		_lty.setZero();
		_param = NULL;
		_inDim1 = 0;
		_inDim2 = 0;
		_inDim3 = 0;
		_outDim = 0;
	}

	TriNode(TriParam* param, MatrixXd (*f)(const MatrixXd&), MatrixXd (*f_deri)(const MatrixXd&, const MatrixXd&)) {
		_x1 = NULL;
		_x2 = NULL;
		_x3 = NULL;
		setFunctions(f, f_deri);
		_y.setZero();
		_ly.setZero();
		_ty.setZero();
		_lty.setZero();
		setParam(param);
	}

	inline void setParam(TriParam* param) {
		_param = param;
		_inDim1 = _param->_W1.inDim();
		_inDim2 = _param->_W2.inDim();
		_inDim3 = _param->_W3.inDim();
		_outDim = _param->_W1.outDim();
		if (!_param->_bUseB) {
			cout << "please check whether _bUseB is true, usually this should be true for non-linear layer" << endl;
		}
	}

	// define the activation function and its derivation form
	inline void setFunctions(MatrixXd (*f)(const MatrixXd&), MatrixXd (*f_deri)(const MatrixXd&, const MatrixXd&)) {
		_f = f;
		_f_deri = f_deri;
	}

public:
	void forward(MatrixXd& x1, MatrixXd& x2, MatrixXd& x3) {
		assert(_param != NULL);

		_ty = _param->_W1.val * x1 + _param->_W2.val * x2 + _param->_W3.val * x3; 
		for (int idx = 0; idx < _ty.cols(); idx++) {
			_ty.row(idx) += _param->_b.val.row(0);
		}

		_y = _f(_ty);
		_x1 = &x1;
		_x2 = &x2;
		_x2 = &x3;
	}

	void backward(MatrixXd& lx1, MatrixXd& lx2, MatrixXd& lx3) {
		assert(_param != NULL);

		_lty = _ly.cwiseProduct(_f_deri(_ty, _y));

		_param->_W1.grad += _lty * _x1->transpose();
		_param->_W2.grad += _lty * _x2->transpose();
		_param->_W3.grad += _lty * _x3->transpose();

		for (int idx = 0; idx < _y.cols(); idx++) {
			_param->_b.grad.row(0) += _lty.row(idx);
		}

		if (lx1.size() == 0) {
			lx1 = MatrixXd::Zero(_x1->rows(), _x1->cols());
		}

		if (lx2.size() == 0) {
			lx2 = MatrixXd::Zero(_x2->rows(), _x2->cols());
		}

		if (lx3.size() == 0) {
			lx3 = MatrixXd::Zero(_x3->rows(), _x3->cols());
		}

		lx1 += _param->_W1.val.transpose() * _lty;
		lx2 += _param->_W2.val.transpose() * _lty;
		lx3 += _param->_W3.val.transpose() * _lty;
	}

};

#endif /* TRIOP_H_ */
