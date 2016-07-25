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

class UniParam {
public:
	Param _W;
	Param _b;

	bool _bUseB;

public:
	UniParam() {
		_bUseB = true;
	}

	inline void exportAdaParams(AdaUpdate &ada) {
		ada.addParam(&_W);
		if (_bUseB) {
			ada.addParam(&_b);
		}
	}

	inline void initial(int nOSize, int nISize, bool bUseB = true, int seed = 0) {
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
class UniNode {
public:
	MatrixXd *_x;
	MatrixXd _y, _ly;
	MatrixXd _ty, _lty;  // t means temp, _ty is to save temp vector before activation

	int _inDim, _outDim;

	UniParam* _param;

	MatrixXd (*_f)(const MatrixXd&);   // activation function
	MatrixXd (*_f_deri)(const MatrixXd&, const MatrixXd&);  // derivation function of activation function

public:
	UniNode() {
		_x = NULL;
		_f = tanh;
		_f_deri = tanh_deri;
		_y.setZero();
		_ly.setZero();
		_ty.setZero();
		_lty.setZero();
		_param = NULL;
		_inDim = _outDim = 0;
	}

	UniNode(UniParam* param) {
		_x = NULL;
		_f = tanh;
		_f_deri = tanh_deri;
		_y.setZero();
		_ly.setZero();
		_ty.setZero();
		_lty.setZero();
		setParam(param);
	}

	UniNode(MatrixXd (*f)(const MatrixXd&), MatrixXd (*f_deri)(const MatrixXd&, const MatrixXd&)) {
		_x = NULL;
		setFunctions(f, f_deri);
		_y.setZero();
		_ly.setZero();
		_ty.setZero();
		_lty.setZero();
		_param = NULL;
		_inDim = _outDim = 0;
	}

	UniNode(UniParam* param, MatrixXd (*f)(const MatrixXd&), MatrixXd (*f_deri)(const MatrixXd&, const MatrixXd&)) {
		_x = NULL;
		setFunctions(f, f_deri);
		_y.setZero();
		_ly.setZero();
		_ty.setZero();
		_lty.setZero();
		setParam(param);
	}

	inline void setParam(UniParam* param) {
		_param = param;
		_inDim = _param->_W.inDim();
		_outDim = _param->_W.outDim();
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
	void forward(MatrixXd& x) {
		assert(_param != NULL);

		_ty = _param->_W.val * x;
		for (int idx = 0; idx < _ty.cols(); idx++) {
			_ty.row(idx) += _param->_b.val.row(0);
		}

		_y = _f(_ty);
		_x = &x;
	}

	void backward(MatrixXd& lx) {
		assert(_param != NULL);

		_lty = _ly.cwiseProduct(_f_deri(_ty, _y));

		_param->_W.grad += _lty * _x->transpose();

		for (int idx = 0; idx < _y.cols(); idx++) {
			_param->_b.grad.row(0) += _lty.row(idx);
		}

		if (lx.size() == 0) {
			lx = MatrixXd::Zero(_x->rows(), _x->cols());
		}

		lx += _param->_W.val.transpose() * _lty;

	}

};

// Linear Node, ofen used for computing output
// input nodes should be specified by forward function
// for input variables, we exploit column vector,
// which means a concrete input vector x_i is represented by x(0, i), x(1, i), ..., x(n, i)
class LinearNode {
public:
	MatrixXd *_x;
	MatrixXd _y, _ly;

	int _inDim, _outDim;

	UniParam* _param;

public:
	LinearNode() {
		_x = NULL;
		_y.setZero();
		_ly.setZero();
		_param = NULL;
		_inDim = _outDim = 0;
	}

	LinearNode(UniParam* param) {
		_x = NULL;
		_y.setZero();
		_ly.setZero();
		setParam(param);
	}

	inline void setParam(UniParam* param) {
		_param = param;
		_inDim = _param->_W.inDim();
		_outDim = _param->_W.outDim();
		if (param->_bUseB) {
			cout << "please check whether _bUseB is false, usually this should be false for linear layer" << endl;
		}
	}

public:
	void forward(MatrixXd& x) {
		assert(_param != NULL);
		_y = _param->_W.val * x;
		_x = &x;
	}

	void backward(MatrixXd& lx) {
		assert(_param != NULL);

		_param->_W.grad += _ly * _x->transpose();

		if (lx.size() == 0) {
			lx = MatrixXd::Zero(_x->rows(), _x->cols());
		}

		lx += _param->_W.val.transpose() * _ly;

	}

};

#endif /* UNIOP_H_ */
