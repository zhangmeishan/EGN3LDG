/*
 * Param.h
 *
 *  Created on: Jul 25, 2016
 *      Author: mason
 */

#ifndef PARAM_H_
#define PARAM_H_

#include "Eigen/Dense"
#include "Utils.h"
using namespace Eigen;

struct Param {
	MatrixXd val;
	MatrixXd grad;
	MatrixXd eg;

	inline void initial(int outDim, int inDim) {
		val = MatrixXd(outDim, inDim).unaryExpr(ptr_fun(urand));
		grad = MatrixXd::Zero(outDim, inDim);
		eg = MatrixXd::Zero(outDim, inDim);
	}

	inline int outDim() {
		return val.rows();
	}

	inline int inDim() {
		return val.cols();
	}

	inline void clearGrad() {
		grad.setZero();
	}


	inline void updateAdagrad(double alpha, double reg, double eps){
		grad = grad + val * reg;
		eg = eg.array() + grad.array().square();
		val = val.array() - grad.array() * alpha / (eg.array().sqrt() + eps);
	}

};

#endif /* PARAM_H_ */
