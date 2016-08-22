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
#include "BaseParam.h"

// Notice: aux is an auxiliary variable to help parameter updating
struct Param : BaseParam{
	Mat aux;

	// allow sparse and dense parameters have different parameter initialization methods
	inline void initial(int outDim, int inDim) {
		val = Mat(outDim, inDim).unaryExpr(ptr_fun(urand));
		grad = Mat::Zero(outDim, inDim);
		aux = Mat::Zero(outDim, inDim);
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

	inline void updateAdagrad(dtype alpha, dtype reg, dtype eps) {
		grad = grad + val * reg;
		aux = aux.array() + grad.array().square();
		val = val.array() - grad.array() * alpha / (aux.array() + eps).sqrt();
	}

	inline void randpoint(int& idx, int &idy, int seed){
		srand(seed);
		//select indexes randomly		
		std::vector<int> idRows, idCols;
		idRows.clear();
		idCols.clear();
		for (int i = 0; i < val.rows(); i++)
			idRows.push_back(i);
		for (int i = 0; i < val.cols(); i++)
			idCols.push_back(i);

		random_shuffle(idRows.begin(), idRows.end());
		random_shuffle(idCols.begin(), idCols.end());

		idx = idRows[0];
		idy = idCols[0];
	}

	inline dtype squareGradNorm(){
		dtype sumNorm = 0.0;
		for (int i = 0; i < grad.rows(); i++){
			for (int j = 0; j < grad.cols(); j++){
				sumNorm += grad(i, j) * grad(i, j);
			}
		}

		return sumNorm;
	}

	inline void rescaleGrad(dtype scale){
		grad = grad * scale;
	}
};

#endif /* PARAM_H_ */
