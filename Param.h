/*
 * Param.h
 *
 *  Created on: Jul 25, 2016
 *      Author: mason
 */

#ifndef PARAM_H_
#define PARAM_H_

#include "Eigen/Dense"
#include "BaseParam.h"

 // Notice: aux is an auxiliary variable to help parameter updating
struct Param : BaseParam {
	Tensor2D aux;

	// allow sparse and dense parameters have different parameter initialization methods
	inline void initial(int outDim, int inDim, AlignedMemoryPool* mem = NULL) {
		val.init(outDim, inDim, mem);
		grad.init(outDim, inDim, mem);
		aux.init(outDim, inDim, mem);

		dtype bound = sqrt(6.0 / (outDim + inDim + 1));
		val.random(bound);
	}

	inline int outDim() {
		return val.row;
	}

	inline int inDim() {
		return val.col;
	}

	inline void clearGrad() {
		grad.zero();
	}

	inline void updateAdagrad(dtype alpha, dtype reg, dtype eps) {
		grad.vec() = grad.vec() + val.vec() * reg;
		aux.vec() = aux.vec() + grad.vec().square();
		val.vec() = val.vec() - grad.vec() * alpha / (aux.vec() + eps).sqrt();
	}

	inline void randpoint(int& idx, int &idy) {
		//select indexes randomly		
		std::vector<int> idRows, idCols;
		idRows.clear();
		idCols.clear();
		for (int i = 0; i < val.row; i++)
			idRows.push_back(i);
		for (int i = 0; i < val.col; i++)
			idCols.push_back(i);

		random_shuffle(idRows.begin(), idRows.end());
		random_shuffle(idCols.begin(), idCols.end());

		idy = idRows[0];
		idx = idCols[0];
	}

	inline dtype squareGradNorm() {
		dtype sumNorm = 0.0;
		for (int i = 0; i < grad.size; i++) {
			sumNorm += grad.v[i] * grad.v[i];
		}
		return sumNorm;
	}

	inline void rescaleGrad(dtype scale) {
		grad.vec() = grad.vec() * scale;
	}

	inline void save(std::ofstream &os)const {
		val.save(os);
		aux.save(os);
	}

	inline void load(std::ifstream &is, AlignedMemoryPool* mem = NULL) {
		val.load(is, mem);
		aux.load(is, mem);
	}
};

#endif /* PARAM_H_ */
