/*
 * BaseParam.h
 *
 *  Created on: Jul 25, 2016
 *      Author: mason
 */

#ifndef BasePARAM_H_
#define BasePARAM_H_

#include "Eigen/Dense"
#include "Utils.h"
using namespace Eigen;

struct BaseParam {
	Mat val;
	Mat grad;

public:
	virtual inline void initial(int outDim, int inDim) = 0;
	virtual inline void updateAdagrad(dtype alpha, dtype reg, dtype eps) = 0;
	virtual inline int outDim() = 0;
	virtual inline int inDim() = 0;
	virtual inline void clearGrad() = 0;

	// Choose one point randomly
	virtual inline void randpoint(int& idx, int &idy, int seed) = 0;

};

#endif /* BasePARAM_H_ */
