/*
 * SparseParam.h
 *
 *  Created on: Jul 25, 2016
 *      Author: mason
 */

#ifndef SPARSEPARAM_H_
#define SPARSEPARAM_H_

#include "Eigen/Dense"
#include "Utils.h"
#include "BaseParam.h"

// Notice: aux is an auxiliary variable to help parameter updating
// The in-out dimension definiation is different with dense parameters.
struct SparseParam : BaseParam{
	Mat aux;
	hash_set<int> _indexers;

	// allow sparse and dense parameters have different parameter initialization methods
	inline void initial(int outDim, int inDim) {
		val = Mat(inDim, outDim).unaryExpr(ptr_fun(urand));
		grad = Mat::Zero(inDim, outDim);
		aux = Mat::Zero(inDim, outDim);
		_indexers.clear();
	}

	inline void clearGrad() {
		static hash_set<int>::iterator it;
		for (it = _indexers.begin(); it != _indexers.end(); ++it) {
			int index = *it;
			grad.row(index).setZero();
		}
		_indexers.clear();
	}
	
	inline int outDim() {
		return val.cols();
	}

	inline int inDim() {
		return val.rows();
	}	

	inline void updateAdagrad(dtype alpha, dtype reg, dtype eps) {
		static hash_set<int>::iterator it;
		for (it = _indexers.begin(); it != _indexers.end(); ++it) {
			int index = *it;
			grad.row(index) = grad.row(index) + val.row(index) * reg;
			aux.row(index) = aux.row(index).array() + grad.row(index).array().square();
			val.row(index) = val.row(index).array() - grad.row(index).array() * alpha / (aux.row(index).array() + eps).sqrt();
		}
	}

	inline void randpoint(int& idx, int &idy, int seed){
		srand(seed);

		//select indexes randomly		
		std::vector<int> idRows, idCols;
		idRows.clear();
		idCols.clear();
		static hash_set<int>::iterator it;
		for (it = _indexers.begin(); it != _indexers.end(); ++it) {
			idRows.push_back(*it);
		}
		for (int i = 0; i < val.cols(); i++){
			idCols.push_back(i);
		}

		random_shuffle(idRows.begin(), idRows.end());
		random_shuffle(idCols.begin(), idCols.end());

		idx = idRows[0];
		idy = idCols[0];
	}

};

#endif /* SPARSEPARAM_H_ */
