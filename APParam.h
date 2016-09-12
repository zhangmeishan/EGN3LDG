/*
 * APParam.h
 *
 *  Created on: Jul 25, 2016
 *      Author: mason
 */

#ifndef AVGPARAM_H_
#define AVGPARAM_H_

#include "Eigen/Dense"
#include "Utils.h"
#include "BaseParam.h"

// Notice: aux is an auxiliary variable to help parameter updating
// The in-out dimension definiation is different with dense parameters.
struct APParam : BaseParam{
	Mat aux;
	unordered_set<int> indexers;
	int max_update;
	VectorXi last_update;

	// allow sparse and dense parameters have different parameter initialization methods
	inline void initial(int outDim, int inDim) {
		val = Mat::Zero(inDim, outDim);
		grad = Mat::Zero(inDim, outDim);
		aux = Mat::Zero(inDim, outDim);
		indexers.clear();
		max_update = 0;
		last_update = VectorXi::Zero(inDim);
	}

	inline void clearGrad() {
		static unordered_set<int>::iterator it;
		for (it = indexers.begin(); it != indexers.end(); ++it) {
			int index = *it;
			grad.row(index).setZero();
		}
		indexers.clear();
	}
	
	inline int outDim() {
		return val.cols();
	}

	inline int inDim() {
		return val.rows();
	}	

	inline void updateAdagrad(dtype alpha, dtype reg, dtype eps) {
		static unordered_set<int>::iterator it;
		max_update++;

		for (it = indexers.begin(); it != indexers.end(); ++it) {
			int index = *it;
			aux.row(index) += (max_update - last_update(index)) * val.row(index) - grad.row(index);
			val.row(index) = val.row(index) - grad.row(index);
			last_update(index) = max_update;
		}
	}

	inline void randpoint(int& idx, int &idy){
		//select indexes randomly		
		std::vector<int> idRows, idCols;
		idRows.clear();
		idCols.clear();
		static unordered_set<int>::iterator it;
		for (it = indexers.begin(); it != indexers.end(); ++it) {
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

	inline dtype squareGradNorm(){
		static unordered_set<int>::iterator it;
		dtype sumNorm = 0.0;
		for (it = indexers.begin(); it != indexers.end(); ++it) {
			int index = *it;
			for (int idx = 0; idx < grad.cols(); idx++){
				sumNorm += grad(index, idx) * grad(index, idx);
			}
		}

		return sumNorm;
	}

	inline void rescaleGrad(dtype scale){
		static unordered_set<int>::iterator it;
		for (it = indexers.begin(); it != indexers.end(); ++it) {
			int index = *it;
			grad.row(index) = grad.row(index) * scale;
		}
	}


	inline Mat value(int featId, bool bTrain = false) {
		if (bTrain)
			return val.row(featId);
		else
			return sumWeight(featId).array() * 1.0 / max_update;
	}

	inline Mat sumWeight(int featId) {
		if (last_update(featId) < max_update) {
			int times = max_update - last_update(featId);
			aux.row(featId) += val.row(featId) * times;
			last_update(featId) = max_update;
		}

		return aux.row(featId);
	}

};

#endif /* AVGPARAM_H_ */
