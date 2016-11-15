/*
 * SparseParam.h
 *
 *  Created on: Jul 25, 2016
 *      Author: mason
 */

#ifndef SPARSEPARAM_H_
#define SPARSEPARAM_H_

#include "BaseParam.h"

// Notice: aux is an auxiliary variable to help parameter updating
// The in-out dimension definiation is different with dense parameters.
struct SparseParam : BaseParam{
	Tensor2D aux;
	unordered_set<int> indexers;

	// allow sparse and dense parameters have different parameter initialization methods
	inline void initial(int outDim, int inDim, AlignedMemoryPool* mem = NULL) {
		//not in the aligned memory pool
		val.init(outDim, inDim); 
		val.random(0.01);
		grad.init(outDim, inDim); 
		aux.init(outDim, inDim); 
		indexers.clear();
	}

	inline void clearGrad() {
		static unordered_set<int>::iterator it;
		for (it = indexers.begin(); it != indexers.end(); ++it) {
			int index = *it;
			for (int idx = 0; idx < grad.row; idx++){
				grad[index][idx] = 0;
			}
		}
		indexers.clear();
	}
	
	inline int outDim() {
		return val.row;
	}

	inline int inDim() {
		return val.col;
	}	

	inline void updateAdagrad(dtype alpha, dtype reg, dtype eps) {
		static unordered_set<int>::iterator it;
		for (it = indexers.begin(); it != indexers.end(); ++it) {
			int index = *it;
			for (int idx = 0; idx < grad.row; idx++){
				grad[index][idx] = grad[index][idx] + val[index][idx] * reg;
				aux[index][idx] = aux[index][idx] + grad[index][idx] * grad[index][idx];
				val[index][idx] = val[index][idx] - grad[index][idx] * alpha / sqrt(aux[index][idx] + eps);
			}
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

		for (int i = 0; i < val.row; i++){
			idCols.push_back(i);
		}

		random_shuffle(idRows.begin(), idRows.end());
		random_shuffle(idCols.begin(), idCols.end());

		idy = idRows[0];
		idx = idCols[0];
	}

	inline dtype squareGradNorm(){
		static unordered_set<int>::iterator it;
		dtype sumNorm = 0.0;
		for (it = indexers.begin(); it != indexers.end(); ++it) {
			int index = *it;
			for (int idx = 0; idx < val.row; idx++){
				sumNorm += grad[index][idx] * grad[index][idx];
			}
		}

		return sumNorm;
	}

	inline void rescaleGrad(dtype scale){
		static unordered_set<int>::iterator it;
		for (it = indexers.begin(); it != indexers.end(); ++it) {
			int index = *it;
			for (int idx = 0; idx < val.row; idx++){
				grad[index][idx] = grad[index][idx] * scale;
			}
		}
	}

	inline void value(const int& featId, Tensor1D& out){
		for (int idx = 0; idx < val.row; idx++){
			out[idx] = val[featId][idx];
		}
	}

	inline void value(const vector<int>& featIds, Tensor1D& out){
		int featNum = featIds.size();
		int featId;
		for (int i = 0; i < featNum; i++){
			featId = featIds[i];
			for (int idx = 0; idx < val.row; idx++){
				out[idx] += val[featId][idx];
			}
		}
	}

	inline void loss(const int& featId, const Tensor1D& loss){
		indexers.insert(featId);
		for (int idx = 0; idx < val.row; idx++){
			grad[featId][idx] += loss[idx];
		}
	}

	inline void loss(const vector<int>& featIds, const Tensor1D& loss){
		int featNum = featIds.size();
		int featId;
		for (int i = 0; i < featNum; i++){
			featId = featIds[i];
			indexers.insert(featId);
			for (int idx = 0; idx < val.row; idx++){
				grad[featId][idx] += loss[idx];
			}
		}
	}
	
};

#endif /* SPARSEPARAM_H_ */
