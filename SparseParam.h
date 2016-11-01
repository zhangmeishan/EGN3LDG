/*
 * SparseParam.h
 *
 *  Created on: Jul 25, 2016
 *      Author: mason
 */

#ifndef SPARSEPARAM_H_
#define SPARSEPARAM_H_

#include "BaseParam.h"
using namespace nr;

// Notice: aux is an auxiliary variable to help parameter updating
// The in-out dimension definiation is different with dense parameters.
struct SparseParam : BaseParam{
	NRMat<dtype> val;
	NRMat<dtype> grad;
	NRMat<dtype> aux;
	unordered_set<int> indexers;

	// allow sparse and dense parameters have different parameter initialization methods
	inline void initial(int outDim, int inDim) {
		val.resize(inDim, outDim); 
		random(val);
		grad.resize(inDim, outDim); grad = 0;
		aux.resize(inDim, outDim); aux = 0;
		indexers.clear();
	}

	inline void clearGrad() {
		static unordered_set<int>::iterator it;
		int outDim = grad.ncols();
		for (it = indexers.begin(); it != indexers.end(); ++it) {
			int index = *it;
			for (int idx = 0; idx < outDim; idx++){
				grad[index][idx] = 0;
			}

		}
		indexers.clear();
	}
	
	inline int outDim() {
		return val.ncols();
	}

	inline int inDim() {
		return val.nrows();
	}	

	inline void updateAdagrad(dtype alpha, dtype reg, dtype eps) {
		static unordered_set<int>::iterator it;
		int outDim = grad.ncols();
		for (it = indexers.begin(); it != indexers.end(); ++it) {
			int index = *it;
			for (int idx = 0; idx < outDim; idx++){
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
		int outDim = val.ncols();
		for (int i = 0; i < outDim; i++){
			idCols.push_back(i);
		}

		random_shuffle(idRows.begin(), idRows.end());
		random_shuffle(idCols.begin(), idCols.end());

		idx = idRows[0];
		idy = idCols[0];
	}

	inline dtype squareGradNorm(){
		static unordered_set<int>::iterator it;
		int outDim = val.ncols();
		dtype sumNorm = 0.0;
		for (it = indexers.begin(); it != indexers.end(); ++it) {
			int index = *it;
			for (int idx = 0; idx < outDim; idx++){
				sumNorm += grad[index][idx] * grad[index][idx];
			}
		}

		return sumNorm;
	}

	inline void rescaleGrad(dtype scale){
		static unordered_set<int>::iterator it;
		int outDim = val.ncols();
		for (it = indexers.begin(); it != indexers.end(); ++it) {
			int index = *it;
			for (int idx = 0; idx < outDim; idx++){
				grad[index][idx] = grad[index][idx] * scale;
			}
		}
	}

	inline void value(const int& featId, Mat& out){
		int outDim = val.ncols();
		if (out.size() != outDim){
			out = Mat::Zero(outDim, 1);
		}

		for (int idx = 0; idx < outDim; idx++){
			out.coeffRef(idx) = val[featId][idx];
		}
	}

	inline void value(const vector<int>& featIds, Mat& out){
		int outDim = val.ncols();
		if (out.size() != outDim){
			out = Mat::Zero(outDim, 1);
		}
		int featNum = featIds.size();
		int featId;
		for (int i = 0; i < featNum; i++){
			featId = featIds[i];
			for (int idx = 0; idx < outDim; idx++){
				out.coeffRef(idx) += val[featId][idx];
			}
		}
	}

	inline void loss(const int& featId, const Mat&loss){
		int outDim = val.ncols();
		indexers.insert(featId);
		for (int idx = 0; idx < outDim; idx++){
			grad[featId][idx] += loss.coeffRef(idx);
		}
	}

	inline void loss(const vector<int>& featIds, const Mat&loss){
		int outDim = val.ncols();
		int featNum = featIds.size();
		int featId;
		for (int i = 0; i < featNum; i++){
			featId = featIds[i];
			indexers.insert(featId);
			for (int idx = 0; idx < outDim; idx++){
				grad[featId][idx] += loss.coeffRef(idx);
			}
		}
	}
	
	inline void value(const int& featId, NRVec<dtype>& out){
		int outDim = val.ncols();
		if (out.size() != outDim){
			out.resize(outDim);
		}

		for (int idx = 0; idx < outDim; idx++){
			out[idx] = val[featId][idx];
		}
	}

	inline void value(const vector<int>& featIds, NRVec<dtype>& out){
		int outDim = val.ncols();
		if (out.size() != outDim){
			out.resize(outDim);
			out = 0;
		}
		int featNum = featIds.size();
		int featId;
		for (int i = 0; i < featNum; i++){
			featId = featIds[i];
			for (int idx = 0; idx < outDim; idx++){
				out[idx] += val[featId][idx];
			}
		}
	}

	inline void loss(const int& featId, const NRVec<dtype>& loss){
		int outDim = val.ncols();
		indexers.insert(featId);
		for (int idx = 0; idx < outDim; idx++){
			grad[featId][idx] += loss[idx];
		}
	}

	inline void loss(const vector<int>& featIds, const NRVec<dtype>& loss){
		int outDim = val.ncols();
		int featNum = featIds.size();
		int featId;
		for (int i = 0; i < featNum; i++){
			featId = featIds[i];
			indexers.insert(featId);
			for (int idx = 0; idx < outDim; idx++){
				grad[featId][idx] += loss[idx];
			}
		}
	}	
};

#endif /* SPARSEPARAM_H_ */
