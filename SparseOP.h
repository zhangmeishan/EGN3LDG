/*
 * UniOP.h
 *
 *  Created on: Jul 20, 2016
 *      Author: mason
 */

#ifndef SPARSEOP_H_
#define SPARSEOP_H_

#include "SparseParam.h"
#include "MyLib.h"
#include "Alphabet.h"
#include "Node.h"

struct SparseParams {
public:
	SparseParam W;
	Alphabet elems;
	int nVSize;
	int nDim;

public:
	SparseParams() {
	}

	inline void exportAdaParams(ModelUpdate& ada) {
		ada.addParam(&W);
	}

	inline void initialWeights(int nOSize, int seed = 0) {
		if (nVSize == 0){
			std::cout << "please check the alphabet" << std::endl;
			return;
		}
		srand(seed);
		nDim = nOSize;
		W.initial(nOSize, nVSize);
	}

	// for sepcial elements such as UNK and NULL, please add insert them into the elem_stat
	// I will not implement another addAlpha function, thus please collect alpha all at once
	inline void initialAlpha(const hash_map<string, int>& elem_stat, int cutOff = 0){
		elems.clear();

		static hash_map<string, int>::const_iterator elem_iter;
		for (elem_iter = elem_stat.begin(); elem_iter != elem_stat.end(); elem_iter++) {
			if (elem_iter->second > cutOff) {
				elems.from_string(elem_iter->first);
			}
		}
		elems.set_fixed_flag(true);
		nVSize = elems.size();
	}

	//random initialization
	inline void initial(const hash_map<string, int>& elem_stat, int cutOff, int nOSize, int seed){
		initialAlpha(elem_stat, cutOff);
		initialWeights(nOSize, seed);
	}

	inline int getFeatureId(const string& strFeat){
		return elems.from_string(strFeat);
	}

};

//only implemented sparse linear node.
//non-linear transformations are not support,
//but we can use other approaches to achieve the same goal.
struct SparseNode : Node {
public:
	vector<int> tx;
	SparseParams* param;

public:
	SparseNode() {
		clear();
	}

	inline void setParam(SparseParams* paramInit) {
		param = paramInit;
		dim = param->nDim;
	}

	inline void clear(){
		Node::clear();
		tx.clear();
		param = NULL;

	}

	inline void clearValue(){
		Node::clearValue();
		tx.clear();
	}

public:
	//notice the output
	void forward(const vector<string>& x) {
		assert(param != NULL);
		val = Mat::Zero(dim, 1);
		static int featId;
		for (int idx = 0; idx < x.size(); idx++) {
			featId = param->getFeatureId(x[idx]);
			if (featId >= 0){
				tx.push_back(featId);
				val.col(0) += param->W.val.row(featId).transpose();
			}
		}
	}

	void backward() {
		assert(param != NULL);
		for (int idx = 0; idx < tx.size(); idx++) {
			param->W.indexers.insert(tx[idx]);
			param->W.grad.row(tx[idx]) += loss.col(0).transpose();
		}
	}

};

#endif /* SPARSEOP_H_ */
