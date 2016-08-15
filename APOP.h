/*
 * AP1O.h
 *
 *  Created on: Jul 20, 2016
 *      Author: mszhang
 */

#ifndef APOP_H_
#define APOP_H_

#include "MyLib.h"
#include "Alphabet.h"
#include "Node.h"

// aiming for Averaged Perceptron, can not coexist with neural networks
// thus we do not use the BaseParam class
struct APParams {
public:
	Alphabet elems;
	hash_set<int> indexers;
	Mat W, gradW, sumW;

	int max_update;
	VectorXi last_update;

	int nVSize;
	int nDim;

public:
	APParams() {
		indexers.clear();
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

	inline void initialWeights(int nOSize, int seed = 0) {
		if (nVSize == 0){
			std::cout << "please check the alphabet" << std::endl;
			return;
		}
		srand(seed);
		nDim = nOSize;
		W = Mat(nVSize, nDim).unaryExpr(ptr_fun(urand));

		gradW = Mat::Zero(nVSize, nDim);
		sumW = Mat::Zero(nVSize, nDim);

		max_update = 0;
		last_update = VectorXi::Zero(nVSize);
	}

	//random initialization
	inline void initial(const hash_map<string, int>& elem_stat, int cutOff, int nOSize, int seed){
		initialAlpha(elem_stat, cutOff);
		initialWeights(nOSize, seed);
	}

	inline Mat get(int featId, bool bTrain = false) {
		if (bTrain)
			return W.row(featId);
		else
			return sumWeight(featId).array() * 1.0 / max_update;
	}

	inline Mat sumWeight(int featId) {
		if (last_update(featId) < max_update) {
			int times = max_update - last_update(featId);
			sumW.row(featId) += W.row(featId) * times;
			last_update(featId) = max_update;
		}

		return sumW.row(featId);
	}

	inline int getFeatureId(const string& strFeat){
		return elems.from_string(strFeat);
	}

	void update() {
		static hash_set<int>::iterator it;
		max_update++;

		for (it = indexers.begin(); it != indexers.end(); ++it) {
			int index = *it;
			sumW.row(index) += (max_update - last_update(index)) * W.row(index) - gradW.row(index);
			W.row(index) = W.row(index) - gradW.row(index);
			last_update(index) = max_update;
		}

		clearGrad();
	}

	void clearGrad() {
		static hash_set<int>::iterator it;
		for (it = indexers.begin(); it != indexers.end(); ++it) {
			int index = *it;
			gradW.row(index).setZero();
		}
		indexers.clear();

	}

};

// a single node;
// input variables are not allocated by current node(s)
struct APNode : Node {

public:
	APParams* param;
	vector<int> tx;

public:
	APNode() {
		clear();
	}

	inline void setParam(APParams* paramInit) {
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

	// initialize inputs at the same times
	inline void forward(const vector<string>& x, bool bTrain = false) {
		assert(param != NULL);
		val = Mat::Zero(dim, 1);
		static int featId;
		for (int idx = 0; idx < x.size(); idx++) {
			featId = param->getFeatureId(x[idx]);
			if (featId >= 0){
				tx.push_back(featId);
				val.col(0) += param->get(featId, bTrain).transpose();
			}
		}		
	}

	//no output losses
	void backward() {
		assert(param != NULL);
		for (int idx = 0; idx < tx.size(); idx++) {
			param->indexers.insert(tx[idx]);
			param->gradW.row(tx[idx]) += loss.col(0).transpose();
		}
	}

};


#endif /* APOP_H_ */
