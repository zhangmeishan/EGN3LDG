/*
 * APOP.h
 *
 *  Created on: Jul 20, 2016
 *      Author: mszhang
 */

#ifndef APOP_H_
#define APOP_H_

#include "MyLib.h"
#include "Alphabet.h"
#include "Node.h"
#include "Graph.h"
#include "APParam.h"

// aiming for Averaged Perceptron, can not coexist with neural networks
struct APParams {
public:	
	APParam W;
	PAlphabet elems;
	int nVSize;
	int nDim;

public:
	APParams() {
		nVSize = 0;
		nDim = 0;
		elems = NULL;
	}

	inline void exportAdaParams(ModelUpdate& ada) {
		ada.addParam(&W);
	}

	inline void initialWeights(int nOSize) {
		if (nVSize == 0){
			std::cout << "please check the alphabet" << std::endl;
			return;
		}
		nDim = nOSize;
		W.initial(nOSize, nVSize);
	}

	//random initialization
	inline void initial(PAlphabet alpha, int nOSize){
		elems = alpha;
		nVSize = elems->size();
		initialWeights(nOSize);
	}

	inline int getFeatureId(const string& strFeat){
		return elems->from_string(strFeat);
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
	inline void forward(Graph* cg, const vector<string>& x, bool bTrain = false) {
		assert(param != NULL);
		val = Mat::Zero(dim, 1);
		static int featId;
		for (int idx = 0; idx < x.size(); idx++) {
			featId = param->getFeatureId(x[idx]);
			if (featId >= 0){
				tx.push_back(featId);
				val.col(0) += param->W.value(featId, bTrain).transpose();
			}
		}	

		cg->addNode(this);
	}

	//no output losses
	void backward() {
		assert(param != NULL);
		for (int idx = 0; idx < tx.size(); idx++) {
			param->W.indexers.insert(tx[idx]);
			param->W.grad.row(tx[idx]) += loss.col(0).transpose();
		}
	}

};

#endif /* APOP_H_ */
