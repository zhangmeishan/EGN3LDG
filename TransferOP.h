/*
 * TransferOP.h
 *
 *  Created on: Jul 20, 2016
 *      Author: mason
 */

#ifndef TransferOP_H_
#define TransferOP_H_

#include "Param.h"
#include "MyLib.h"
#include "Node.h"
#include "Graph.h"

struct TransferParams {
public:
	vector<Param> W;
	PAlphabet elems;
	int nVSize;


public:
	TransferParams() {
		nVSize = 0;
	}

	inline void exportAdaParams(ModelUpdate& ada) {
		for(int idx = 0; idx < nVSize; idx++){
			ada.addParam(&(W[idx]));
		}
	}

	inline void initial(PAlphabet alpha, int nOSize, int nISize, AlignedMemoryPool* mem = NULL) {
		elems = alpha;
		nVSize = elems->size();
        W.resize(nVSize);
		for(int idx = 0; idx < nVSize; idx++){
			W[idx].initial(nOSize, nISize, mem);
		}
	}

    inline int getElemId(const string& strFeat) {
        return elems->from_string(strFeat);
    }
  
  // will add it
	inline void save(std::ofstream &os) const {

	}

  // will add it
	inline void load(std::ifstream &is, AlignedMemoryPool* mem = NULL) {

	}

};



struct TransferNode : Node{
public:
	PNode in;
	int xid;

	TransferParams* param;

public:
	TransferNode() : Node(){
		in = NULL;
		xid = -1;
		param = NULL;

	}


	inline void setParam(TransferParams* paramInit) {
		param = paramInit;
	}
	
	inline void clearValue(){
		Node::clearValue();
		in = NULL;
		xid = -1;
	}	


public:
	void forward(Graph *cg, PNode x, const string& strNorm) {
		in = x;		
		xid = param->getElemId(strNorm);
		if(xid >= 0){
			val.mat() = param->W[xid].val.mat() * in->val.mat();
		}
		else{
			val = 0;
		}

		in->lock++;
		cg->addNode(this);
	}

	void backward() {
		if(xid >= 0){
			param->W[xid].grad.mat() += loss.mat() * in->val.tmat();
			in->loss.mat() += param->W[xid].val.mat().transpose() * loss.mat();
		}
	}


	inline void unlock(){
		in->lock--;
		if(!lossed)return;
		in->lossed = true;
	}

};

#endif /* TransferOP_H_ */
