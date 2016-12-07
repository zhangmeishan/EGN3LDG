#ifndef CONCAT
#define CONCAT

#include "MyLib.h"
#include "Node.h"
#include "Graph.h"


struct ConcatNode : Node{
public:
	int nSize;
	vector<int> inDims;
	vector<PNode> ins;

public:
	ConcatNode() : Node(){
		nSize = 0;
		inDims.clear();
		ins.clear();
	}

	inline void clearValue(){
		Node::clearValue();
	}
	
	inline void init(int dim, dtype dropOut, AlignedMemoryPool* mem = NULL){
		Node::init(dim, -1, mem);
	}

public:
	void forward(Graph *cg, const vector<PNode>& x) {
		if (x.size() == 0){
			std::cout << "empty inputs for concat" << std::endl;
			return;
		}

		ins.clear();
		for (int i = 0; i < x.size(); i++){
			ins.push_back(x[i]);
		}

		forward();

		cg->addNode(this);
	}


	void forward(Graph *cg, PNode x1, PNode x2){
		ins.clear();
		ins.push_back(x1);
		ins.push_back(x2);

		forward();

		cg->addNode(this);
	}

	void forward(Graph *cg, PNode x1, PNode x2, PNode x3){
		ins.clear();
		ins.push_back(x1);
		ins.push_back(x2);
		ins.push_back(x3);

		forward();

		cg->addNode(this);
	}

	void forward(Graph *cg, PNode x1, PNode x2, PNode x3, PNode x4){
		ins.clear();
		ins.push_back(x1);
		ins.push_back(x2);
		ins.push_back(x3);
		ins.push_back(x4);

		forward();

		cg->addNode(this);
	}

	void forward(Graph *cg, PNode x1, PNode x2, PNode x3, PNode x4, PNode x5){
		ins.clear();
		ins.push_back(x1);
		ins.push_back(x2);
		ins.push_back(x3);
		ins.push_back(x4);
		ins.push_back(x5);

		forward();

		cg->addNode(this);
	}

	void forward(Graph *cg, PNode x1, PNode x2, PNode x3, PNode x4, PNode x5, PNode x6){
		ins.clear();
		ins.push_back(x1);
		ins.push_back(x2);
		ins.push_back(x3);
		ins.push_back(x4);
		ins.push_back(x5);
		ins.push_back(x6);

		forward();

		cg->addNode(this);
	}

	void backward(){
		int offset = 0;
		for (int i = 0; i < nSize; ++i){
			for (int idx = 0; idx < inDims[i]; idx++){
				ins[i]->loss[idx] += loss[offset + idx];				
			}
			offset += inDims[i];
		}
	}

	inline void unlock(){
		for (int i = 0; i < nSize; i++){
			ins[i]->lock--;
		}
		if(!lossed)return;
		for (int i = 0; i < nSize; i++){
			ins[i]->lossed = true;
		}		
	}

protected:
	inline void forward(){
		nSize = ins.size();
		inDims.clear();
		int curDim = 0;
		for (int i = 0; i < nSize; ++i){
			inDims.push_back(ins[i]->val.dim);
			curDim += inDims[i];
		}
		if(curDim != dim){
			std::cout << "input dim size not match" << std::endl;
			return;
		}

		int offset = 0;
		for (int i = 0; i < nSize; ++i){
			for (int idx = 0; idx < inDims[i]; idx++){
				val[offset + idx] = ins[i]->val[idx];
			}
			offset += inDims[i];
		}

		for (int i = 0; i < nSize; ++i){
			ins[i]->lock++;
		}
	}

};




#endif
