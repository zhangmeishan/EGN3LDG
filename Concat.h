#ifndef CONCAT
#define CONCAT

#include "MyLib.h"
#include "Node.h"
#include "Graph.h"

struct ConcatNode : Node{
public:
	int nSize, distance;
	vector<int> inDims;
	vector<PNode> ins;

public:
	ConcatNode(){
		clear();
	}

	inline void clear(){
		Node::clear();
		nSize = 0;
		distance = 0;
		inDims.clear();
	}

	inline void clearValue(){
		Node::clearValue();
	}

public:
	// please better restrict col to 1
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

	void forward(Graph *cg, const vector<PNode>& x, int distance) {
		// distance denotes right shift position, if less than zero,
		// denotes right (-distance) values are filled with zeors
		if (x.size() == 0){
			std::cout << "empty inputs for concat" << std::endl;
			return;
		}

		ins.clear();
		for (int i = 0; i < x.size(); i++){
			ins.push_back(x[i]);
		}

		forward(distance);

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
		for (int i = 0; i < ins.size(); i++){
			if (ins[i]->loss.size() == 0){
				ins[i]->loss = Mat::Zero(inDims[i], 1);
			}
		}

		int offset = distance > 0 ? distance : 0;
		for (int i = 0; i < nSize; ++i){
			for (int idx = 0; idx < inDims[i]; idx++){
				ins[i]->loss(idx, 0) += loss(offset + idx, 0);				
			}
			offset += inDims[i];
		}
	}

	inline void unlock(){
		for (int i = 0; i < nSize; i++){
			ins[i]->lock--;
		}
	}

protected:
	inline void forward(){
		assert(ins.size() > 0);

		distance = 0;
		nSize = ins.size();
		inDims.clear();
		dim = 0;
		for (int i = 0; i < nSize; ++i){
			inDims.push_back(ins[i]->val.rows());
			dim += inDims[i];
		}

		val = Mat::Zero(dim, 1);
		int offset = 0;
		for (int i = 0; i < nSize; ++i){
			assert(ins[i]->val.rows() == inDims[i]);
			for (int idx = 0; idx < inDims[i]; idx++){
				val(offset + idx, 0) = ins[i]->val(idx, 0);
			}
			offset += inDims[i];
		}

		for (int i = 0; i < nSize; ++i){
			ins[i]->lock++;
		}
	}

	inline void forward(int distance){
		assert(ins.size() > 0);

		this->distance = distance;
		nSize = ins.size();
		inDims.clear();
		dim = 0;
		for (int i = 0; i < nSize; ++i){
			inDims.push_back(ins[i]->val.rows());
			dim += inDims[i];
		}
		dim += distance;
		val = Mat::Zero(dim, 1);
		int offset = distance > 0 ? distance : 0;
		for (int i = 0; i < nSize; ++i){
			assert(ins[i]->val.rows() == inDims[i]);
			for (int idx = 0; idx < inDims[i]; idx++){
				val(offset + idx, 0) = ins[i]->val(idx, 0);
			}
			offset += inDims[i];
		}

		for (int i = 0; i < nSize; ++i){
			ins[i]->lock++;
		}
	}
};




#endif
