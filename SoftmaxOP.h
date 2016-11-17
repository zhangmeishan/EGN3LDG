#ifndef SoftmaxOP
#define SoftmaxOP

#include "MyLib.h"
#include "Node.h"
#include "Graph.h"
#include "AtomicOP.h"


//default val dim = max_length * atomic_dim
struct SoftmaxNode : Node {
private:
	int maxsize;
public:
	vector<PNode> ins;
	Tensor1D maxv, expv, sumexpv;
	int nSize;
	int unit_dim;

public:
	SoftmaxNode() : Node(){
		ins.clear();
		nSize = 0;
		unit_dim = 0;
		maxsize = max_length;
	}
	
	~SoftmaxNode(){
		ins.clear();
		nSize = 0;
		unit_dim = 0;
	}

	inline void clearValue(){
		Node::clearValue();
		ins.clear();
		nSize = 0;
		expv.zero();
		maxv.zero();
		sumexpv.zero();
	}

	inline void setParam(const int& maxsize) {
		this->maxsize = maxsize;
	}
	
	//note: please this is the unit dim	
	inline void init(int unit_dim, dtype dropOut, AlignedMemoryPool* mem = NULL){
		this->unit_dim = unit_dim;
		Node::init(maxsize * unit_dim, -1, mem);
		expv.init(maxsize * unit_dim, mem);
		maxv.init(unit_dim, mem);
		sumexpv.init(unit_dim, mem);
	}

public:
	// please better restrict col to 1
	void forward(Graph *cg, const vector<PNode>& x) {
		nSize = x.size();
		if (nSize < 1) {
			std::cout << "at least one nodes are required" << std::endl;
			return;
		}

		ins.clear();
		for (int i = 0; i < nSize; i++) {
			ins.push_back(x[i]);
		}

		forward();
		cg->addNode(this);
	}

	void forward(Graph *cg, PNode x1, PNode x2) {
		ins.clear();
		ins.push_back(x1);
		ins.push_back(x2);
		nSize = 2;

		forward();
		cg->addNode(this);
	}

	void forward(Graph *cg, PNode x1, PNode x2, PNode x3) {
		ins.clear();
		ins.push_back(x1);
		ins.push_back(x2);
		ins.push_back(x3);
		nSize = 3;

		forward();
		cg->addNode(this);
	}

	void forward(Graph *cg, PNode x1, PNode x2, PNode x3, PNode x4) {
		ins.clear();
		ins.push_back(x1);
		ins.push_back(x2);
		ins.push_back(x3);
		ins.push_back(x4);
		nSize = 4;

		forward();
		cg->addNode(this);
	}

	void forward(Graph *cg, PNode x1, PNode x2, PNode x3, PNode x4, PNode x5) {
		ins.clear();
		ins.push_back(x1);
		ins.push_back(x2);
		ins.push_back(x3);
		ins.push_back(x4);
		ins.push_back(x5);
		nSize = 5;

		forward();
		cg->addNode(this);
	}

	void forward(Graph *cg, PNode x1, PNode x2, PNode x3, PNode x4, PNode x5, PNode x6) {
		ins.clear();
		ins.push_back(x1);
		ins.push_back(x2);
		ins.push_back(x3);
		ins.push_back(x4);
		ins.push_back(x5);
		ins.push_back(x6);
		nSize = 6;

		forward();
		cg->addNode(this);
	}

public:
	void forward() {
		if(nSize > max_length) std::cout << "length overflow" << std::endl;
			
		for(int idy = 0; idy < unit_dim; idy++){
			maxv[idy] = ins[0]->val[idy];
			for (int idx = 1; idx < nSize; idx++){
				if(ins[idx]->val[idy] > maxv[idy]){
					maxv[idy] = ins[idx]->val[idy];
				}
			}
		}		
		int offset = 0;
		for (int idx = 0; idx < nSize; idx++){
			for(int idy = 0; idy < unit_dim; idy++){
				expv[offset] = exp(ins[idx]->val[idy] - maxv[idy]);
				if(idx == 0){
					sumexpv[idy] = expv[offset];
				}
				else{
					sumexpv[idy] += expv[offset];
				}
				offset++;
			}
		}		
		
		offset = 0;
		for (int idx = 0; idx < nSize; idx++){
			for(int idy = 0; idy < unit_dim; idy++){
				val[offset] = expv[offset] / sumexpv[idy];
				offset++;
			}
		}

		//debug
		/*
		for (int idy = 0; idy < unit_dim; idy++) {
			for (int idx = 0; idx < nSize; idx++) {
				std::cout << val[idx * unit_dim + idy] << " ";
			}
			std::cout << ": sumexp"  << sumexpv[idy] << std::endl;
		}
		*/

		for (int idx = 0; idx < nSize; idx++){
			ins[idx]->lock++;
		}
				
	}

	void backward(){
		int offset = 0;
		for (int idx = 0; idx < nSize; idx++){
			for(int idy = 0; idy < unit_dim; idy++){
				for(int idz = 0; idz < nSize; idz++) {
					if(idx == idz) {
						ins[idz]->loss[idy] += loss[offset] *  val[offset] * (1 - val[offset]);
					}
					else{
						ins[idz]->loss[idy] -= loss[offset] *  val[offset] * val[idz * unit_dim + idy];
					}
				}
				offset++;
			}
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
};


class SoftmaxBuilder {
public:
	SoftmaxNode _softmax;
	vector<SelectionNode> _output;
	int _dim;

public:
	SoftmaxBuilder() {
		clear();
	}

	~SoftmaxBuilder() {
		clear();
	}
public:
	inline void resize(int maxsize) {
		_output.resize(maxsize);
		_softmax.setParam(maxsize);
	}

	inline void clear() {
		_output.clear();
		_dim = 0;
	}

public:
	inline void init(int inDim, AlignedMemoryPool* mem = NULL) {
		_dim = inDim;
		_softmax.init(_dim, -1, mem);
		int maxsize = _output.size();
		for (int idx = 0; idx < maxsize; idx++) {
			_output[idx].init(_dim, -1, mem);
		}
	}

	inline void forward(Graph *cg, const vector<PNode>& x) {
		if (x.size() == 0) {
			std::cout << "empty inputs for softmax builder operation" << std::endl;
			return;
		}
		int nSize = x.size();
		if (x[0]->val.dim != _dim) {
			std::cout << "input dim dose not match for softmax builder operation" << std::endl;
			return;
		}

		_softmax.forward(cg, x);

		int offset = 0;
		for (int idx = 0; idx < nSize; idx++) {
			_output[idx].forward(cg, &_softmax, offset);
			offset += _dim;
		}
	}


};

#endif
