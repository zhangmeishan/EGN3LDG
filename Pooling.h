#ifndef POOLING
#define POOLING

#include "MyLib.h"
#include "Node.h"
#include "Graph.h"

struct PoolNode : Node {
public:
	vector<Tensor1D> masks; 
	vector<PNode> ins;
	int nSize;

public:
	PoolNode() : Node(){
		ins.clear();
	}
	
	~PoolNode(){
		masks.clear();
		ins.clear();
	}

	inline void clearValue(){
		Node::clearValue();
		ins.clear();
	}
	
	inline void setParam(int maxsize){
		masks.resize(maxsize);
	}
	
	inline void init(int dim, dtype dropOut, AlignedMemoryPool* mem = NULL){
		Node::init(dim, -1, mem);
		int count = masks.size();	
		for(int idx = 0; idx < count; idx++){
			masks[idx].init(dim, mem);
		}
	}		

public:

	virtual void forward(Graph *cg, const vector<PNode>& x) = 0;

	void backward(){
		for (int i = 0; i < nSize; i++){
			ins[i]->loss.vec() += loss.vec() * masks[i].vec();			
		}
	}

	inline void unlock(){
		for (int i = 0; i < nSize; i++){
			ins[i]->decrease_loc();
		}
		if(!lossed)return;
		for (int i = 0; i < nSize; i++){
			ins[i]->lossed = true;
		}			
	}
};

struct MaxPoolNode : PoolNode {
public:
	MaxPoolNode() : PoolNode(){
	}

public:
	//Be careful that the row is the dim of input vector, and the col is the number of input vectors
	//Another point is that we change the input vectors directly.
	void forward(Graph *cg, const vector<PNode>& x) {
		if (x.size() == 0 ){
			std::cout << "empty inputs for max pooling" << std::endl;
			return;
		}
		nSize = x.size();
		ins.clear();
		for (int i = 0; i < nSize; i++){
			ins.push_back(x[i]);
		}

		for (int i = 0; i < nSize; ++i){
			if (ins[i]->val.dim != dim){
				std::cout << "input matrixes are not matched" << std::endl;
				clearValue();
				return;
			}
			masks[i].zero();
		}

		for (int idx = 0; idx < dim; idx++){
			int maxIndex = -1;
			for (int i = 0; i < nSize; ++i){
				if (maxIndex == -1 || ins[i]->val[idx] > ins[maxIndex]->val[idx]){
					maxIndex = i;
				}
			}
			masks[maxIndex][idx] = 1.0;
		}

		val.zero();
		for (int i = 0; i < nSize; ++i){
			val.vec() += masks[i].vec() * ins[i]->val.vec();
		}

		for (int i = 0; i < nSize; ++i){
			ins[i]->increase_loc();
		}

		cg->addNode(this);
	}

};


struct SumPoolNode : PoolNode {
public:
	SumPoolNode() : PoolNode(){
	}

public:
	//Be careful that the row is the dim of input vector, and the col is the number of input vectors
	//Another point is that we change the input vectors directly.
	void forward(Graph *cg, const vector<PNode>& x) {
		if (x.size() == 0){
			std::cout << "empty inputs for max pooling" << std::endl;
			return;
		}

		nSize = x.size();
		ins.clear();
		for (int i = 0; i < nSize; i++){
			ins.push_back(x[i]);
		}

		for (int i = 0; i < nSize; ++i){
			if (ins[i]->val.dim != dim){
				std::cout << "input matrixes are not matched" << std::endl;
				clearValue();
				return;
			}
			masks[i] = 1.0;
		}

		val.zero();
		for (int i = 0; i < nSize; ++i){
			val.vec() += masks[i].vec() * ins[i]->val.vec();
		}

		for (int i = 0; i < nSize; ++i){
			ins[i]->increase_loc();
		}

		cg->addNode(this);
	}

};


struct MinPoolNode : PoolNode {
public:
	MinPoolNode() : PoolNode(){
	}

public:
	//Be careful that the row is the dim of input vector, and the col is the number of input vectors
	//Another point is that we change the input vectors directly.
	void forward(Graph *cg, const vector<PNode>& x) {
		if (x.size() == 0){
			std::cout << "empty inputs for max pooling" << std::endl;
			return;
		}
		nSize = x.size();
		ins.clear();
		for (int i = 0; i < nSize; i++){
			ins.push_back(x[i]);
		}

		for (int i = 0; i < nSize; ++i){
			if (ins[i]->val.dim != dim){
				std::cout << "input matrixes are not matched" << std::endl;
				clearValue();
				return;
			}
			masks[i].zero();
		}


		for (int idx = 0; idx < dim; idx++){
			int minIndex = -1;
			for (int i = 0; i < nSize; ++i){
				if (minIndex == -1 || ins[i]->val[idx] < ins[minIndex]->val[idx]){
					minIndex = i;
				}
			}
			masks[minIndex][idx] = 1.0;
		}

		val.zero();
		for (int i = 0; i < nSize; ++i){
			val.vec() += masks[i].vec() * ins[i]->val.vec();
		}

		for (int i = 0; i < nSize; ++i){
			ins[i]->increase_loc();
		}

		cg->addNode(this);
	}

};

struct StdPoolNode : PoolNode {
public:
	StdPoolNode() : PoolNode(){
	}

public:
	//Be careful that the row is the dim of input vector, and the col is the number of input vectors
	//Another point is that we change the input vectors directly.
	void forward(Graph *cg, const vector<PNode>& x) {
		if (x.size() == 0){
			std::cout << "empty inputs for std pooling" << std::endl;
			return;
		}

		nSize = x.size();
		ins.clear();
		for (int i = 0; i < nSize; i++){
			ins.push_back(x[i]);
		}

		for (int i = 0; i < nSize; ++i){
			if (ins[i]->val.dim != dim){
				std::cout << "input matrixes are not matched" << std::endl;
				clearValue();
				return;
			}
		}

		val.zero();
		for (int i = 0; i < nSize; ++i){
			val.vec() += ins[i]->val.vec() * ins[i]->val.vec();
		}
		val.vec() = val.vec().sqrt();

		for (int i = 0; i < nSize; ++i){
			masks[i].vec() = ins[i]->val.vec() / val.vec();
		}

		for (int i = 0; i < nSize; ++i){
			ins[i]->increase_loc();
		}

		cg->addNode(this);
	}

};


struct AvgPoolNode : PoolNode {
public:
	AvgPoolNode(){
	}

public:
	//Be careful that the row is the dim of input vector, and the col is the number of input vectors
	//Another point is that we change the input vectors directly.
	void forward(Graph *cg, const vector<PNode>& x) {
		if (x.size() == 0){
			std::cout << "empty inputs for avg pooling" << std::endl;
			return;
		}

		nSize = x.size();
		ins.clear();
		for (int i = 0; i < nSize; i++){
			ins.push_back(x[i]);
		}

		for (int i = 0; i < nSize; ++i){
			if (ins[i]->val.dim != dim){
				std::cout << "input matrixes are not matched" << std::endl;
				clearValue();
				return;
			}
			masks[i] = 1.0 / nSize;
		}

		val.zero();
		for (int i = 0; i < nSize; ++i){
			val.vec() += masks[i].vec() * ins[i]->val.vec();
		}

		for (int i = 0; i < nSize; ++i){
			ins[i]->increase_loc();
		}

		cg->addNode(this);
	}
};



#endif
