#ifndef BasicGraph
#define BasicGraph

#include "Eigen/Dense"
#include "Node.h"
#include "MyLib.h"

using namespace Eigen;

// one Node means a vector
// the col should be 1, because we aimed for NLP only
struct Graph {

protected:
	vector<PNode> execs; //backward

public:
	Graph(){
		execs.clear();
	}

public:
	inline void clearValue(){
		for (int idx = 0; idx < execs.size(); idx++){
			execs[idx]->clearValue();
		}
		execs.clear();
	}

	inline void backward(){
		for (int idx = execs.size() - 1; idx >= 0; idx--){
			execs[idx]->backward();
		}
	}

	inline void addNode(PNode x){
		execs.push_back(x);
	}

public: // virtual functions
	virtual inline void clear(){
		execs.clear();
	}
	//virtual inline void createNodes(...) = 0; // create nodes, as large as possible
	//virtual inline void initial(...) = 0;  // initial params
	//virtual inline void forward(...) = 0; // define graph, and compute

};

// one very useful function to collect pointers of derived nodes
template<typename DerivedNode>
inline vector<PNode> getPNodes(vector<DerivedNode>& inputs, int size){
	int usedSize = inputs.size();
	if (size >= 0 && size < usedSize) usedSize = size;
	vector<PNode> pnodes;
	for (int idx = 0; idx < usedSize; idx++){
		pnodes.push_back(&(inputs[idx]));
	}

	return pnodes;
}

template<typename DerivedNode>
inline vector<PNode> getPNodes(DerivedNode inputs[], int size){
	//int usedSize = inputs.;
	//if (size >= 0 && size < usedSize) usedSize = size;
	int usedSize = size;
	vector<PNode> pnodes;
	for (int idx = 0; idx < usedSize; idx++){
		pnodes.push_back(&(inputs[idx]));
	}

	return pnodes;
}

#endif
