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
	vector<PNode> pnodes;
	vector<PNode> execs; //backward

public:
	Graph(){
		pnodes.clear();
		execs.clear();
	}

public: 
	inline void clearValue(){
		for (int idx = 0; idx < execs.size(); idx++){
			execs[idx]->clearValue();
		}
		pnodes.clear();
		execs.clear();
	}

	inline void backward(){
		for (int idx = execs.size() - 1; idx >= 0; idx--){
			execs[idx]->backward();
		}
	}

public: // virtual functions
	virtual inline void clear(){
		pnodes.clear();
		execs.clear();
	}
	//virtual inline void createNodes(...) = 0; // create nodes, as large as possible
	//virtual inline void initial(...) = 0;  // initial params
	//virtual inline void forward(...) = 0; // define graph, and compute

protected:
	//size is a must variable
	template<typename DerivedNode>
	inline vector<PNode>& getPNodes(vector<DerivedNode>& inputs, int size){
		int usedSize = inputs.size();
		if (size >= 0 && size < usedSize) usedSize = size;
		pnodes.clear();
		for (int idx = 0; idx < usedSize; idx++){
			pnodes.push_back(&(inputs[idx]));
		}

		return pnodes;
	}

};



#endif
