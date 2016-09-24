#ifndef BasicGraph
#define BasicGraph

#include "Eigen/Dense"
#include "Node.h"
#include "MyLib.h"

using namespace Eigen;

// one Node means a vector
// the col should be 1, because we aimed for NLP only
struct Graph {

public:
	bool train;
protected:
	vector<PNode> execs; //backward
	//vector<PNode> exports; //backward

public:
	Graph(){
		execs.clear();
		//exports.clear();
	}

public:
	inline void clearValue(const bool& bTrain = false){
		static int count;
		count = execs.size();
		for (int idx = 0; idx < count; idx++){
			execs[idx]->clearValue();
		}
		execs.clear();
		//exports.clear();
		train = bTrain;
	}

	inline void backward(){
		static int count;
		count = execs.size();
		for (int idx = count - 1; idx >= 0; idx--){
			//
			if (execs[idx]->lock > 0){
				//execs[idx]->backward();
				std::cout << "bug exists, please check: " << execs[idx]->sid << " " << execs[idx]->lock << std::endl;
				continue;  // impossible.....
			}
			else if (execs[idx]->lock == 0) {
				if (execs[idx]->loss.size() > 0){
					execs[idx]->backward();
				}
				execs[idx]->unlock();
			}
			else {
				std::cout << "bug exists, please check: impossible neglative lock value" << std::endl;
			}
		}
	}

	inline void addNode(PNode x){
		execs.push_back(x);
		//std::cout << x->sid << std::endl;
	}

	//some nodes are exported for output, define them
	//root nodes not aiming for outputs are not allowed
	//inline void exportNode(PNode x){
		//x->lock++;
		//exports.push_back(x);
	//}

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
