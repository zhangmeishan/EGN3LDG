/*
 * SparseCOP.h
 *
 *  Created on: Jul 20, 2016
 *      Author: mszhang
 */

#ifndef SparseCOP_H_
#define SparseCOP_H_

#include "COPUtils.h"
#include "Alphabet.h"
#include "Node.h"
#include "Graph.h"
#include "SparseParam.h"

struct SparseC1Params {
public:
	SparseParam W;
	int nDim;
	unordered_map<C1Feat, int> hash2id;
	int nVSize;
	int bound;
protected:
	C1Feat feat;
public:
	SparseC1Params() {
		nDim = 0;
		nVSize = 0;
		bound = 0;
		hash2id.clear();
	}

	inline void exportAdaParams(ModelUpdate& ada) {
		if (nVSize > 0)ada.addParam(&W);
	}

	inline void initialWeights() {
		if (nVSize <= 0){
			std::cout << "please check the alphabet" << std::endl;
			return;
		}
		W.initial(nDim, nVSize);
	}

	//random initialization
	inline void initial(int outputDim = 1){
		hash2id.clear();
		bound = 0;
		nDim = outputDim;
	}

	inline int getFeatureId(const int& id, const bool& bTrain){
		if (id < 0){
			return -1;
		}
		feat.setId(id);
		if (hash2id.find(feat) != hash2id.end()){
			return hash2id[feat];
		}
		if (bTrain){
			if (bound < nVSize){
				hash2id[feat] = bound;
				bound++;
			}
			else{ //bound must have an above zero value
				hash2id[feat] = -1; // discard
			}
			return hash2id[feat];
		}
		else{
			return -1;
		}
	}

	inline void collectFeature(const int& id){
		if (id < 0){
			return;
		}
		feat.setId(id);
		if (hash2id.find(feat) == hash2id.end() && bound < maxCapacity){
			hash2id[feat] = bound;
			bound++;
		}
	}

	inline void setFixed(int base){
		nVSize = (bound + 1) * base;
		if (nVSize > maxCapacity) nVSize = maxCapacity;
		initialWeights();
	}

};

// a single node;
// input variables are not allocated by current node(s)
struct SparseC1Node : Node {
public:
	SparseC1Params* param;
	int tx;

public:
	SparseC1Node() : Node() {
		param = NULL;
		tx = -1;
	}

	inline void setParam(SparseC1Params* paramInit) {
		param = paramInit;
	}

	inline void clearValue(){
		Node::clearValue();
		tx = -1;
	}

public:
	inline void forward(Graph* cg, const int& x1){
		//assert(param != NULL);
		static int featId;
		featId = param->getFeatureId(x1, cg->train);
		if (featId < 0){
			tx = -1;
			return;
		}
		tx = featId;
		param->W.value(tx, val);
		cg->addNode(this);
	}

	//no output losses
	void backward() {
		//assert(param != NULL);
		if (tx >= 0){
			param->W.loss(tx, loss);
		}
	}
};



//a sparse feature has two actomic features
struct SparseC2Params {
public:
	SparseParam W;
	int nDim;
	unordered_map<C2Feat, int> hash2id;
	int nVSize;
	int bound;
protected:
	C2Feat feat;
public:
	SparseC2Params() {
		nDim = 0;
		nVSize = 0;
		bound = 0;
		hash2id.clear();
	}

	inline void exportAdaParams(ModelUpdate& ada) {
		if (nVSize > 0)ada.addParam(&W);
	}

	inline void initialWeights() {
		if (nVSize <= 0){
			std::cout << "please check the alphabet" << std::endl;
			return;
		}
		W.initial(nDim, nVSize);
	}

	//random initialization
	inline void initial(int outputDim = 1){
		hash2id.clear();
		bound = 0;
		nDim = outputDim;
	}

	//important!!! if > nHVSize, using LW, otherwise, using HW
	inline int getFeatureId(const int& id1, const int& id2, const bool& bTrain){
		if (id1 < 0 || id2 < 0){
			return -1;
		}
		feat.setId(id1, id2);
		if (hash2id.find(feat) != hash2id.end()){
			return hash2id[feat];
		}
		if (bTrain){
			if (bound < nVSize){
				hash2id[feat] = bound;
				bound++;
			}
			else{ //bound must have an above zero value
				hash2id[feat] = -1; // discard
			}
			return hash2id[feat];
		}
		else{
			return -1;
		}
	}

	inline void collectFeature(const int& id1, const int& id2){
		if (id1 < 0 || id2 < 0){
			return;
		}
		feat.setId(id1, id2);
		if (hash2id.find(feat) == hash2id.end() && bound < maxCapacity){
			hash2id[feat] = bound;
			bound++;
		}
	}

	inline void setFixed(int base){
		nVSize = (bound + 1) * base;
		if (nVSize > maxCapacity) nVSize = maxCapacity;
		initialWeights();
	}
};

// a single node;
// input variables are not allocated by current node(s)
struct SparseC2Node : Node {
public:
	SparseC2Params* param;
	int tx;

public:
	SparseC2Node() : Node() {
		param = NULL;
		tx = -1;
	}

	inline void setParam(SparseC2Params* paramInit) {
		param = paramInit;
	}

	inline void clearValue(){
		Node::clearValue();
		tx = -1;
	}

public:
	inline void forward(Graph* cg, const int& x1, const int& x2) {
		//assert(param != NULL);
		static int featId;
		featId = param->getFeatureId(x1, x2, cg->train);
		if (featId < 0){
			tx = -1;
			return;
		}
		tx = featId;
		param->W.value(tx, val);
		cg->addNode(this);
	}

	void backward() {
		//assert(param != NULL);
		if (tx >= 0){
			param->W.loss(tx, loss);
		}
	}
};


//a sparse feature has three actomic features
struct SparseC3Params {
public:
	SparseParam W;
	int nDim;
	unordered_map<C3Feat, int> hash2id;
	int nVSize;
	int bound;
protected:
	C3Feat feat;
public:
	SparseC3Params() {
		nDim = 0;
		nVSize = 0;
		bound = 0;
		hash2id.clear();
	}

	inline void exportAdaParams(ModelUpdate& ada) {
		if (nVSize > 0)ada.addParam(&W);
	}

	inline void initialWeights() {
		if (nVSize <= 0){
			std::cout << "please check the alphabet" << std::endl;
			return;
		}
		W.initial(nDim, nVSize);
	}

	//random initialization
	inline void initial(int outputDim = 1){
		hash2id.clear();
		bound = 0;
		nDim = outputDim;
	}

	inline int getFeatureId(const int& id1, const int& id2, const int& id3, const bool& bTrain){
		if (id1 < 0 || id2 < 0 || id3 < 0){
			return -1;
		}
		feat.setId(id1, id2, id3);
		if (hash2id.find(feat) != hash2id.end()){
			return hash2id[feat];
		}
		if (bTrain){
			if (bound < nVSize){
				hash2id[feat] = bound;
				bound++;
			}
			else{ //bound must have an above zero value
				hash2id[feat] = -1; // discard
			}
			return hash2id[feat];
		}
		else{
			return -1;
		}
	}

	inline void collectFeature(const int& id1, const int& id2, const int& id3){
		if (id1 < 0 || id2 < 0 || id3 < 0){
			return;
		}
		feat.setId(id1, id2, id3);
		if (hash2id.find(feat) == hash2id.end() && bound < maxCapacity){
			hash2id[feat] = bound;
			bound++;
		}
	}

	inline void setFixed(int base){
		nVSize = (bound + 1) * base;
		if (nVSize > maxCapacity) nVSize = maxCapacity;
		initialWeights();
	}
};

// a single node;
// input variables are not allocated by current node(s)
struct SparseC3Node : Node {
public:
	SparseC3Params* param;
	int tx;

public:
	SparseC3Node() : Node() {
		param = NULL;
		tx = -1;
	}

	inline void setParam(SparseC3Params* paramInit) {
		param = paramInit;
	}

	inline void clearValue(){
		Node::clearValue();
		tx = -1;
	}

public:
	inline void forward(Graph* cg, const int& x1, const int& x2, const int& x3) {
		//assert(param != NULL);
		static int featId;
		featId = param->getFeatureId(x1, x2, x3, cg->train);
		if (featId < 0){
			tx = -1;
			return;
		}
		tx = featId;
		param->W.value(tx, val);
		cg->addNode(this);
	}

	void backward() {
		//assert(param != NULL);
		if (tx >= 0){
			param->W.loss(tx, loss);
		}
	}
};


//a sparse feature has four actomic features
struct SparseC4Params {
public:
	SparseParam W;
	int nDim;
	unordered_map<C4Feat, int> hash2id;
	int nVSize;
	int bound;
protected:
	C4Feat feat;
public:
	SparseC4Params() {
		nDim = 0;
		nVSize = 0;
		bound = 0;
		hash2id.clear();
	}

	inline void exportAdaParams(ModelUpdate& ada) {
		if (nVSize > 0)ada.addParam(&W);
	}

	inline void initialWeights() {
		if (nVSize <= 0){
			std::cout << "please check the alphabet" << std::endl;
			return;
		}
		W.initial(nDim, nVSize);
	}

	//random initialization
	inline void initial(int outputDim = 1){
		hash2id.clear();
		bound = 0;
		nDim = outputDim;
	}

	inline int getFeatureId(const int& id1, const int& id2, const int& id3, const int& id4, const bool& bTrain){
		if (id1 < 0 || id2 < 0 || id3 < 0 || id4 < 0){
			return -1;
		}
		feat.setId(id1, id2, id3, id4);
		if (hash2id.find(feat) != hash2id.end()){
			return hash2id[feat];
		}
		if (bTrain){
			if (bound < nVSize){
				hash2id[feat] = bound;
				bound++;
			}
			else{ //bound must have an above zero value
				hash2id[feat] = -1; // discard
			}
			return hash2id[feat];
		}
		else{
			return -1;
		}
	}

	inline void collectFeature(const int& id1, const int& id2, const int& id3, const int& id4){
		if (id1 < 0 || id2 < 0 || id3 < 0 || id4 < 0){
			return;
		}
		feat.setId(id1, id2, id3, id4);
		if (hash2id.find(feat) == hash2id.end() && bound < maxCapacity){
			hash2id[feat] = bound;
			bound++;
		}
	}

	inline void setFixed(int base){
		nVSize = (bound + 1) * base;
		if (nVSize > maxCapacity) nVSize = maxCapacity;
		initialWeights();
	}


};

// a single node;
// input variables are not allocated by current node(s)
struct SparseC4Node : Node {
public:
	SparseC4Params* param;
	int tx;

public:
	SparseC4Node() : Node() {
		param = NULL;
		tx = -1;
	}

	inline void setParam(SparseC4Params* paramInit) {
		param = paramInit;
	}

	inline void clearValue(){
		Node::clearValue();
		tx = -1;
	}

public:
	inline void forward(Graph* cg, const int& x1, const int& x2, const int& x3, const int& x4) {
		//assert(param != NULL);
		static int featId;
		featId = param->getFeatureId(x1, x2, x3, x4, cg->train);
		if (featId < 0){
			tx = -1;
			return;
		}
		tx = featId;
		param->W.value(tx, val);
		cg->addNode(this);
	}

	void backward() {
		//assert(param != NULL);
		if (tx >= 0){
			param->W.loss(tx, loss);
		}
	}
};


//a sparse feature has five actomic features
struct SparseC5Params {
public:
	SparseParam W;
	int nDim;
	unordered_map<C5Feat, int> hash2id;
	int nVSize;
	int bound;
protected:
	C5Feat feat;
public:
	SparseC5Params() {
		nDim = 0;
		nVSize = 0;
		bound = 0;
		hash2id.clear();
	}

	inline void exportAdaParams(ModelUpdate& ada) {
		if (nVSize > 0)ada.addParam(&W);
	}

	inline void initialWeights() {
		if (nVSize <= 0){
			std::cout << "please check the alphabet" << std::endl;
			return;
		}
		W.initial(nDim, nVSize);
	}

	//random initialization
	inline void initial(int outputDim = 1){
		hash2id.clear();
		bound = 0;
		nDim = outputDim;
	}

	inline int getFeatureId(const int& id1, const int& id2, const int& id3, const int& id4, const int& id5, const bool& bTrain){
		if (id1 < 0 || id2 < 0 || id3 < 0 || id4 < 0 || id5 < 0){
			return -1;
		}
		feat.setId(id1, id2, id3, id4, id5);
		if (hash2id.find(feat) != hash2id.end()){
			return hash2id[feat];
		}
		if (bTrain){
			if (bound < nVSize){
				hash2id[feat] = bound;
				bound++;
			}
			else{ //bound must have an above zero value
				hash2id[feat] = -1; // discard
			}
			return hash2id[feat];
		}
		else{
			return -1;
		}
	}

	inline void collectFeature(const int& id1, const int& id2, const int& id3, const int& id4, const int& id5){
		if (id1 < 0 || id2 < 0 || id3 < 0 || id4 < 0 || id5 < 0){
			return;
		}
		feat.setId(id1, id2, id3, id4, id5);
		if (hash2id.find(feat) == hash2id.end() && bound < maxCapacity){
			hash2id[feat] = bound;
			bound++;
		}
	}

	inline void setFixed(int base){
		nVSize = (bound + 1) * base;
		if (nVSize > maxCapacity) nVSize = maxCapacity;
		initialWeights();
	}
};

// a single node;
// input variables are not allocated by current node(s)
struct SparseC5Node : Node {
public:
	SparseC5Params* param;
	int tx;

public:
	SparseC5Node() : Node() {
		param = NULL;
		tx = -1;
	}

	inline void setParam(SparseC5Params* paramInit) {
		param = paramInit;
	}

	inline void clearValue(){
		Node::clearValue();
		tx = -1;
	}

public:
	inline void forward(Graph* cg, const int& x1, const int& x2, const int& x3, const int& x4, const int& x5) {
		//assert(param != NULL);
		static int featId;
		featId = param->getFeatureId(x1, x2, x3, x4, x5, cg->train);
		if (featId < 0){
			tx = -1;
			return;
		}
		tx = featId;
		param->W.value(tx, val);
		cg->addNode(this);
	}

	void backward() {
		//assert(param != NULL);
		if (tx >= 0){
			param->W.loss(tx, loss);
		}
	}
};


//a sparse feature has six actomic features
struct SparseC6Params {
public:
	SparseParam W;
	int nDim;
	unordered_map<C6Feat, int> hash2id;
	int nVSize;
	int bound;
protected:
	C6Feat feat;
public:
	SparseC6Params() {
		nDim = 0;
		nVSize = 0;
		bound = 0;
		hash2id.clear();
	}

	inline void exportAdaParams(ModelUpdate& ada) {
		if (nVSize > 0)ada.addParam(&W);
	}

	inline void initialWeights() {
		if (nVSize <= 0){
			std::cout << "please check the alphabet" << std::endl;
			return;
		}
		W.initial(nDim, nVSize);
	}

	//random initialization
	inline void initial(int outputDim = 1){
		hash2id.clear();
		bound = 0;
		nDim = outputDim;
	}

	inline int getFeatureId(const int& id1, const int& id2, const int& id3, const int& id4, const int& id5, const int& id6, const bool& bTrain){
		if (id1 < 0 || id2 < 0 || id3 < 0 || id4 < 0 || id5 < 0 || id6 < 0){
			return -1;
		}
		feat.setId(id1, id2, id3, id4, id5, id6);
		if (hash2id.find(feat) != hash2id.end()){
			return hash2id[feat];
		}
		if (bTrain){
			if (bound < nVSize){
				hash2id[feat] = bound;
				bound++;
			}
			else{ //bound must have an above zero value
				hash2id[feat] = -1; // discard
			}
			return hash2id[feat];
		}
		else{
			return -1;
		}
	}

	inline void collectFeature(const int& id1, const int& id2, const int& id3, const int& id4, const int& id5, const int& id6){
		if (id1 < 0 || id2 < 0 || id3 < 0 || id4 < 0 || id5 < 0 || id6 < 0){
			return;
		}
		feat.setId(id1, id2, id3, id4, id5, id6);
		if (hash2id.find(feat) == hash2id.end() && bound < maxCapacity){
			hash2id[feat] = bound;
			bound++;
		}
	}

	inline void setFixed(int base){
		nVSize = (bound + 1) * base;
		if (nVSize > maxCapacity) nVSize = maxCapacity;
		initialWeights();
	}
};

// a single node;
// input variables are not allocated by current node(s)
struct SparseC6Node : Node {
public:
	SparseC6Params* param;
	int tx;

public:
	SparseC6Node() : Node() {
		param = NULL;
		tx = -1;
	}

	inline void setParam(SparseC6Params* paramInit) {
		param = paramInit;
	}

	inline void clearValue(){
		Node::clearValue();
		tx = -1;
	}

public:
	inline void forward(Graph* cg, const int& x1, const int& x2, const int& x3, const int& x4, const int& x5, const int& x6) {
		//assert(param != NULL);
		static int featId;
		featId = param->getFeatureId(x1, x2, x3, x4, x5, x6, cg->train);
		if (featId < 0){
			tx = -1;
			return;
		}
		tx = featId;
		param->W.value(tx, val);
		cg->addNode(this);
	}

	void backward() {
		//assert(param != NULL);
		if (tx >= 0){
			param->W.loss(tx, loss);
		}
	}
};

//a sparse feature has seven actomic features
struct SparseC7Params {
public:
	SparseParam W;
	int nDim;
	unordered_map<C7Feat, int> hash2id;
	int nVSize;
	int bound;
protected:
	C7Feat feat;
public:
	SparseC7Params() {
		nDim = 0;
		nVSize = 0;
		bound = 0;
		hash2id.clear();
	}

	inline void exportAdaParams(ModelUpdate& ada) {
		if (nVSize > 0)ada.addParam(&W);
	}

	inline void initialWeights() {
		if (nVSize <= 0){
			std::cout << "please check the alphabet" << std::endl;
			return;
		}
		W.initial(nDim, nVSize);
	}

	//random initialization
	inline void initial(int outputDim = 1){
		hash2id.clear();
		bound = 0;
		nDim = outputDim;
	}

	inline int getFeatureId(const int& id1, const int& id2, const int& id3, const int& id4, const int& id5, const int& id6, const int& id7, const bool& bTrain){
		if (id1 < 0 || id2 < 0 || id3 < 0 || id4 < 0 || id5 < 0 || id6 < 0 || id7 < 0){
			return -1;
		}
		feat.setId(id1, id2, id3, id4, id5, id6, id7);
		if (hash2id.find(feat) != hash2id.end()){
			return hash2id[feat];
		}
		if (bTrain){
			if (bound < nVSize){
				hash2id[feat] = bound;
				bound++;
			}
			else{ //bound must have an above zero value
				hash2id[feat] = -1; // discard
			}
			return hash2id[feat];
		}
		else{
			return -1;
		}
	}

	inline void collectFeature(const int& id1, const int& id2, const int& id3, const int& id4, const int& id5, const int& id6, const int& id7){
		if (id1 < 0 || id2 < 0 || id3 < 0 || id4 < 0 || id5 < 0 || id6 < 0 || id7 < 0){
			return;
		}
		feat.setId(id1, id2, id3, id4, id5, id6, id7);
		if (hash2id.find(feat) == hash2id.end() && bound < maxCapacity){
			hash2id[feat] = bound;
			bound++;
		}
	}

	inline void setFixed(int base){
		nVSize = (bound + 1) * base;
		if (nVSize > maxCapacity) nVSize = maxCapacity;
		initialWeights();
	}
};

// a single node;
// input variables are not allocated by current node(s)
struct SparseC7Node : Node {
public:
	SparseC7Params* param;
	int tx;

public:
	SparseC7Node() : Node() {
		param = NULL;
		tx = -1;
	}

	inline void setParam(SparseC7Params* paramInit) {
		param = paramInit;
	}

	inline void clearValue(){
		Node::clearValue();
		tx = -1;
	}

public:
	inline void forward(Graph* cg, const int& x1, const int& x2, const int& x3, const int& x4, const int& x5, const int& x6, const int& x7) {
		//assert(param != NULL);
		static int featId;
		featId = param->getFeatureId(x1, x2, x3, x4, x5, x6, x7, cg->train);
		if (featId < 0){
			tx = -1;
			return;
		}
		tx = featId;
		param->W.value(tx, val);
		cg->addNode(this);
	}

	void backward() {
		//assert(param != NULL);
		if (tx >= 0){
			param->W.loss(tx, loss);
		}
	}
};

struct SparseCParams {
public:
	SparseParam W;
	int nDim;
	unordered_map<CFeat, int> hash2id;
	int nVSize;
	int bound;
protected:
	CFeat feat;
public:
	SparseCParams() {
		nDim = 0;
		nVSize = 0;
		bound = 0;
		hash2id.clear();
	}

	inline void exportAdaParams(ModelUpdate& ada) {
		if (nVSize > 0)ada.addParam(&W);
	}

	inline void initialWeights() {
		if (nVSize <= 0){
			std::cout << "please check the alphabet" << std::endl;
			return;
		}
		W.initial(nDim, nVSize);
	}

	//random initialization
	inline void initial(int outputDim = 1){
		hash2id.clear();
		bound = 0;
		nDim = outputDim;
	}

	inline int getFeatureId(const CFeat& f, const bool& bTrain){
		if (!f.valid){
			return -1;
		}
		if (hash2id.find(f) != hash2id.end()){
			return hash2id[f];
		}
		if (bTrain && bound < nVSize){
			hash2id[f] = bound;
			bound++;
			return bound - 1;
		}
		else{
			return -1;
		}
	}

	inline void collectFeature(const CFeat& f){
		if (!f.valid){
			return;
		}
		if (hash2id.find(f) == hash2id.end() && bound < maxCapacity){
			hash2id[f] = bound;
			bound++;
		}
	}

	inline void setFixed(int base){
		nVSize = (bound <= 0) ? 1 : bound * base;
		if (nVSize > maxCapacity){
			nVSize = maxCapacity;
			std::cout << "reach max size" << std::endl;
		}
		initialWeights();
	}

};

// a single node;
// input variables are not allocated by current node(s)
struct SparseCNode : Node {
public:
	SparseCParams* param;
	vector<int> ins;
	int nSize;

public:
	SparseCNode() : Node() {
		param = NULL;
		ins.clear();
		nSize = 0;
	}

	inline void setParam(SparseCParams* paramInit) {
		param = paramInit;
	}

	inline void clearValue(){
		Node::clearValue();
		ins.clear();
		nSize = 0;
	}

public:
	inline void forward(Graph* cg, const vector<CFeat*>& xs){
		//assert(param != NULL);
		static int featId, count;
		count = xs.size();
		for (int idx = 0; idx < count; idx++){
			featId = param->getFeatureId(*(xs[idx]), cg->train);
			if (featId >= 0){
				ins.push_back(featId);
				nSize++;
			}
		}
		param->W.value(ins, val);
		cg->addNode(this);
	}

	//no output losses
	void backward() {
		//assert(param != NULL);
		param->W.loss(ins, loss);
	}
};

#endif /* SparseCOP_H_ */
