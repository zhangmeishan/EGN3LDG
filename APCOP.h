/*
 * AP1O.h
 *
 *  Created on: Jul 20, 2016
 *      Author: mszhang
 */

#ifndef APCOP_H_
#define APCOP_H_

#include "COPUtils.h"
#include "Alphabet.h"
#include "Node.h"
#include "Graph.h"
#include "APParam.h"

// aiming for Averaged Perceptron, can not coexist with neural networks
// thus we do not use the BaseParam class
struct APC1Params {
public:	
	APParam HW; // for high frequency features
	APParam LW; // for low frequency features
	PAlphabet elems; //please first sort the actomic features according to their accuracy
	int nHVSize, nLVSize;
	int nDim;

public:
	APC1Params() {
		nHVSize = nLVSize = 0;
		elems = NULL;
		nDim = 0;
	}

	inline void exportAdaParams(ModelUpdate& ada) {
		if (nHVSize > 0)ada.addParam(&HW);
		if (nLVSize > 0)ada.addParam(&LW);
	}

	inline void initialWeights(int nOSize) {
		if (nHVSize == 0 || nLVSize == 0){
			std::cout << "please check the alphabet" << std::endl;
			return;
		}
		nDim = nOSize;
		if (nHVSize > 0) HW.initial(nOSize, nHVSize);
		if (nLVSize > 0) LW.initial(nOSize, nLVSize);
	}

	//random initialization
	inline void initial(PAlphabet alpha, int nOSize, int cutoff, int lowCapacity = 10000){
		elems = alpha;
		nHVSize = elems->aboveThreshold(cutoff);
		nLVSize = elems->size() - nHVSize;
		if (nLVSize > lowCapacity) nLVSize = lowCapacity;
		initialWeights(nOSize);
	}

	//important!!! if > nHVSize, using LW, otherwise, using HW
	inline blong getFeatureId(const string& strFeat){
		blong id = elems->from_string(strFeat);
		if (id >= nHVSize){
			if (nLVSize > 0){
				return nHVSize + ((id - nHVSize) % nLVSize);  //very simple hash strategy
			}
			else{
				return -1;
			}
		}
		if (id < 0){
			return -1;
		}
		return id;
	}

};

// a single node;
// input variables are not allocated by current node(s)
struct APC1Node : Node {
public:
	APC1Params* param;
	int tx;
	int mode;  //-1, invalid; 0, high; 1, low

public:
	APC1Node() {
		clear();
	}

	inline void setParam(APC1Params* paramInit) {
		param = paramInit;
		dim = param->nDim;
	}

	inline void clear(){
		Node::clear();
		tx = -1;
		mode = -1;
		param = NULL;
	}

	inline void clearValue(){
		Node::clearValue();
		tx = -1;
		mode = -1;
	}

public:
	// initialize inputs at the same times
	inline void forward(Graph* cg, const string& x, bool bTrain = false) {
		assert(param != NULL);
		static int featId;
		val = Mat::Zero(dim, 1);
		featId = param->getFeatureId(x);
		if (featId < 0){
			tx = -1;
			mode = -1;
		}
		else if (featId < param->nHVSize){
			tx = featId;
			mode = 0;
			val.col(0) += param->HW.value(tx, bTrain).transpose();
		}
		else{
			tx = featId - param->nHVSize;
			mode = 1;
			val.col(0) += param->LW.value(tx, bTrain).transpose();
		}

		cg->addNode(this);
	}

	//no output losses
	void backward() {
		assert(param != NULL);
		if (mode == 0){
			param->HW.indexers.insert(tx);
			param->HW.grad.row(tx) += loss.col(0).transpose();
		}
		else if (mode == 1){
			param->LW.indexers.insert(tx);
			param->LW.grad.row(tx) += loss.col(0).transpose();
		}
	}

};



//a sparse feature has two actomic features
struct APC2Params {
public:
	APParam HW; // for high frequency features
	APParam LW; // for low frequency features
	PAlphabet elems1, elems2; //please first sort the actomic features according to their accuracy
	int nHVSize, nLVSize;
	int nDim;

private:
	int nHVSize1, nHVSize2;

public:
	APC2Params() {
		nHVSize = nLVSize = 0;
		elems1 = elems2 = NULL;
		nDim = 0;
		nHVSize1 = nHVSize2 = 0;
	}

	inline void exportAdaParams(ModelUpdate& ada) {
		if (nHVSize > 0)ada.addParam(&HW);
		if (nLVSize > 0)ada.addParam(&LW);
	}

	inline void initialWeights(int nOSize) {
		if (nHVSize == 0 || nLVSize == 0){
			std::cout << "please check the alphabet" << std::endl;
			return;
		}
		nDim = nOSize;
		if (nHVSize > 0) HW.initial(nOSize, nHVSize);
		if (nLVSize > 0) LW.initial(nOSize, nLVSize);
	}

	//random initialization
	inline void initial(PAlphabet alpha1, PAlphabet alpha2, int nOSize, int cutoff1, int cutoff2, int lowCapacity = 10000){
		elems1 = alpha1;
		nHVSize1 = elems1->aboveThreshold(cutoff1);

		elems2 = alpha2;
		nHVSize2 = elems2->aboveThreshold(cutoff2);

		nHVSize = multiply(nHVSize1, nHVSize2);
		if (nHVSize < 0){
			std::cout << "APC2Params: too much features, please increase the cutoff value." << std::endl;
			return;
		}

		nLVSize = lowCapacity;
		int nAllSize = multiply(elems1->size(), elems2->size());
		int nRemainSize =  nAllSize - nHVSize;
		if (nLVSize > nRemainSize && nAllSize >= 0) nLVSize = nRemainSize;

		int totalSize = nHVSize + nLVSize;
		if (totalSize < 0){
			std::cout << "APC2Params: too much features, please increase the cutoff value." << std::endl;
			return;
		}

		initialWeights(nOSize);
	}

	//important!!! if > nHVSize, using LW, otherwise, using HW
	inline int getFeatureId(const string& strFeat1, const string& strFeat2){
		static int curIndex;
		int id1 = elems1->from_string(strFeat1);
		int id2 = elems2->from_string(strFeat2);
		if (id1 >= nHVSize1 || id2 >= nHVSize2){
			if (nLVSize > 0){
				curIndex = ((blong)(id1)* (blong)(elems2->size()) + (blong)(id2)-(blong)(nHVSize)) % (blong)(nLVSize);
				if (curIndex < 0) curIndex = curIndex + nLVSize;
				if (curIndex < 0 || curIndex >= nLVSize) std::cout << "APC2Params: impossible mod operation" << std::endl;
				return nHVSize + curIndex;  //very simple hash strategy
			}
			else{
				return -1;
			}
		}
		if (id1 < 0 || id2 < 0){
			return -1;
		}
		return id1 * nHVSize2 + id2;
	}

};

// a single node;
// input variables are not allocated by current node(s)
struct APC2Node : Node {
public:
	APC2Params* param;
	int tx;
	int mode;  //-1, invalid; 0, high; 1, low

public:
	APC2Node() {
		clear();
	}

	inline void setParam(APC2Params* paramInit) {
		param = paramInit;
		dim = param->nDim;
	}

	inline void clear(){
		Node::clear();
		tx = -1;
		mode = -1;
		param = NULL;
	}

	inline void clearValue(){
		Node::clearValue();
		tx = -1;
		mode = -1;
	}

public:
	// initialize inputs at the same times
	inline void forward(Graph* cg, const string& x1, const string& x2, bool bTrain = false) {
		assert(param != NULL);
		static int featId;
		val = Mat::Zero(dim, 1);
		featId = param->getFeatureId(x1, x2);
		if (featId < 0){
			tx = -1;
			mode = -1;
		}
		else if (featId < param->nHVSize){
			tx = featId;
			mode = 0;
			val.col(0) += param->HW.value(tx, bTrain).transpose();
		}
		else{
			tx = featId - param->nHVSize;
			mode = 1;
			val.col(0) += param->LW.value(tx, bTrain).transpose();
		}

		cg->addNode(this);
	}

	//no output losses
	void backward() {
		assert(param != NULL);
		if (mode == 0){
			param->HW.indexers.insert(tx);
			param->HW.grad.row(tx) += loss.col(0).transpose();
		}
		else if (mode == 1){
			param->LW.indexers.insert(tx);
			param->LW.grad.row(tx) += loss.col(0).transpose();
		}
	}

};


//a sparse feature has three actomic features
struct APC3Params {
public:
	APParam HW; // for high frequency features
	APParam LW; // for low frequency features
	PAlphabet elems1, elems2, elems3; //please first sort the actomic features according to their accuracy
	int nHVSize, nLVSize;
	int nDim;

private:
	int nHVSize1, nHVSize2, nHVSize3;

public:
	APC3Params() {
		nHVSize = nLVSize = 0;
		elems1 = elems2 = elems3 = NULL;
		nDim = 0;
		nHVSize1 = nHVSize2 = nHVSize3 = 0;
	}

	inline void exportAdaParams(ModelUpdate& ada) {
		if (nHVSize > 0)ada.addParam(&HW);
		if (nLVSize > 0)ada.addParam(&LW);
	}

	inline void initialWeights(int nOSize) {
		if (nHVSize == 0 || nLVSize == 0){
			std::cout << "please check the alphabet" << std::endl;
			return;
		}
		nDim = nOSize;
		if (nHVSize > 0) HW.initial(nOSize, nHVSize);
		if (nLVSize > 0) LW.initial(nOSize, nLVSize);
	}

	//random initialization
	inline void initial(PAlphabet alpha1, PAlphabet alpha2, PAlphabet alpha3, int nOSize, int cutoff1, int cutoff2, int cutoff3, int lowCapacity = 1000){
		elems1 = alpha1;
		nHVSize1 = elems1->aboveThreshold(cutoff1);

		elems2 = alpha2;
		nHVSize2 = elems2->aboveThreshold(cutoff2);

		elems3 = alpha3;
		nHVSize3 = elems3->aboveThreshold(cutoff3);

		nHVSize = multiply(nHVSize1, nHVSize2, nHVSize3);
		if (nHVSize < 0){
			std::cout << "APC3Params: too much features, please increase the cutoff value." << std::endl;
			return;
		}

		nLVSize = lowCapacity;
		int nAllSize = multiply(elems1->size(), elems2->size(), elems3->size());
		int nRemainSize =  nAllSize - nHVSize;
		if (nLVSize > nRemainSize && nAllSize >= 0) nLVSize = nRemainSize;

		int totalSize = nHVSize + nLVSize;
		if (totalSize < 0){
			std::cout << "APC3Params: too much features, please increase the cutoff value." << std::endl;
			return;
		}

		initialWeights(nOSize);
	}

	//important!!! if > nHVSize, using LW, otherwise, using HW
	inline int getFeatureId(const string& strFeat1, const string& strFeat2, const string& strFeat3){
		static int curIndex;
		int id1 = elems1->from_string(strFeat1);
		int id2 = elems2->from_string(strFeat2);
		int id3 = elems3->from_string(strFeat3);
		if (id1 >= nHVSize1 || id2 >= nHVSize2 || id3 >= nHVSize3){
			if (nLVSize > 0){
				curIndex = ((blong)(id1)* (blong)(elems2->size()) + (blong)(id2)) % (blong)(nLVSize);
				curIndex = ((blong)(curIndex)* (blong)(elems3->size()) + (blong)(id3) -(blong)(nHVSize)) % (blong)(nLVSize);
				if (curIndex < 0) curIndex = curIndex + nLVSize;
				if (curIndex < 0 || curIndex >= nLVSize) std::cout << "APC3Params: impossible mod operation" << std::endl;
				return nHVSize + curIndex;  //very simple hash strategy
			}
			else{
				return -1;
			}
		}
		if (id1 < 0 || id2 < 0 || id3 < 0){
			return -1;
		}
		curIndex = id1 * nHVSize2 + id2;
		curIndex = curIndex * nHVSize3 + id3;
		return curIndex;
	}

};

// a single node;
// input variables are not allocated by current node(s)
struct APC3Node : Node {
public:
	APC3Params* param;
	int tx;
	int mode;  //-1, invalid; 0, high; 1, low

public:
	APC3Node() {
		clear();
	}

	inline void setParam(APC3Params* paramInit) {
		param = paramInit;
		dim = param->nDim;
	}

	inline void clear(){
		Node::clear();
		tx = -1;
		mode = -1;
		param = NULL;
	}

	inline void clearValue(){
		Node::clearValue();
		tx = -1;
		mode = -1;
	}

public:
	// initialize inputs at the same times
	inline void forward(Graph* cg, const string& x1, const string& x2, const string& x3, bool bTrain = false) {
		assert(param != NULL);
		static int featId;
		val = Mat::Zero(dim, 1);
		featId = param->getFeatureId(x1, x2, x3);
		if (featId < 0){
			tx = -1;
			mode = -1;
		}
		else if (featId < param->nHVSize){
			tx = featId;
			mode = 0;
			val.col(0) += param->HW.value(tx, bTrain).transpose();
		}
		else{
			tx = featId - param->nHVSize;
			mode = 1;
			val.col(0) += param->LW.value(tx, bTrain).transpose();
		}

		cg->addNode(this);
	}

	//no output losses
	void backward() {
		assert(param != NULL);
		if (mode == 0){
			param->HW.indexers.insert(tx);
			param->HW.grad.row(tx) += loss.col(0).transpose();
		}
		else if (mode == 1){
			param->LW.indexers.insert(tx);
			param->LW.grad.row(tx) += loss.col(0).transpose();
		}
	}

};


//a sparse feature has four actomic features
struct APC4Params {
public:
	APParam HW; // for high frequency features
	APParam LW; // for low frequency features
	PAlphabet elems1, elems2, elems3, elems4; //please first sort the actomic features according to their accuracy
	int nHVSize, nLVSize;
	int nDim;

private:
	int nHVSize1, nHVSize2, nHVSize3, nHVSize4;

public:
	APC4Params() {
		nHVSize = nLVSize = 0;
		elems1 = elems2 = elems3 = elems4 = NULL;
		nDim = 0;
		nHVSize1 = nHVSize2 = nHVSize3 = nHVSize4 = 0;
	}

	inline void exportAdaParams(ModelUpdate& ada) {
		if (nHVSize > 0)ada.addParam(&HW);
		if (nLVSize > 0)ada.addParam(&LW);
	}

	inline void initialWeights(int nOSize) {
		if (nHVSize == 0 || nLVSize == 0){
			std::cout << "please check the alphabet" << std::endl;
			return;
		}
		nDim = nOSize;
		if (nHVSize > 0) HW.initial(nOSize, nHVSize);
		if (nLVSize > 0) LW.initial(nOSize, nLVSize);
	}

	//random initialization
	inline void initial(PAlphabet alpha1, PAlphabet alpha2, PAlphabet alpha3, PAlphabet alpha4, int nOSize, int cutoff1, int cutoff2, int cutoff3, int cutoff4, int lowCapacity = 1000){
		elems1 = alpha1;
		nHVSize1 = elems1->aboveThreshold(cutoff1);

		elems2 = alpha2;
		nHVSize2 = elems2->aboveThreshold(cutoff2);

		elems3 = alpha3;
		nHVSize3 = elems3->aboveThreshold(cutoff3);

		elems4 = alpha4;
		nHVSize4 = elems4->aboveThreshold(cutoff4);

		nHVSize = multiply(nHVSize1, nHVSize2, nHVSize3, nHVSize4);
		if (nHVSize < 0){
			std::cout << "APC4Params: too much features, please increase the cutoff value." << std::endl;
			return;
		}

		nLVSize = lowCapacity;
		int nAllSize = multiply(elems1->size(), elems2->size(), elems3->size(), elems4->size());
		int nRemainSize =  nAllSize - nHVSize;
		if (nLVSize > nRemainSize && nAllSize >= 0) nLVSize = nRemainSize;

		int totalSize = nHVSize + nLVSize;
		if (totalSize < 0){
			std::cout << "APC4Params: too much features, please increase the cutoff value." << std::endl;
			return;
		}

		initialWeights(nOSize);
	}

	//important!!! if > nHVSize, using LW, otherwise, using HW
	inline int getFeatureId(const string& strFeat1, const string& strFeat2, const string& strFeat3, const string& strFeat4){
		static int curIndex;
		int id1 = elems1->from_string(strFeat1);
		int id2 = elems2->from_string(strFeat2);
		int id3 = elems3->from_string(strFeat3);
		int id4 = elems4->from_string(strFeat4);
		if (id1 >= nHVSize1 || id2 >= nHVSize2 || id3 >= nHVSize3 || id4 >= nHVSize4){
			if (nLVSize > 0){
				curIndex = ((blong)(id1)* (blong)(elems2->size()) + (blong)(id2)) % (blong)(nLVSize);
				curIndex = ((blong)(curIndex)* (blong)(elems3->size()) + (blong)(id3)) % (blong)(nLVSize);
				curIndex = ((blong)(curIndex)* (blong)(elems4->size()) + (blong)(id4)-(blong)(nHVSize)) % (blong)(nLVSize);
				if (curIndex < 0) curIndex = curIndex + nLVSize;
				if (curIndex < 0 || curIndex >= nLVSize) std::cout << "APC4Params: impossible mod operation" << std::endl;
				return nHVSize + curIndex;  //very simple hash strategy
			}
			else{
				return -1;
			}
		}
		if (id1 < 0 || id2 < 0 || id3 < 0 || id4 < 0){
			return -1;
		}
		curIndex = id1 * nHVSize2 + id2;
		curIndex = curIndex * nHVSize3 + id3;
		curIndex = curIndex * nHVSize4 + id4;
		return curIndex;
	}

};

// a single node;
// input variables are not allocated by current node(s)
struct APC4Node : Node {
public:
	APC4Params* param;
	int tx;
	int mode;  //-1, invalid; 0, high; 1, low

public:
	APC4Node() {
		clear();
	}

	inline void setParam(APC4Params* paramInit) {
		param = paramInit;
		dim = param->nDim;
	}

	inline void clear(){
		Node::clear();
		tx = -1;
		mode = -1;
		param = NULL;
	}

	inline void clearValue(){
		Node::clearValue();
		tx = -1;
		mode = -1;
	}

public:
	// initialize inputs at the same times
	inline void forward(Graph* cg, const string& x1, const string& x2, const string& x3, const string& x4, bool bTrain = false) {
		assert(param != NULL);
		static int featId;
		val = Mat::Zero(dim, 1);
		featId = param->getFeatureId(x1, x2, x3, x4);
		if (featId < 0){
			tx = -1;
			mode = -1;
		}
		else if (featId < param->nHVSize){
			tx = featId;
			mode = 0;
			val.col(0) += param->HW.value(tx, bTrain).transpose();
		}
		else{
			tx = featId - param->nHVSize;
			mode = 1;
			val.col(0) += param->LW.value(tx, bTrain).transpose();
		}

		cg->addNode(this);
	}

	//no output losses
	void backward() {
		assert(param != NULL);
		if (mode == 0){
			param->HW.indexers.insert(tx);
			param->HW.grad.row(tx) += loss.col(0).transpose();
		}
		else if (mode == 1){
			param->LW.indexers.insert(tx);
			param->LW.grad.row(tx) += loss.col(0).transpose();
		}
	}

};


//a sparse feature has five actomic features
struct APC5Params {
public:
	APParam HW; // for high frequency features
	APParam LW; // for low frequency features
	PAlphabet elems1, elems2, elems3, elems4, elems5; //please first sort the actomic features according to their accuracy
	int nHVSize, nLVSize;
	int nDim;

private:
	int nHVSize1, nHVSize2, nHVSize3, nHVSize4, nHVSize5;

public:
	APC5Params() {
		nHVSize = nLVSize = 0;
		elems1 = elems2 = elems3 = elems4 = elems5 = NULL;
		nDim = 0;
		nHVSize1 = nHVSize2 = nHVSize3 = nHVSize4 = nHVSize5 = 0;
	}

	inline void exportAdaParams(ModelUpdate& ada) {
		if (nHVSize > 0)ada.addParam(&HW);
		if (nLVSize > 0)ada.addParam(&LW);
	}

	inline void initialWeights(int nOSize) {
		if (nHVSize == 0 || nLVSize == 0){
			std::cout << "please check the alphabet" << std::endl;
			return;
		}
		nDim = nOSize;
		if (nHVSize > 0) HW.initial(nOSize, nHVSize);
		if (nLVSize > 0) LW.initial(nOSize, nLVSize);
	}

	//random initialization
	inline void initial(PAlphabet alpha1, PAlphabet alpha2, PAlphabet alpha3, PAlphabet alpha4, PAlphabet alpha5,
		int nOSize, int cutoff1, int cutoff2, int cutoff3, int cutoff4, int cutoff5, int lowCapacity = 1000){
		elems1 = alpha1;
		nHVSize1 = elems1->aboveThreshold(cutoff1);

		elems2 = alpha2;
		nHVSize2 = elems2->aboveThreshold(cutoff2);

		elems3 = alpha3;
		nHVSize3 = elems3->aboveThreshold(cutoff3);

		elems4 = alpha4;
		nHVSize4 = elems4->aboveThreshold(cutoff4);

		elems5 = alpha5;
		nHVSize5 = elems5->aboveThreshold(cutoff5);

		nHVSize = multiply(nHVSize1, nHVSize2, nHVSize3, nHVSize4, nHVSize5);
		if (nHVSize < 0){
			std::cout << "APC5Params: too much features, please increase the cutoff value." << std::endl;
			return;
		}

		nLVSize = lowCapacity;
		int nAllSize = multiply(elems1->size(), elems2->size(), elems3->size(), elems4->size(), elems5->size());
		int nRemainSize =  nAllSize - nHVSize;
		if (nLVSize > nRemainSize && nAllSize >= 0) nLVSize = nRemainSize;

		int totalSize = nHVSize + nLVSize;
		if (totalSize < 0){
			std::cout << "APC5Params: too much features, please increase the cutoff value." << std::endl;
			return;
		}

		initialWeights(nOSize);
	}

	//important!!! if > nHVSize, using LW, otherwise, using HW
	inline int getFeatureId(const string& strFeat1, const string& strFeat2, const string& strFeat3, const string& strFeat4, const string& strFeat5){
		static int curIndex;
		int id1 = elems1->from_string(strFeat1);
		int id2 = elems2->from_string(strFeat2);
		int id3 = elems3->from_string(strFeat3);
		int id4 = elems4->from_string(strFeat4);
		int id5 = elems5->from_string(strFeat5);
		if (id1 >= nHVSize1 || id2 >= nHVSize2 || id3 >= nHVSize3 || id4 >= nHVSize4 || id5 >= nHVSize5){
			if (nLVSize > 0){
				curIndex = ((blong)(id1)* (blong)(elems2->size()) + (blong)(id2)) % (blong)(nLVSize);
				curIndex = ((blong)(curIndex)* (blong)(elems3->size()) + (blong)(id3)) % (blong)(nLVSize);
				curIndex = ((blong)(curIndex)* (blong)(elems4->size()) + (blong)(id4)) % (blong)(nLVSize);
				curIndex = ((blong)(curIndex)* (blong)(elems5->size()) + (blong)(id5)-(blong)(nHVSize)) % (blong)(nLVSize);
				if (curIndex < 0) curIndex = curIndex + nLVSize;
				if (curIndex < 0 || curIndex >= nLVSize) std::cout << "APC5Params: impossible mod operation" << std::endl;
				return nHVSize + curIndex;  //very simple hash strategy
			}
			else{
				return -1;
			}
		}
		if (id1 < 0 || id2 < 0 || id3 < 0 || id4 < 0 || id5 < 0){
			return -1;
		}
		curIndex = id1 * nHVSize2 + id2;
		curIndex = curIndex * nHVSize3 + id3;
		curIndex = curIndex * nHVSize4 + id4;
		curIndex = curIndex * nHVSize5 + id5;
		return curIndex;
	}

};

// a single node;
// input variables are not allocated by current node(s)
struct APC5Node : Node {
public:
	APC5Params* param;
	int tx;
	int mode;  //-1, invalid; 0, high; 1, low

public:
	APC5Node() {
		clear();
	}

	inline void setParam(APC5Params* paramInit) {
		param = paramInit;
		dim = param->nDim;
	}

	inline void clear(){
		Node::clear();
		tx = -1;
		mode = -1;
		param = NULL;
	}

	inline void clearValue(){
		Node::clearValue();
		tx = -1;
		mode = -1;
	}

public:
	// initialize inputs at the same times
	inline void forward(Graph* cg, const string& x1, const string& x2, const string& x3, const string& x4, const string& x5, bool bTrain = false) {
		assert(param != NULL);
		static int featId;
		val = Mat::Zero(dim, 1);
		featId = param->getFeatureId(x1, x2, x3, x4, x5);
		if (featId < 0){
			tx = -1;
			mode = -1;
		}
		else if (featId < param->nHVSize){
			tx = featId;
			mode = 0;
			val.col(0) += param->HW.value(tx, bTrain).transpose();
		}
		else{
			tx = featId - param->nHVSize;
			mode = 1;
			val.col(0) += param->LW.value(tx, bTrain).transpose();
		}

		cg->addNode(this);
	}

	//no output losses
	void backward() {
		assert(param != NULL);
		if (mode == 0){
			param->HW.indexers.insert(tx);
			param->HW.grad.row(tx) += loss.col(0).transpose();
		}
		else if (mode == 1){
			param->LW.indexers.insert(tx);
			param->LW.grad.row(tx) += loss.col(0).transpose();
		}
	}

};


//a sparse feature has six actomic features
struct APC6Params {
public:
	APParam HW; // for high frequency features
	APParam LW; // for low frequency features
	PAlphabet elems1, elems2, elems3, elems4, elems5, elems6; //please first sort the actomic features according to their accuracy
	int nHVSize, nLVSize;
	int nDim;

private:
	int nHVSize1, nHVSize2, nHVSize3, nHVSize4, nHVSize5, nHVSize6;

public:
	APC6Params() {
		nHVSize = nLVSize = 0;
		elems1 = elems2 = elems3 = elems4 = elems5 = elems6 = NULL;
		nDim = 0;
		nHVSize1 = nHVSize2 = nHVSize3 = nHVSize4 = nHVSize5 = nHVSize6 = 0;
	}

	inline void exportAdaParams(ModelUpdate& ada) {
		if (nHVSize > 0)ada.addParam(&HW);
		if (nLVSize > 0)ada.addParam(&LW);
	}

	inline void initialWeights(int nOSize) {
		if (nHVSize == 0 || nLVSize == 0){
			std::cout << "please check the alphabet" << std::endl;
			return;
		}
		nDim = nOSize;
		if (nHVSize > 0) HW.initial(nOSize, nHVSize);
		if (nLVSize > 0) LW.initial(nOSize, nLVSize);
	}

	//random initialization
	inline void initial(PAlphabet alpha1, PAlphabet alpha2, PAlphabet alpha3, PAlphabet alpha4, PAlphabet alpha5, PAlphabet alpha6,
		int nOSize, int cutoff1, int cutoff2, int cutoff3, int cutoff4, int cutoff5, int cutoff6, int lowCapacity = 1000){
		elems1 = alpha1;
		nHVSize1 = elems1->aboveThreshold(cutoff1);

		elems2 = alpha2;
		nHVSize2 = elems2->aboveThreshold(cutoff2);

		elems3 = alpha3;
		nHVSize3 = elems3->aboveThreshold(cutoff3);

		elems4 = alpha4;
		nHVSize4 = elems4->aboveThreshold(cutoff4);

		elems5 = alpha5;
		nHVSize5 = elems5->aboveThreshold(cutoff5);

		elems6 = alpha6;
		nHVSize6 = elems6->aboveThreshold(cutoff6);

		nHVSize = multiply(nHVSize1, nHVSize2, nHVSize3, nHVSize4, nHVSize5, nHVSize6);
		if (nHVSize < 0){
			std::cout << "APC6Params: too much features, please increase the cutoff value." << std::endl;
			return;
		}

		nLVSize = lowCapacity;
		int nAllSize = multiply(elems1->size(), elems2->size(), elems3->size(), elems4->size(), elems5->size(), elems6->size());
		int nRemainSize =  nAllSize - nHVSize;
		if (nLVSize > nRemainSize && nAllSize >= 0) nLVSize = nRemainSize;

		int totalSize = nHVSize + nLVSize;
		if (totalSize < 0){
			std::cout << "APC6Params: too much features, please increase the cutoff value." << std::endl;
			return;
		}

		initialWeights(nOSize);
	}

	//important!!! if > nHVSize, using LW, otherwise, using HW
	inline int getFeatureId(const string& strFeat1, const string& strFeat2, const string& strFeat3, const string& strFeat4, const string& strFeat5, const string& strFeat6){
		static int curIndex;
		int id1 = elems1->from_string(strFeat1);
		int id2 = elems2->from_string(strFeat2);
		int id3 = elems3->from_string(strFeat3);
		int id4 = elems4->from_string(strFeat4);
		int id5 = elems5->from_string(strFeat5);
		int id6 = elems6->from_string(strFeat6);
		if (id1 >= nHVSize1 || id2 >= nHVSize2 || id3 >= nHVSize3 || id4 >= nHVSize4 || id5 >= nHVSize5 || id6 >= nHVSize6){
			if (nLVSize > 0){
				curIndex = ((blong)(id1)* (blong)(elems2->size()) + (blong)(id2)) % (blong)(nLVSize);
				curIndex = ((blong)(curIndex)* (blong)(elems3->size()) + (blong)(id3)) % (blong)(nLVSize);
				curIndex = ((blong)(curIndex)* (blong)(elems4->size()) + (blong)(id4)) % (blong)(nLVSize);
				curIndex = ((blong)(curIndex)* (blong)(elems5->size()) + (blong)(id5)) % (blong)(nLVSize);
				curIndex = ((blong)(curIndex)* (blong)(elems6->size()) + (blong)(id6)-(blong)(nHVSize)) % (blong)(nLVSize);
				if (curIndex < 0) curIndex = curIndex + nLVSize;
				if (curIndex < 0 || curIndex >= nLVSize) std::cout << "APC6Params: impossible mod operation" << std::endl;
				return nHVSize + curIndex;  //very simple hash strategy
			}
			else{
				return -1;
			}
		}
		if (id1 < 0 || id2 < 0 || id3 < 0 || id4 < 0 || id5 < 0 || id6 < 0){
			return -1;
		}
		curIndex = id1 * nHVSize2 + id2;
		curIndex = curIndex * nHVSize3 + id3;
		curIndex = curIndex * nHVSize4 + id4;
		curIndex = curIndex * nHVSize5 + id5;
		curIndex = curIndex * nHVSize6 + id6;
		return curIndex;
	}

};

// a single node;
// input variables are not allocated by current node(s)
struct APC6Node : Node {
public:
	APC6Params* param;
	int tx;
	int mode;  //-1, invalid; 0, high; 1, low

public:
	APC6Node() {
		clear();
	}

	inline void setParam(APC6Params* paramInit) {
		param = paramInit;
		dim = param->nDim;
	}

	inline void clear(){
		Node::clear();
		tx = -1;
		mode = -1;
		param = NULL;
	}

	inline void clearValue(){
		Node::clearValue();
		tx = -1;
		mode = -1;
	}

public:
	// initialize inputs at the same times
	inline void forward(Graph* cg, const string& x1, const string& x2, const string& x3, const string& x4, const string& x5, const string& x6, bool bTrain = false) {
		assert(param != NULL);
		static int featId;
		val = Mat::Zero(dim, 1);
		featId = param->getFeatureId(x1, x2, x3, x4, x5, x6);
		if (featId < 0){
			tx = -1;
			mode = -1;
		}
		else if (featId < param->nHVSize){
			tx = featId;
			mode = 0;
			val.col(0) += param->HW.value(tx, bTrain).transpose();
		}
		else{
			tx = featId - param->nHVSize;
			mode = 1;
			val.col(0) += param->LW.value(tx, bTrain).transpose();
		}

		cg->addNode(this);
	}

	//no output losses
	void backward() {
		assert(param != NULL);
		if (mode == 0){
			param->HW.indexers.insert(tx);
			param->HW.grad.row(tx) += loss.col(0).transpose();
		}
		else if (mode == 1){
			param->LW.indexers.insert(tx);
			param->LW.grad.row(tx) += loss.col(0).transpose();
		}
	}

};

//a sparse feature has seven actomic features
struct APC7Params {
public:
	APParam HW; // for high frequency features
	APParam LW; // for low frequency features
	PAlphabet elems1, elems2, elems3, elems4, elems5, elems6, elems7; //please first sort the actomic features according to their accuracy
	int nHVSize, nLVSize;
	int nDim;

private:
	int nHVSize1, nHVSize2, nHVSize3, nHVSize4, nHVSize5, nHVSize6, nHVSize7;

public:
	APC7Params() {
		nHVSize = nLVSize = 0;
		elems1 = elems2 = elems3 = elems4 = elems5 = elems6 = elems7 = NULL;
		nDim = 0;
		nHVSize1 = nHVSize2 = nHVSize3 = nHVSize4 = nHVSize5 = nHVSize6 = nHVSize7 = 0;
	}

	inline void exportAdaParams(ModelUpdate& ada) {
		if (nHVSize > 0)ada.addParam(&HW);
		if (nLVSize > 0)ada.addParam(&LW);
	}

	inline void initialWeights(int nOSize) {
		if (nHVSize == 0 || nLVSize == 0){
			std::cout << "please check the alphabet" << std::endl;
			return;
		}
		nDim = nOSize;
		if (nHVSize > 0) HW.initial(nOSize, nHVSize);
		if (nLVSize > 0) LW.initial(nOSize, nLVSize);
	}

	//random initialization
	inline void initial(PAlphabet alpha1, PAlphabet alpha2, PAlphabet alpha3, PAlphabet alpha4, PAlphabet alpha5, PAlphabet alpha6, PAlphabet alpha7,
		int nOSize, int cutoff1, int cutoff2, int cutoff3, int cutoff4, int cutoff5, int cutoff6, int cutoff7, int lowCapacity = 1000){
		elems1 = alpha1;
		nHVSize1 = elems1->aboveThreshold(cutoff1);

		elems2 = alpha2;
		nHVSize2 = elems2->aboveThreshold(cutoff2);

		elems3 = alpha3;
		nHVSize3 = elems3->aboveThreshold(cutoff3);

		elems4 = alpha4;
		nHVSize4 = elems4->aboveThreshold(cutoff4);

		elems5 = alpha5;
		nHVSize5 = elems5->aboveThreshold(cutoff5);

		elems6 = alpha6;
		nHVSize6 = elems6->aboveThreshold(cutoff6);

		elems7 = alpha7;
		nHVSize7 = elems7->aboveThreshold(cutoff7);

		nHVSize = multiply(nHVSize1, nHVSize2, nHVSize3, nHVSize4, nHVSize5, nHVSize6, nHVSize7);
		if (nHVSize < 0){
			std::cout << "APC7Params: too much features, please increase the cutoff value." << std::endl;
			return;
		}

		nLVSize = lowCapacity;
		int nAllSize = multiply(elems1->size(), elems2->size(), elems3->size(), elems4->size(), elems5->size(), elems6->size(), elems7->size());
		int nRemainSize =  nAllSize - nHVSize;
		if (nLVSize > nRemainSize && nAllSize >= 0) nLVSize = nRemainSize;

		int totalSize = nHVSize + nLVSize;
		if (totalSize < 0){
			std::cout << "APC7Params: too much features, please increase the cutoff value." << std::endl;
			return;
		}

		initialWeights(nOSize);
	}

	//important!!! if > nHVSize, using LW, otherwise, using HW
	inline int getFeatureId(const string& strFeat1, const string& strFeat2, const string& strFeat3, const string& strFeat4, const string& strFeat5, const string& strFeat6, const string& strFeat7){
		static int curIndex;
		int id1 = elems1->from_string(strFeat1);
		int id2 = elems2->from_string(strFeat2);
		int id3 = elems3->from_string(strFeat3);
		int id4 = elems4->from_string(strFeat4);
		int id5 = elems5->from_string(strFeat5);
		int id6 = elems6->from_string(strFeat6);
		int id7 = elems7->from_string(strFeat7);
		if (id1 >= nHVSize1 || id2 >= nHVSize2 || id3 >= nHVSize3 || id4 >= nHVSize4 || id5 >= nHVSize5 || id6 >= nHVSize6 || id7 >= nHVSize7){
			if (nLVSize > 0){
				curIndex = ((blong)(id1)* (blong)(elems2->size()) + (blong)(id2)) % (blong)(nLVSize);
				curIndex = ((blong)(curIndex)* (blong)(elems3->size()) + (blong)(id3)) % (blong)(nLVSize);
				curIndex = ((blong)(curIndex)* (blong)(elems4->size()) + (blong)(id4)) % (blong)(nLVSize);
				curIndex = ((blong)(curIndex)* (blong)(elems5->size()) + (blong)(id5)) % (blong)(nLVSize);
				curIndex = ((blong)(curIndex)* (blong)(elems6->size()) + (blong)(id6)) % (blong)(nLVSize);
				curIndex = ((blong)(curIndex)* (blong)(elems7->size()) + (blong)(id7)-(blong)(nHVSize)) % (blong)(nLVSize);
				if (curIndex < 0) curIndex = curIndex + nLVSize;
				if (curIndex < 0 || curIndex >= nLVSize) std::cout << "APC7Params: impossible mod operation" << std::endl;
				return nHVSize + curIndex;  //very simple hash strategy
			}
			else{
				return -1;
			}
		}
		if (id1 < 0 || id2 < 0 || id3 < 0 || id4 < 0 || id5 < 0 || id6 < 0 || id7 < 0){
			return -1;
		}
		curIndex = id1 * nHVSize2 + id2;
		curIndex = curIndex * nHVSize3 + id3;
		curIndex = curIndex * nHVSize4 + id4;
		curIndex = curIndex * nHVSize5 + id5;
		curIndex = curIndex * nHVSize6 + id6;
		curIndex = curIndex * nHVSize7 + id7;
		return curIndex;
	}

};

// a single node;
// input variables are not allocated by current node(s)
struct APC7Node : Node {
public:
	APC7Params* param;
	int tx;
	int mode;  //-1, invalid; 0, high; 1, low

public:
	APC7Node() {
		clear();
	}

	inline void setParam(APC7Params* paramInit) {
		param = paramInit;
		dim = param->nDim;
	}

	inline void clear(){
		Node::clear();
		tx = -1;
		mode = -1;
		param = NULL;
	}

	inline void clearValue(){
		Node::clearValue();
		tx = -1;
		mode = -1;
	}

public:
	// initialize inputs at the same times
	inline void forward(Graph* cg, const string& x1, const string& x2, const string& x3, const string& x4, const string& x5, const string& x6, const string& x7, bool bTrain = false) {
		assert(param != NULL);
		static int featId;
		val = Mat::Zero(dim, 1);
		featId = param->getFeatureId(x1, x2, x3, x4, x5, x6, x7);
		if (featId < 0){
			tx = -1;
			mode = -1;
		}
		else if (featId < param->nHVSize){
			tx = featId;
			mode = 0;
			val.col(0) += param->HW.value(tx, bTrain).transpose();
		}
		else{
			tx = featId - param->nHVSize;
			mode = 1;
			val.col(0) += param->LW.value(tx, bTrain).transpose();
		}

		cg->addNode(this);
	}

	//no output losses
	void backward() {
		assert(param != NULL);
		if (mode == 0){
			param->HW.indexers.insert(tx);
			param->HW.grad.row(tx) += loss.col(0).transpose();
		}
		else if (mode == 1){
			param->LW.indexers.insert(tx);
			param->LW.grad.row(tx) += loss.col(0).transpose();
		}
	}

};

#endif /* APCOP_H_ */
