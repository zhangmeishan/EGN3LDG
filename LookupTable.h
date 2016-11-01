#ifndef _LOOKUPTABLE_H_
#define _LOOKUPTABLE_H_

#include "SparseParam.h"
#include "MyLib.h"
#include "Alphabet.h"
#include "Node.h"
#include "Graph.h"

#include <Eigen/Dense>

using namespace Eigen;

struct LookupTable {
public:
	PAlphabet elems;
	SparseParam E;
	bool bFineTune;
	int nDim;
	int nVSize;
	int nUNKId;

public:

	LookupTable() {
		nVSize = 0;
		nDim = 0;
		elems = NULL;
		nUNKId = -1;
		bFineTune = false;
	}

	//random initialization
	inline void initial(PAlphabet alpha, int dim, bool bFineTune = true){
		elems = alpha;
		nVSize = elems->size();
		nUNKId = elems->from_string(unknownkey);
		initialWeights(dim, bFineTune);
	}

	//initialization by pre-trained embeddings
	inline void initial(PAlphabet alpha, const string& inFile, bool bFineTune = true){
		elems = alpha;
		nVSize = elems->size();
		nUNKId = elems->from_string(unknownkey);
		initialWeights(inFile, bFineTune);
	}

	inline void initialWeights(int dim, bool tune) {
		if (nVSize == 0){
			std::cout << "please check the alphabet" << std::endl;
			return;
		}
		nDim = dim;
		E.initial(nDim, nVSize);
		for (int idx = 0; idx < nVSize; idx++){
			norm2one(E.val, idx);
		}

		bFineTune = tune;
	}

	// default should be fineTune, just for initialization
	inline void initialWeights(const string& inFile, bool tune) {
		if (nVSize == 0){
			std::cout << "please check the alphabet" << std::endl;
			return;
		}

		static ifstream inf;
		if (inf.is_open()) {
			inf.close();
			inf.clear();
		}
		inf.open(inFile.c_str());

		static string strLine, curWord;
		static int wordId;

		static vector<string> sLines;
		sLines.clear();
		while (1) {
			if (!my_getline(inf, strLine)) {
				break;
			}
			if (!strLine.empty()){
				sLines.push_back(strLine);
			}
		}
		inf.close();

		//find the first line, decide the wordDim;
		static vector<string> vecInfo;
		split_bychar(sLines[0], vecInfo, ' ');
		nDim = vecInfo.size() - 1;

		E.initial(nDim, nVSize);
		E.val = 0;

		std::cout << "word embedding dim is " << nDim << std::endl;

		bool bHasUnknown = false;
		unordered_set<int> indexers;
		VectorXd sum = VectorXd::Zero(nDim);
		int count = 0;
		for (int idx = 0; idx < sLines.size(); idx++){
			split_bychar(sLines[idx], vecInfo, ' ');
			if (vecInfo.size() != nDim + 1) {
				std::cout << "error embedding file" << std::endl;
			}
			curWord = vecInfo[0];
			//we assume the keys are normalized
			wordId = elems->from_string(curWord);
			if (wordId >= 0) {
				count++;
				if (nUNKId == wordId){
					bHasUnknown = true;
				}
				indexers.insert(wordId);

				for (int idy = 0; idy < nDim; idy++) {
					dtype curValue = atof(vecInfo[idy + 1].c_str());
					sum(idy) += curValue;
					E.val[wordId][idy] += curValue;
				}
			}
		}

		if (nUNKId >= 0 && !bHasUnknown){
			for (int idx = 0; idx < nDim; idx++) {
				E.val[nUNKId][idx] = sum(idx) / count;
			}
			indexers.insert(nUNKId);
			count++;
			std::cout << unknownkey << " not found, using averaged value to initialize." << std::endl;
		}

		int oovWords = 0;
		for (int id = 0; id < nVSize; id++) {
			if (indexers.find(id) == indexers.end()) {
				oovWords++;
				for (int idy = 0; idy < nDim; idy++){
					E.val[id][idy] = nUNKId >= 0 ? E.val[nUNKId][idy] : sum(idy) / count;
				}
			}
		}

		std::cout << "OOV num is " << oovWords << ", total num is " << nVSize << ", embedding oov ratio is " << oovWords * 1.0 / nVSize << std::endl;

		bFineTune = tune;
		if (bFineTune){
			for (int idx = 0; idx < nVSize; idx++){
				norm2one(E.val, idx);
			}
		}
	
	}

	inline void exportAdaParams(ModelUpdate& ada) {
		if (bFineTune) {
			ada.addParam(&E);
		}
	}


	inline int getElemId(const string& strFeat){
		return elems->from_string(strFeat);
	}

};

struct LookupNode : Node {
public:
	LookupTable* param;
	int xid;

public:
	LookupNode() {
		clear();
	}

	inline void setParam(LookupTable* paramInit) {
		param = paramInit;
		dim = param->nDim;
	}

	inline void clear(){
		Node::clear();
		xid = -1;
		param = NULL;
	}

	inline void clearValue(){
		Node::clearValue();
		xid = -1;
	}

public:
	//notice the output
	//this should be leaf nodes
	void forward(Graph *cg, const string& strNorm) {
		assert(param != NULL);
		xid = param->getElemId(strNorm);
		if (xid < 0 && param->nUNKId >= 0){
			xid = param->nUNKId;
		}
		if (xid >= 0){
			param->E.value(xid, val);
		}
		else{
			if (param->bFineTune)std::cout << "Caution: unknown words are not modeled !" << std::endl;
			val = Mat::Zero(dim, 1);
		}

		cg->addNode(this);
	}

	void backward() {
		assert(param != NULL);
		if (xid >= 0 && param->bFineTune){
			param->E.loss(xid, loss);
		}
	}

};

#endif /*_LOOKUPTABLE_H*/
