#ifndef _LOOKUPTABLE_H_
#define _LOOKUPTABLE_H_

#include "SparseParam.h"
#include "MyLib.h"
#include "Alphabet.h"

#include <Eigen/Dense>

using namespace Eigen;

struct LookupTable {
public:
	Alphabet _elems;
	SparseParam _E;
	bool _bFineTune;
	int _nDim;
	int _nVSize;
	int _nUNKId;

public:

	LookupTable() {
	}

	//random initialization
	inline void initial(const hash_map<string, int>& elem_stat, int cutOff, int nDim, int seed, bool bFineTune){
		initialAlpha(elem_stat, cutOff);
		initialWeights(nDim, seed, bFineTune);
	}

	//initialization by pre-trained embeddings
	inline void initial(const hash_map<string, int>& elem_stat, int cutOff, const string& inFile, bool bFineTune){
		initialAlpha(elem_stat, cutOff);
		initialWeights(inFile, bFineTune);
	}

	// for sepcial elements such as UNK and NULL, please add insert them into the elem_stat
	// I will not implement another addAlpha function, thus please collect alpha all at once
	inline void initialAlpha(const hash_map<string, int>& elem_stat, int cutOff = 0){
		_elems.clear();

		static hash_map<string, int>::const_iterator elem_iter;
		for (elem_iter = elem_stat.begin(); elem_iter != elem_stat.end(); elem_iter++) {
			if (elem_iter->second > cutOff) {
				_elems.from_string(elem_iter->first);
			}
		}
		_elems.set_fixed_flag(true);
		_nVSize = _elems.size();
		_nUNKId = _elems.from_string(unknownkey);
	}

	inline void initialWeights(int nDim, int seed = 0, bool bFineTune = true) {
		if (_nVSize == 0){
			std::cout << "please check the alphabet" << std::endl;
			return;
		}
		_nDim = nDim;
		srand(seed);
		_E.initial(_nDim, _nVSize);
		for (int idx = 0; idx < _nVSize; idx++){
			norm2one(_E.val, idx);
		}

		_bFineTune = bFineTune;
	}

	// default should be fineTune, just for initialization
	inline void initialWeights(const string& inFile, bool bFineTune = true) {
		if (_nVSize == 0){
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
		_nDim = vecInfo.size() - 1;

		_E.initial(_nDim, _nVSize);
		_E.val.setZero();

		std::cout << "word embedding dim is " << _nDim << std::endl;

		bool bHasUnknown = false;
		hash_set<int> indexers;
		VectorXd sum = VectorXd::Zero(_nDim);
		int count = 0;
		for (int idx = 0; idx < sLines.size(); idx++){
			split_bychar(sLines[idx], vecInfo, ' ');
			if (vecInfo.size() != _nDim + 1) {
				std::cout << "error embedding file" << std::endl;
			}
			curWord = vecInfo[0];
			//we assume the keys are normalized
			wordId = _elems.from_string(curWord);
			if (wordId >= 0) {
				count++;
				if (_nUNKId == wordId){
					bHasUnknown = true;
				}
				indexers.insert(wordId);

				for (int idy = 0; idy < _nDim; idy++) {
					dtype curValue = atof(vecInfo[idy + 1].c_str());
					sum(idy) += curValue;
					_E.val(wordId, idy) += curValue;
				}
			}
		}

		if (_nUNKId >= 0 && !bHasUnknown){
			for (int idx = 0; idx < _nDim; idx++) {
				_E.val(_nUNKId, idx) = sum(idx) / count;
			}
			indexers.insert(_nUNKId);
			count++;
			std::cout << unknownkey << " not found, using averaged value to initialize." << std::endl;
		}

		int oovWords = 0;
		for (int id = 0; id < _nVSize; id++) {
			if (indexers.find(id) == indexers.end()) {
				oovWords++;
				for (int idy = 0; idy < _nDim; idy++){
					_E.val(id, idy) = _nUNKId >= 0 ? _E.val(_nUNKId, idy) : sum(idy) / count;
				}
			}
		}

		std::cout << "OOV num is " << oovWords << ", total num is " << _nVSize << ", embedding oov ratio is " << oovWords * 1.0 / _nVSize << std::endl;

		for (int idx = 0; idx < _nVSize; idx++){
			norm2one(_E.val, idx);
		}

		_bFineTune = bFineTune;
	}

	inline void exportAdaParams(ModelUpdate& ada) {
		if (_bFineTune) {
			ada.addParam(&_E);
		}
	}


	inline int getElemId(const string& strFeat){
		return _elems.from_string(strFeat);
	}

};

struct LookupNode {
public:
	LookupTable* _param;
	int _xid;
	Mat _y;
	Mat _ly;

	int _inDim;
	int _outDim;

public:
	LookupNode() {
		clear();
	}

	LookupNode(LookupTable* param) {
		_xid = -1;
		_y.setZero();
		_ly.setZero();
		setParam(param);
	}

	inline void setParam(LookupTable* param) {
		_param = param;
		_inDim = _param->_nVSize;
		_outDim = _param->_nDim;
	}

	inline void clear(){
		_xid = -1;
		_y.setZero();
		_ly.setZero();
		_param = NULL;
		_inDim = _outDim = 0;
	}

	inline void clearValue(){
		_xid = -1;
		_y.setZero();
		_ly.setZero();
	}

public:
	//notice the output
	void forward(const string& strNorm) {
		assert(_param != NULL);
		_xid = _param->getElemId(strNorm);
		if (_xid < 0 && _param->_nUNKId >= 0){
			_xid = _param->_nUNKId;
		}
		if (_xid >= 0){
			_y = _param->_E.val.row(_xid).transpose();
		}
		else{
			std::cout << "Caution: unknown words are not modeled !" << std::endl;
			_y = Mat::Zero(_outDim, 1);
		}
	}

	void backward() {
		assert(_param != NULL);
		if (_xid >= 0 && _param->_bFineTune){
			_param->_E.grad.row(_xid) += _ly.col(0).transpose();
			_param->_E._indexers.insert(_xid);
		}
	}

};

#endif /*_LOOKUPTABLE_H*/
