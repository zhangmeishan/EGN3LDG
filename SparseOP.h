/*
 * UniOP.h
 *
 *  Created on: Jul 20, 2016
 *      Author: mason
 */

#ifndef SPARSEOP_H_
#define SPARSEOP_H_

#include "SparseParam.h"
#include "MyLib.h"
#include "Alphabet.h"

struct SparseParams {
public:
	SparseParam _W;
	Alphabet _elems;
	int _nVSize;
	int _nDim;

public:
	SparseParams() {
	}

	inline void exportAdaParams(ModelUpdate& ada) {
		ada.addParam(&_W);
	}

	inline void initialWeights(int nOSize, int seed = 0) {
		if (_nVSize == 0){
			std::cout << "please check the alphabet" << std::endl;
			return;
		}
		srand(seed);
		_nDim = nOSize;
		_W.initial(nOSize, _nVSize);
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
	}

	//random initialization
	inline void initial(const hash_map<string, int>& elem_stat, int cutOff, int nOSize, int seed){
		initialAlpha(elem_stat, cutOff);
		initialWeights(nOSize, seed);
	}

	inline int getFeatureId(const string& strFeat){
		return _elems.from_string(strFeat);
	}

};

//only implemented sparse linear node.
//non-linear transformations are not support,
//but we can use other approaches to achieve the same goal.
struct SparseNode {
public:
	vector<int> _tx;
	Mat _y, _ly;

	int _inDim, _outDim;

	SparseParams* _param;

public:
	SparseNode() {
		clear();
	}

	SparseNode(SparseParams* param) {
		_tx.clear();
		_y.setZero();
		_ly.setZero();
		setParam(param);
	}

	inline void setParam(SparseParams* param) {
		_param = param;
		_inDim = _param->_nVSize;
		_outDim = _param->_nDim;
	}

	inline void clear(){
		_tx.clear();
		_y.setZero();
		_ly.setZero();
		_param = NULL;
		_inDim = _outDim = 0;
	}

	inline void clearValue(){
		_tx.clear();
		_y.setZero();
		_ly.setZero();
	}

public:
	//notice the output
	void forward(const vector<string>& x) {
		assert(_param != NULL);
		_y = Mat::Zero(_outDim, 1);
		static int featId;
		for (int idx = 0; idx < x.size(); idx++) {
			featId = _param->getFeatureId(x[idx]);
			if (featId < _inDim && featId >= 0){
				_tx.push_back(featId);
				_y.col(0) += _param->_W.val.row(featId).transpose();
			}
		}
	}

	void backward() {
		assert(_param != NULL);
		for (int idx = 0; idx < _tx.size(); idx++) {
			_param->_W._indexers.insert(_tx[idx]);
			_param->_W.grad.row(_tx[idx]) += _ly.col(0).transpose();
		}

	}

};

#endif /* SPARSEOP_H_ */
