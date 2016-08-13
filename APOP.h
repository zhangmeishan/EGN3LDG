/*
 * AP1O.h
 *
 *  Created on: Jul 20, 2016
 *      Author: mszhang
 */

#ifndef APOP_H_
#define APOP_H_

#include "MyLib.h"
#include "Alphabet.h"

// aiming for Averaged Perceptron, can not coexist with neural networks
// thus we do not use the BaseParam class
struct APParams {
public:
	Alphabet _elems;
	hash_set<int> _indexers;
	Mat _W, _gradW, _sumW;

	int _max_update;
	VectorXi _last_update;

	int _nVSize;
	int _nDim;

public:
	APParams() {
		_indexers.clear();
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

	inline void initialWeights(int nOSize, int seed = 0) {
		if (_nVSize == 0){
			std::cout << "please check the alphabet" << std::endl;
			return;
		}
		srand(seed);
		_nDim = nOSize;
		_W = Mat(_nVSize, _nDim).unaryExpr(ptr_fun(urand));

		_gradW = Mat::Zero(_nVSize, _nDim);
		_sumW = Mat::Zero(_nVSize, _nDim);

		_max_update = 0;
		_last_update = VectorXi::Zero(_nVSize);
	}

	//random initialization
	inline void initial(const hash_map<string, int>& elem_stat, int cutOff, int nOSize, int seed){
		initialAlpha(elem_stat, cutOff);
		initialWeights(nOSize, seed);
	}

	inline Mat get(int featId, bool bTrain = false) {
		if (bTrain)
			return _W.row(featId);
		else
			return sumWeight(featId).array() * 1.0 / _max_update;
	}

	inline Mat sumWeight(int featId) {
		if (_last_update(featId) < _max_update) {
			int times = _max_update - _last_update(featId);
			_sumW.row(featId) += _W.row(featId) * times;
			_last_update(featId) = _max_update;
		}

		return _sumW.row(featId);
	}

	inline int getFeatureId(const string& strFeat){
		return _elems.from_string(strFeat);
	}

	void update() {
		static hash_set<int>::iterator it;
		_max_update++;

		for (it = _indexers.begin(); it != _indexers.end(); ++it) {
			int index = *it;
			_sumW.row(index) += (_max_update - _last_update(index)) * _W.row(index) - _gradW.row(index);
			_W.row(index) = _W.row(index) - _gradW.row(index);
			_last_update(index) = _max_update;
		}

		clearGrad();
	}

	void clearGrad() {
		static hash_set<int>::iterator it;
		for (it = _indexers.begin(); it != _indexers.end(); ++it) {
			int index = *it;
			_gradW.row(index).setZero();
		}
		_indexers.clear();

	}

};

// a single node;
// input variables are not allocated by current node(s)
struct APNode {

public:
	APParams* _param;
	vector<int> _tx;
	Mat _y;
	Mat _ly;

	int _inDim;
	int _outDim;

public:

	APNode() {
		clear();
	}

	inline void setParam(APParams* param) {
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

	// initialize inputs at the same times
	inline void forward(const vector<string>& x, bool bTrain = false) {
		assert(_param != NULL);
		_y = Mat::Zero(_outDim, 1);
		static int featId;
		for (int idx = 0; idx < x.size(); idx++) {
			featId = _param->getFeatureId(x[idx]);
			if (featId < _inDim && featId >= 0){
				_tx.push_back(featId);
				_y.col(0) += _param->get(featId, bTrain).transpose();
			}
		}
	}

	//no output losses
	void backward() {
		assert(_param != NULL);
		for (int idx = 0; idx < _tx.size(); idx++) {
			_param->_indexers.insert(_tx[idx]);
			_param->_gradW.row(_tx[idx]) += _ly.col(0).transpose();
		}
	}

};


#endif /* APOP_H_ */
