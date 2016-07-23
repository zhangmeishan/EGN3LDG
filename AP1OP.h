/*
 * AP1O.h
 *
 *  Created on: Jul 20, 2016
 *      Author: mason
 */

#ifndef AP1OP_H_
#define AP1OP_H_

#include "Eigen/Dense"
#include "MyLib.h"

using namespace Eigen;

// sparse features
class AP1Param {
public:
	hash_set<int> _indexers;
	MatrixXd _W, _gradW, _sumW;

	int _max_update;
	VectorXi _last_update;

public:
	AP1Param() {
		_indexers.clear();
	}

	inline void initial(int nISize, int seed = 0) {
		srand(seed);
		_W = MatrixXd(nISize, 1).unaryExpr(ptr_fun(urand));

		_gradW = MatrixXd::Zero(nISize, 1);
		_sumW = MatrixXd::Zero(nISize, 1);

		_max_update = 0;
		_last_update = VectorXi::Zero(nISize);
	}

	inline double get(int featId, bool bTrain = false) {
		if (featId >= _W.rows() || featId < 0){
			return 0.0;
		}
		if (bTrain)
			return _W(featId, 0);
		else
			return sumWeight(featId) * 1.0 / _max_update;
	}

	inline double sumWeight(int featId) {
		if (_last_update(featId) < _max_update) {
			int times = _max_update - _last_update[featId];
			_sumW(featId, 0) += _W(featId, 0) * times;
			_last_update(featId) = _max_update;
		}

		return _sumW(featId);
	}

	void update() {
		static hash_set<int>::iterator it;
		_max_update++;

		for (it = _indexers.begin(); it != _indexers.end(); ++it) {
			int index = *it;
			_sumW(index, 0) += (_max_update - _last_update(index)) * _W(index, 0) - _gradW(index, 0);
			_W(index, 0) = _W(index, 0) - _gradW(index, 0);
			_last_update(index) = _max_update;
		}

		clearGrad();
	}

	void clearGrad() {
		static hash_set<int>::iterator it;
		for (it = _indexers.begin(); it != _indexers.end(); ++it) {
			int index = *it;
			_gradW(index, 0) = 0.0;
		}
		_indexers.clear();

	}

};

// a single node;
// input variables are not allocated by current node(s)
class AP1Node {

public:
	AP1Param* _param;
	vector<int>* _x;
	double _y;
	double _ly;

public:

	AP1Node() {
		_param = NULL;
		_x = NULL;
		_y = _ly = 0.0;
	}

	inline void setParam(AP1Param* param) {
		_param = param;
	}

	// initialize inputs at the same times
	inline void forward(vector<int>& x, bool bTrain = false) {
		_y = 0.0;
		_x = &x;
		static int featId;
		for (int idx = 0; idx < _x->size(); idx++) {
			featId = (*_x)[idx];
			_y += _param->get(featId, bTrain);
		}
	}

	//no output losses
	void backward() {
		for (int idx = 0; idx < _x->size(); idx++) {
			int featId = (*_x)[idx];
			if (featId >= _param->_W.rows() || featId < 0)
				continue;
			_param->_indexers.insert(featId);
			_param->_gradW(featId, 0) += _ly;
		}
	}

};


#endif /* AP1OP_H_ */
