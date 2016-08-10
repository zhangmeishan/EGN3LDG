#ifndef _LOOKUPTABLE_H_
#define _LOOKUPTABLE_H_

#include "Param.h" 
#include "MyLib.h"

#include <Eigen/Dense>

using namespace Eigen;

class LookupTable {
public:

	MatrixXd _E, _gradE, _eg2E, _ftE;
	bool _bFineTune;
	int _nDim;
	int _nVSize;

	int _max_update;

	hash_set<int> _indexers;
	NRVec<int> _last_update;
public:

	LookupTable() {
		_indexers.clear();
	}

	inline void initial(const NRMat<double>& wordEmb) {
		_nVSize = wordEmb.nrows();
		_nDim = wordEmb.ncols();

		_gradE.resize(_nVSize, _nDim);
		_eg2E.resize(_nVSize, _nDim);
		_ftE.resize(_nVSize, _nDim);

		assign(_E, wordEmb);
		for (int idx = 0; idx < _nVSize; idx++)
			norm2one(_E, idx);

		_bFineTune = true;

		_max_update = 0;
		_last_update.resize(_nVSize);
		_last_update = 0;

	}

	inline double squarenormAll() {
		double result = 0;
		static hash_set<int>::iterator it;
		for (int idx = 0; idx < _nDim; idx++)
			for (it = _indexers.begin(); it != _indexers.end(); ++it)
				result += _gradE(*it, idx) * _gradE(*it, idx);

		return result;
	}

	inline void scaleGrad(double scale) {
		static hash_set<int>::iterator it;
		for (int idx = 0; idx < _nDim; idx++)
			for (it = _indexers.begin(); it != _indexers.end(); ++it)
				_gradE(*it, idx) = _gradE(*it, idx) * scale;
	}

	inline void setEmbFineTune(bool bFineTune) {
		_bFineTune = bFineTune;
	}

	void GetEmb(int id, MatrixXd& y) {
		y = _E.row(id);
	}

	void EmbLoss(int id, MatrixXd& ly) {
		if (!_bFineTune)
			return;
		_gradE.row(id) += ly;
		_indexers.insert(id);
	}

	void updateSparseWeight(int wordId) {
		if (!_bFineTune)
			return;
		if (_last_update[wordId] < _max_update) {
			int times = _max_update - _last_update[wordId];
			_E.row(wordId).array() *=
					(times * _ftE.row(wordId).array().log()).exp();
			_last_update[wordId] = _max_update;
		}
	}

	void updateAdaGrad(double alpha, double reg, double eps) {
		if (!_bFineTune)
			return;
		static hash_set<int>::iterator it;
		_max_update++;
		MatrixXd sqrt_eg2E;

		for (it = _indexers.begin(); it != _indexers.end(); ++it) {
			int index = *it;
			_eg2E.row(index).array() += _gradE.row(index).array().square();
			sqrt_eg2E = (_eg2E.row(index).array() + eps).sqrt();
			_E.row(index) = (_E.row(index).array() * sqrt_eg2E.array()
					- _gradE.row(index).array() * alpha)
					/ (alpha * reg + sqrt_eg2E.array());
			_ftE.row(index) = (sqrt_eg2E.array()
					/ (alpha * reg + sqrt_eg2E.array())).matrix();
		}
		clearGrad();
	}

	void clearGrad() {
		static hash_set<int>::iterator it;
		for (it = _indexers.begin(); it != _indexers.end(); ++it) {
			int index = *it;
			_gradE.row(index).setZero();
		}
		_indexers.clear();
	}
};

#endif /*_LOOKUPTABLE_H*/
