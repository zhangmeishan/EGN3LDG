/*
 * ModelUpdate.h
 *
 *  Created on: Jul 20, 2016
 *      Author: mason
 */

#ifndef ModelUpdate_H_
#define ModelUpdate_H_

#include "BaseParam.h"
#include "MyLib.h"


class ModelUpdate {

public:
	vector<BaseParam*> _params;

	dtype _reg, _alpha, _eps;

public:
	ModelUpdate(){
		_params.clear();

		_reg = 1e-8;
		_alpha = 0.01;
		_eps = 1e-8;
	}


public:

	inline void addParam(BaseParam* param){
		_params.push_back(param);
	}


	inline void update(){
		for(int idx = 0; idx < _params.size(); idx++){
			_params[idx]->updateAdagrad(_alpha, _reg, _eps);
			_params[idx]->clearGrad();
		}
	}

	inline void update(dtype maxScale){
		dtype sumNorm = 0.0;
		for (int idx = 0; idx < _params.size(); idx++){
			sumNorm += _params[idx]->squareGradNorm();
		}
		if (std::isnan(sumNorm) || sumNorm > 1e20){ //too large
			clearGrad();
			return;
		}
		dtype norm = sqrt(sumNorm);
		if (norm > maxScale){
			dtype scale = maxScale / norm;
			for (int idx = 0; idx < _params.size(); idx++){
				_params[idx]->rescaleGrad(scale);
			}
		}

		update();
	}

	inline void clearGrad(){
		for(int idx = 0; idx < _params.size(); idx++){
			_params[idx]->clearGrad();
		}
	}

	inline void clear(){
		_params.clear();
	}


};



#endif /* ModelUpdate_H_ */
