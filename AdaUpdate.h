/*
 * AdaUpdate.h
 *
 *  Created on: Jul 20, 2016
 *      Author: mason
 */

#ifndef ADAUPDATE_H_
#define ADAUPDATE_H_

#include "Param.h"
#include "MyLib.h"


class AdaUpdate {

public:
	vector<Param*> _params;

	double _reg, _alpha, _eps;

public:
	AdaUpdate(){
		_params.clear();

		_reg = 1e-8;
		_alpha = 0.01;
		_eps = 1e-8;
	}


public:

	inline void addParam(Param* param){
		_params.push_back(param);
	}


	inline void update(){
		for(int idx = 0; idx < _params.size(); idx++){
			_params[idx]->updateAdagrad(_alpha, _reg, _eps);
			_params[idx]->clearGrad();
		}
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



#endif /* ADAUPDATE_H_ */
