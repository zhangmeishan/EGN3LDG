/*
 * AdaUpdate.h
 *
 *  Created on: Jul 20, 2016
 *      Author: mason
 */

#ifndef ADAUPDATE_H_
#define ADAUPDATE_H_

#include "Eigen/Dense"
#include "Utils.h"
#include "MyLib.h"

using namespace Eigen;

class AdaUpdate {

public:
	vector<MatrixXd*> _params;
	vector<MatrixXd*> _grad_params;
	vector<MatrixXd*> _eg2_params;

	double _reg, _alpha, _eps;

public:
	AdaUpdate(){
		_params.clear();
		_grad_params.clear();
		_eg2_params.clear();

		_reg = 1e-8;
		_alpha = 0.01;
		_eps = 1e-8;
	}


public:

	inline void addParam(MatrixXd* param, MatrixXd* grad_param, MatrixXd* eg2_param){
		if(!check()){
			std::cout << "error parameter sets" << std::endl;
			return;
		}
		_params.push_back(param);
		_grad_params.push_back(grad_param);
		_eg2_params.push_back(eg2_param);
	}


	inline void update(){
		if(!check()){
			std::cout << "error parameter sets" << std::endl;
			return;
		}
		static MatrixXd tmp;
		for(int idx = 0; idx < _params.size(); idx++){
			*(_grad_params[idx]) = *(_grad_params[idx]) +  *(_params[idx]) * _reg;
			*(_eg2_params[idx]) = *(_eg2_params[idx]) + _grad_params[idx]->cwiseProduct(*(_grad_params[idx]));
			tmp = _grad_params[idx]->array() * _alpha / ( _eg2_params[idx]->array().sqrt() + _eps);
			*(_params[idx]) = *(_params[idx]) - tmp;
		}

		clearGrad();
	}

	inline void clearGrad(){
		for(int idx = 0; idx < _params.size(); idx++){
			_grad_params[idx]->setZero();
		}
	}

	inline void clear(){
		_params.clear();
		_grad_params.clear();
		_eg2_params.clear();
	}


protected:

	inline bool check(){
		if(_params.size() != _grad_params.size() || _params.size() != _eg2_params.size()){
			return false;
		}
		return true;
	}

};



#endif /* ADAUPDATE_H_ */
