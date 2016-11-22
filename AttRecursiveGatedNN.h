#ifndef _AttRecursiveGatedNN_H_
#define _AttRecursiveGatedNN_H_

#include "BiOP.h"
#include "SoftmaxOP.h"

struct AttRecursiveGatedParams{
	BiParams  _reset_left_param;
	BiParams  _reset_right_param;
	BiParams  _update_left_param;
	BiParams  _update_right_param;
	BiParams  _update_tilde_param;
	BiParams _recursive_tilde_param;

	inline void initial(int nOSize, int nISize, AlignedMemoryPool* mem = NULL) {
		_reset_left_param.initial(nOSize, nOSize, nISize, false, mem);
		_reset_right_param.initial(nOSize, nOSize, nISize, false, mem);
		_update_left_param.initial(nOSize, nOSize, nISize, false, mem);
		_update_right_param.initial(nOSize, nOSize, nISize, false, mem);
		_update_tilde_param.initial(nOSize, nOSize, nISize, false, mem);
		_recursive_tilde_param.initial(nOSize, nOSize, nOSize, false, mem);
	}

	inline void exportAdaParams(ModelUpdate& ada){
		_reset_left_param.exportAdaParams(ada);
		_reset_right_param.exportAdaParams(ada);
		_update_left_param.exportAdaParams(ada);
		_update_right_param.exportAdaParams(ada);
		_update_tilde_param.exportAdaParams(ada);
	}

	inline int inDim(){
		return _reset_left_param.W2.inDim();
	}

	inline int outDim(){
		return _reset_left_param.W2.outDim();
	}
};

class AttRecursiveGatedBuilder{
public:
	int _outDim;
	int _inDim;

	BiNode _reset_left;
	BiNode  _reset_right;
	BiNode  _update_left;
	BiNode  _update_right;
	BiNode _recursive_tilde;
	BiNode _update_tilde;

	PMultNode _mul_left;
	PMultNode _mul_right;

	AttRecursiveGatedParams* _param;

	SoftmaxBuilder _softmax_layer;
	vector<PMultNode> _muls;
	PAddNode _output;

	AttRecursiveGatedBuilder(){
		clear();
		_softmax_layer.resize(3);
		_muls.resize(3);
	}

	~AttRecursiveGatedBuilder(){
		clear();
	}

	inline void clear(){
		_outDim = 0;
		_inDim = 0;
	}

	inline void init(AttRecursiveGatedParams* paramInit, dtype dropout, AlignedMemoryPool* mem = NULL){
		_param = paramInit;
		_inDim = paramInit->inDim();
		_outDim = paramInit->outDim();

		_reset_left.setParam(&_param->_reset_left_param);
		_reset_right.setParam(&_param->_reset_right_param);
		_update_left.setParam(&_param->_update_left_param);
		_update_right.setParam(&_param->_update_right_param);
		_update_tilde.setParam(&_param->_update_tilde_param);
		_recursive_tilde.setParam(&_param->_recursive_tilde_param);
		_mul_left.init(_outDim, -1, mem);
		_mul_right.init(_outDim, -1, mem);
		_reset_left.init(_outDim, -1, mem);
		_reset_right.init(_outDim, -1, mem);
		_update_left.init(_outDim, -1, mem);
		_update_right.init(_outDim, -1, mem);
		_update_tilde.init(_outDim, -1, mem);
		_recursive_tilde.init(_outDim, -1, mem);
		_reset_left.setFunctions(&fsigmoid, &dsigmoid);
		_reset_right.setFunctions(&fsigmoid, &dsigmoid);

		_update_left.setFunctions(&fexp, &dexp);
		_update_right.setFunctions(&fexp, &dexp);
		_update_tilde.setFunctions(&fexp, &dexp);

		_softmax_layer.init(_outDim, mem);

		for (int idx = 0; idx < 3; idx++)
			_muls[idx].init(_outDim, -1, mem);
		_output.init(_outDim, dropout, mem);
	}

	inline void forward(Graph *cg, PNode left, PNode right, PNode target){
		vector<PNode> x;
		x.push_back(left);
		x.push_back(right);
		x.push_back(target);
		forward(cg, x);
	}

	// 0 left, 1 right, 2target
	inline void forward(Graph *cg, const vector<PNode>& x){
		if (x.size() != 3) {
			std::cout << "please check the input of AttRecursiveGated" << std::endl;
			return;
		}
		_reset_left.forward(cg, x[0], x[2]);
		_reset_right.forward(cg, x[1], x[2]);
		_mul_left.forward(cg, &_reset_left, x[0]);
		_mul_right.forward(cg, &_reset_right, x[1]);
		_recursive_tilde.forward(cg, &_mul_left, &_mul_right);

		_update_left.forward(cg, x[0], x[2]);
		_update_right.forward(cg, x[1], x[2]);
		_update_tilde.forward(cg, &_recursive_tilde, x[2]);

		vector<PNode> _update_nodes;
		_update_nodes.push_back(&_update_left);
		_update_nodes.push_back(&_update_right);
		_update_nodes.push_back(&_update_tilde);

		_softmax_layer.forward(cg, _update_nodes);
		_muls[0].forward(cg, x[0], &_softmax_layer._output[0]);
		_muls[1].forward(cg, x[1], &_softmax_layer._output[1]);
		_muls[2].forward(cg, &_update_tilde, &_softmax_layer._output[2]);
		_output.forward(cg, getPNodes(_muls, 3));
	}
};

#endif