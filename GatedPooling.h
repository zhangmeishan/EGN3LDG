#ifndef _Gate_pool_param_H_
#define _Gate_pool_param_H_

#include "UniOP.h"
#include "SoftmaxOP.h"
#include "Pooling.h"
#include "ModelUpdate.h"

struct GatedPoolParam{
	UniParams _uni_gate_param;

	inline void initial(int nOSize, int nISize, AlignedMemoryPool* mem = NULL){
		_uni_gate_param.initial(nOSize, nISize, true, mem);
	}

	inline int inDim(){
		return _uni_gate_param.W.inDim();
	}

	inline int outDim(){
		return _uni_gate_param.W.outDim();
	}

	inline void exportAdaParams(ModelUpdate& ada) {
		_uni_gate_param.exportAdaParams(ada);
	}
};

class GatedPoolBuilder {

public:
	int _nSize;
	int _inDim;
	int _outDim;

	vector<UniNode> _uni_gate;

	SoftmaxBuilder _softmax_project;

	vector<PMultNode> _mul;

	PAddNode  _output;

	GatedPoolParam* _param;

	GatedPoolBuilder(){
		clear();
	}

	~GatedPoolBuilder(){
		clear();
	}

	inline void clear(){
		_uni_gate.clear();
	}

	inline void resize(int maxsize){
		_uni_gate.resize(maxsize);
		_mul.resize(maxsize);
		_softmax_project.resize(maxsize);
	}

	inline void init(GatedPoolParam *paramInit, AlignedMemoryPool *mem = NULL){
		_param = paramInit;
		int maxsize = _uni_gate.size();
		_inDim = _param->inDim();
		_outDim = _param->outDim();
		for (int idx = 0; idx < maxsize; idx++) {
			_uni_gate[idx].setParam(&_param->_uni_gate_param);
			_uni_gate[idx].init(_outDim, -1, mem);
		}
		_softmax_project.init(_outDim, mem);
		for (int idx = 0; idx < maxsize; idx++)
			_mul[idx].init(_outDim, -1, mem);
		_output.init(_outDim, -1, mem);
	}

	inline void forward(Graph *cg, const vector<PNode>& x){
		if (x.size() == 0) {
			std::cout << "empty inputs for GatedPoolBuilder operation" << std::endl;
			return;
		}
		_nSize = x.size();
		if (x[0]->val.dim != _inDim) {
			std::cout << "input dim does not for GatedPoolBuilder operation" << std::endl;
			return;
		}
		for (int idx = 0; idx < _nSize; idx++)
			_uni_gate[idx].forward(cg, x[idx]);
		_softmax_project.forward(cg, getPNodes(_uni_gate, _nSize));
		for (int idx = 0; idx < _nSize; idx++)
			_mul[idx].forward(cg, &_softmax_project._output[idx], &_uni_gate[idx]);
		_output.forward(cg, getPNodes(_mul, _nSize));
	}
};

#endif
