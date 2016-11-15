#ifndef GRNN
#define GRNN

#include "BiOP.h"
#include "UniOP.h"
#include "Graph.h"

struct GRNNParams{
	BiParams _rnn;
	BiParams _rnn_update;
	BiParams _rnn_reset;

	GRNNParams() {
	}

	inline void exportAdaParams(ModelUpdate& ada){
		_rnn.exportAdaParams(ada);
		_rnn_update.exportAdaParams(ada);
		_rnn_reset.exportAdaParams(ada);
	}

	inline void initial(int nOSize, int nISize, AlignedMemoryPool* mem = NULL){
		_rnn_update.initial(nOSize, nOSize, nISize, true, mem);
		_rnn_reset.initial(nOSize, nOSize, nISize, true, mem);
		_rnn.initial(nOSize, nOSize, nISize, true, mem);
	}

	inline int inDim() {
		return _rnn.W2.inDim();
	}

	inline int outDim(){
		return _rnn.W2.outDim();
	}
};

class GRNNBuilder{
public:
	int _nSize;
	int _inDim;
	int _outDim;
	
	Node _bucket_zero;
	Node _bucket_one;
	vector<BiNode> _rnn_update_nodes;
	vector<BiNode> _rnn_reset_nodes;
	vector<PMultNode> _y_temp_nodes;
	vector<PSubNode> _sub_nodes;
	vector<PMultNode> _mult_nodes_1;
	vector<PMultNode> _mult_nodes_2;
	vector<PAddNode> _add_node;
	vector<BiNode> _rnn_nodes;
	
public:
	vector<PAddNode> _output;

	GRNNParams* _params;
	bool _left2right;

public:
	~GRNNBuilder(){
		clear();
	}

	GRNNBuilder(){
		clear();
	}

public:
	inline void init(GRNNParams* paramInit, dtype dropout, bool left2right = true, AlignedMemoryPool* mem = NULL) {
		_params = paramInit;
		_inDim = _params->_rnn.W2.inDim();
		_outDim = _params->_rnn.W2.outDim();
		int maxsize = _rnn_nodes.size();
		for (int idx = 0; idx < maxsize; idx++) {
			_rnn_update_nodes[idx].setParam(&_params->_rnn_update);			
			_rnn_reset_nodes[idx].setParam(&_params->_rnn_reset);
			_rnn_nodes[idx].setParam(&_params->_rnn);
			_rnn_update_nodes[idx].setFunctions(&fsigmoid, &dsigmoid);
			_rnn_reset_nodes[idx].setFunctions(&fsigmoid, &dsigmoid);			
			_rnn_nodes[idx].setFunctions(&ftanh, &dtanh);
		}
		_left2right = left2right;
		
		for (int idx = 0; idx < maxsize; idx++) {
			_rnn_update_nodes[idx].init(_outDim, -1, mem);
			_rnn_reset_nodes[idx].init(_outDim, -1, mem);
			_rnn_nodes[idx].init(_outDim, -1, mem);
			_y_temp_nodes[idx].init(_outDim, -1, mem);
			_sub_nodes[idx].init(_outDim, -1, mem);
			_mult_nodes_1[idx].init(_outDim, -1, mem);
			_mult_nodes_2[idx].init(_outDim, -1, mem);
			_add_node[idx].init(_outDim, -1, mem);
			_output[idx].init(_outDim, dropout, mem);			
		}
		
		_bucket_zero.init(_outDim, -1, mem);
		_bucket_one.init(_outDim, -1, mem);
		_bucket_one.val = 1.0;
	}

	inline void resize(int maxsize) {
		_rnn_update_nodes.resize(maxsize);
		_rnn_reset_nodes.resize(maxsize);
		_rnn_nodes.resize(maxsize);
		_y_temp_nodes.resize(maxsize);
		_sub_nodes.resize(maxsize);
		_mult_nodes_1.resize(maxsize);
		_mult_nodes_2.resize(maxsize);
		_add_node.resize(maxsize);
		_output.resize(maxsize);
	}

	inline void clear(){
		_nSize = 0;
		_inDim = 0;
		_outDim = 0;
		_left2right = true;
		_params = NULL;
		_rnn_update_nodes.clear();
		_rnn_reset_nodes.clear();
		_rnn_nodes.clear();
		_y_temp_nodes.clear();
		_sub_nodes.clear();
		_mult_nodes_1.clear();
		_mult_nodes_2.clear();
		_add_node.clear();
		_output.clear();
	}
	
	inline void forward(Graph* cg, const vector<PNode>& x) {
		if (x.size() == 0) {
			std::cout << "empty inputs for GRNN operation" << std::endl;
			return;
		}
		_nSize = x.size();
		if (x[0]->val.dim != _inDim) {
			std::cout << "input dim dose not match for seg operation" << std::endl;
			return;
		}
		if (_left2right)
			left2right_forward(cg, x);
		else
			right2left_forward(cg, x);
	}
protected:
	/*
		y_reset_i = sigmoid ( y_i-1 * w_reset_1 + x_i * w_reset_2 + b_reset)
		y_update_i = sigmoid ( y_i-1 * w_update_1 + x_i * w_update_2 + b_update )
		y_temp_i = y_reset_i * y_i-1
		y_rnn_i = tanh (y_temp_i * w_rnn_1 + x_i * w_rnn_2 + b_rnn)
		y_i = (1 - y_update_i) * y_i-1 + y_update_i * y_rnn_i
	*/
	inline void left2right_forward(Graph *cg, const vector<PNode>& x) {
		for (int idx = 0; idx < _nSize; idx++) {
			if (idx == 0) {
				_rnn_update_nodes[idx].forward(cg, &_bucket_zero, x[idx]);
				_rnn_nodes[idx].forward(cg, &_bucket_zero, x[idx]);
				_mult_nodes_2[idx].forward(cg, &_rnn_update_nodes[idx], &_rnn_nodes[idx]);
				_output[idx].forward(cg, &_bucket_zero, &_mult_nodes_2[idx]);
			} else {
				_rnn_reset_nodes[idx].forward(cg, &_output[idx - 1], x[idx]); 
				_rnn_update_nodes[idx].forward(cg, &_output[idx - 1], x[idx]);
				_y_temp_nodes[idx].forward(cg, &_rnn_reset_nodes[idx], &_output[idx - 1]);
				_rnn_nodes[idx].forward(cg, &_y_temp_nodes[idx], x[idx]);
				_sub_nodes[idx].forward(cg, &_bucket_one, &_rnn_update_nodes[idx]);
				_mult_nodes_1[idx].forward(cg, &_sub_nodes[idx], &_output[idx - 1]);
				_mult_nodes_2[idx].forward(cg, &_rnn_update_nodes[idx], &_rnn_nodes[idx]);
				_output[idx].forward(cg, &_mult_nodes_1[idx], &_mult_nodes_2[idx]);
			}
		}
	}

	inline void right2left_forward(Graph *cg, const vector<PNode>& x) {
		for (int idx = _nSize - 1; idx >= 0; idx--) {
			if (idx == _nSize - 1) {
				_rnn_update_nodes[idx].forward(cg, &_bucket_zero, x[idx]);
				_rnn_nodes[idx].forward(cg, &_bucket_zero, x[idx]);
				_mult_nodes_2[idx].forward(cg, &_rnn_update_nodes[idx], &_rnn_nodes[idx]);
				_output[idx].forward(cg, &_bucket_zero, &_mult_nodes_2[idx]);
			} else {
				_rnn_reset_nodes[idx].forward(cg, &_output[idx + 1], x[idx]);
				_rnn_update_nodes[idx].forward(cg, &_output[idx + 1], x[idx]);
				_y_temp_nodes[idx].forward(cg, &_rnn_reset_nodes[idx], &_output[idx + 1]);
				_rnn_nodes[idx].forward(cg, &_y_temp_nodes[idx], x[idx]);
				_sub_nodes[idx].forward(cg, &_bucket_one, &_rnn_update_nodes[idx]);
				_mult_nodes_1[idx].forward(cg, &_sub_nodes[idx], &_output[idx + 1]);
				_mult_nodes_2[idx].forward(cg, &_rnn_update_nodes[idx], &_rnn_nodes[idx]);
				_output[idx].forward(cg, &_mult_nodes_1[idx], &_mult_nodes_2[idx]);
			}
		}
	}

};

class IncGRNNBuilder{
public:
	int _nSize;
	int _inDim;
	int _outDim;
	
	Node _bucket_zero;
	Node _bucket_one;
	BiNode _rnn_update_node;
	BiNode _rnn_reset_node;
	PMultNode _y_temp_node;
	PSubNode _sub_node;
	PMultNode _mult_node_1;
	PMultNode _mult_node_2;
	PAddNode _add_node;
	BiNode _rnn_node;
	PAddNode _output;

	GRNNParams* _params;

public:
	~IncGRNNBuilder(){
		clear();
	}

	IncGRNNBuilder(){
		clear();
	}

public:
	inline void init(GRNNParams* paramInit, dtype dropout, AlignedMemoryPool* mem = NULL) {
		_params = paramInit;
		_inDim = _params->_rnn.W2.inDim();
		_outDim = _params->_rnn.W2.outDim();

		_rnn_update_node.setParam(&_params->_rnn_update);		
		_rnn_reset_node.setParam(&_params->_rnn_reset);		
		_rnn_node.setParam(&_params->_rnn);
		_rnn_update_node.setFunctions(&fsigmoid, &dsigmoid);
		_rnn_reset_node.setFunctions(&fsigmoid, &dsigmoid);
		_rnn_node.setFunctions(&ftanh, &dtanh);
		
		_rnn_update_node.init(_outDim, -1, mem);
		_rnn_reset_node.init(_outDim, -1, mem);
		_rnn_node.init(_outDim, -1, mem);
		_y_temp_node.init(_outDim, -1, mem);
		_sub_node.init(_outDim, -1, mem);
		_mult_node_1.init(_outDim, -1, mem);
		_mult_node_2.init(_outDim, -1, mem);
		_add_node.init(_outDim, -1, mem);
		_output.init(_outDim, dropout, mem);		

		_bucket_zero.init(_outDim, -1, mem);
		_bucket_one.init(_outDim, -1, mem);
		_bucket_one.val = 1.0;
	}


	inline void clear(){
		_nSize = 0;
		_inDim = 0;
		_outDim = 0;
		_params = NULL;
	}
	
public:
	inline void left2right_forward(Graph *cg, PNode x, IncGRNNBuilder* prev = NULL) {
		if (prev == NULL) {
			_rnn_update_node.forward(cg, &_bucket_zero, x);
			_rnn_node.forward(cg, &_bucket_zero, x);
			_mult_node_2.forward(cg, &_rnn_update_node, &_rnn_node);
			_output.forward(cg, &_bucket_zero, &_mult_node_2);
		} else {
			_rnn_reset_node.forward(cg, &prev->_output, x); 
			_rnn_update_node.forward(cg, &prev->_output, x);
			_y_temp_node.forward(cg, &_rnn_reset_node, &prev->_output);
			_rnn_node.forward(cg, &_y_temp_node, x);
			_sub_node.forward(cg, &_bucket_one, &_rnn_update_node);
			_mult_node_1.forward(cg, &_sub_node, &prev->_output);
			_mult_node_2.forward(cg, &_rnn_update_node, &_rnn_node);
			_output.forward(cg, &_mult_node_1, &_mult_node_2);
		}
	}

};
#endif