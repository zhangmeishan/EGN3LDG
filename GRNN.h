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

	inline void initial(int nOSize, int nISize){
		_rnn_update.initial(nOSize, nOSize, nISize, true);
		_rnn_reset.initial(nOSize, nOSize, nISize, true);
		_rnn.initial(nOSize, nOSize, nISize, true);
	}

	inline int inDim() {
		return _rnn.W2.inDim();
	}

	inline int outDim(){
		return _rnn.W2.outDim();
	}
};

class GRNNBuilder :NodeBuilder{
public:
	int _nSize;
	int _inDim;
	int _outDim;
	
private:
	Node _bucket_zero;
	Node _bucket_one;
	Node _bucket_neg_one;
	vector<BiNode> _rnn_update_nodes;
	vector<BiNode> _rnn_reset_nodes;
	vector<PMultNode> _y_temp_nodes;
	vector<PMultNode> _mult_nodes_1;
	vector<PMultNode> _mult_nodes_2;
	vector<PMultNode> _mult_neg_one;
	vector<PAddNode> _add_node;
	vector<BiNode> _rnn_nodes;
public:
	vector<PAddNode> _output;
	vector<DropNode> _grnn_drop;

	GRNNParams* _params;
	bool _left2right;

	~GRNNBuilder(){
		clear();
	}

	GRNNBuilder(){
		clear();
	}

	inline void resize(int maxsize) {
		_rnn_update_nodes.resize(maxsize);
		_rnn_reset_nodes.resize(maxsize);
		_rnn_nodes.resize(maxsize);
		_y_temp_nodes.resize(maxsize);
		_mult_nodes_1.resize(maxsize);
		_mult_nodes_2.resize(maxsize);
		_mult_neg_one.resize(maxsize);
		_add_node.resize(maxsize);
		_output.resize(maxsize);
		_grnn_drop.resize(maxsize);
	}

	inline void clear(){
		_nSize = 0;
		_inDim = 0;
		_outDim = 0;
		_left2right = true;
		_params = NULL;
		_rnn_update_nodes.clear();
		_rnn_reset_nodes.clear();
		_y_temp_nodes.clear();
		_mult_nodes_1.clear();
		_mult_nodes_2.clear();
		_rnn_nodes.clear();
		_grnn_drop.clear();
	}

	inline void setParam(GRNNParams* paramInit, dtype dropout, bool left2right = true) {
		_params = paramInit;
		_inDim = _params->_rnn.W2.inDim();
		_outDim = _params->_rnn.W2.outDim();
		for (int idx = 0; idx < _rnn_nodes.size(); idx++)
		{
			_rnn_update_nodes[idx].setParam(&_params->_rnn_update);
			_rnn_update_nodes[idx].setFunctions(&sigmoid, &sigmoid_deri);
			_rnn_reset_nodes[idx].setParam(&_params->_rnn_reset);
			_rnn_reset_nodes[idx].setFunctions(&sigmoid, &sigmoid_deri);
			_rnn_nodes[idx].setParam(&_params->_rnn);
			_rnn_nodes[idx].setFunctions(&tanh, &tanh_deri);
			_grnn_drop[idx].setDropValue(dropout);
		}
		_left2right = left2right;
		_bucket_zero.val = Mat::Zero(_outDim, 1);
		_bucket_one.val = Mat::Ones(_outDim, 1);
		_bucket_neg_one.val = Mat::Ones(_outDim, 1) * -1;
	}

	inline void forward(Graph* cg, const vector<PNode>& x) {
		if (x.size() == 0) {
			std::cout << "empty inputs for GRNN operation" << std::endl;
			return;
		}
		_nSize = x.size();
		if (x[0]->val.rows() != _inDim) {
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
				_mult_neg_one[idx].forward(cg, &_bucket_neg_one, &_rnn_update_nodes[idx]);
				_add_node[idx].forward(cg, &_bucket_one, &_mult_neg_one[idx]);
				_mult_nodes_1[idx].forward(cg, &_add_node[idx], &_output[idx - 1]);
				_mult_nodes_2[idx].forward(cg, &_rnn_update_nodes[idx], &_rnn_nodes[idx]);
				_output[idx].forward(cg, &_mult_nodes_1[idx], &_mult_nodes_2[idx]);
			}
			//_grnn_drop[idx].forward(cg, &_output[idx]);
		}
		for (int idx = 0; idx < _nSize; idx++)
			_grnn_drop[idx].forward(cg, &_output[idx]);
	}

	inline void right2left_forward(Graph *cg, const vector<PNode>& x) {
		for (int idx = _nSize - 1; idx >= 0; idx--) {
			if (idx == _nSize - 1) {
				_rnn_update_nodes[idx].forward(cg, &_bucket_zero, x[idx]);
				_rnn_nodes[idx].forward(cg, &_bucket_zero, x[idx]);
				_mult_nodes_2[idx].forward(cg, &_rnn_update_nodes[idx], &_rnn_nodes[idx]);
				_output[idx].forward(cg, &_bucket_zero, &_mult_nodes_2[idx]);
			}
			else {
				_rnn_reset_nodes[idx].forward(cg, &_output[idx + 1], x[idx]);
				_rnn_update_nodes[idx].forward(cg, &_output[idx + 1], x[idx]);
				_y_temp_nodes[idx].forward(cg, &_rnn_reset_nodes[idx], &_output[idx + 1]);
				_rnn_nodes[idx].forward(cg, &_y_temp_nodes[idx], x[idx]);
				_mult_neg_one[idx].forward(cg, &_bucket_neg_one, &_rnn_update_nodes[idx]);
				_add_node[idx].forward(cg, &_bucket_one, &_mult_neg_one[idx]);
				_mult_nodes_1[idx].forward(cg, &_add_node[idx], &_output[idx + 1]);
				_mult_nodes_2[idx].forward(cg, &_rnn_update_nodes[idx], &_rnn_nodes[idx]);
				_output[idx].forward(cg, &_mult_nodes_1[idx], &_mult_nodes_2[idx]);

			}
			//_grnn_drop[idx].forward(cg, &_output[idx]);
		}
		for (int idx = 0; idx < _nSize; idx++)
			_grnn_drop[idx].forward(cg, &_output[idx]);
	}

};
#endif