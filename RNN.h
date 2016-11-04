#ifndef RNN
#define RNN

#include "BiOP.h"
#include "Graph.h"

struct RNNParams {
	BiParams _rnn;

	RNNParams() {
	}

	inline void exportAdaParams(ModelUpdate& ada) {
		_rnn.exportAdaParams(ada);
	}

	inline void initial(int nOSize, int nISize) {
		_rnn.initial(nOSize, nOSize, nISize, true);
	}

	inline int inDim(){
			return _rnn.W2.inDim();
	}

	inline int outDim(){
			return _rnn.W2.outDim();
	}

};

class RNNBuilder :NodeBuilder{
public:
	int _nSize;
	int _inDim;
	int _outDim;

	Node _bucket;
	vector<BiNode> _rnn_nodes;
	vector<DropNode> _rnn_drop;
	RNNParams* _params;
	bool _left2right;

	~RNNBuilder() {
		clear();
	}

	RNNBuilder() {
		clear();
	}

	inline void clear() {
		_nSize = 0;
		_inDim = 0;
		_outDim = 0;
		_left2right = true;
		_bucket.clear();
		_rnn_nodes.clear();
		_rnn_drop.clear();
		_params = NULL;
	}

	inline void resize(int maxsize){
		_rnn_nodes.resize(maxsize);
		_rnn_drop.resize(maxsize);
	}

	inline bool empty() {
		return _rnn_drop.empty();
	}

	inline void setParam(RNNParams* paramInit, dtype dropout, bool left2right = true) {
		_params = paramInit;
		_inDim = _params->_rnn.W2.inDim();
		_outDim = _params->_rnn.W2.outDim();
		for (int idx = 0; idx < _rnn_nodes.size(); idx++){
			_rnn_nodes[idx].setParam(&_params->_rnn);
			_rnn_drop[idx].setDropValue(dropout);
		}
		_left2right = left2right;
		_bucket.val = Mat::Zero(_outDim, 1);
	}
	
	inline void forward(Graph *cg, const vector<PNode>& x) {
		if (x.size() == 0){
			std::cout << "empty inputs for RNN operation" << std::endl;
			return;
		}
		_nSize = x.size();
		if (x[0]->val.rows() != _inDim)
		{
			std::cout << "input dim dose not match for seg operation" << std::endl;
			return;
		}
		if (_left2right)
			left2right_forward(cg, x);
		else
			right2left_forward(cg, x);
	}
protected:
	inline void left2right_forward(Graph *cg, const vector<PNode>& x) {
		for (int idx = 0; idx < _nSize; idx++) {
			if (idx == 0)
				_rnn_nodes[idx].forward(cg, &_bucket, x[idx]);
			else
				_rnn_nodes[idx].forward(cg, &_rnn_nodes[idx - 1], x[idx]);
		}
		for (int idx = 0; idx < _nSize; idx++) 
			_rnn_drop[idx].forward(cg, &_rnn_nodes[idx]);
	}
	inline void right2left_forward(Graph *cg, const vector<PNode>& x) {
		for (int idx = _nSize - 1; idx >= 0; idx--) {
			if (idx == _nSize - 1)
				_rnn_nodes[idx].forward(cg, &_bucket, x[idx]);
			else
				_rnn_nodes[idx].forward(cg, &_rnn_nodes[idx + 1], x[idx]);
		}
		for (int idx = 0; idx < _nSize; idx++) 
			_rnn_drop[idx].forward(cg, &_rnn_nodes[idx]);
	}
	
};

#endif