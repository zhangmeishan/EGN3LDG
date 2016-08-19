#ifndef LSTM
#define LSTM

#include "MyLib.h"
#include "Node.h"
#include "TriOP.h"
#include "BiOP.h"
#include "AtomicOP.h"

struct LSTMParams {
	TriParams input;
	TriParams output;
	TriParams forget;
	BiParams cell;

	LSTMParams() {
	}

	inline void exportAdaParams(ModelUpdate& ada) {
		input.exportAdaParams(ada);
		output.exportAdaParams(ada);
		forget.exportAdaParams(ada);
		cell.exportAdaParams(ada);
	}

	inline void initial(int nOSize, int nISize, int seed = 0) {
		input.initial(nOSize, nOSize, nOSize, nISize, true, seed);
		output.initial(nOSize, nOSize, nOSize, nISize, true, seed + 1);
		forget.initial(nOSize, nOSize, nOSize, nISize, true, seed + 2);
		cell.initial(nOSize, nOSize, nISize, true, seed + 3);
	}
};

// standard LSTM using tanh as activation function
// other conditions are not implemented unless they are clear
class LSTMBuilder : NodeBuilder{
public:
	int _nSize;
	int _inDim;
	int _outDim;

	vector<TriNode> _inputgates;
	vector<TriNode> _forgetgates;
	vector<BiNode> _halfcells;

	vector<PMultNode> _inputfilters;
	vector<PMultNode> _forgetfilters;

	vector<PAddNode> _cells;

	vector<TriNode> _outputgates;

	vector<TanhNode> _halfhiddens;

	vector<PMultNode> _hiddens;

	Node bucket;

	LSTMParams* _param;

	vector<PNode> _execNodes;

	bool _left2right;

public:
	LSTMBuilder(){

	}

	~LSTMBuilder(){

	}

public:
	inline void setParam(LSTMParams* paramInit, bool left2right = true) {
		_param = paramInit;
		_inDim = _param->input.W3.inDim();
		_outDim = _param->input.W3.outDim();

		for (int idx = 0; idx < _inputgates.size(); idx++){
			_inputgates[idx].setParam(&_param->input);
			_forgetgates[idx].setParam(&_param->forget);
			_outputgates[idx].setParam(&_param->output);
			_halfcells[idx].setParam(&_param->cell);
			_inputgates[idx].setFunctions(&sigmoid, &sigmoid_deri);
			_forgetgates[idx].setFunctions(&sigmoid, &sigmoid_deri);
			_outputgates[idx].setFunctions(&sigmoid, &sigmoid_deri);
			_halfcells[idx].setFunctions(&tanh, &tanh_deri);
		}

		_left2right = left2right;
		bucket.val = Mat::Zero(_outDim, 1);
	}

	inline void resize(int maxsize){
		_inputgates.resize(maxsize);
		_forgetgates.resize(maxsize);
		_halfcells.resize(maxsize);
		_inputfilters.resize(maxsize);
		_forgetfilters.resize(maxsize);
		_cells.resize(maxsize);
		_outputgates.resize(maxsize);
		_halfhiddens.resize(maxsize);
		_hiddens.resize(maxsize);
	}

	inline void clear(){
		_inputgates.clear();
		_forgetgates.clear();
		_halfcells.clear();
		_inputfilters.clear();
		_forgetfilters.clear();
		_cells.clear();
		_outputgates.clear();
		_halfhiddens.clear();
		_hiddens.clear();
	}

public:
	inline void forward(const vector<PNode>& x){
		if (x.size() == 0){
			std::cout << "empty inputs for lstm operation" << std::endl;
			return;
		}

		_nSize = x.size();
		if (x[0]->val.rows() != _inDim){
			std::cout << "input dim does not match for seg operation" << std::endl;
			return;
		}
		_execNodes.clear();

		if (_left2right){
			left2right_forward(x);
			//faked_forward(x);
		}
		else{
			right2left_forward(x);
			//faked_forward(x);
		}
	}

	inline void traverseNodes(vector<PNode> &exec){
		for (int idx = 0; idx < _execNodes.size(); idx++){
			exec.push_back(_execNodes[idx]);
		}
	}

protected:

	inline void faked_forward(const vector<PNode>& x){
		for (int idx = 0; idx < _nSize; idx++){
			_inputgates[idx].forward(&bucket, &bucket, x[idx]);
			_execNodes.push_back(&_inputgates[idx]);

			
			_halfcells[idx].forward(&bucket, x[idx]);
			_execNodes.push_back(&_halfcells[idx]);

			_inputfilters[idx].forward(&_halfcells[idx], &_inputgates[idx]);
			_execNodes.push_back(&_inputfilters[idx]);

			
			//_cells[idx].forward(&_inputfilters[idx], &bucket);
			//_execNodes.push_back(&_cells[idx]);

			_halfhiddens[idx].forward(&_inputfilters[idx]);
			_execNodes.push_back(&_halfhiddens[idx]);
	
			//_outputgates[idx].forward(&bucket, &_cells[idx], x[idx]);
			//_execNodes.push_back(&_outputgates[idx]);

			//_hiddens[idx].forward(&_cells[idx], &_outputgates[idx]);
			//_execNodes.push_back(&_hiddens[idx]);
			
		}
	}

	inline void left2right_forward(const vector<PNode>& x){
		for (int idx = 0; idx < _nSize; idx++){
			if (idx == 0){
				_inputgates[idx].forward(&bucket, &bucket, x[idx]);
				_execNodes.push_back(&_inputgates[idx]);

				_halfcells[idx].forward(&bucket, x[idx]);
				_execNodes.push_back(&_halfcells[idx]);

				_inputfilters[idx].forward(&_halfcells[idx], &_inputgates[idx]);
				_execNodes.push_back(&_inputfilters[idx]);

				_cells[idx].forward(&_inputfilters[idx], &bucket);
				_execNodes.push_back(&_cells[idx]);

				_halfhiddens[idx].forward(&_cells[idx]);
				_execNodes.push_back(&_halfhiddens[idx]);

				_outputgates[idx].forward(&bucket, &_cells[idx], x[idx]);
				_execNodes.push_back(&_outputgates[idx]);

				_hiddens[idx].forward(&_halfhiddens[idx], &_outputgates[idx]);
				_execNodes.push_back(&_hiddens[idx]);
			}
			else{
				_inputgates[idx].forward(&_hiddens[idx - 1], &_cells[idx - 1], x[idx]);
				_execNodes.push_back(&_inputgates[idx]);

				_outputgates[idx].forward(&_hiddens[idx - 1], &_cells[idx - 1], x[idx]);
				_execNodes.push_back(&_outputgates[idx]);

				_halfcells[idx].forward(&_hiddens[idx - 1], x[idx]);
				_execNodes.push_back(&_halfcells[idx]);

				_inputfilters[idx].forward(&_halfcells[idx], &_inputgates[idx]);
				_execNodes.push_back(&_inputfilters[idx]);

				_forgetfilters[idx].forward(&_cells[idx - 1], &_outputgates[idx]);
				_execNodes.push_back(&_forgetfilters[idx]);

				_cells[idx].forward(&_inputfilters[idx], &_forgetfilters[idx]);
				_execNodes.push_back(&_cells[idx]);

				_halfhiddens[idx].forward(&_cells[idx]);
				_execNodes.push_back(&_halfhiddens[idx]);

				_outputgates[idx].forward(&_hiddens[idx - 1], &_cells[idx], x[idx]);
				_execNodes.push_back(&_outputgates[idx]);

				_hiddens[idx].forward(&_halfhiddens[idx], &_outputgates[idx]);
				_execNodes.push_back(&_hiddens[idx]);
			}
		}
	}

	inline void right2left_forward(const vector<PNode>& x){
		for (int idx = _nSize - 1; idx >= 0; idx--){
			if (idx == _nSize - 1){
				_inputgates[idx].forward(&bucket, &bucket, x[idx]);
				_execNodes.push_back(&_inputgates[idx]);

				_halfcells[idx].forward(&bucket, x[idx]);
				_execNodes.push_back(&_halfcells[idx]);

				_inputfilters[idx].forward(&_halfcells[idx], &_inputgates[idx]);
				_execNodes.push_back(&_inputfilters[idx]);

				_cells[idx].forward(&_inputfilters[idx], &bucket);
				_execNodes.push_back(&_cells[idx]);

				_halfhiddens[idx].forward(&_cells[idx]);
				_execNodes.push_back(&_halfhiddens[idx]);

				_outputgates[idx].forward(&bucket, &_cells[idx], x[idx]);
				_execNodes.push_back(&_outputgates[idx]);

				_hiddens[idx].forward(&_halfhiddens[idx], &_outputgates[idx]);
				_execNodes.push_back(&_hiddens[idx]);
			}
			else{
				_inputgates[idx].forward(&_hiddens[idx + 1], &_cells[idx + 1], x[idx]);
				_execNodes.push_back(&_inputgates[idx]);

				_outputgates[idx].forward(&_hiddens[idx + 1], &_cells[idx + 1], x[idx]);
				_execNodes.push_back(&_outputgates[idx]);

				_halfcells[idx].forward(&_hiddens[idx + 1], x[idx]);
				_execNodes.push_back(&_halfcells[idx]);

				_inputfilters[idx].forward(&_halfcells[idx], &_inputgates[idx]);
				_execNodes.push_back(&_inputfilters[idx]);

				_forgetfilters[idx].forward(&_cells[idx + 1], &_outputgates[idx]);
				_execNodes.push_back(&_forgetfilters[idx]);

				_cells[idx].forward(&_inputfilters[idx], &_forgetfilters[idx]);
				_execNodes.push_back(&_cells[idx]);

				_halfhiddens[idx].forward(&_cells[idx]);
				_execNodes.push_back(&_halfhiddens[idx]);

				_outputgates[idx].forward(&_hiddens[idx + 1], &_cells[idx], x[idx]);
				_execNodes.push_back(&_outputgates[idx]);

				_hiddens[idx].forward(&_halfhiddens[idx], &_outputgates[idx]);
				_execNodes.push_back(&_hiddens[idx]);
			}
		}
	}


};


#endif
