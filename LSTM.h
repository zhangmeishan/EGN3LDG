#ifndef LSTM
#define LSTM

#include "MyLib.h"
#include "Node.h"
#include "TriOP.h"
#include "BiOP.h"
#include "AtomicOP.h"
#include "Graph.h"

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

	inline int inDim(){
		return input.W3.inDim();
	}

	inline int outDim(){
		return input.W3.outDim();
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

	vector<DropNode> _hiddens_drop;

	Node bucket;

	LSTMParams* _param;


	bool _left2right;



public:
	LSTMBuilder(){
		clear();
	}

	~LSTMBuilder(){
		clear();
	}

public:
	inline void setParam(LSTMParams* paramInit, dtype dropout, bool left2right = true) {
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

			_hiddens_drop[idx].setDropValue(dropout);
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

		_hiddens_drop.resize(maxsize);
	}

	//whether vectors have been allocated
	inline bool empty(){
		return _hiddens.empty();
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
		_left2right = true;
		_param = NULL;
		bucket.clear();
		_nSize = 0;
		_inDim = 0;
		_outDim = 0;

		_hiddens_drop.clear();
	}

public:
	inline void forward(Graph *cg, const vector<PNode>& x, bool bTrain){
		if (x.size() == 0){
			std::cout << "empty inputs for lstm operation" << std::endl;
			return;
		}

		_nSize = x.size();
		if (x[0]->val.rows() != _inDim){
			std::cout << "input dim does not match for seg operation" << std::endl;
			return;
		}

		if (_left2right){
			left2right_forward(cg, x, bTrain);
		}
		else{
			right2left_forward(cg, x, bTrain);
		}

	}


protected:
	inline void left2right_forward(Graph *cg, const vector<PNode>& x, bool bTrain){
		for (int idx = 0; idx < _nSize; idx++){
			if (idx == 0){
				_inputgates[idx].forward(cg, &bucket, &bucket, x[idx]);

				_halfcells[idx].forward(cg, &bucket, x[idx]);

				_inputfilters[idx].forward(cg, &_halfcells[idx], &_inputgates[idx]);

				_cells[idx].forward(cg, &_inputfilters[idx], &bucket);

				_halfhiddens[idx].forward(cg, &_cells[idx]);

				_outputgates[idx].forward(cg, &bucket, &_cells[idx], x[idx]);

				_hiddens[idx].forward(cg, &_halfhiddens[idx], &_outputgates[idx]);

				_hiddens_drop[idx].forward(cg, &_hiddens[idx], bTrain);
			}
			else{
				_inputgates[idx].forward(cg, &_hiddens_drop[idx - 1], &_cells[idx - 1], x[idx]);

				_forgetgates[idx].forward(cg, &_hiddens_drop[idx - 1], &_cells[idx - 1], x[idx]);

				_halfcells[idx].forward(cg, &_hiddens_drop[idx - 1], x[idx]);

				_inputfilters[idx].forward(cg, &_halfcells[idx], &_inputgates[idx]);

				_forgetfilters[idx].forward(cg, &_cells[idx - 1], &_forgetgates[idx]);

				_cells[idx].forward(cg, &_inputfilters[idx], &_forgetfilters[idx]);

				_halfhiddens[idx].forward(cg, &_cells[idx]);

				_outputgates[idx].forward(cg, &_hiddens_drop[idx - 1], &_cells[idx], x[idx]);

				_hiddens[idx].forward(cg, &_halfhiddens[idx], &_outputgates[idx]);

				_hiddens_drop[idx].forward(cg, &_hiddens[idx], bTrain);
			}
		}
	}

	inline void right2left_forward(Graph *cg, const vector<PNode>& x, bool bTrain){
		for (int idx = _nSize - 1; idx >= 0; idx--){
			if (idx == _nSize - 1){
				_inputgates[idx].forward(cg, &bucket, &bucket, x[idx]);

				_halfcells[idx].forward(cg, &bucket, x[idx]);

				_inputfilters[idx].forward(cg, &_halfcells[idx], &_inputgates[idx]);

				_cells[idx].forward(cg, &_inputfilters[idx], &bucket);

				_halfhiddens[idx].forward(cg, &_cells[idx]);

				_outputgates[idx].forward(cg, &bucket, &_cells[idx], x[idx]);

				_hiddens[idx].forward(cg, &_halfhiddens[idx], &_outputgates[idx]);

				_hiddens_drop[idx].forward(cg, &_hiddens[idx], bTrain);
			}
			else{
				_inputgates[idx].forward(cg, &_hiddens_drop[idx + 1], &_cells[idx + 1], x[idx]);

				_forgetgates[idx].forward(cg, &_hiddens_drop[idx + 1], &_cells[idx + 1], x[idx]);

				_halfcells[idx].forward(cg, &_hiddens_drop[idx + 1], x[idx]);

				_inputfilters[idx].forward(cg, &_halfcells[idx], &_inputgates[idx]);

				_forgetfilters[idx].forward(cg, &_cells[idx + 1], &_forgetgates[idx]);

				_cells[idx].forward(cg, &_inputfilters[idx], &_forgetfilters[idx]);

				_halfhiddens[idx].forward(cg, &_cells[idx]);

				_outputgates[idx].forward(cg, &_hiddens_drop[idx + 1], &_cells[idx], x[idx]);

				_hiddens[idx].forward(cg, &_halfhiddens[idx], &_outputgates[idx]);

				_hiddens_drop[idx].forward(cg, &_hiddens[idx], bTrain);
			}
		}
	}


};


#endif
