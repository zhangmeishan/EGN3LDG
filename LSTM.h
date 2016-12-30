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

	inline void initial(int nOSize, int nISize, AlignedMemoryPool* mem = NULL) {
		input.initial(nOSize, nOSize, nOSize, nISize, true, mem);
		output.initial(nOSize, nOSize, nOSize, nISize, true, mem);
		forget.initial(nOSize, nOSize, nOSize, nISize, true, mem);
		cell.initial(nOSize, nOSize, nISize, true, mem);
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
class LSTMBuilder{
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

	Node _bucket;

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
	inline void init(LSTMParams* paramInit, dtype dropout, bool left2right = true, AlignedMemoryPool* mem = NULL) {
		_param = paramInit;
		_inDim = _param->input.W3.inDim();
		_outDim = _param->input.W3.outDim();
		
		int maxsize = _inputgates.size();
		for (int idx = 0; idx < maxsize; idx++){
			_inputgates[idx].setParam(&_param->input);
			_forgetgates[idx].setParam(&_param->forget);
			_outputgates[idx].setParam(&_param->output);
			_halfcells[idx].setParam(&_param->cell);
			_inputgates[idx].setFunctions(&fsigmoid, &dsigmoid);
			_forgetgates[idx].setFunctions(&fsigmoid, &dsigmoid);
			_outputgates[idx].setFunctions(&fsigmoid, &dsigmoid);
			_halfcells[idx].setFunctions(&ftanh, &dtanh);
		}
		_left2right = left2right;
		
		for (int idx = 0; idx < maxsize; idx++){
			_inputgates[idx].init(_outDim, -1, mem);
			_forgetgates[idx].init(_outDim, -1, mem);
			_halfcells[idx].init(_outDim, -1, mem);
			_inputfilters[idx].init(_outDim, -1, mem);
			_forgetfilters[idx].init(_outDim, -1, mem);
			_cells[idx].init(_outDim, -1, mem);
			_outputgates[idx].init(_outDim, -1, mem);
			_halfhiddens[idx].init(_outDim, -1, mem);
			_hiddens[idx].init(_outDim, dropout, mem);			
		}
				
		_bucket.init(_outDim, -1, mem);
		_bucket.set_bucket();
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
		
		_left2right = true;
		_param = NULL;
		_nSize = 0;
		_inDim = 0;
		_outDim = 0;
	}

public:
	inline void forward(Graph *cg, const vector<PNode>& x){
		if (x.size() == 0){
			std::cout << "empty inputs for lstm operation" << std::endl;
			return;
		}

		_nSize = x.size();
		if (x[0]->val.dim != _inDim){
			std::cout << "input dim does not match for seg operation" << std::endl;
			return;
		}

		if (_left2right){
			left2right_forward(cg, x);
		}
		else{
			right2left_forward(cg, x);
		}

	}


protected:
	inline void left2right_forward(Graph *cg, const vector<PNode>& x){
		for (int idx = 0; idx < _nSize; idx++){
			if (idx == 0){
				_inputgates[idx].forward(cg, &_bucket, &_bucket, x[idx]);

				_halfcells[idx].forward(cg, &_bucket, x[idx]);

				_inputfilters[idx].forward(cg, &_halfcells[idx], &_inputgates[idx]);

				_cells[idx].forward(cg, &_inputfilters[idx], &_bucket);

				_halfhiddens[idx].forward(cg, &_cells[idx]);

				_outputgates[idx].forward(cg, &_bucket, &_cells[idx], x[idx]);

				_hiddens[idx].forward(cg, &_halfhiddens[idx], &_outputgates[idx]);
			}
			else{
				_inputgates[idx].forward(cg, &_hiddens[idx - 1], &_cells[idx - 1], x[idx]);

				_forgetgates[idx].forward(cg, &_hiddens[idx - 1], &_cells[idx - 1], x[idx]);

				_halfcells[idx].forward(cg, &_hiddens[idx - 1], x[idx]);

				_inputfilters[idx].forward(cg, &_halfcells[idx], &_inputgates[idx]);

				_forgetfilters[idx].forward(cg, &_cells[idx - 1], &_forgetgates[idx]);

				_cells[idx].forward(cg, &_inputfilters[idx], &_forgetfilters[idx]);

				_halfhiddens[idx].forward(cg, &_cells[idx]);

				_outputgates[idx].forward(cg, &_hiddens[idx - 1], &_cells[idx], x[idx]);

				_hiddens[idx].forward(cg, &_halfhiddens[idx], &_outputgates[idx]);
			}
		}
	}

	inline void right2left_forward(Graph *cg, const vector<PNode>& x){
		for (int idx = _nSize - 1; idx >= 0; idx--){
			if (idx == _nSize - 1){
				_inputgates[idx].forward(cg, &_bucket, &_bucket, x[idx]);

				_halfcells[idx].forward(cg, &_bucket, x[idx]);

				_inputfilters[idx].forward(cg, &_halfcells[idx], &_inputgates[idx]);

				_cells[idx].forward(cg, &_inputfilters[idx], &_bucket);

				_halfhiddens[idx].forward(cg, &_cells[idx]);

				_outputgates[idx].forward(cg, &_bucket, &_cells[idx], x[idx]);

				_hiddens[idx].forward(cg, &_halfhiddens[idx], &_outputgates[idx]);
			}
			else{
				_inputgates[idx].forward(cg, &_hiddens[idx + 1], &_cells[idx + 1], x[idx]);

				_forgetgates[idx].forward(cg, &_hiddens[idx + 1], &_cells[idx + 1], x[idx]);

				_halfcells[idx].forward(cg, &_hiddens[idx + 1], x[idx]);

				_inputfilters[idx].forward(cg, &_halfcells[idx], &_inputgates[idx]);

				_forgetfilters[idx].forward(cg, &_cells[idx + 1], &_forgetgates[idx]);

				_cells[idx].forward(cg, &_inputfilters[idx], &_forgetfilters[idx]);

				_halfhiddens[idx].forward(cg, &_cells[idx]);

				_outputgates[idx].forward(cg, &_hiddens[idx + 1], &_cells[idx], x[idx]);

				_hiddens[idx].forward(cg, &_halfhiddens[idx], &_outputgates[idx]);
			}
		}
	}
};

class IncLSTMBuilder{
public:
	int _nSize;
	int _inDim;
	int _outDim;

	TriNode _inputgate;
	TriNode _forgetgate;
	BiNode _halfcell;

	PMultNode _inputfilter;
	PMultNode _forgetfilter;

	PAddNode _cell;

	TriNode _outputgate;

	TanhNode _halfhidden;

	PMultNode _hidden;

	Node _bucket;

	LSTMParams* _param;


public:
	IncLSTMBuilder(){
		clear();
	}

	~IncLSTMBuilder(){
		clear();
	}
	
	void clear(){
		_nSize = 0;
		_inDim = 0;
		_outDim = 0;
		_param = NULL;
	}

public:
	inline void init(LSTMParams* paramInit, dtype dropout, AlignedMemoryPool* mem = NULL) {
		_param = paramInit;
		_inDim = _param->input.W3.inDim();
		_outDim = _param->input.W3.outDim();

		_inputgate.setParam(&_param->input);
		_forgetgate.setParam(&_param->forget);
		_outputgate.setParam(&_param->output);
		_halfcell.setParam(&_param->cell);
		_inputgate.setFunctions(&fsigmoid, &dsigmoid);
		_forgetgate.setFunctions(&fsigmoid, &dsigmoid);
		_outputgate.setFunctions(&fsigmoid, &dsigmoid);
		_halfcell.setFunctions(&ftanh, &dtanh);

		_inputgate.init(_outDim, -1, mem);
		_forgetgate.init(_outDim, -1, mem);
		_halfcell.init(_outDim, -1, mem);
		_inputfilter.init(_outDim, -1, mem);
		_forgetfilter.init(_outDim, -1, mem);
		_cell.init(_outDim, -1, mem);
		_outputgate.init(_outDim, -1, mem);
		_halfhidden.init(_outDim, -1, mem);
		_hidden.init(_outDim, dropout, mem);		

		_bucket.init(_outDim, -1, mem);
		_bucket.set_bucket();
	}


public:
	inline void forward(Graph *cg, PNode x, IncLSTMBuilder* prev = NULL){
		if (prev == NULL){
			_inputgate.forward(cg, &_bucket, &_bucket, x);

			_halfcell.forward(cg, &_bucket, x);

			_inputfilter.forward(cg, &_halfcell, &_inputgate);

			_cell.forward(cg, &_inputfilter, &_bucket);

			_halfhidden.forward(cg, &_cell);

			_outputgate.forward(cg, &_bucket, &_cell, x);

			_hidden.forward(cg, &_halfhidden, &_outputgate);			
			
			_nSize = 1;
		}
		else{
			_inputgate.forward(cg, &prev->_hidden, &prev->_cell, x);

			_forgetgate.forward(cg, &prev->_hidden, &prev->_cell, x);

			_halfcell.forward(cg, &prev->_hidden, x);

			_inputfilter.forward(cg, &_halfcell, &_inputgate);

			_forgetfilter.forward(cg, &prev->_cell, &_forgetgate);

			_cell.forward(cg, &_inputfilter, &_forgetfilter);

			_halfhidden.forward(cg, &_cell);

			_outputgate.forward(cg, &prev->_hidden, &_cell, x);

			_hidden.forward(cg, &_halfhidden, &_outputgate);
			
			_nSize = prev->_nSize + 1;
		}
	}	

};


#endif
