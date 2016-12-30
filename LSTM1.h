#ifndef LSTM1
#define LSTM1

#include "MyLib.h"
#include "Node.h"
#include "TriOP.h"
#include "BiOP.h"
#include "AtomicOP.h"
#include "Graph.h"

struct LSTM1Params {
	BiParams input;
	BiParams output;
	BiParams forget;
	BiParams cell;

	LSTM1Params() {
	}

	inline void exportAdaParams(ModelUpdate& ada) {
		input.exportAdaParams(ada);
		output.exportAdaParams(ada);
		forget.exportAdaParams(ada);
		cell.exportAdaParams(ada);
	}

	inline void initial(int nOSize, int nISize, AlignedMemoryPool* mem = NULL) {
		input.initial(nOSize, nOSize, nISize, true, mem);
		output.initial(nOSize, nOSize, nISize, true, mem);
		forget.initial(nOSize, nOSize, nISize, true, mem);
		cell.initial(nOSize, nOSize, nISize, true, mem);
	}

	inline int inDim(){
		return input.W2.inDim();
	}

	inline int outDim(){
		return input.W2.outDim();
	}

	inline void save(std::ofstream &os) const {
		input.save(os);
		output.save(os);
		forget.save(os);
		cell.save(os);
	}

	inline void load(std::ifstream &is, AlignedMemoryPool* mem = NULL) {
		input.load(is, mem);
		output.load(is, mem);
		forget.load(is, mem);
		cell.load(is, mem);
	}

};

// standard LSTM1 using tanh as activation function
// other conditions are not implemented unless they are clear
class LSTM1Builder{
public:
	int _nSize;
	int _inDim;
	int _outDim;

	vector<BiNode> _inputgates;
	vector<BiNode> _forgetgates;
	vector<BiNode> _halfcells;

	vector<PMultNode> _inputfilters;
	vector<PMultNode> _forgetfilters;

	vector<PAddNode> _cells;

	vector<BiNode> _outputgates;

	vector<TanhNode> _halfhiddens;

	vector<PMultNode> _hiddens;

	Node _bucket;

	LSTM1Params* _param;

	bool _left2right;

public:
	LSTM1Builder(){
		clear();
	}

	~LSTM1Builder(){
		clear();
	}

public:
	inline void init(LSTM1Params* paramInit, dtype dropout, bool left2right = true, AlignedMemoryPool* mem = NULL) {
		_param = paramInit;
		_inDim = _param->input.W2.inDim();
		_outDim = _param->input.W2.outDim();
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
				_inputgates[idx].forward(cg, &_bucket, x[idx]);

				_halfcells[idx].forward(cg, &_bucket, x[idx]);

				_inputfilters[idx].forward(cg, &_halfcells[idx], &_inputgates[idx]);

				_cells[idx].forward(cg, &_inputfilters[idx], &_bucket);

				_halfhiddens[idx].forward(cg, &_cells[idx]);

				_outputgates[idx].forward(cg, &_bucket, x[idx]);

				_hiddens[idx].forward(cg, &_halfhiddens[idx], &_outputgates[idx]);
			}
			else{
				_inputgates[idx].forward(cg, &_hiddens[idx - 1], x[idx]);

				_forgetgates[idx].forward(cg, &_hiddens[idx - 1], x[idx]);

				_halfcells[idx].forward(cg, &_hiddens[idx - 1], x[idx]);

				_inputfilters[idx].forward(cg, &_halfcells[idx], &_inputgates[idx]);

				_forgetfilters[idx].forward(cg, &_cells[idx - 1], &_forgetgates[idx]);

				_cells[idx].forward(cg, &_inputfilters[idx], &_forgetfilters[idx]);

				_halfhiddens[idx].forward(cg, &_cells[idx]);

				_outputgates[idx].forward(cg, &_hiddens[idx - 1], x[idx]);

				_hiddens[idx].forward(cg, &_halfhiddens[idx], &_outputgates[idx]);
			}
		}
	}

	inline void right2left_forward(Graph *cg, const vector<PNode>& x){
		for (int idx = _nSize - 1; idx >= 0; idx--){
			if (idx == _nSize - 1){
				_inputgates[idx].forward(cg, &_bucket, x[idx]);

				_halfcells[idx].forward(cg, &_bucket, x[idx]);

				_inputfilters[idx].forward(cg, &_halfcells[idx], &_inputgates[idx]);

				_cells[idx].forward(cg, &_inputfilters[idx], &_bucket);

				_halfhiddens[idx].forward(cg, &_cells[idx]);

				_outputgates[idx].forward(cg, &_bucket, x[idx]);

				_hiddens[idx].forward(cg, &_halfhiddens[idx], &_outputgates[idx]);
			}
			else{
				_inputgates[idx].forward(cg, &_hiddens[idx + 1], x[idx]);

				_forgetgates[idx].forward(cg, &_hiddens[idx + 1], x[idx]);

				_halfcells[idx].forward(cg, &_hiddens[idx + 1], x[idx]);

				_inputfilters[idx].forward(cg, &_halfcells[idx], &_inputgates[idx]);

				_forgetfilters[idx].forward(cg, &_cells[idx + 1], &_forgetgates[idx]);

				_cells[idx].forward(cg, &_inputfilters[idx], &_forgetfilters[idx]);

				_halfhiddens[idx].forward(cg, &_cells[idx]);

				_outputgates[idx].forward(cg, &_hiddens[idx + 1], x[idx]);

				_hiddens[idx].forward(cg, &_halfhiddens[idx], &_outputgates[idx]);
			}
		}
	}


};

class IncLSTM1Builder{
public:
	int _nSize;
	int _inDim;
	int _outDim;

  IncLSTM1Builder* _pPrev;

	BiNode _inputgate;
	BiNode _forgetgate;
	BiNode _halfcell;

	PMultNode _inputfilter;
	PMultNode _forgetfilter;

	PAddNode _cell;

	BiNode _outputgate;

	TanhNode _halfhidden;

	PMultNode _hidden;

	Node _bucket;

	LSTM1Params* _param;


public:
	IncLSTM1Builder(){
		clear();
	}

	~IncLSTM1Builder(){
		clear();
	}
	
	void clear(){
		_nSize = 0;
		_inDim = 0;
		_outDim = 0;
		_param = NULL;
    _pPrev = NULL;
	}

public:
	inline void init(LSTM1Params* paramInit, dtype dropout, AlignedMemoryPool* mem = NULL) {
		_param = paramInit;
		_inDim = _param->input.W2.inDim();
		_outDim = _param->input.W2.outDim();

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
	inline void forward(Graph *cg, PNode x, IncLSTM1Builder* prev = NULL){
		if (prev == NULL){
			_inputgate.forward(cg, &_bucket, x);

			_halfcell.forward(cg, &_bucket, x);

			_inputfilter.forward(cg, &_halfcell, &_inputgate);

			_cell.forward(cg, &_inputfilter, &_bucket);

			_halfhidden.forward(cg, &_cell);

			_outputgate.forward(cg, &_bucket, x);

			_hidden.forward(cg, &_halfhidden, &_outputgate);
			
			_nSize = 1;
		}
		else{
			_inputgate.forward(cg, &(prev->_hidden), x);

			_forgetgate.forward(cg, &(prev->_hidden), x);

			_halfcell.forward(cg, &(prev->_hidden), x);

			_inputfilter.forward(cg, &_halfcell, &_inputgate);

			_forgetfilter.forward(cg, &(prev->_cell), &_forgetgate);

			_cell.forward(cg, &_inputfilter, &_forgetfilter);

			_halfhidden.forward(cg, &_cell);

			_outputgate.forward(cg, &(prev->_hidden), x);

			_hidden.forward(cg, &_halfhidden, &_outputgate);
			
			_nSize = prev->_nSize + 1;
		}

    _pPrev = prev;
	}	

};

#endif
