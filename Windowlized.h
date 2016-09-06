#ifndef WINDOWLIZED
#define WINDOWLIZED

#include "MyLib.h"
#include "Node.h"
#include "Concat.h"
#include "Graph.h"

class WindowBuilder : NodeBuilder{
public:
	int _context;
	int _window;
	int _nSize;
	int _inDim;
	int _outDim;

	vector<ConcatNode> _outputs;

	Node bucket;

public:
	WindowBuilder(){
		clear();
	}

	~WindowBuilder(){
		clear();
	}


	inline void resize(int maxsize){
		_outputs.resize(maxsize);
	}

	inline void clear(){
		_outputs.clear();
		_context = 0;
		_window = 0;
		_nSize = 0;
		_inDim = 0;
		_outDim = 0;
	}


	inline void setContext(int context){
		_context = context;
		_window = 2 * _context + 1;
	}

public:	

	inline void forward(Graph *cg, const vector<PNode>& x){
		if (x.size() == 0){
			std::cout << "empty inputs for windowlized operation" << std::endl;
			return;
		}

		_nSize = x.size();
		_inDim = x[0]->val.rows();
		_outDim = _window * _inDim;

		bucket.clear();
		bucket.val = Mat::Zero(_inDim, 1);

		vector<PNode> in_nodes(_window);
		for (int idx = 0; idx < _nSize; idx++){
			int offset = 0;
			in_nodes[offset++] = x[idx];
			for (int j = 1; j <= _context; j++){
				in_nodes[offset++] = idx - j >= 0 ? x[idx - j] : &bucket;
				in_nodes[offset++] = idx + j < _nSize ? x[idx + j] : &bucket;
			}
			_outputs[idx].forward(cg, in_nodes);
		}
	}

};


#endif