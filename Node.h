#ifndef BasicNode
#define BasicNode

#include "MyTensor.h"


// one Node means a vector
// the col should be 1, because we aimed for NLP only
struct Node {
public:
	Tensor1D val;
	Tensor1D loss;
public:
	int dim;
	int lock;  //node can backward only when lock = 0;

	int sid;
	bool lossed;
	bool executed;

//for dropout only
public:
	Tensor1D mask;
	bool usedrop;
	dtype dropvalue;
	
public:
	Node(){
		dim = 0;
		lock = 0;
		sid = rand();

		lossed = false;
		executed = false;
		usedrop = false;
		dropvalue = -1.0;
	}

public:
	virtual inline void clearValue(){
		val = 0;
		loss = 0;
		lock = 0;
		lossed = false;
		executed = false;
	}
	
	virtual inline void init(int dim, dtype dropOut, AlignedMemoryPool* mem = NULL){
		this->dim = dim;
		val.init(dim, mem);
		loss.init(dim, mem);
		if (dropOut >= 0 && dropOut <= 1){
			dropvalue = dropOut;
			usedrop = true;
			mask.init(dim, mem);
		}
		else{
			dropvalue = -1;
			usedrop = false;
		}
		
	}

	virtual inline void backward(){
	}

	virtual inline void unlock(){
	}

	inline void applydrop_forward(bool train){
		if (usedrop)
		{			
			if (train){
				mask = 1;
				std::vector<int> indexes;
				for (int i = 0; i < val.dim; ++i)
					indexes.push_back(i);

				int dropNum = (int)(val.dim * dropvalue);
				
				for (int i = 0; i < dropNum; i++){
					mask[indexes[i]] = 0.0;
				}
			}
			else{
				mask = 1.0 - dropvalue;
			}

			val.vec() = val.vec() * mask.vec();
		}
	}

	inline void applydrop_backward(){
		if (usedrop)
		{
			loss.vec() = loss.vec() * mask.vec();
		}
	}

	//virtual inline void forward(Graph *cg, ...) = 0

};

typedef  Node* PNode;




#endif
