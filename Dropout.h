#ifndef DROPOUT
#define DROPOUT

#include "Eigen/Dense"
#include "MyLib.h"

using namespace Eigen;

class DropNode {
public:
	Mat _mask;
	dtype _prob;
	
public:
	DropNode(){
		_mask.setOne();
		_prob = 1.0;
	}
	
	DropNode(duble prob){
		_mask.setOne();
		_prob = prob;
	}	
	
public:
	//Be careful that the row is the dim of input vector, and the col is the number of input vectors
	//Another point is that we change the input vectors directly.
	void forward(Mat& x) {
		_mask = Mat::One(x.rows(), x.cols());

		std::vector<int> indexes;
		for (int i = 0; i < x.rows(); ++i)
			indexes.push_back(i);
		
		int dropNum =   (int) (x.rows() * dropOut);
		
		for(int j = 0; j < x.cols(); j++){
			random_shuffle(indexes.begin(), indexes.end());
			for(int i = 0; i < dropNum; i++){
				_mask(i, j) = 0.0;
			}
		}
		
		x = x.array() * _mask.array();
	}
	
	
	void backward(Mat& lx){
		lx = lx.array() * _mask.array();
	}
	
}


#endif
