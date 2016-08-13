#ifndef DROPOUT
#define DROPOUT

#include "MyLib.h"

struct PoolNode {
public:
	vector<Mat> _masks;
	Mat _y;
	Mat _ly;
	
public:
	PoolNode(){
		_masks.clear();
		_y.setZero();
		_ly.setZero();
	}

	inline void clearValue(){
		_masks.clear();
		_y.setZero();
		_ly.setZero();
	}

public:	

	virtual void forward(vector<PMat>& px) = 0;
	
	void backward(vector<PMat>& plx){
		if (plx.size() != _masks.size()){
			std::cout << "forward size does not equal backward size" << std::endl;
			return;
		}
		
		for (int i = 0; i < plx.size(); i++){
			if (plx[i]->size() == 0){
				*(plx[i]) = Mat::Zero(_ly.rows(), _ly.cols());
			}
			*(plx[i]) = plx[i]->array() + _ly.array() * _masks[i].array();
		}
	}
};

struct MaxPoolNode: PoolNode {

public:
	MaxPoolNode(){
	}
	
public:
	//Be careful that the row is the dim of input vector, and the col is the number of input vectors
	//Another point is that we change the input vectors directly.
	void forward(vector<PMat>& px) {
		if(px.size() == 0){
			std::cout << "empty inputs for max pooling" << std::endl;
			return;
		}
		
		int rows = px[0]->rows(), cols = px[0]->cols();
		_masks.resize(px.size());

		for (int i = 0; i < px.size(); ++i){
			if (px[i]->rows() != rows || px[i]->cols() != cols){
				std::cout << "input matrixes are not matched" << std::endl;
				_masks.clear();
				return;
			}
			_masks[i] = Mat::Zero(rows, cols);
		}
		
		for(int idx = 0; idx < rows; idx++){
			for(int idy = 0; idy < cols; idy++){
				int maxIndex = -1;
				for (int i = 0; i < px.size(); ++i){
					if (maxIndex == -1 || (*(px[i]))(idx, idy) >(*(px[maxIndex]))(idx, idy)){
						maxIndex = i;
					}
				}
				 _masks[maxIndex](idx, idy) = 1.0;
			}
		}
		
		_y = Mat::Zero(rows, cols);
		for (int i = 0; i < px.size(); ++i){
			_y = _y.array() + _masks[i].array() * px[i]->array();
		}
	}
		
};


#endif
