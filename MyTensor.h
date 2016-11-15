#ifndef BasicTensor
#define BasicTensor


#include "Eigen/Dense"
#include <unsupported/Eigen/CXX11/Tensor>
#include "Mem.h"
#include "MyLib.h"

using namespace Eigen;

struct Tensor1D {
private:
	size_t memsize;	
	AlignedMemoryPool* mempool;
public:
	dtype *v;
	int dim;
	
	Tensor1D(){
		memsize = 0;
		dim = 0;
		v = NULL;
	}
	
	~Tensor1D(){
		memsize = 0;
		dim = 0;
		if(!mempool){
			delete[] v;
		}
		else{
			mempool = NULL;  //globally delete the allocated mem.
		}
		v = NULL;
	}
	
	//please call this function before using it really. must! must! must!
	//only this function allocates memories
	inline void init(int dim, AlignedMemoryPool* mem = NULL){
		this->dim = dim;
		v = NULL;
		if(mem != NULL){
			v = (dtype*)mem->allocate(dim * sizeof(dtype), memsize);
		}
		
		if(v){
			mempool = mem;
		}
		else {
			mempool = NULL;
			v = new dtype[dim];
			memsize = dim * sizeof(dtype);
		}
		zero();
	}
	
	inline void zero(){
		if(v)memset((void*)v, 0, memsize);;
	}
	
	const Mat mat() const {
		return Mat(v, dim, 1);
	}
	
	Mat mat() {
		return Mat(v, dim, 1);
	}	
	
	const Mat tmat() const {
		return Mat(v, 1, dim);
	}
	
	Mat tmat() {
		return Mat(v, 1, dim);
	}
	
	const Vec vec() const {
		return Vec(v, dim);
	}
	
	Vec vec() {
		return Vec(v, dim);
	}
	
	inline dtype& operator[](const int i) { 
	  return v[i];  // no boundary check?
	}

	inline const dtype& operator[](const int i) const{
		return v[i];  // no boundary check?
	}
	
	inline Tensor1D& operator=(const dtype &a) { // assign a to every element
	  for (int i = 0; i < dim; i++)
	    v[i] = a;
	  return *this;
	}
	
	inline Tensor1D& operator=(const vector<dtype> &a) { // assign a to every element
	  for (int i = 0; i < dim; i++)
	    v[i] = a[i];
	  return *this;
	}
	
	inline Tensor1D& operator=(const NRVec<dtype> &a) { // assign a to every element
	  for (int i = 0; i < dim; i++)
	    v[i] = a[i];
	  return *this;
	}
	
	inline Tensor1D& operator=(const Tensor1D &a) { // assign a to every element
	  for (int i = 0; i < dim; i++)
	    v[i] = a[i];
	  return *this;
	}
	
	inline void random(dtype bound){
		dtype min = -bound, max = bound;
		for (int i = 0; i < dim; i++){			
			v[i] =  (dtype(rand()) / RAND_MAX) * (max - min) + min;
		}
	}

	
};



struct Tensor2D {
private:
	size_t memsize;	
	AlignedMemoryPool* mempool;
public:
	dtype *v;
	int col, row, size;
	
	Tensor2D(){
		memsize = 0;
		col = row = 0;
		size = 0;
		v = NULL;
	}

	~Tensor2D(){
		memsize = 0;
		col = row = 0;
		size = 0;
		if(!mempool){
			delete[] v;
		}
		else{
			mempool = NULL;  //globally delete the allocated mem.
		}
		v = NULL;
	}
		
	//please call this function before using it really. must! must! must!
	//only this function allocates memories
	inline void init(int row, int col, AlignedMemoryPool* mem = NULL){
		this->col = col;
		this->row = row;
		size = col * row;
		if(mem){
			v = (dtype*)mem->allocate(size * sizeof(dtype), memsize);
		}
		
		if(v){
			mempool = mem;
		}
		else {
			mempool = NULL;
			v = new dtype[size];
			memsize = size * sizeof(dtype);
		}
		zero();
	}
	
	inline void zero(){
		if(v)memset((void*)v, 0, memsize);;
	}
	
	const Mat mat() const {
		return Mat(v, row, col);
	}
	
	Mat mat() {
		return Mat(v, row, col);
	}
	
	const Vec vec() const {
		return Vec(v, size);
	}
	
	Vec vec() {
		return Vec(v, size);
	}

	
	//use it carefully, first col, then row, because rows are allocated successively
	inline dtype* operator[](const int icol) {
	  return &(v[icol*row]);  // no boundary check?
	}
	
	inline const dtype* operator[](const int icol) const {
	  return &(v[icol*row]);  // no boundary check?
	}
	
	//use it carefully
	inline Tensor2D& operator=(const dtype &a) { // assign a to every element
	  for (int i = 0; i < size; i++)
	    v[i] = a;
	  return *this;
	}
	
	inline Tensor2D& operator=(const vector<dtype> &a) { // assign a to every element
	  for (int i = 0; i < size; i++)
	    v[i] = a[i];
	  return *this;
	}
	
	inline Tensor2D& operator=(const vector<vector<dtype> > &a) { // assign a to every element
		int offset = 0;
	  for (int i = 0; i < row; i++){
	  	for (int j = 0; j < col; j++) {
	    	v[offset] = a[i][j];
	    	offset++;
	    }
	  }
	  return *this;
	}
	
	inline Tensor2D& operator=(const NRMat<dtype> &a) { // assign a to every element
		int offset = 0;
	  for (int i = 0; i < row; i++){
	  	for (int j = 0; j < col; j++) {
	    	v[offset] = a[i][j];
	    	offset++;
	    }
	  }
	  return *this;
	}
	
	inline Tensor2D& operator=(const Tensor2D &a) { // assign a to every element
	  for (int i = 0; i < size; i++)
	    v[i] = a.v[i];
	  return *this;
	}
	
	inline void random(dtype bound){
		dtype min = -bound, max = bound;
		for (int i = 0; i < size; i++){			
			v[i] =  (dtype(rand()) / RAND_MAX) * (max - min) + min;
		}
	}

	inline void norm2one() { //every col is normalized (for embeddings only)
		dtype sum;
		for (int idx = 0; idx < col; idx++) {
			sum = 0.000001;
			for (int idy = 0; idy < row; idy++) {
				sum += (*this)[idx][idy] * (*this)[idx][idy];
			}
			dtype scale = sqrt(sum);
			for (int idy = 0; idy < row; idy++) {
				(*this)[idx][idy] /= scale;
			}
		}
	}

};


//useful functions
inline dtype fequal(const dtype& x) {
	return x;
}

inline dtype ftanh(const dtype& x) {
	return tanh(x);
}

inline dtype fsigmoid(const dtype& x) {
	return 1.0 / (1.0 + exp(-x));
}

inline dtype frelu(const dtype& x) {
	if (x <= 0) return 0;
	return x;
}

inline dtype fexp(const dtype& x) {
	return exp(x);
}

//derive function
inline dtype dequal(const dtype& x, const dtype& y) {
	return 1;
}

inline dtype dtanh(const dtype& x, const dtype& y) {
	return (1 + y) * (1 - y);
}

inline dtype dsigmoid(const dtype& x, const dtype& y) {
	return (1 - y) * y;
}

inline dtype drelu(const dtype& x, const dtype& y) {
	if (x <= 0) return 0;
	return 1;
}

inline dtype dexp(const dtype& x, const dtype& y) {
	return y;
}





#endif
