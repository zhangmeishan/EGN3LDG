/*
 * Utils.h
 *
 *  Created on: Jul 20, 2016
 *      Author: mason
 *
 *  Some of the functions are borrowed from
 *     https://github.com/oir/deep-recurrent
 */

#ifndef _LIBN3L_UTILS_H_
#define _LIBN3L_UTILS_H_

#include <iostream>
#include <fstream>
#include <sstream>
#include <cstdio>
#include <algorithm>
#include <cmath>
#include <string>
#include <vector>
#include <map>
#include <set>
#include <iterator>
#include <cassert>
#include "Eigen/Dense"
#include "MyLib.h"

#define uint unsigned int

using namespace Eigen;
using namespace std;

inline istream& operator>>(istream& s, Mat& m) {
	for (uint i = 0; i < m.rows(); i++)
		for (uint j = 0; j < m.cols(); j++)
			s >> m(i, j);
	return s;
}

inline istream& operator>>(istream& s, VectorXd& m) {
	for (uint i = 0; i < m.size(); i++)
		s >> m(i);
	return s;
}

// this is used for randomly initializing an Eigen matrix
inline dtype urand(dtype bound) {
	dtype min = -bound, max = bound;
	return (dtype(rand()) / RAND_MAX) * (max - min) + min;
}

inline dtype ftanh(dtype x) {
	return tanh(x);
}

inline dtype fsigmoid(dtype x) {
	return 1.0 / (1.0 + exp(-x));
}

inline dtype frelu(dtype x) {
	if (x <= 0) return 0;
	return x;
}

inline dtype frelu_deri(dtype x) {
	if (x <= 0) return 0;
	return 1;
}

inline Mat smaxentp(const Mat &y, const Mat &r) {
	return y - r;
}

inline Mat relu(const Mat &x) {
	return x.unaryExpr(ptr_fun(frelu));
}

inline Mat relu_deri(const Mat &x, const Mat &y) {
	return y.unaryExpr(ptr_fun(frelu_deri));
}

inline Mat tanh(const Mat &x) {
	return x.unaryExpr(ptr_fun(ftanh));
}

inline Mat tanh_deri(const Mat &x, const Mat &y) {
	return (1.0 + y.array()) * (1.0 - y.array());
}

inline Mat sigmoid(const Mat &x) {
	return x.unaryExpr(ptr_fun(fsigmoid));
}

inline Mat sigmoid_deri(const Mat &x, const Mat &y) {
	return (1 - y.array()) * y.array();
}


inline Mat equal(const Mat &x) {
	return x;
}

inline Mat equal_deri(const Mat &x, const Mat &y) {
	return Mat::Ones(y.rows(), y.cols());
}

inline void random(NRMat<dtype>& nr, dtype bound = 0.01){
	int rows = nr.nrows(),cols = nr.ncols();
	for(int idx = 0; idx < rows; idx++){
		for(int idy = 0; idy < cols; idy++){
			nr[idx][idy] = urand(0.01);
		}
	}
}

inline void random(Mat& m, dtype bound = 0.01){
	int rows = m.rows(),cols = m.cols();
	for(int idx = 0; idx < rows; idx++){
		for(int idy = 0; idy < cols; idy++){
			m.coeffRef(idx, idy) = urand(0.01);
		}
	}
}

inline bool nonzeroloss(const NRVec<dtype>& loss){
	int dim = loss.size();
	dtype t;
	for(int idx = 0; idx < dim; idx++){
		t = loss[idx];
		if(t >= 1e-8 || t <= -1e-8){
			return true;
		}
	}
	return false;
}

inline bool nonzeroloss(const Mat& loss){
	int dim = loss.size();
	dtype t;
	for(int idx = 0; idx < dim; idx++){
		t = loss(idx);
		if(t >= 1e-8 || t <= -1e-8){
			return true;
		}
	}
	return false;
}

/*write by yunan*/
inline void assign(Mat &m, const NRMat<dtype>& wnr)
{
	int rows = wnr.nrows();
	int cols = wnr.ncols();
	m.resize(rows, cols);
	for(int i = 0; i < rows; i++)	
		for(int j = 0; j < cols; j++)
			m(i, j) = wnr[i][j];
}

/*write by yunan*/
inline void norm2one(Mat &w, int idx) {
	dtype sum = 0.000001;
	for (int idy = 0; idy < w.cols(); idy++) {
		sum += w(idx,idy) * w(idx,idy);
	}
	dtype scale = sqrt(sum);
	for (int idy = 0; idy < w.cols(); idy++)
		w(idx,idy) /= scale;
}

inline void norm2one(NRMat<dtype> &w, int idx) {
	dtype sum = 0.000001;
	for (int idy = 0; idy < w.ncols(); idy++) {
		sum += w[idx][idy] * w[idx][idy];
	}
	dtype scale = sqrt(sum);
	for (int idy = 0; idy < w.ncols(); idy++)
		w[idx][idy] = w[idx][idy] / scale;
}

inline dtype str2double(const string& s) {
	istringstream i(s);
	dtype x;
	if (!(i >> x))
		return 0;
	return x;
}

// index of max in a vector
inline uint argmax(const VectorXd& x) {
	dtype max = x(0);
	uint maxi = 0;
	for (uint i = 1; i < x.size(); i++) {
		if (x(i) > max) {
			max = x(i);
			maxi = i;
		}
	}
	return maxi;
}



#endif
