/*
 * Utils.h
 *
 *  Created on: Jul 20, 2016
 *      Author: mason
 *
 *  Some of the functions are borrowed from
 *     https://github.com/oir/deep-recurrent
 */

#ifndef _UTILS_H_
#define _UTILS_H_

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

inline istream& operator>>(istream& s, MatrixXd& m) {
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


inline MatrixXd softmax(const MatrixXd &x) {
	RowVectorXd m = x.colwise().maxCoeff();
	MatrixXd t = (x.rowwise() - m).array().exp();
	return t.array().rowwise() / t.colwise().sum().array();
}

inline MatrixXd smaxentp(const MatrixXd &y, const MatrixXd &r) {
	return y - r;
}

inline MatrixXd relu(const MatrixXd &x) {
	return x.array().max(0);
}

inline MatrixXd tanh(const MatrixXd &x) {
	return x.array().tan();
}

inline MatrixXd equal(const MatrixXd &x) {
	return x;
}

inline MatrixXd tanh_deri(const MatrixXd &x, const MatrixXd &y) {
	return (1 + y.array()) * (1 - y.array());
}

inline MatrixXd equal_deri(const MatrixXd &x, const MatrixXd &y) {
	return y;
}

/*write by yunan*/
inline void assign(MatrixXd &m, const NRMat<double>& wnr)
{
	int rows = wnr.nrows();
	int cols = wnr.ncols();
	m.resize(rows, cols);
	for(int i = 0; i < rows; i++)	
		for(int j = 0; j < cols; j++)
			m(i, j) = wnr[i][j];
}

/*write by yunan*/
inline void norm2one(MatrixXd &w, int idx) {
	double sum = 0.000001;
	for (int idy = 0; idy < w.cols(); idy++) {
		sum += w(idx,idy) * w(idx,idy);
	}
	double scale = sqrt(sum);
	for (int idy = 0; idy < w.cols(); idy++)
		w(idx,idy) /= scale;
}

inline double str2double(const string& s) {
	istringstream i(s);
	double x;
	if (!(i >> x))
		return 0;
	return x;
}

// index of max in a vector
inline uint argmax(const VectorXd& x) {
	double max = x(0);
	uint maxi = 0;
	for (uint i = 1; i < x.size(); i++) {
		if (x(i) > max) {
			max = x(i);
			maxi = i;
		}
	}
	return maxi;
}

// this is used for randomly initializing an Eigen matrix
inline double urand(double dummy) {
	double min = -0.01, max = 0.01;
	return (double(rand()) / RAND_MAX) * (max - min) + min;
}


#endif
