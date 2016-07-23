/*
 * Utils.h
 *
 *  Created on: Jul 20, 2016
 *      Author: mason
 *
 *  Some of the functions are borrowed from
 *     https://github.com/oir/deep-recurrent
 */

#ifndef UTILS_LIBN3L
#define UTILS_LIBN3L

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

#define uint unsigned int

using namespace Eigen;
using namespace std;

istream& operator>>(istream& s, MatrixXd& m) {
	for (uint i = 0; i < m.rows(); i++)
		for (uint j = 0; j < m.cols(); j++)
			s >> m(i, j);
	return s;
}

istream& operator>>(istream& s, VectorXd& m) {
	for (uint i = 0; i < m.size(); i++)
		s >> m(i);
	return s;
}


MatrixXd softmax(const MatrixXd &x) {
	RowVectorXd m = x.colwise().maxCoeff();
	MatrixXd t = (x.rowwise() - m).array().exp();
	return t.array().rowwise() / t.colwise().sum().array();
}

MatrixXd smaxentp(const MatrixXd &y, const MatrixXd &r) {
	return y - r;
}

MatrixXd relu(const MatrixXd &x) {
	return x.array().max(0);
}

MatrixXd tanh(const MatrixXd &x) {
	return x.array().tan();
}

MatrixXd equal(const MatrixXd &x) {
	return x;
}

MatrixXd tanh_deri(const MatrixXd &x, const MatrixXd &y) {
	return (1 + y.array()) * (1 - y.array());
}

MatrixXd equal_deri(const MatrixXd &x, const MatrixXd &y) {
	return y;
}


double str2double(const string& s) {
	istringstream i(s);
	double x;
	if (!(i >> x))
		return 0;
	return x;
}

// index of max in a vector
uint argmax(const VectorXd& x) {
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
double urand(double dummy) {
	double min = -0.01, max = 0.01;
	return (double(rand()) / RAND_MAX) * (max - min) + min;
}


#endif
