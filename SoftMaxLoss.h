#ifndef _SOFTMAXLOSS_H_
#define _SOFTMAXLOSS_H_

#include "MyLib.h"
#include "Metric.h"
#include <Eigen/Dense>

using namespace Eigen;

inline double softmax_loss(const MatrixXd &output, const vector<int> &answer,
		MatrixXd &loutput, Metric & eval, int batchsize = 1) {
	int dim2 = output.rows();
	int odim2 = loutput.rows();
	int labelsize = answer.size();

	if (labelsize != odim2 || dim2 != odim2) {
		std::cerr << "softmax_loss error: dim size invalid" << std::endl;
	}

	MatrixXd scores(dim2, 1);
	scores.setZero();
	loutput.setZero();
	double cost = 0.0;

	int optLabel = -1;
	for (int i = 0; i < dim2; ++i) {
		if (answer[i] >= 0) {
			if (optLabel < 0 || output(i, 0) > output(optLabel, 0))
				optLabel = i;
		}
	}

	double sum1 = 0, sum2 = 0, maxScore = output(optLabel, 0);
	for (int i = 0; i < dim2; ++i) {
		scores(i, 0) = -1e10;
		if (answer[i] >= 0) {
			scores(i, 0) = exp(output(i, 0) - maxScore);
			if (answer[i] == 1)
				sum1 += scores(i, 0);
			sum2 += scores(i, 0);
		}
	}
	cost += (log(sum2) - log(sum1)) / batchsize;
	if (answer[optLabel] == 1)
		eval.correct_label_count++;
	eval.overall_label_count++;

	for (int i = 0; i < dim2; ++i) {
		if (answer[i] >= 0) {
			loutput(i, 0) = (scores(i, 0) / sum2 - answer[i]) / batchsize;
		}
	}

	return cost;
}

inline void softmax_predict(const MatrixXd& output, int& result) {
	int dim2 = output.rows();

	int optLabel = -1;
	for (int i = 0; i < dim2; ++i) {
		if (optLabel < 0 || output(i, 0) > output(optLabel, 0))
			optLabel = i;
	}
	result = optLabel;
}

//
inline double softmax_cost(const MatrixXd& output, const vector<int>& answer,
		int batchsize = 1) {
	int dim2 = output.rows();
	int labelsize = answer.size();

	if (labelsize != dim2) {
		std::cerr << "softmax_cost error: dim size invalid" << std::endl;
	}

	MatrixXd scores(dim2, 1);
	scores.setZero();
	double cost = 0.0;

	int optLabel = -1;
	for (int i = 0; i < dim2; ++Mi) {
		if (answer[i] >= 0) {
			if (optLabel < 0 || output(i, 0) > output(optLabel, 0))
				optLabel = i;
		}
	}

	double sum1 = 0.0, sum2 = 0.0, maxScore = output(optLabel, 0);
	for (int i = 0; i < dim2; ++i) {
		scores(i, 0) = -1e10;
		if (answer[i] >= 0) {
			scores(i, 0) = exp(output(i, 0) - maxScore);
			if (answer[i] == 1)
				sum1 += scores(i, 0);
			sum2 += scores(i, 0);
		}
	}
	cost += (log(sum2) - log(sum1)) / batchsize;
	return cost;
}

#endif /* _SOFTMAXLOSS_H_ */
