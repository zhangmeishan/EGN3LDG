#ifndef _SOFTMAXLOSS_H_
#define _SOFTMAXLOSS_H_

#include "MyLib.h"
#include "Metric.h"
#include <Eigen/Dense>

using namespace Eigen;

struct SoftMaxLoss{
public:
	inline dtype loss(PNode x, const vector<dtype> &answer, Metric& eval, int batchsize = 1){
		int nDim = x->dim;
		int labelsize = answer.size();
		if (labelsize != nDim) {
			std::cerr << "softmax_loss error: dim size invalid" << std::endl;
			return -1.0;
		}
		if(x->loss.size() == 0){
			x->loss = Mat::Zero(nDim, 1);
		}

		Mat scores = Mat::Zero(nDim, 1);

		dtype cost = 0.0;

		int optLabel = -1;
		for (int i = 0; i < nDim; ++i) {
			if (answer[i] >= 0) {
				if (optLabel < 0 || x->val(i, 0) > x->val(optLabel, 0))
					optLabel = i;
			}
		}

		dtype sum1 = 0, sum2 = 0, maxScore = x->val(optLabel, 0);
		for (int i = 0; i < nDim; ++i) {
			scores(i, 0) = -1e10;
			if (answer[i] >= 0) {
				scores(i, 0) = exp(x->val(i, 0) - maxScore);
				if (answer[i] == 1)
					sum1 += scores(i, 0);
				sum2 += scores(i, 0);
			}
		}
		cost += (log(sum2) - log(sum1)) / batchsize;
		if (answer[optLabel] == 1)
			eval.correct_label_count++;
		eval.overall_label_count++;

		for (int i = 0; i < nDim; ++i) {
			if (answer[i] >= 0) {
				x->loss(i, 0) = (scores(i, 0) / sum2 - answer[i]) / batchsize;
			}
		}

		x->lock--;
		return cost;

	}

	inline void predict(PNode x, int& y){
		int nDim = x->dim;

		int optLabel = -1;
		for (int i = 0; i < nDim; ++i) {
			if (optLabel < 0 || x->val(i, 0) >  x->val(optLabel, 0))
				optLabel = i;
		}
		y = optLabel;
	}

	inline dtype cost(PNode x, const vector<dtype> &answer, int batchsize = 1){
		int nDim = x->dim;
		int labelsize = answer.size();
		if (labelsize != nDim) {
			std::cerr << "softmax_loss error: dim size invalid" << std::endl;
			return -1.0;
		}

		Mat scores = Mat::Zero(nDim, 1);

		dtype cost = 0.0;

		int optLabel = -1;
		for (int i = 0; i < nDim; ++i) {
			if (answer[i] >= 0) {
				if (optLabel < 0 || x->val(i, 0) > x->val(optLabel, 0))
					optLabel = i;
			}
		}

		dtype sum1 = 0, sum2 = 0, maxScore = x->val(optLabel, 0);
		for (int i = 0; i < nDim; ++i) {
			scores(i, 0) = -1e10;
			if (answer[i] >= 0) {
				scores(i, 0) = exp(x->val(i, 0) - maxScore);
				if (answer[i] == 1)
					sum1 += scores(i, 0);
				sum2 += scores(i, 0);
			}
		}
		cost += (log(sum2) - log(sum1)) / batchsize;
		return cost;
	}

};


#endif /* _SOFTMAXLOSS_H_ */
