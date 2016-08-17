#ifndef _Semi0CRFMLLOSS_H_
#define _Semi0CRFMLLOSS_H_

#include "MyLib.h"
#include "Metric.h"
#include <Eigen/Dense>
#include "Param.h"

using namespace Eigen;

struct Semi0CRFMLLoss{
public:
	int labelSize;
	vector<dtype> buffer;
	dtype eps;
	vector<int> maxLen;

public:
	Semi0CRFMLLoss(){
		labelSize = 0;
		buffer.clear();
		eps = 1e-20;
		maxLen.clear();
	}

	~Semi0CRFMLLoss(){
		labelSize = 0;
		buffer.clear();
		maxLen.clear();
	}

public:
	inline void initial(const vector<int>& lens, int seed = 0){
		labelSize = lens.size();
		maxLen.resize(labelSize);
		for (int idx = 0; idx < labelSize; idx++){
			maxLen[idx] = lens[idx];
		}
	}


public:
	// 
	inline dtype loss(const NRMat<PNode>& x, const vector<vector<vector<dtype> > >& answer, Metric& eval, int batchsize = 1){
		/*
		assert(x.nrows() > 0 && x.ncols() == x.nrows() && x.nrows() == answer.size() && x.nrows() == answer[0].size());
		int nDim = x[0][0]->dim;
		if (labelSize != nDim || labelSize != answer[0][0].size()) {
			std::cerr << "semi crf max likelihood loss error: label size invalid" << std::endl;
			return -1.0;
		}*/

		int seq_size = x.nrows();

		for (int idx = 0; idx < seq_size; idx++) {
			for (int idy = idx; idy < seq_size; idy++) {
				if (x[idx][idy]->loss.size() == 0){
					x[idx][idy]->loss = Mat::Zero(labelSize, 1);
				}
			}
		}


		// comute alpha values, only the above parts are valid
		NRMat3d<dtype> alpha(seq_size, seq_size, labelSize);
		NRMat3d<dtype> alpha_answer(seq_size, seq_size, labelSize);
		alpha = 0.0; alpha_answer = 0.0;
		for (int idx = 0; idx < seq_size; idx++) {
			for (int i = 0; i < labelSize; ++i) {
				int maxEnd = seq_size;
				if (maxLen[i] > 0 && maxEnd > idx + maxLen[i]) maxEnd = idx + maxLen[i];
				for (int idy = idx; idy < maxEnd; idy++) {
					// can be changed with probabilities in future work
					if (idx == 0) {
						alpha[idx][idy][i] = x[idx][idy]->val(i, 0);
						alpha_answer[idx][idy][i] = x[idx][idy]->val(i, 0) + log(answer[idx][idy][i] + eps);
					}
					else {
						buffer.clear();
						for (int j = 0; j < labelSize; ++j) {
							int minStart = 0;
							if (maxLen[j] > 0 && minStart < idx - maxLen[j]) minStart = idx - maxLen[j];
							for (int idz = minStart; idz < idx; idz++){
								buffer.push_back(x[idx][idy]->val(i, 0) + alpha[idz][idx - 1][j]);
							}
						}
						alpha[idx][idy][i] = logsumexp(buffer);

						buffer.clear();
						for (int j = 0; j < labelSize; ++j) {
							int minStart = 0;
							if (maxLen[j] > 0 && minStart < idx - maxLen[j]) minStart = idx - maxLen[j];
							for (int idz = minStart; idz < idx; idz++){
								buffer.push_back(x[idx][idy]->val(i, 0) + alpha_answer[idz][idx - 1][j]);
							}
						}
						alpha_answer[idx][idy][i] = logsumexp(buffer) + log(answer[idx][idy][i] + eps);
					}

				}
			}
		}

		// loss computation
		buffer.clear();
		for (int j = 0; j < labelSize; ++j) {
			int minStart = 0;
			if (maxLen[j] > 0 && minStart < seq_size - maxLen[j]) minStart = seq_size - maxLen[j];
			for (int idz = minStart; idz < seq_size; idz++){
				buffer.push_back(alpha[idz][seq_size - 1][j]);
			}
		}
		dtype logZ = logsumexp(buffer);

		buffer.clear();
		for (int j = 0; j < labelSize; ++j) {
			int minStart = 0;
			if (maxLen[j] > 0 && minStart < seq_size - maxLen[j]) minStart = seq_size - maxLen[j];
			for (int idz = minStart; idz < seq_size; idz++){
				buffer.push_back(alpha_answer[idz][seq_size - 1][j]);
			}
		}
		dtype logZ_answer = logsumexp(buffer);
		dtype cost = (logZ - logZ_answer) / batchsize;

		// comute belta values
		NRMat3d<dtype> belta(seq_size, seq_size, labelSize);
		NRMat3d<dtype> belta_answer(seq_size, seq_size, labelSize);
		belta = 0.0; belta_answer = 0.0;
		for (int idy = seq_size - 1; idy >= 0; idy--) {
			for (int i = 0; i < labelSize; ++i) {
				int minStart = 0;
				if (maxLen[i] > 0 && minStart < idy + 1 - maxLen[i]) minStart = idy + 1 - maxLen[i];
				for (int idx = minStart; idx <= idy; idx++){
					if (idy == seq_size - 1) {
						belta[idx][idy][i] = 0.0;
						belta_answer[idx][idy][i] = log(answer[idx][idy][i] + eps);
					}
					else {
						buffer.clear();
						for (int j = 0; j < labelSize; ++j) {
							int maxEnd = seq_size;
							if (maxLen[j] > 0 && maxEnd > idy + 1 + maxLen[j]) maxEnd = idy + 1 + maxLen[j];
							for (int idz = idy + 1; idz < maxEnd; idz++){
								buffer.push_back(x[idy + 1][idz]->val(j, 0) + belta[idy + 1][idz][j]);
							}
						}
						belta[idx][idy][i] = logsumexp(buffer);

						buffer.clear();
						for (int j = 0; j < labelSize; ++j) {
							int maxEnd = seq_size;
							if (maxLen[j] > 0 && maxEnd > idy + 1 + maxLen[j]) maxEnd = idy + 1 + maxLen[j];
							for (int idz = idy + 1; idz < maxEnd; idz++){
								buffer.push_back(x[idy + 1][idz]->val(j, 0) + belta_answer[idy + 1][idz][j]);
							}
						}
						belta_answer[idx][idy][i] = logsumexp(buffer) + log(answer[idx][idy][i] + eps);
					}
				}
			}
		}

		//compute margins
		NRMat3d<dtype> margin(seq_size, seq_size, labelSize);
		NRMat3d<dtype> margin_answer(seq_size, seq_size, labelSize);
		margin = 0.0; margin_answer = 0.0;
		for (int idx = 0; idx < seq_size; idx++) {
			for (int i = 0; i < labelSize; ++i) {
				int maxEnd = seq_size;
				if (maxLen[i] > 0 && maxEnd > idx + maxLen[i]) maxEnd = idx + maxLen[i];
				for (int idy = idx; idy < maxEnd; idy++) {
					margin[idx][idy][i] = exp(alpha[idx][idy][i] + belta[idx][idy][i] - logZ);
					margin_answer[idx][idy][i] = exp(alpha_answer[idx][idy][i] + belta_answer[idx][idy][i] - logZ_answer);
				}
			}
		}

		//compute loss
		for (int idx = 0; idx < seq_size; idx++) {
			for (int i = 0; i < labelSize; ++i) {
				int maxEnd = seq_size;
				if (maxLen[i] > 0 && maxEnd > idx + maxLen[i]) maxEnd = idx + maxLen[i];
				for (int idy = idx; idy < maxEnd; idy++) {
					if (margin_answer[idx][idy][i] > 0.5){
						eval.overall_label_count++;
						if (margin[idx][idy][i] > 0.5){
							eval.correct_label_count++;
						}
					}
					x[idx][idy]->loss(i, 0) = (margin[idx][idy][i] - margin_answer[idx][idy][i]) / batchsize;
				}
			}
		}

		return cost;
	}

	//viterbi decode algorithm
	inline void predict(const NRMat<PNode>& x, NRMat<int>& y){
		/*
		assert(x.nrows() > 0 && x.ncols() == x.nrows());
		int nDim = x[0][0]->dim;
		if (labelSize != nDim) {
			std::cerr << "semi crf max likelihood predict error: labelSize size invalid" << std::endl;
			return;
		}*/

		int seq_size = x.nrows();

		NRMat3d<dtype> maxScores(seq_size, seq_size, labelSize);
		NRMat3d<int> maxLastLabels(seq_size, seq_size, labelSize);
		NRMat3d<int> maxLastStarts(seq_size, seq_size, labelSize);
		NRMat3d<int> maxLastEnds(seq_size, seq_size, labelSize);

		maxScores = 0.0; maxLastLabels = -2; 
		maxLastStarts = -2; maxLastEnds = -2;
		for (int idx = 0; idx < seq_size; idx++) {
			for (int i = 0; i < labelSize; ++i) {
				int maxEnd = seq_size;
				if (maxLen[i] > 0 && maxEnd > idx + maxLen[i]) maxEnd = idx + maxLen[i];
				for (int idy = idx; idy < maxEnd; idy++) {
					// can be changed with probabilities in future work
					if (idx == 0) {
						maxScores[idx][idy][i] = x[idx][idy]->val(i, 0);
						maxLastLabels[idx][idy][i] = -1;
						maxLastStarts[idx][idy][i] = -1;
						maxLastEnds[idx][idy][i] = -1;
					}
					else {
						int maxLastLabel = -1;
						int maxLastStart = -1;
						int maxLastEnd = -1;
						dtype maxscore = 0.0;
						for (int j = 0; j < labelSize; ++j) {
							int minStart = 0;
							if (maxLen[j] > 0 && minStart < idx - maxLen[j]) minStart = idx - maxLen[j];
							for (int idz = minStart; idz < idx; idz++){
								dtype curScore = maxScores[idz][idx - 1][j];
								if (maxLastLabel == -1 || curScore > maxscore){
									maxLastLabel = j;
									maxLastStart = idz;
									maxLastEnd = idx - 1;
								}
							}
						}
						maxScores[idx][idy][i] = x[idx][idy]->val(i, 0) + maxscore;
						maxLastLabels[idx][idy][i] = maxLastLabel;
						maxLastStarts[idx][idy][i] = maxLastStart;
						maxLastEnds[idx][idy][i] = maxLastEnd;
					}

				}
			}
		}

		// below zero denotes no such segment
		y.resize(seq_size, seq_size);
		y = -1;

		dtype maxFinalScore = 0.0;
		int maxFinalLabel = -1;
		int maxFinalStart = -1;
		int maxFinalEnd = -1;
		for (int j = 0; j < labelSize; ++j) {
			int minStart = 0;
			if (maxLen[j] > 0 && minStart < seq_size - maxLen[j]) minStart = seq_size - maxLen[j];
			for (int idz = minStart; idz < seq_size; idz++){
				dtype curScore = maxScores[idz][seq_size - 1][j];
				if (maxFinalLabel == -1 || curScore > maxFinalScore){
					maxFinalLabel = j;
					maxFinalStart = idz;
					maxFinalEnd = seq_size - 1;
					maxFinalScore = curScore;
				}
			}
		}

		y[maxFinalStart][maxFinalEnd] = maxFinalLabel;

		while (1){
			int lastLabel = maxLastLabels[maxFinalStart][maxFinalEnd][maxFinalLabel];
			int lastStart = maxLastStarts[maxFinalStart][maxFinalEnd][maxFinalLabel];
			int lastEnd = maxLastEnds[maxFinalStart][maxFinalEnd][maxFinalLabel];

			if (lastStart < 0){
				assert(maxFinalStart == 0);
				break;
			}

			y[lastStart][lastEnd] = lastLabel;
			maxFinalLabel = lastLabel;
			maxFinalStart = lastStart;
			maxFinalEnd = lastEnd;
		}
	}

	inline dtype cost(const NRMat<PNode>& x, const vector<vector<vector<dtype> > >& answer, int batchsize = 1){
		/*
		assert(x.nrows() > 0 && x.ncols() == x.nrows() && x.nrows() == answer.size() && x.nrows() == answer[0].size());
		int nDim = x[0][0]->dim;
		if (labelSize != nDim || labelSize != answer[0][0].size()) {
			std::cerr << "semi crf max likelihood loss error: label size invalid" << std::endl;
			return -1.0;
		}*/

		int seq_size = x.nrows();
		// comute alpha values, only the above parts are valid
		NRMat3d<dtype> alpha(seq_size, seq_size, labelSize);
		NRMat3d<dtype> alpha_answer(seq_size, seq_size, labelSize);
		alpha = 0.0; alpha_answer = 0.0;
		for (int idx = 0; idx < seq_size; idx++) {
			for (int i = 0; i < labelSize; ++i) {
				int maxEnd = seq_size;
				if (maxLen[i] > 0 && maxEnd > idx + maxLen[i]) maxEnd = idx + maxLen[i];
				for (int idy = idx; idy < maxEnd; idy++) {
					// can be changed with probabilities in future work
					if (idx == 0) {
						alpha[idx][idy][i] = x[idx][idy]->val(i, 0);
						alpha_answer[idx][idy][i] = x[idx][idy]->val(i, 0) + log(answer[idx][idy][i] + eps);
					}
					else {
						buffer.clear();
						for (int j = 0; j < labelSize; ++j) {
							int minStart = 0;
							if (maxLen[j] > 0 && minStart < idx - maxLen[j]) minStart = idx - maxLen[j];
							for (int idz = minStart; idz < idx; idz++){
								buffer.push_back(x[idx][idy]->val(i, 0) + alpha[idz][idx - 1][j]);
							}
						}
						alpha[idx][idy][i] = logsumexp(buffer);

						buffer.clear();
						for (int j = 0; j < labelSize; ++j) {
							int minStart = 0;
							if (maxLen[j] > 0 && minStart < idx - maxLen[j]) minStart = idx - maxLen[j];
							for (int idz = minStart; idz < idx; idz++){
								buffer.push_back(x[idx][idy]->val(i, 0) + alpha_answer[idz][idx - 1][j]);
							}
						}
						alpha_answer[idx][idy][i] = logsumexp(buffer) + log(answer[idx][idy][i] + eps);
					}

				}
			}
		}

		// loss computation
		buffer.clear();
		for (int j = 0; j < labelSize; ++j) {
			int minStart = 0;
			if (maxLen[j] > 0 && minStart < seq_size - maxLen[j]) minStart = seq_size - maxLen[j];
			for (int idz = minStart; idz < seq_size; idz++){
				buffer.push_back(alpha[idz][seq_size - 1][j]);
			}
		}
		dtype logZ = logsumexp(buffer);

		buffer.clear();
		for (int j = 0; j < labelSize; ++j) {
			int minStart = 0;
			if (maxLen[j] > 0 && minStart < seq_size - maxLen[j]) minStart = seq_size - maxLen[j];
			for (int idz = minStart; idz < seq_size; idz++){
				buffer.push_back(alpha_answer[idz][seq_size - 1][j]);
			}
		}
		dtype logZ_answer = logsumexp(buffer);

		return (logZ - logZ_answer) / batchsize;
	}
};


#endif /* _Semi0CRFMLLOSS_H_ */
