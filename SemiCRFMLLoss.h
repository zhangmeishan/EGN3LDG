#ifndef _SemiCRFMLLOSS_H_
#define _SemiCRFMLLOSS_H_

#include "MyLib.h"
#include "Metric.h"
#include "Param.h"
#include "Node.h"

struct SemiCRFMLLoss{
public:
	int labelSize;
	vector<dtype> buffer;
	dtype eps;
	vector<int> maxLens;
	int maxLen;
	Param T; 


public:
	SemiCRFMLLoss(){
		labelSize = 0;
		buffer.clear();
		eps = 1e-20;
		maxLens.clear();
		maxLen = 0;
	}

	~SemiCRFMLLoss(){
		labelSize = 0;
		buffer.clear();
		maxLens.clear();
		maxLen = 0;
	}

public:
	inline void initial(const vector<int>& lens, int maxLength){
		labelSize = lens.size();
		maxLen = maxLength;
		maxLens.resize(labelSize);
		for (int idx = 0; idx < labelSize; idx++){
			maxLens[idx] = lens[idx];
		}
		T.initial(labelSize, labelSize); //not in the aligned memory pool
	}

	inline void exportAdaParams(ModelUpdate& ada){
		ada.addParam(&T);
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
		//int maxLength = x.ncols();

		for (int idx = 0; idx < seq_size; idx++) {
			for (int dist = 0; dist < seq_size - idx && dist < maxLen; dist++) {
				x[idx][dist]->lossed = true;
			}
		}


		// comute alpha values, only the above parts are valid
		NRMat3d<dtype> alpha(seq_size, maxLen, labelSize);
		NRMat3d<dtype> alpha_answer(seq_size, maxLen, labelSize);
		alpha = 0.0; alpha_answer = 0.0;
		for (int idx = 0; idx < seq_size; idx++) {
			for (int i = 0; i < labelSize; ++i) {
				for (int dist = 0; dist < seq_size - idx && dist < maxLens[i]; dist++) {
					// can be changed with probabilities in future work
					if (idx == 0) {
						alpha[idx][dist][i] = x[idx][dist]->val[i];
						alpha_answer[idx][dist][i] = x[idx][dist]->val[i] + log(answer[idx][dist][i] + eps);
					}
					else {
						buffer.clear();
						for (int j = 0; j < labelSize; ++j) {
							for (int prevdist = 1; prevdist <= idx && prevdist <= maxLens[j]; prevdist++) {
								buffer.push_back(T.val[j][i] + x[idx][dist]->val[i] + alpha[idx - prevdist][prevdist - 1][j]);
							}
						}
						alpha[idx][dist][i] = logsumexp(buffer);

						buffer.clear();
						for (int j = 0; j < labelSize; ++j) {
							for (int prevdist = 1; prevdist <= idx && prevdist <= maxLens[j]; prevdist++) {
								buffer.push_back(T.val[j][i] + x[idx][dist]->val[i] + alpha_answer[idx - prevdist][prevdist - 1][j]);
							}
						}
						alpha_answer[idx][dist][i] = logsumexp(buffer) + log(answer[idx][dist][i] + eps);
					}

				}
			}
		}

		// loss computation
		buffer.clear();
		for (int j = 0; j < labelSize; ++j) {
			for (int dist = 1; dist <= seq_size && dist <= maxLens[j]; dist++) {
				buffer.push_back(alpha[seq_size - dist][dist - 1][j]);
			}
		}
		dtype logZ = logsumexp(buffer);

		buffer.clear();
		for (int j = 0; j < labelSize; ++j) {
			for (int dist = 1; dist <= seq_size && dist <= maxLens[j]; dist++) {
				buffer.push_back(alpha_answer[seq_size - dist][dist - 1][j]);
			}
		}
		dtype logZ_answer = logsumexp(buffer);

		dtype cost = (logZ - logZ_answer) / batchsize;

		// comute belta values
		NRMat3d<dtype> belta(seq_size, maxLen, labelSize);
		NRMat3d<dtype> belta_answer(seq_size, maxLen, labelSize);
		belta = 0.0; belta_answer = 0.0;
		for (int idx = seq_size; idx > 0; idx--) {
			for (int i = 0; i < labelSize; ++i) {
				for (int dist = 1; dist <= idx && dist <= maxLens[i]; dist++) {
					if (idx == seq_size) {
						belta[idx - dist][dist - 1][i] = 0.0;
						belta_answer[idx - dist][dist - 1][i] = log(answer[idx - dist][dist - 1][i] + eps);
					}
					else {
						buffer.clear();
						for (int j = 0; j < labelSize; ++j) {
							for (int nextdist = 0; nextdist < seq_size - idx && nextdist < maxLens[j]; nextdist++) {
								buffer.push_back(T.val[i][j] + x[idx][nextdist]->val[j] + belta[idx][nextdist][j]);
							}
						}
						belta[idx - dist][dist - 1][i] = logsumexp(buffer);

						buffer.clear();
						for (int j = 0; j < labelSize; ++j) {
							for (int nextdist = 0; nextdist < seq_size - idx && nextdist < maxLens[j]; nextdist++) {
								buffer.push_back(T.val[i][j] + x[idx][nextdist]->val[j] + belta_answer[idx][nextdist][j]);
							}
						}
						belta_answer[idx - dist][dist - 1][i] = logsumexp(buffer) + log(answer[idx - dist][dist - 1][i] + eps);
					}
				}
			}
		}

		//compute margins
		NRMat3d<dtype> margin(seq_size, maxLen, labelSize);
		NRMat3d<dtype> margin_answer(seq_size, maxLen, labelSize);
		NRMat<dtype> trans(labelSize, labelSize);
		NRMat<dtype> trans_answer(labelSize, labelSize);
		margin = 0.0; margin_answer = 0.0;
		trans = 0.0; trans_answer = 0.0;
		for (int idx = 0; idx < seq_size; idx++) {
			for (int i = 0; i < labelSize; ++i) {
				for (int dist = 0; dist < seq_size - idx && dist < maxLens[i]; dist++) {
					margin[idx][dist][i] = exp(alpha[idx][dist][i] + belta[idx][dist][i] - logZ);
					margin_answer[idx][dist][i] = exp(alpha_answer[idx][dist][i] + belta_answer[idx][dist][i] - logZ_answer);

					if (idx > 0) {
						for (int j = 0; j < labelSize; ++j) {
							for (int prevdist = 1; prevdist <= idx && prevdist <= maxLens[j]; prevdist++) {
								dtype logvalue = alpha[idx - prevdist][prevdist - 1][j] + x[idx][dist]->val[i] + T.val[j][i] + belta[idx][dist][i] - logZ;
								trans[j][i] += exp(logvalue);
								logvalue = alpha_answer[idx - prevdist][prevdist - 1][j] + x[idx][dist]->val[i] + T.val[j][i] + belta_answer[idx][dist][i] - logZ_answer;
								trans_answer[j][i] += exp(logvalue);
							}
						}
					}
				}
			}
		}

		//compute transition matrix losses
		for (int i = 0; i < labelSize; ++i) {
			for (int j = 0; j < labelSize; ++j) {
				T.grad[i][j] += trans[i][j] - trans_answer[i][j];
			}
		}

		//compute loss
		for (int idx = 0; idx < seq_size; idx++) {
			for (int i = 0; i < labelSize; ++i) {
				for (int dist = 0; dist < seq_size - idx && dist < maxLens[i]; dist++) {
					if (margin_answer[idx][dist][i] > 0.5){
						eval.overall_label_count++;
						if (margin[idx][dist][i] > 0.5){
							eval.correct_label_count++;
						}
					}
					x[idx][dist]->loss[i] = (margin[idx][dist][i] - margin_answer[idx][dist][i]) / batchsize;
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

		NRMat3d<dtype> maxScores(seq_size, maxLen, labelSize);
		NRMat3d<int> maxLastLabels(seq_size, maxLen, labelSize);
		NRMat3d<int> maxLastStarts(seq_size, maxLen, labelSize);
		NRMat3d<int> maxLastDists(seq_size, maxLen, labelSize);

		maxScores = 0.0; maxLastLabels = -2; 
		maxLastStarts = -2; maxLastDists = -2;
		for (int idx = 0; idx < seq_size; idx++) {
			for (int i = 0; i < labelSize; ++i) {
				for (int dist = 0; dist < seq_size - idx && dist < maxLens[i]; dist++) {
					// can be changed with probabilities in future work
					if (idx == 0) {
						maxScores[idx][dist][i] = x[idx][dist]->val[i];
						maxLastLabels[idx][dist][i] = -1;
						maxLastStarts[idx][dist][i] = -1;
						maxLastDists[idx][dist][i] = -1;
					}
					else {
						int maxLastLabel = -1;
						int maxLastStart = -1;
						int LastDist = -1;
						dtype maxscore = 0.0;
						for (int j = 0; j < labelSize; ++j) {
							for (int prevdist = 1; prevdist <= idx && prevdist <= maxLens[j]; prevdist++) {
								dtype curScore = T.val[j][i] + x[idx][dist]->val[i] + maxScores[idx - prevdist][prevdist - 1][j];
								if (maxLastLabel == -1 || curScore > maxscore){
									maxLastLabel = j;
									maxLastStart = idx - prevdist;
									LastDist = prevdist - 1;
									maxscore = curScore;
								}
							}
						}
						maxScores[idx][dist][i] = maxscore;
						maxLastLabels[idx][dist][i] = maxLastLabel;
						maxLastStarts[idx][dist][i] = maxLastStart;
						maxLastDists[idx][dist][i] = LastDist;
					}

				}
			}
		}

		// below zero denotes no such segment
		y.resize(seq_size, maxLen);
		y = -1;
		dtype maxFinalScore = 0.0;
		int maxFinalLabel = -1;
		int maxFinalStart = -1;
		int maxFinalDist = -1;
		for (int j = 0; j < labelSize; ++j) {
			for (int dist = 1; dist <= seq_size && dist <= maxLens[j]; dist++) {
				dtype curScore = maxScores[seq_size - dist][dist - 1][j];
				if (maxFinalLabel == -1 || curScore > maxFinalScore){
					maxFinalLabel = j;
					maxFinalStart = seq_size - dist;
					maxFinalDist = dist - 1;
					maxFinalScore = curScore;
				}
			}
		}

		y[maxFinalStart][maxFinalDist] = maxFinalLabel;

		while (1){
			int lastLabel = maxLastLabels[maxFinalStart][maxFinalDist][maxFinalLabel];
			int lastStart = maxLastStarts[maxFinalStart][maxFinalDist][maxFinalLabel];
			int lastDist = maxLastDists[maxFinalStart][maxFinalDist][maxFinalLabel];

			if (lastStart < 0){
				assert(maxFinalStart == 0);
				break;
			}

			y[lastStart][lastDist] = lastLabel;
			maxFinalLabel = lastLabel;
			maxFinalStart = lastStart;
			maxFinalDist = lastDist;
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
		NRMat3d<dtype> alpha(seq_size, maxLen, labelSize);
		NRMat3d<dtype> alpha_answer(seq_size, maxLen, labelSize);
		alpha = 0.0; alpha_answer = 0.0;
		for (int idx = 0; idx < seq_size; idx++) {
			for (int i = 0; i < labelSize; ++i) {
				for (int dist = 0; dist < seq_size - idx && dist < maxLens[i]; dist++) {
					// can be changed with probabilities in future work
					if (idx == 0) {
						alpha[idx][dist][i] = x[idx][dist]->val[i];
						alpha_answer[idx][dist][i] = x[idx][dist]->val[i] + log(answer[idx][dist][i] + eps);
					}
					else {
						buffer.clear();
						for (int j = 0; j < labelSize; ++j) {
							for (int prevdist = 1; prevdist <= idx && prevdist <= maxLens[j]; prevdist++) {
								buffer.push_back(T.val[j][i] + x[idx][dist]->val[i] + alpha[idx - prevdist][prevdist - 1][j]);
							}
						}
						alpha[idx][dist][i] = logsumexp(buffer);

						buffer.clear();
						for (int j = 0; j < labelSize; ++j) {
							for (int prevdist = 1; prevdist <= idx && prevdist <= maxLens[j]; prevdist++) {
								buffer.push_back(T.val[j][i] + x[idx][dist]->val[i] + alpha_answer[idx - prevdist][prevdist - 1][j]);
							}
						}
						alpha_answer[idx][dist][i] = logsumexp(buffer) + log(answer[idx][dist][i] + eps);
					}

				}
			}
		}

		// loss computation
		buffer.clear();
		for (int j = 0; j < labelSize; ++j) {
			for (int dist = 1; dist <= seq_size && dist <= maxLens[j]; dist++) {
				buffer.push_back(alpha[seq_size - dist][dist - 1][j]);
			}
		}
		dtype logZ = logsumexp(buffer);

		buffer.clear();
		for (int j = 0; j < labelSize; ++j) {
			for (int dist = 1; dist <= seq_size && dist <= maxLens[j]; dist++) {
				buffer.push_back(alpha_answer[seq_size - dist][dist - 1][j]);
			}
		}
		dtype logZ_answer = logsumexp(buffer);

		return (logZ - logZ_answer) / batchsize;
	}
};


#endif /* _SemiCRFMLLOSS_H_ */
