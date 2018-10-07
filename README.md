EGN3LDG
===========================
A lightweight neural network library based on dynamic graph for natural language processing
* The library supports only cpu, where tensors are implemented based on eigen.
* The library is fast if only cpu is available. (eigen can be speeded up by MKL)
* To use this library, just include the directory in your code and call it by "#include N3LDG.h"
 
## Installation:
Download and include the directory
### Prerequisitions:
#### EIGEN
You can get EIGEN from http://eigen.tuxfamily.org/index.php?title=Main_Page



If you have any problem, please send an email to mason.zms@gmail.com
## Examples:
Some examples are realeased at:
* https://github.com/zhangmeishan/NNTranSegmentor
* https://github.com/zhangmeishan/NNTranJSTagger
* https://github.com/zhangmeishan/N3LDGClassifier

## Histories:
* LibN3L:A Lightweight Package for Neural NLP. (Zhang et al., LREC 2016); initial
* LibN3L-2.0, auto-differentiation
* EGN3LDG, dynamic batching, much faster (if limited gpu resource like me, a good choice)
* N3LDG (enable gpu), in progress, just for interest since dynet exists.

## Authors:
Zhang Meishan, Yu Nan