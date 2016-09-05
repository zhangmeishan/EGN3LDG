/*
 * AP1O.h
 *
 *  Created on: Jul 20, 2016
 *      Author: mszhang
 */

#ifndef COPUtil_H_
#define COPUtil_H_
#include "MyLib.h"

// all v values are larger than zero

inline int multiply(int v1, int v2){
	int s = v1 * v2;
	blong bs = (blong)(v1)* (blong)(v2);
	if (bs != s){
		return -1;
	}

	return s;
}

inline int multiply(int v1, int v2, int v3){
	int s = multiply(v1, v2);

	if (s < 0){
		return -1;
	}

	return multiply(s, v3);
}


inline int multiply(int v1, int v2, int v3, int v4){
	int s = multiply(v1, v2, v3);

	if (s < 0){
		return -1;
	}

	return multiply(s, v4);
}

inline int multiply(int v1, int v2, int v3, int v4, int v5){
	int s = multiply(v1, v2, v3, v4);

	if (s < 0){
		return -1;
	}

	return multiply(s, v5);
}


inline int multiply(int v1, int v2, int v3, int v4, int v5, int v6){
	int s = multiply(v1, v2, v3, v4, v5);

	if (s < 0){
		return -1;
	}

	return multiply(s, v6);
}

inline int multiply(int v1, int v2, int v3, int v4, int v5, int v6, int v7){
	int s = multiply(v1, v2, v3, v4, v5, v6);

	if (s < 0){
		return -1;
	}

	return multiply(s, v7);
}


#endif /* COPUtil_H_ */
