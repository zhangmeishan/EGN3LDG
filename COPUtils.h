/*
 * COPUtils.h
 *
 *  Created on: Jul 20, 2016
 *      Author: mszhang
 */

#ifndef COPUtil_H_
#define COPUtil_H_
#include "MyLib.h"

const static int maxCapacity = 1<<30;

inline void hash_combine(size_t& seed, const int& v){
	seed ^= v + 0x9e3779b9 + (seed << 6) + (seed >> 2);
	//seed = 131 * seed + v;
}

inline size_t hash_func(const int& v1, const int& v2){
	static size_t curIndex;
	curIndex = 0;
	hash_combine(curIndex, v1);
	hash_combine(curIndex, v2);
	return curIndex;
}

inline size_t hash_func(const int& v1, const int& v2, const int& v3){
	static size_t curIndex;
	curIndex = 0;
	hash_combine(curIndex, v1);
	hash_combine(curIndex, v2);
	hash_combine(curIndex, v3);
	return curIndex;
}

inline size_t hash_func(const int& v1, const int& v2, const int& v3, const int& v4){
	static size_t curIndex;
	curIndex = 0;
	hash_combine(curIndex, v1);
	hash_combine(curIndex, v2);
	hash_combine(curIndex, v3);
	hash_combine(curIndex, v4);
	return curIndex;
}

inline size_t hash_func(const int& v1, const int& v2, const int& v3, const int& v4, const int& v5){
	static size_t curIndex;
	curIndex = 0;
	hash_combine(curIndex, v1);
	hash_combine(curIndex, v2);
	hash_combine(curIndex, v3);
	hash_combine(curIndex, v4);
	hash_combine(curIndex, v5);
	return curIndex;
}

inline size_t hash_func(const int& v1, const int& v2, const int& v3, const int& v4, const int& v5, const int& v6){
	static size_t curIndex;
	curIndex = 0;
	hash_combine(curIndex, v1);
	hash_combine(curIndex, v2);
	hash_combine(curIndex, v3);
	hash_combine(curIndex, v4);
	hash_combine(curIndex, v5);
	hash_combine(curIndex, v6);
	return curIndex;
}

inline size_t hash_func(const int& v1, const int& v2, const int& v3, const int& v4, const int& v5, const int& v6, const int& v7){
	static size_t curIndex;
	curIndex = 0;
	hash_combine(curIndex, v1);
	hash_combine(curIndex, v2);
	hash_combine(curIndex, v3);
	hash_combine(curIndex, v4);
	hash_combine(curIndex, v5);
	hash_combine(curIndex, v6);
	hash_combine(curIndex, v7);
	return curIndex;
}

struct C1Feat{
protected:
	int id;
	size_t seed;
public:
	bool operator == (const C1Feat& a) const {
		return (a.id == id);
	}
	void setId(const int& v){
		id = v;
		seed = id;
	}
	size_t hash_value() const{
		return seed;
	}	
};

struct C2Feat{
protected:
	int id1, id2;
	size_t seed;
public:
	bool operator == (const C2Feat& a) const {
		return (a.id1 == id1 && a.id2 == id2);
	}
	void setId(const int& v1, const int& v2){
		id1 = v1;
		id1 = v2;
		seed = hash_func(v1, v2);
	}
	std::size_t hash_value() const{
		return seed;
	}
};

struct C3Feat{
protected:
	int id1, id2, id3;
	size_t seed;
public:
	bool operator == (const C3Feat& a) const {
		return (a.id1 == id1 && a.id2 == id2 && a.id3 == id3);
	}
	void setId(const int& v1, const int& v2, const int& v3){
		id1 = v1;
		id2 = v2;
		id3 = v3;
		seed = hash_func(v1, v2, v3);
	}
	std::size_t hash_value() const {
		return seed;
	}
};

struct C4Feat{
protected:
	int id1, id2, id3, id4;
	size_t seed;
public:
	bool operator == (const C4Feat& a) const {
		return (a.id1 == id1 && a.id2 == id2 && a.id3 == id3 && a.id4 == id4);
	}
	void setId(const int& v1, const int& v2, const int& v3, const int& v4){
		id1 = v1;
		id2 = v2;
		id3 = v3;
		id4 = v4;
		seed = hash_func(v1, v2, v3, v4);
	}
	std::size_t hash_value() const {
		return seed;
	}
};

struct C5Feat{
protected:
	int id1, id2, id3, id4, id5;
	size_t seed;
public:
	bool operator == (const C5Feat& a) const {
		return (a.id1 == id1 && a.id2 == id2 && a.id3 == id3 && a.id4 == id4 && a.id5 == id5);
	}
	void setId(const int& v1, const int& v2, const int& v3, const int& v4, const int& v5){
		id1 = v1;
		id2 = v2;
		id3 = v3;
		id4 = v4;
		id5 = v5;
		seed = hash_func(v1, v2, v3, v4, v5);
	}
	std::size_t hash_value() const {
		return seed;
	}
};

struct C6Feat{
protected:
	int id1, id2, id3, id4, id5, id6;
	size_t seed;
public:
	bool operator == (const C6Feat& a) const {
		return (a.id1 == id1 && a.id2 == id2 && a.id3 == id3 && a.id4 == id4 && a.id5 == id5 && a.id6 == id6);
	}
	void setId(const int& v1, const int& v2, const int& v3, const int& v4, const int& v5, const int& v6){
		id1 = v1;
		id2 = v2;
		id3 = v3;
		id4 = v4;
		id5 = v5;
		id6 = v6;
		seed = hash_func(v1, v2, v3, v4, v5, v6);
	}
	std::size_t hash_value() const {
		return seed;
	}
};

struct C7Feat{
protected:
	int id1, id2, id3, id4, id5, id6, id7;
	size_t seed;
public:
	bool operator == (const C7Feat& a) const {
		return (a.id1 == id1 && a.id2 == id2 && a.id3 == id3 && a.id4 == id4 && a.id5 == id5 && a.id6 == id6 && a.id7 == id7);
	}
	void setId(const int& v1, const int& v2, const int& v3, const int& v4, const int& v5, const int& v6, const int& v7){
		id1 = v1;
		id2 = v2;
		id3 = v3;
		id4 = v4;
		id5 = v5;
		id6 = v6;
		id7 = v7;
		seed = hash_func(v1, v2, v3, v4, v5, v6, v7);
	}
	std::size_t hash_value() const {
		return seed;
	}
};


namespace std {
	template<>
	struct hash < C1Feat > {
	public:
		size_t operator()(const C1Feat& s)const{
			return s.hash_value();
		}
	};

	template<>
	struct hash < C2Feat > {
	public:
		size_t operator()(const C2Feat& s)const{
			return s.hash_value();
		}
	};

	template<>
	struct hash < C3Feat > {
	public:
		size_t operator()(const C3Feat& s)const{
			return s.hash_value();
		}
	};


	template<>
	struct hash < C4Feat > {
	public:
		size_t operator()(const C4Feat& s)const{
			return s.hash_value();
		}
	};

	template<>
	struct hash < C5Feat > {
	public:
		size_t operator()(const C5Feat& s)const{
			return s.hash_value();
		}
	};

	template<>
	struct hash < C6Feat > {
	public:
		size_t operator()(const C6Feat& s)const{
			return s.hash_value();
		}
	};


	template<>
	struct hash < C7Feat > {
	public:
		size_t operator()(const C7Feat& s)const{
			return s.hash_value();
		}
	};

};

#endif /* COPUtil_H_ */
