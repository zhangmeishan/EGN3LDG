#ifndef N3L_MEM_H
#define N3L_MEM_H

#include <vector>
#include <cstdlib>
#include <cstring>
#include <iostream>
#if !_WINDOWS
#include <sys/shm.h>
#include <sys/mman.h>
#endif

#include <fcntl.h>
#if !_WINDOWS
#include <mm_malloc.h>
#endif
//This is the code copied from dynet: only CPU model

class AlignedMemoryPool {
private:
	const static int align = 32;
	const static size_t unit_size = 1 << 30;
private:
	char** mem;
public:
	size_t capacity;
	size_t used, index;
	float required;  // orcale required size
private:
	inline std::size_t round_up_align(std::size_t n) const {
		if (align < 2) return n;
		return ((n + align - 1) / align) * align;
	}

	void malloc(std::size_t n) {
		capacity = n;
		mem = new char*[capacity];
		for (int idx = 0; idx < capacity; idx++) {
			mem[idx] = (char*)_mm_malloc(unit_size, align);
			if (!mem[idx]) {
				std::cerr << "CPU memory allocation failed capacity=" << idx << " align=" << align << std::endl;
			}
			zero((void*)(mem[idx]), unit_size);
		}
	}

	void free() {
		for (int idx = 0; idx < capacity; idx++) {
			_mm_free((void*)mem[idx]);
		}
		delete[] mem;
		mem = NULL;
	}

public:
	AlignedMemoryPool(size_t cap) {
		used = 0;
		index = 0;
		capacity = cap;
		if (capacity > 0) {
			malloc(capacity);
		}
		required = 0;
	}

	~AlignedMemoryPool() {
		if (capacity > 0) {
			free();
		}
		capacity = 0;
		used = 0;
		index = 0;
	}

public:
	inline void zero(void* p, std::size_t n) {
		memset(p, 0, n);
	}

	void* allocate(size_t n, size_t& aligned) {
		aligned = round_up_align(n);
		required += aligned * 1.0 / unit_size; //never mind if the memory is allocated successfully.
		if (used == capacity || aligned > unit_size) {
			//std::cout << "allocated size bigger than unit size!" << std::endl;
			return NULL;
		}
		void* res = NULL;
		if (aligned + index <= unit_size) {
			//std::cout << "aligned memory is full: " << capacity << ")\n";
			res = mem[used] + index;
			index += aligned;
			if (index == unit_size) {
				used++;
				index = 0;
			}
		}
		else if (used + 1 < capacity) {
			used++;
			index = 0;
			res = mem[used] + index;
			index += aligned;
		}

		return res;
	}

	// zeros out the amount of allocations
	void zero_allocated_memory() {
		if (used == 0 && index == 0) return;
		for (int idx = 0; idx < used; idx++) {
			zero((void*)(mem[idx]), unit_size);
		}
		zero((void*)(mem[used]), unit_size);
	}

};

#endif
