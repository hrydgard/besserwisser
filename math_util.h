#pragma once

#include <cstdio>
#include <vector>
#include <cmath>
#include <cstdint>

#define ARRAY_SIZE(x) (sizeof(x)/sizeof(x[0]))

struct ivec3 {
	int x, y, z;
};

inline float Sigmoid(float f) {
	return 1.0f / (1.0f + expf(-f));
}

// TODO: SSE2 it up.
inline float Dot(const float *a, const float *b, size_t size) {
	float sum = 0.0f;
	for (size_t i = 0; i < size; i++) {
		sum += a[i] * b[i];
	}
	return sum;
}

// Simple array operations. Most are highly optimized.
float DotSSE(const float *a, const float *b, size_t size);
float DotAVX(const float *a, const float *b, size_t size);
float Sum(const float *a, size_t size);
float SumAVX(const float *a, size_t size);
float SumSquaresAVX(const float *a, size_t size);
void Accumulate(float *a, const float *b, size_t size);
void AccumulateScaledSquares(float *a, const float *b, float scale, size_t size);
void AccumulateScaledVector(float *sum, const float *a, float factorA, size_t size);
void AccumulateScaledVectors(float *sum, const float *a, float factorA, const float *b, float factorB, size_t size);
void ScaleInPlace(float *a, float factor, size_t size);
void ClampDownToZero(float *a, const float *b, size_t size);
void BytesToFloat(float *a, const uint8_t *b, size_t size, float scale, float offset);

// LAPACK stuff
void SaxpyAVX(size_t size, float a, const float *x, float *y);

// NOTE: This can return -1 if all the input is INFINITY or if there are NaNs.
int FindMinIndex(const float *data, size_t size);
int FindMaxIndex(const float *data, size_t size);

void FloatNoise(float *data, size_t size, float scale = 1.0f, float bias = 0.0f);
void GaussianNoise(float *data, size_t size, float scale);  // Centered around 0 with unit stddev before scaling.

void PrintFloatVector(const char *name, const float *x, size_t size, size_t maxSize = 10);
int DiffVectors(const float *a, const float *b, size_t size, float tolerance, size_t maxDiffCount = 10);

inline float sqr(float x) {
	return x * x;
}

inline uint32_t swap32(uint32_t x) {
#ifdef _MSC_VER
	return _byteswap_ulong(x);
#else
	return (x >> 24) | ((x >> 8) & 0xFF00) | ((x << 8) & 0xFF0000) | (x << 24);
#endif
}

inline uint32_t readBE32(FILE *f) {
	uint32_t x;
	fread(&x, 1, 4, f);
	return swap32(x);
}

struct Tensor {
	Tensor() : w(0), h(0), d(0), data(nullptr) {}
	Tensor(int _w, int _h, int _d) : w(_w), h(_h), d(_d), data(nullptr) {
		data = new float[w * h * d];
	}
	~Tensor() {
		delete[] data;
	}

	void operator=(Tensor &&tensor) {
		delete[] data;
		w = tensor.w;
		h = tensor.h;
		d = tensor.d;
		data = tensor.data;
	}
	Tensor(Tensor&& tensor) noexcept : data(tensor.data), w(tensor.w), h(tensor.h), d(tensor.d) {}

	int GetDataSize() const {
		return w * h * d;
	}

	int w, h, d;
	float *data;

	float &At(int x, int y, int z) {
		return data[z * w * h + y * w + x];
	}
};

std::vector<std::vector<int>> GenerateRandomSubsets(size_t count, size_t setSize);
std::vector<int> GetFullSet(size_t count);

/*
Tensor LoadImageAsTensor(std::string path, bool monochrome) {
	FILE *f = fopen(path.c_str(), "rb");
	int w, h, comp;
	stbi_uc *data = stbi_load_from_file(f, &w, &h, &comp, 3);

	// TODO: Load all three channels.
	Tensor tensor(w, h, 1);
	for (int y = 0; y < h; y++) {
		for (int x = 0; x < w; x++) {
			tensor.At(x, y, 0) = ByteToFloat(data[(y * w + h) * 3]);
		}
	}
}*/
