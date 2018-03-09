#pragma once

#include <cstdio>
#include <vector>
#include <cmath>

#define ARRAY_SIZE(x) (sizeof(x)/sizeof(x[0]))

struct ivec3 {
	int x, y, z;
};

inline float ByteToFloat(uint8_t b) {
	return (float)b * (1.0f / 255.0f);
}

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
void AccumulateScaledVectors(float *sum, const float *a, float factorA, const float *b, float factorB, size_t size);
void ScaleInPlace(float *a, float factor, size_t size);
void ClampDownToZero(float *a, const float *b, size_t size);

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
	return _byteswap_ulong(x);
}

inline uint32_t readBE32(FILE *f) {
	uint32_t x;
	fread(&x, 1, 4, f);
	return swap32(x);
}

struct Tensor {
	Tensor() : w(0), h(0), d(0), data(nullptr) {}
	Tensor(int _w, int _h, int _d) : w(_w), h(_h), d(_d) {
		data = new float[w * h * d];
	}
	~Tensor() {
		delete[] data;
	}

	void operator=(Tensor &&tensor) {
		delete[] data;
		data = tensor.data;
		w = tensor.w;
		h = tensor.h;
		d = tensor.d;
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

struct DataVector {
	~DataVector() {
		delete[] data;
	}
	float *data = nullptr;
	size_t size = 0;
	ivec3 dim{};  // Dimensions the data should be interpreted at. Will tag along on the ride.
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
