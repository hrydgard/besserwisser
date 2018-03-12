#include <random>
#include <algorithm>
#include "math_util.h"
#include <immintrin.h>

#ifdef _MSC_VER
#define USE_AVX
#define USE_SSE
#else
#ifdef __AVX__
#define USE_AVX
#else
#warning __AVX__ not defined
#endif
#ifdef __SSE__
#define USE_SSE
#else
#warning __SSE__ not defined
#endif
#endif

// #define USE_FMA

inline float HorizontalSum(__m128 v) {
	__m128 shuf = _mm_movehdup_ps(v);        // broadcast elements 3,1 to 2,0
	__m128 sums = _mm_add_ps(v, shuf);
	shuf = _mm_movehl_ps(shuf, sums); // high half -> low half
	sums = _mm_add_ss(sums, shuf);
	return _mm_cvtss_f32(sums);
}

#if defined(USE_AVX)
inline float HorizontalSum(__m256 v) {
	float sumAVX = 0;
	__m256 hsum = _mm256_hadd_ps(v, v);
	hsum = _mm256_add_ps(hsum, _mm256_permute2f128_ps(hsum, hsum, 0x1));
	__m128 hsum128 = _mm_hadd_ps(_mm256_castps256_ps128(hsum), _mm256_castps256_ps128(hsum));
	return _mm_cvtss_f32(hsum128);
}
#endif

void ClampDownToZero(float *a, const float *b, size_t size) {
#ifdef USE_AVX
	__m256 zero = _mm256_setzero_ps();
	while (size >= 8) {
		_mm256_storeu_ps(a, _mm256_max_ps(zero, _mm256_loadu_ps(b)));
		a += 8;
		b += 8;
		size -= 8;
	}
#elif defined(USE_SSE)
	__m128 zero = _mm_setzero_ps();
	while (size >= 4) {
		_mm_store_ps(a, _mm_max_ps(zero, _mm_load_ps(b)));
		a += 4;
		b += 4;
		size -= 4;
	}
#endif
	for (int i = 0; i < size; i++) {
		a[i] = std::max(0.0f, b[i]);
	}
}

float DotAVX(const float *a, const float *b, size_t size) {
	float sum = 0.0f;
#ifdef USE_AVX
	if (size >= 16) {
		__m256 sumWide1 = _mm256_setzero_ps();
		__m256 sumWide2 = _mm256_setzero_ps();
		while (size >= 16) {
#if 1
			sumWide1 = _mm256_add_ps(sumWide1, _mm256_mul_ps(_mm256_loadu_ps(a), _mm256_loadu_ps(b)));
			sumWide2 = _mm256_add_ps(sumWide2, _mm256_mul_ps(_mm256_loadu_ps(a + 8), _mm256_loadu_ps(b + 8)));
#else
			sumWide1 = _mm256_fmadd_ps(_mm256_loadu_ps(a), _mm256_loadu_ps(b), sumWide1);
			sumWide2 = _mm256_fmadd_ps(_mm256_loadu_ps(a + 8), _mm256_loadu_ps(b + 8), sumWide2);
#endif
			a += 16;
			b += 16;
			size -= 16;
		}
		sum = HorizontalSum(_mm256_add_ps(sumWide1, sumWide2));
	}
#elif defined(USE_SSE)
	if (size >= 8) {
		__m128 sumWide1 = _mm_setzero_ps();
		__m128 sumWide2 = _mm_setzero_ps();
		while (size >= 8) {
			sumWide1 = _mm_add_ps(sumWide1, _mm_mul_ps(_mm_loadu_ps(a), _mm_loadu_ps(b)));
			sumWide2 = _mm_add_ps(sumWide2, _mm_mul_ps(_mm_loadu_ps(a + 4), _mm_loadu_ps(b + 4)));
			a += 8;
			b += 8;
			size -= 8;
		}
		sum = HorizontalSum(_mm_add_ps(sumWide1, sumWide2));
	} else {
		sum = 0.0f;
	}
#endif

	for (size_t i = 0; i < size; i++) {
		sum += a[i] * b[i];
	}
	return sum;
}

float Sum(const float *a, size_t size) {
	float sum = 0.0f;
	for (size_t i = 0; i < size; i++) {
		sum += a[i];
	}
	return sum;
}

float SumAVX(const float *a, size_t size) {
	float sum = 0.0f;
#if defined(USE_AVX)
	if (size >= 16) {
		__m256 sumWide1 = _mm256_setzero_ps();
		__m256 sumWide2 = _mm256_setzero_ps();
		while (size >= 16) {
			sumWide1 = _mm256_add_ps(sumWide1, _mm256_loadu_ps(a));
			sumWide2 = _mm256_add_ps(sumWide2, _mm256_loadu_ps(a + 8));
			a += 16;
			size -= 16;
		}
		sum = HorizontalSum(_mm256_add_ps(sumWide1, sumWide2));
	}
#elif defined(USE_SSE)
	if (size >= 8) {
		__m128 sumWide1 = _mm_setzero_ps();
		__m128 sumWide2 = _mm_setzero_ps();
		while (size >= 8) {
			sumWide1 = _mm_add_ps(sumWide1, _mm_loadu_ps(a));
			sumWide2 = _mm_add_ps(sumWide2, _mm_loadu_ps(a + 4));
			a += 8;
			size -= 8;
		}
		sum = HorizontalSum(_mm_add_ps(sumWide1, sumWide2));
	}
#endif
	for (size_t i = 0; i < size; i++) {
		sum += a[i];
	}
	return sum;
}

float SumSquaresAVX(const float *a, size_t size) {
	float sum = 0.0f;
#if defined(USE_AVX)
	if (size >= 16) {
		__m256 sumWide1 = _mm256_setzero_ps();
		__m256 sumWide2 = _mm256_setzero_ps();
		while (size >= 16) {
			__m256 x = _mm256_loadu_ps(a);
			__m256 y = _mm256_loadu_ps(a + 8);
#if !defined(USE_FMA)
			sumWide1 = _mm256_add_ps(sumWide1, _mm256_mul_ps(x, x));
			sumWide2 = _mm256_add_ps(sumWide2, _mm256_mul_ps(y, y));
#else
			sumWide1 = _mm256_fmadd_ps(x, x, sumWide1);
			sumWide2 = _mm256_fmadd_ps(y, y, sumWide2);
#endif
			a += 16;
			size -= 16;
		}
		sum = HorizontalSum(_mm256_add_ps(sumWide1, sumWide2));
	} else {
		sum = 0.0f;
	}
#elif defined(USE_SSE)
	if (size >= 8) {
		__m128 sumWide1 = _mm_setzero_ps();
		__m128 sumWide2 = _mm_setzero_ps();
		while (size >= 8) {
			__m128 x = _mm_loadu_ps(a);
			__m128 y = _mm_loadu_ps(a + 4);
#if !defined(USE_FMA)
			sumWide1 = _mm_add_ps(sumWide1, _mm_mul_ps(x, x));
			sumWide2 = _mm_add_ps(sumWide2, _mm_mul_ps(y, y));
#else
			sumWide1 = _mm_fmadd_ps(x, x, sumWide1);
			sumWide2 = _mm_fmadd_ps(y, y, sumWide2);
#endif
			a += 8;
			size -= 8;
		}
		sum = HorizontalSum(_mm_add_ps(sumWide1, sumWide2));
	} else {
		sum = 0.0f;
	}
#endif
	for (size_t i = 0; i < size; i++) {
		float x = a[i];
		sum += x * x;
	}
	return sum;
}

void SaxpyAVX(size_t size, float a, const float *x, float *y) {
#if defined(USE_AVX)
	__m256 factor = _mm256_set1_ps(a);
	while (size >= 8) {
		__m256 sum = _mm256_add_ps(_mm256_mul_ps(factor, _mm256_loadu_ps(x)), _mm256_loadu_ps(y));
		_mm256_storeu_ps(y, sum);
		x += 8;
		y += 8;
		size -= 8;
	}
#elif defined(USE_SSE)
	__m128 factor = _mm_set1_ps(a);
	while (size >= 8) {
		__m128 sum = _mm_add_ps(_mm_mul_ps(factor, _mm_load_ps(x)), _mm_load_ps(y));
		_mm_store_ps(y, sum);
		x += 4;
		y += 4;
		size -= 4;
	}
#endif
	for (int i = 0; i < size; i++)
		y[i] = a*x[i] + y[i];
}

void AccumulateScaledVector(float *d, const float *a, float factorA, size_t size) {
#if defined(USE_AVX)
	__m256 factorAwide = _mm256_set1_ps(factorA);
	while (size >= 8) {
		__m256 prev = _mm256_loadu_ps(d);
		__m256 sum = _mm256_mul_ps(factorAwide, _mm256_loadu_ps(a));
		_mm256_storeu_ps(d, _mm256_add_ps(prev, sum));
		a += 8;
		d += 8;
		size -= 8;
	}
#elif defined(USE_SSE)
	__m128 factorAwide = _mm_set1_ps(factorA);
	while (size >= 4) {
		__m128 prev = _mm_load_ps(d);
		__m128 sum = _mm_mul_ps(factorAwide, _mm_load_ps(a));
		_mm_store_ps(d, _mm_add_ps(prev, sum));
		a += 4;
		d += 4;
		size -= 4;
	}
#endif
	for (int i = 0; i < size; i++)
		d[i] += factorA * a[i];
}

void AccumulateScaledVectors(float *d, const float *a, float factorA, const float *b, float factorB, size_t size) {
#if defined(USE_AVX)
	__m256 factorAwide = _mm256_set1_ps(factorA);
	__m256 factorBwide = _mm256_set1_ps(factorB);
	while (size >= 8) {
		__m256 prev = _mm256_loadu_ps(d);
		__m256 sum = _mm256_add_ps(
			_mm256_mul_ps(factorAwide, _mm256_loadu_ps(a)),
			_mm256_mul_ps(factorBwide, _mm256_loadu_ps(b)));
		_mm256_storeu_ps(d, _mm256_add_ps(prev, sum));
		a += 8;
		b += 8;
		d += 8;
		size -= 8;
	}
#elif defined(USE_SSE)
	__m128 factorAwide = _mm_set1_ps(factorA);
	__m128 factorBwide = _mm_set1_ps(factorB);
	while (size >= 8) {
		__m128 prev = _mm_loadu_ps(d);
		__m128 sum = _mm_add_ps(
			_mm_mul_ps(factorAwide, _mm_loadu_ps(a)),
			_mm_mul_ps(factorBwide, _mm_loadu_ps(b)));
		_mm_storeu_ps(d, _mm_add_ps(prev, sum));
		a += 4;
		b += 4;
		d += 4;
		size -= 4;
	}
#endif
	for (int i = 0; i < size; i++)
		d[i] += factorA * a[i] + factorB * b[i];
}

void Accumulate(float *a, const float *b, size_t size) {
#if defined(USE_AVX)
	while (size >= 8) {
		__m256 sum = _mm256_add_ps(_mm256_loadu_ps(a), _mm256_loadu_ps(b));
		_mm256_storeu_ps(a, sum);
		a += 8;
		b += 8;
		size -= 8;
	}
#elif defined(USE_SSE)
	while (size >= 4) {
		__m128 sum = _mm_add_ps(_mm_loadu_ps(a), _mm_loadu_ps(b));
		_mm_store_ps(a, sum);
		a += 4;
		b += 4;
		size -= 4;
	}
#endif
	for (size_t i = 0; i < size; i++) {
		a[i] += b[i];
	}
}

void ScaleInPlace(float *a, float factor, size_t size) {
#if defined(USE_AVX)
	__m256 factor8 = _mm256_set_ps(factor, factor, factor, factor, factor, factor, factor, factor);
	while (size >= 8) {
		__m256 product = _mm256_mul_ps(_mm256_loadu_ps(a), factor8);
		_mm256_storeu_ps(a, product);
		a += 8;
		size -= 8;
	}
#elif defined(USE_SSE)
	__m128 factor8 = _mm_set_ps(factor, factor, factor, factor);
	while (size >= 8) {
		__m128 product = _mm_mul_ps(_mm_load_ps(a), factor8);
		_mm_store_ps(a, product);
		a += 4;
		size -= 4;
	}
#endif
	for (size_t i = 0; i < size; i++) {
		a[i] *= factor;
	}
}

void AccumulateScaledSquares(float *a, const float *b, float scale, size_t size) {
#if defined(USE_AVX)
	__m256 factor = _mm256_set1_ps(scale);
	while (size >= 8) {
		__m256 bvalue = _mm256_loadu_ps(b);
		__m256 sum = _mm256_add_ps(_mm256_loadu_ps(a), _mm256_mul_ps(factor, _mm256_mul_ps(bvalue, bvalue)));
		_mm256_storeu_ps(a, sum);
		a += 8;
		b += 8;
		size -= 8;
	}
#elif defined(USE_SSE)
	__m128 factor = _mm_set1_ps(scale);
	while (size >= 4) {
		__m128 bvalue = _mm_loadu_ps(b);
		__m128 sum = _mm_add_ps(_mm_loadu_ps(a), _mm_mul_ps(factor, _mm_mul_ps(bvalue, bvalue)));
		_mm_store_ps(a, sum);
		a += 4;
		b += 4;
		size -= 4;
	}
#endif
	for (size_t i = 0; i < size; i++) {
		a[i] += scale * b[i] * b[i];
	}
}

void BytesToFloat(float *a, const uint8_t *b, size_t size, float scale, float offset) {
	for (int x = 0; x < size; x++) {
		uint8_t val = ((const uint8_t *)b)[x];
		float v = (float)val * scale + offset;
		a[x] = v;
	}
}

// argmin(data[i], i)
int FindMinIndex(const float *data, size_t size) {
	float minValue = INFINITY;
	int index = -1;
	for (size_t i = 0; i < size; i++) {
		if (data[i] < minValue) {
			index = (int)i;
			minValue = data[i];
		}
	}
	return index;
}

int FindMaxIndex(const float *data, size_t size) {
	float maxValue = -INFINITY;
	int index = -1;
	for (size_t i = 0; i < size; i++) {
		if (data[i] > maxValue) {
			index = (int)i;
			maxValue = data[i];
		}
	}
	return index;
}

void FloatNoise(float *data, size_t size, float scale, float bias) {
	scale /= (float)RAND_MAX;
	for (size_t i = 0; i < size; i++) {
		data[i] = (float)rand() * scale + bias;
	}
}

void GaussianNoise(float *data, size_t size, float scale) {
	std::random_device rd{};
	std::mt19937 gen{ rd() };
	std::normal_distribution<float> normal;
	for (size_t i = 0; i < size; i++) {
		data[i] = normal(gen) * scale;
	}
}

std::vector<std::vector<int>> GenerateRandomSubsets(size_t count, size_t setSize) {
	std::vector<int> all;
	for (int i = 0; i < count; i++) {
		all.push_back(i);
	}
	std::random_device rd;
	std::mt19937 g(rd());
	std::shuffle(all.begin(), all.end(), g);

	size_t setCount = count / setSize;
	std::vector<std::vector<int>> sets(setCount);
	for (size_t i = 0; i < setCount; i++) {
		sets[i].reserve(setSize);
		for (size_t j = 0; j < setSize; j++) {
			sets[i].push_back(all[i * setSize + j]);
		}
	}
	return sets;
}

std::vector<int> GetFullSet(size_t count) {
	std::vector<int> all;
	all.reserve(count);
	for (int i = 0; i < count; i++) {
		all.push_back(i);
	}
	return all;
}


void PrintFloatVector(const char *name, const float *x, size_t size, size_t maxSize) {
	printf("%s: (", name);
	for (size_t i = 0; i < size; i++) {
		if (i != size - 1) {
			printf("%0.5f, ", x[i]);
			if (i >= maxSize) {
				printf("... / %d)\n", (int)size);
				return;
			}
		} else {
			printf("%0.5f)\n", x[i]);
		}
	}
}

int DiffVectors(const float *a, const float *b, size_t size, float tolerance, size_t maxDiffCount) {
	int diffCount = 0;
	for (size_t i = 0; i < size; i++) {
		float diff = fabsf(a[i] - b[i]);
		if (diff > tolerance) {
			if (diffCount < maxDiffCount)
				printf("DIFF: a[%d] (%f) != b[%d] (%f)\n", (int)i, a[i], (int)i, b[i]);
			diffCount++;
		}
	}
	if (diffCount >= maxDiffCount) {
		printf("(too many diffs: %d/%d)\n", diffCount, (int)size);
	}
	if (diffCount == 0) {
		printf("vectors equal (of size %d)!\n", (int)size);
	}
	return diffCount;
}
