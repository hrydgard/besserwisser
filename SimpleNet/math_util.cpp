#include <random>
#include "math_util.h"
#include <immintrin.h>

inline float HorizontalSum(__m128 v) {
	__m128 shuf = _mm_movehdup_ps(v);        // broadcast elements 3,1 to 2,0
	__m128 sums = _mm_add_ps(v, shuf);
	shuf = _mm_movehl_ps(shuf, sums); // high half -> low half
	sums = _mm_add_ss(sums, shuf);
	return _mm_cvtss_f32(sums);
}

inline float HorizontalSum(__m256 v) {
	float sumAVX = 0;
	__m256 hsum = _mm256_hadd_ps(v, v);
	hsum = _mm256_add_ps(hsum, _mm256_permute2f128_ps(hsum, hsum, 0x1));
	__m128 hsum128 = _mm_hadd_ps(_mm256_castps256_ps128(hsum), _mm256_castps256_ps128(hsum));
	return _mm_cvtss_f32(hsum128);
}

float DotSSE(const float *a, const float *b, size_t size) {
	float sum;
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
	for (size_t i = 0; i < size; i++) {
		sum += a[i] * b[i];
	}
	return sum;
}

float DotAVX(const float *a, const float *b, size_t size) {
	float sum;
	if (size >= 16) {
		__m256 sumWide1 = _mm256_setzero_ps();
		__m256 sumWide2 = _mm256_setzero_ps();
		while (size >= 16) {
			sumWide1 = _mm256_add_ps(sumWide1, _mm256_mul_ps(_mm256_loadu_ps(a), _mm256_loadu_ps(b)));
			sumWide2 = _mm256_add_ps(sumWide2, _mm256_mul_ps(_mm256_loadu_ps(a + 8), _mm256_loadu_ps(b + 8)));
			a += 16;
			b += 16;
			size -= 16;
		}
		sum = HorizontalSum(_mm256_add_ps(sumWide1, sumWide2));
	} else {
		sum = 0.0f;
	}
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
	float sum;
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
	} else {
		sum = 0.0f;
	}
	for (size_t i = 0; i < size; i++) {
		sum += a[i];
	}
	return sum;
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

std::vector<std::vector<int>> GenerateRandomSubsets(int count, int setSize) {
	std::vector<int> all;
	for (int i = 0; i < count; i++) {
		all.push_back(i);
	}
	std::random_shuffle(all.begin(), all.end());

	int setCount = count / setSize;
	std::vector<std::vector<int>> sets(setCount);
	for (int i = 0; i < setCount; i++) {
		sets[i].reserve(setSize);
		for (int j = 0; j < setSize; j++) {
			sets[i].push_back(all[i * setSize + j]);
		}
	}
	return sets;
}

void PrintFloatVector(const char *name, const float *x, size_t size, int maxSize) {
	printf("%s: (", name);
	for (int i = 0; i < size; i++) {
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
