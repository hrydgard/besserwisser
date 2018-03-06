#include <random>
#include "math_util.h"
#include <pmmintrin.h>

inline float hsum_ps_sse3(__m128 v) {
	__m128 shuf = _mm_movehdup_ps(v);        // broadcast elements 3,1 to 2,0
	__m128 sums = _mm_add_ps(v, shuf);
	shuf = _mm_movehl_ps(shuf, sums); // high half -> low half
	sums = _mm_add_ss(sums, shuf);
	return        _mm_cvtss_f32(sums);
}

float DotSSE(const float *a, const float *b, size_t size) {
	float sum;
	if (size >= 16) {
		__m128 sumWide1 = _mm_setzero_ps();
		__m128 sumWide2 = _mm_setzero_ps();
		while (size >= 8) {
			sumWide1 = _mm_add_ps(sumWide1, _mm_mul_ps(_mm_loadu_ps(a), _mm_loadu_ps(b)));
			sumWide2 = _mm_add_ps(sumWide2, _mm_mul_ps(_mm_loadu_ps(a + 4), _mm_loadu_ps(b + 4)));
			a += 8;
			b += 8;
			size -= 8;
		}
		sum = hsum_ps_sse3(_mm_add_ps(sumWide1, sumWide2));
	} else {
		sum = 0.0f;
	}
	for (size_t i = 0; i < size; i++) {
		sum += a[i] * b[i];
	}
	return sum;
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
