#include "layer.h"
#include "network.h"
#include "math_util.h"

void Layer::AccumulateGradientSum() {
	Accumulate(gradientSum, gradient, numGradients);
}

void FcLayer::Forward(const float *input) {
	// Just a matrix*vector multiplication.
	for (int y = 0; y < numNeurons; y++) {
		neurons[y] = DotAVX(input, &weights[y * numInputs], numInputs);
	}
}

void FcLayer::Backward(const float *prev_data, const float *next_gradient) {
	float regStrength = network_->hyperParams.regStrength;

	// Partial derivative
	for (int y = 0; y < numNeurons; y++) {
		for (int x = 0; x < numInputs; x++) {
			int index = y * numInputs + x;
			// The derivative of a multiplication with respect to a variable is the other variable.
			// Then do the chain rule multiplication.
			gradient[index] = prev_data[x] * next_gradient[y] + regStrength * weights[index];

			assert(fabsf(gradient[index]) < 1000.0f);
		}
	}
}

void SVMLossLayer::Forward(const float *input) {
	assert(label != -1);
	float sum = 0.0f;
	for (size_t i = 0; i < numInputs; i++) {
		if (i != label) {
			sum += std::max(0.0f, input[i] - input[label] + 1.0f);
		}
	}
	neurons[0] = sum;
}

void SVMLossLayer::Backward(const float *prev_data, const float *next_gradient) {
	// There's no input gradient, we compute the original dL/dz gradient.
	assert(!next_gradient);

	// Note! The function we are differentiating here is the complete loss,
	// that is the sum of all the "neurons". We're differentiating the entire sum
	// with respect to each row.
	// The "correct" level is involved in all the rows so it needs a summing loop,
	// while the others can be computed directly.
	int positive_count = 0;
	for (size_t i = 0; i < numInputs; i++) {
		if (i != label)
			positive_count += (prev_data[i] - prev_data[label] + 1.0f) > 0.0f;
	}

	for (size_t i = 0; i < numInputs; i++) {
		if (i == label) {
			gradient[i] = -(float)positive_count;
		} else {
			gradient[i] = (prev_data[i] - prev_data[label] + 1.0f) > 0.0f ? 1.0f : 0.0f;
		}
	}
	PrintFloatVector("SVMgradient", gradient, numInputs);
}

void SoftMaxLayer::Forward(const float *input) {
	float expSum = 0.0f;
	for (size_t i = 0; i < numInputs; i++) {
		expSum += expf(input[i]);
	}
	neurons[0] = -logf(expf(input[label]) / expSum);
}

void SoftMaxLayer::Backward(const float *prev_data, const float *next_gradient) {
	assert(!next_gradient);
	// TODO: Implement. Forward should cache the p(k) values, computed in a loop. Could also recompute them from prev_data.
	// http://cs231n.github.io/neural-networks-case-study/#together
}

void ReluLayer::Forward(const float *input) {
	// TODO: This can be very easily SIMD'd.
	for (int i = 0; i < numNeurons; i++) {
		neurons[i] = std::max(0.0f, input[i]);
	}
}

void ReluLayer::Backward(const float *prev_data, const float *input) {
	for (int i = 0; i < numNeurons; i++) {
		gradient[i] = neurons[i] > 0.0f ? input[i] : 0.0f;
	}
}
