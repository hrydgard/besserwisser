#include "layer.h"
#include "network.h"
#include "math_util.h"

// Used to sum up the results of each minibatch before updating the weights.
// If we multithread this, we'll want one gradientSum vector per core, and sum them
// all up at the end.
void Layer::AccumulateGradientSum() {
	Accumulate(gradientSum, gradient, numGradients);
}

void Layer::ScaleGradientSum(float factor) {
	ScaleInPlace(gradientSum, factor, numGradients);
}

void FcLayer::Forward(const float *input) {
	// Just a matrix*vector multiplication.
	for (int y = 0; y < numData; y++) {
		data[y] = DotAVX(input, &weights[y * numInputs], numInputs);
	}
}

void FcLayer::Backward(const float *prev_data, const float *next_gradient) {
	float regStrength = network_->hyperParams.regStrength;

	// Partial derivative.
	for (int y = 0; y < numData; y++) {
		for (int x = 0; x < numInputs; x++) {
			int index = y * numInputs + x;
			// The derivative of a multiplication with respect to a variable is the other variable.
			// Then do the chain rule multiplication. Remember to add on the partial derivative
			// of the regularization function, which turns out to be very simple.
			// Also note that we regularize the biases if they've been baked into weights, we don't care.
			// The literature says that it really doesn't seem to matter but is unclear on why.
			gradient[index] = prev_data[x] * next_gradient[y] + regStrength * weights[index];
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
	data[0] = sum;
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
}

void SoftMaxLayer::Forward(const float *input) {
	float expSum = 0.0f;
	for (size_t i = 0; i < numInputs; i++) {
		expSum += expf(input[i]);
	}
	data[0] = -logf(expf(input[label]) / expSum);
}

void SoftMaxLayer::Backward(const float *prev_data, const float *next_gradient) {
	assert(!next_gradient);
	// TODO: Implement. Forward should cache the p(k) values, computed in a loop. Could also recompute them from prev_data.
	// http://cs231n.github.io/neural-networks-case-study/#together
}

void ReluLayer::Forward(const float *input) {
	// TODO: This can be very easily SIMD'd.
	for (int i = 0; i < numData; i++) {
		data[i] = std::max(0.0f, input[i]);
	}
}

void ReluLayer::Backward(const float *prev_data, const float *input) {
	for (int i = 0; i < numData; i++) {
		gradient[i] = data[i] > 0.0f ? input[i] : 0.0f;
	}
}

void Relu6Layer::Forward(const float *input) {
	// TODO: This can be very easily SIMD'd.
	for (int i = 0; i < numData; i++) {
		data[i] = std::max(0.0f, std::min(input[i], 6.0f));
	}
}

void Relu6Layer::Backward(const float *prev_data, const float *input) {
	for (int i = 0; i < numData; i++) {
		gradient[i] = data[i] > 6.0f ? 0.0f : (data[i] > 0.0f ? input[i] : 0.0f);
	}
}
