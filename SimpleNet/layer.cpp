#include "layer.h"

void FcLayer::Forward(const float *input) {
	// Just a matrix*vector multiplication.
	for (int y = 0; y < numNeurons; y++) {
		neurons[y] = DotAVX(input, &weights[y * numInputs], numInputs);
	}
}

void FcLayer::Backward(const float *prev_data, const float *next_gradient) {
	// Partial derivative
	for (int y = 0; y < numNeurons; y++) {
		for (int x = 0; x < numInputs; x++) {
			int index = y * numInputs + x;
			// The derivative of a multiplication with respect to a variable is the other variable.
			// Then do the chain rule multiplication.
			gradient[index] = weights[index] * next_gradient[y];
		}
	}
}

void SVMLossLayer::Forward(const float *input) {
	assert(label != -1);
	for (size_t i = 0; i < numNeurons; i++) {
		if (i == label) {
			neurons[i] = 0.0f;
		} else {
			neurons[i] = std::max(0.0f, input[i] - input[label] + 1.0f);
		}
	}
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
	for (size_t i = 0; i < numNeurons; i++) {
		if (i != label)
			positive_count += (prev_data[i] - prev_data[label] + 1.0f) > 0.0f;
	}

	for (size_t i = 0; i < numNeurons; i++) {
		if (i == label) {
			gradient[i] = -(float)positive_count;
		} else {
			gradient[i] = (prev_data[i] - prev_data[label] + 1.0f) > 0.0f ? 1.0f : 0.0f;
		}
	}
	PrintFloatVector("SVMgradient", gradient, numNeurons);
}

void ReluLayer::Forward(const float *input) {
	// TODO: This can be very easily SIMD'd.
	for (int i = 0; i < numNeurons; i++) {
		neurons[i] = std::max(0.0f, input[i]);
	}
}

void ReluLayer::Backward(const float *prev_data, const float *input) {
	for (int i = 0; i < numNeurons; i++) {
		gradient[i] = neurons[i] > 0.0f ? 1.0f : 0.0f;
	}
}
