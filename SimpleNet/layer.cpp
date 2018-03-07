#include "layer.h"

void FcLayer::Forward(const float *input) {
	// Just a matrix*vector multiplication.
	for (int y = 0; y < numNeurons; y++) {
		neurons[y] = DotAVX(input, &weights[y * numInputs], numInputs);
	}
}

void FcLayer::Backward(const float *input) {
	// Partial derivative
	for (int y = 0; y < numNeurons; y++) {
		for (int x = 0; x < numInputs; x++) {
			int index = y * numInputs + x;
			// The derivative of a multiplication with respect to a variable is the other variable.
			gradient[index] = neurons[y] * input[y];
		}
	}
}

void SVMLossLayer::Forward(const float *input) {
	float truth = input[label];
	assert(label != -1);
	for (size_t i = 0; i < numNeurons; i++) {
		if (i == label) {
			neurons[i] = 0.0f;
		} else {
			neurons[i] = std::max(0.0f, input[i] - truth + 1.0f);
		}
	}
}

void SVMLossLayer::Backward(const float *input) {
	// Input unused here, this is the original dL/dz gradient.
	assert(!input);
	for (size_t i = 0; i < numNeurons; i++) {
		if (i == label) {
			gradient[i] = 0.0f;
		} else {
			gradient[i] = neurons[i] >= 0.0f ? 1.0f : 0.0f;
		}
	}
}