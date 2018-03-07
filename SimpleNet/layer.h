#pragma once

#include <cassert>
#include <algorithm>

#include "math_util.h"

enum class LayerType {
	// These should be enough to get good results.
	UNDEFINED,
	IMAGE,  // Doesn't do anything, just a static image.
	FC,  // Fully connected. Same as a regular linear classifier matrix.
	RELU,  // Best non-linearity.
	SOFTMAX_LOSS,
	SVM_LOSS,  // This seems to converge faster than softmax.

	// Future
	CONV,
	MAXPOOL,  // Downsamples by 2x in X and Y but not Z.
};

class Layer {
public:
	virtual ~Layer() {
		delete[] neurons;
		delete[] weights;
		delete[] gradient;
	}

	virtual void Initialize() {}
	virtual void Forward(const float *input) = 0;   // input = The neurons from the previous layer
	virtual void Backward(const float *input) = 0;  // input = The gradients from the next layer

	LayerType type;
	ivec3 inputDim{};  // How the input should be interpreted. Important for some layer types.

	int numInputs = 0;
	int numNeurons = 0;
	int numWeights = 0;  // for convenience.
	int numGradients = 0;

	// State
	float *neurons = nullptr;  // Vector.

														 // Trained. Only read from in forward pass, updated after computing gradients.
	float *weights = nullptr;  // Matrix (inputs * neurons)

														 // Backward gradient
	float *gradient = nullptr;

	// Truth. Used by softmaxloss and svmloss layers.
	int label = -1;
};

class ReluLayer : public Layer {
public:
	ReluLayer() { type = LayerType::RELU; }
	void Forward(const float *input) override {
		// TODO: This can be very easily SIMD'd.
		for (int i = 0; i < numNeurons; i++) {
			neurons[i] = std::max(0.0f, input[i]);
		}
	}
	void Backward(const float *input) override {
		for (int i = 0; i < numNeurons; i++) {
			gradient[i] = neurons[i] > 0.0f ? 1.0f : 0.0f;
		}
	}
};

class InputLayer : public Layer {
public:
	InputLayer() {}
	void Forward(const float *input) override {}
	void Backward(const float *input) override {}
};

class ImageLayer : public InputLayer {
public:
	ImageLayer() { type = LayerType::IMAGE; }
};

class FcLayer : public Layer {
public:
	FcLayer() { type = LayerType::FC; }
	void Forward(const float *input) override;
	void Backward(const float *input) override;
};

class SVMLossLayer : public Layer {
public:
	SVMLossLayer() { type = LayerType::SVM_LOSS; }
	void Forward(const float *input) override;
	void Backward(const float *input) override;
};