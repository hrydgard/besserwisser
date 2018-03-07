#pragma once

#include <cassert>
#include <cstring>
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

class NeuralNetwork;

class Layer {
public:
	Layer(NeuralNetwork *network) : network_(network) {}

	virtual ~Layer() {
		delete[] neurons;
		delete[] weights;
		delete[] gradient;
	}

	virtual void Initialize() {}
	virtual void Forward(const float *input) = 0;   // input = The neurons from the previous layer
	virtual void Backward(const float *prev_data, const float *next_gradient) = 0;  // input = The gradients from the next layer
	void ClearGradients() {
		if (gradient) {
			memset(gradient, 0, sizeof(gradient[0]) * numGradients);
		}
	}

	LayerType type;
	ivec3 inputDim{};  // How the input should be interpreted. Important for some layer types.

	int numInputs = 0;
	int numNeurons = 0;
	int numWeights = 0;  // for convenience.
	int numGradients = 0;

	// State. There's way too much state! Possibly should be separated out so the rest
	// can be shared between threads or something.
	float *neurons = nullptr;  // Vector.

														 // Trained. Only read from in forward pass, updated after computing gradients.
	float *weights = nullptr;  // Matrix (inputs * neurons)

														 // Backward gradient
	float *gradient = nullptr;

	// Truth. Used by softmaxloss and svmloss layers.
	int label = -1;

protected:
	NeuralNetwork *network_;
};

class ReluLayer : public Layer {
public:
	ReluLayer(NeuralNetwork *network) : Layer(network) { type = LayerType::RELU; }
	void Forward(const float *input) override;
	void Backward(const float *prev_data, const float *next_gradient) override;
};

class InputLayer : public Layer {
public:
	InputLayer(NeuralNetwork *network) : Layer(network) {}
	void Forward(const float *input) override {}
	void Backward(const float *prev_data, const float *next_gradient) override {}
};

class ImageLayer : public InputLayer {
public:
	ImageLayer(NeuralNetwork *network) : InputLayer(network) { type = LayerType::IMAGE; }
};

class FcLayer : public Layer {
public:
	FcLayer(NeuralNetwork *network) : Layer(network) { type = LayerType::FC; }
	void Forward(const float *input) override;
	void Backward(const float *prev_data, const float *next_gradient) override;
};

class SVMLossLayer : public Layer {
public:
	SVMLossLayer(NeuralNetwork *network) : Layer(network) { type = LayerType::SVM_LOSS; }
	void Forward(const float *input) override;
	void Backward(const float *prev_data, const float *next_gradient) override;
};