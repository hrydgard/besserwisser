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
	RELU6,  // Variant of RELU.
	SOFTMAX_LOSS,
	SVM_LOSS,

	// Future
	CONV,
	MAXPOOL,  // Downsamples by 2x in X and Y but not Z.
};

class NeuralNetwork;

class Layer {
public:
	Layer(NeuralNetwork *network) : network_(network) {}

	virtual ~Layer() {
		delete[] data;
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
	void AccumulateGradientSum();
	void ScaleGradientSum(float factor);

	LayerType type;
	ivec3 inputDim{};  // How the input should be interpreted. Important for some layer types.

	int numInputs = 0;
	int numData = 0;
	int numWeights = 0;  // for convenience.
	int numGradients = 0;

	// State. There's way too much state! Possibly should be separated out so the rest
	// can be shared between threads or something.
	float *data = nullptr;  // Vector.

	// Trained. Only read from in forward pass, updated after computing gradients.
	float *weights = nullptr;  // Matrix (inputs * neurons)

	// Gradient to backpropagate to the next step.
	float *gradient = nullptr;

	// Used in batch training only to keep intermediate data.
	float *gradientSum = nullptr;

	// Truth. Used by softmaxloss and svmloss layers.
	int label = -1;

protected:
	NeuralNetwork *network_;
};

// Simple RELU activation. No max.
class ReluLayer : public Layer {
public:
	ReluLayer(NeuralNetwork *network) : Layer(network) { type = LayerType::RELU; }
	void Forward(const float *input) override;
	void Backward(const float *prev_data, const float *next_gradient) override;
};

// RELU but with a hardcoded max of 6. Available on Android's Neural API.
class Relu6Layer : public Layer {
public:
	Relu6Layer(NeuralNetwork *network) : Layer(network) { type = LayerType::RELU; }
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

// Fully connected neural layer.
// Note that the back propagation code of this layer requires regularization to be performed
// the usual way. Should probably incorporate that contribution as a separate bias layer somehow
// when we change things to a DAG.
class FcLayer : public Layer {
public:
	FcLayer(NeuralNetwork *network) : Layer(network) { type = LayerType::FC; }
	void Forward(const float *input) override;
	void Backward(const float *prev_data, const float *next_gradient) override;
};

// Outputs the loss as a single float, and caches data in weights to be able to perform
// back propagation.
class SVMLossLayer : public Layer {
public:
	SVMLossLayer(NeuralNetwork *network) : Layer(network) { type = LayerType::SVM_LOSS; }
	void Forward(const float *input) override;
	void Backward(const float *prev_data, const float *next_gradient) override;
};

// Outputs the loss as a single float, and caches data in weights to be able to perform
// back propagation.
class SoftMaxLayer : public Layer {
	SoftMaxLayer(NeuralNetwork *network) : Layer(network) { type = LayerType::SOFTMAX_LOSS; }
	void Forward(const float *input) override;
	void Backward(const float *prev_data, const float *next_gradient) override;
};

// Convolutional image layer.
/*
class ConvLayer : public Layer {
	ConvLayer(NeuralNetwork *network) : Layer(network) {}
};*/