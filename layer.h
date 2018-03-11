#pragma once

#include <cassert>
#include <cstring>
#include <algorithm>

#include "math_util.h"
#include "blob.h"

enum class LayerType {
	// These should be enough to get good results.
	UNDEFINED,
	IMAGE,  // Doesn't do anything, just a static image.
	FC,  // Fully connected. Same as a regular linear classifier matrix.
	RELU,  // Best non-linearity.
	RELU6,  // Variant of RELU.
	SIGMOID,  // Old school alternative to RELU, usually worse.
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
		delete[] gradient;
	}

	virtual void Initialize() {}
	virtual void Forward(int miniBatchSize, const float *input) = 0;   // input = The neurons from the previous layer
	virtual void Backward(int miniBatchSize, const float *prev_data, const float *next_gradient) = 0;  // input = The gradients from the next layer

	virtual void ClearDeltaWeightSum() {}
	virtual void ScaleDeltaWeightSum(float factor) {}
	virtual float GetRegularizationLoss() { return 0.0f; }
	virtual void UpdateWeights(float trainingSpeed) {}

	LayerType type;
	std::string name;  // optional

	int inputSize = 0;
	int dataSize = 0;

	int count = 1;  // Batch size. Allows for better vectorization when we move to GPU.

	// State (image content, neurons, whatever). All nodes have this.
	float *data = nullptr;  // Vector.

	// Gradient to backpropagate to the previous step.
	float *gradient = nullptr;

protected:
	NeuralNetwork *network_;
};


class ActivationLayer : public Layer {
public:
	ActivationLayer(NeuralNetwork *network) : Layer(network) {}
	void Initialize() override;
};

class SigmoidLayer : public ActivationLayer {
public:
	SigmoidLayer(NeuralNetwork *network) : ActivationLayer(network) { type = LayerType::SIGMOID; }
	void Forward(int miniBatchSize, const float *input) override;
	void Backward(int miniBatchSize, const float *prev_data, const float *next_gradient) override;
};

// Simple RELU activation. No max.
class ReluLayer : public ActivationLayer {
public:
	ReluLayer(NeuralNetwork *network) : ActivationLayer(network) { type = LayerType::RELU; }
	void Forward(int miniBatchSize, const float *input) override;
	void Backward(int miniBatchSize, const float *prev_data, const float *next_gradient) override;
};

class LeakyReluLayer : public ActivationLayer {
public:
	LeakyReluLayer(NeuralNetwork *network) : ActivationLayer(network) { type = LayerType::RELU; }
	void Forward(int miniBatchSize, const float *input) override;
	void Backward(int miniBatchSize, const float *prev_data, const float *next_gradient) override;
	float coef = 0.01f;
};

// RELU but with a hardcoded max of 6. Available on Android's Neural API.
class Relu6Layer : public ActivationLayer {
public:
	Relu6Layer(NeuralNetwork *network) : ActivationLayer(network) { type = LayerType::RELU6; }
	void Forward(int miniBatchSize, const float *input) override;
	void Backward(int miniBatchSize, const float *prev_data, const float *next_gradient) override;
};

class InputLayer : public Layer {
public:
	InputLayer(NeuralNetwork *network) : Layer(network) {}
	void Forward(int miniBatchSize, const float *input) override {}
	void Backward(int miniBatchSize, const float *prev_data, const float *next_gradient) override {}
};

class ImageLayer : public InputLayer {
public:
	ImageLayer(NeuralNetwork *network) : InputLayer(network) { type = LayerType::IMAGE; }

	const Blob **blobs;

	void Initialize();
	// Unpacks the example image from the dataset.
	void Forward(int miniBatchSize, const float *input) override;
};

// Fully connected neural layer.
// Note that the back propagation code of this layer requires regularization to be performed
// the usual way. Should probably incorporate that contribution as a separate bias layer somehow
// when we change things to a DAG.
class FcLayer : public Layer {
public:
	FcLayer(NeuralNetwork *network) : Layer(network) { type = LayerType::FC; }
	~FcLayer() {
		delete[] weights;
		delete[] deltaWeightSum;
	}

	void Initialize() override;

	void Forward(int miniBatchSize, const float *input) override;
	void Backward(int miniBatchSize, const float *prev_data, const float *next_gradient) override;

	void ClearDeltaWeightSum() override;
	void ScaleDeltaWeightSum(float factor) override;

	float GetRegularizationLoss() override;
	void UpdateWeights(float trainingSpeed) override;

	int numWeights = 0;  // for convenience.

	// Trained. Only read from in forward pass, updated after computing gradients.
	float *weights = nullptr;  // Matrix (inputs * neurons)

	// Used in batch training only to keep intermediate data.
	float *deltaWeightSum = nullptr;

	// The first layer after an image doesn't need to backprop - use this.
	bool skipBackProp = false;
};

// Convolutional neural layer.
class ConvLayer : public Layer {
public:
	ConvLayer(NeuralNetwork *network) : Layer(network) { type = LayerType::CONV; }
	~ConvLayer() {
		delete[] weights;
		delete[] deltaWeightSum;
	}

	void Initialize() override;
	void Forward(int miniBatchSize, const float *input) override;
	void Backward(int miniBatchSize, const float *prev_data, const float *next_gradient) override;

	// TODO: Share these with FcLayer (common base class TrainableLayer?)
	void ClearDeltaWeightSum() override;
	void ScaleDeltaWeightSum(float factor) override;
	float GetRegularizationLoss() override;
	void UpdateWeights(float trainingSpeed) override;

	Dim inputDim;
	Dim kernelDim;
	Dim outputDim;
	int padding = 0;

	// Filter
	float *weights;
	int numWeights;  // kernelSize*kernelSize*(inputDim.depth * outputDim.depth) ??
	float bias;

	// Used in batch training only to keep intermediate data.
	float *deltaWeightSum = nullptr;

	// The first layer after an image doesn't need to backprop - use this.
	bool skipBackProp = false;
};

// Outputs the loss as a single float.
class LossLayer : public Layer {
public:
	LossLayer(NeuralNetwork *network) : Layer(network) {}
	~LossLayer() {}

	void Initialize() override;

	// Truth. Used by softmaxloss and svmloss layers.
	int *labels = nullptr;
};

class SVMLossLayer : public LossLayer {
public:
	SVMLossLayer(NeuralNetwork *network) : LossLayer(network) { type = LayerType::SVM_LOSS; }
	void Forward(int miniBatchSize, const float *input) override;
	void Backward(int miniBatchSize, const float *prev_data, const float *next_gradient) override;
};

// What this computes is more correctly known as "Cross entropy loss"
class SoftMaxLossLayer : public LossLayer {
public:
	SoftMaxLossLayer(NeuralNetwork *network) : LossLayer(network) { type = LayerType::SOFTMAX_LOSS; }
	void Forward(int miniBatchSize, const float *input) override;
	void Backward(int miniBatchSize, const float *prev_data, const float *next_gradient) override;
};

// Always factor 2 for now.
class MaxPoolLayer : public Layer {
public:
	MaxPoolLayer(NeuralNetwork *network) : Layer(network) { type = LayerType::MAXPOOL; }
	~MaxPoolLayer() {
		delete[] maxIndex;
	}

	void Initialize() override;
	void Forward(int miniBatchSize, const float *input) override;
	void Backward(int miniBatchSize, const float *prev_data, const float *next_gradient) override;

	Dim inputDim;
	Dim outputDim;  // We only pool in X,Y directions so depth will be the same.

	uint8_t *maxIndex;  // In each 2x2 element, cache which index was the max to avoid recomputation.
};