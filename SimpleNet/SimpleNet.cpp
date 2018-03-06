﻿#include <cmath>
#include <cstdint>
#include <string>
#include <vector>
#include <algorithm>
#include <cstdio>

#include "math_util.h"
#include "mnist.h"

enum class LayerType {
	UNDEFINED,
	FC,  // Fully connected. Same as a regular linear classifier matrix.
	CONV,
	POOL,  // Downsamples by 2x in X and Y but not Z
	RELU,  // Best non-linearity.
	IMAGE,  // Doesn't do anything, just a static image.
	SOFTMAX,
	SVM,
};

typedef float(*LossFunction)(const float *data, size_t size, int correctLabel);

struct NeuralLayer {
	LayerType type = LayerType::UNDEFINED;
	ivec3 inputDim{};  // How the input should be interpreted. Important for some layer types.

	int numInputs = 0;
	int numNeurons = 0;
	int numWeights = 0;  // for convenience.

	// Trained:
	float *weights = nullptr;  // Matrix (inputs * neurons)
	float *neurons = nullptr;  // Vector.

};

struct NeuralNetwork {
	std::vector<NeuralLayer *> layers;
	LossFunction lossFunction;
};

// Layers own their own outputs.
void Forward(const NeuralLayer &layer, const float *input) {
	Tensor output;
	switch (layer.type) {
	case LayerType::RELU: {
		size_t sz = output.GetDataSize();
		for (int i = 0; i < layer.numInputs; i++) {
			layer.neurons[i] = std::max(0.0f, input[i]);
		}
		break;
	}
	case LayerType::FC: {
		// Just a matrix*vector multiplication.
		for (int y = 0; y < layer.numNeurons; y++) {
			layer.neurons[y] = DotSSE(input, &layer.weights[y * layer.numInputs], layer.numInputs);
		}
		break;
	}
	case LayerType::SOFTMAX:
		
		break;
	case LayerType::CONV:
		break;
	case LayerType::POOL:
		break;
	case LayerType::IMAGE:
		// Do nothing.
		break;
	}
}

void Backward(const NeuralLayer &layer, const float *gradients) {
	switch (layer.type) {
	case LayerType::RELU:
		break;
	}
}

// Inference.
void RunNetwork(NeuralNetwork &network) {
	for (int i = 1; i < network.layers.size(); i++) {
		Forward(*network.layers[i], network.layers[i - 1]->neurons);
	}
}

void InitializeNetwork(NeuralNetwork &network) {
	for (int i = 1; i < network.layers.size(); i++) {
		NeuralLayer &layer = *network.layers[i];
		assert(layer.numInputs == network.layers[i - 1]->numNeurons);
		layer.neurons = new float[layer.numNeurons];
		switch (layer.type) {
		case LayerType::FC:
			layer.numWeights = layer.numNeurons * layer.numInputs;
			layer.weights = new float[layer.numWeights]{};
			GaussianNoise(layer.weights, layer.numWeights);
			break;
		case LayerType::RELU:
			assert(layer.numNeurons == layer.numInputs);
			break;
		case LayerType::IMAGE:
			break;
		case LayerType::SOFTMAX:
		case LayerType::SVM:
			assert(layer.numNeurons == layer.numInputs);
			break;
		}
	}
}

float ComputeSVMLoss(const float *data, size_t size, int correctLabel) {
	float loss = 0.0f;
	float truth = data[correctLabel];
	for (size_t i = 0; i < size; i++) {
		if (i == truth)
			continue;
		loss += std::max(0.0f, data[i] - truth + 1.0f);
	}
	return loss;
}

float ComputeSoftMaxLoss(const float *data, size_t size, int correctLabel) {
	float sumExp = 0.0f;
	for (size_t i = 0; i < size; i++) {
		float exped = expf(data[i]);
		sumExp += exped;
	}
	float normalized = expf(data[correctLabel]) / sumExp;
	return -logf(normalized);
}

// argmax(data[i], i)
int Judge(const float *data, size_t size) {
	float maxValue = 0.0f;
	int argmax = -1;
	for (size_t i = 0; i < size; i++) {
		if (data[i] > maxValue) {
			argmax = (int)i;
			maxValue = data[i];
		}
	}
	return argmax;
}

struct DataSet {
	std::vector<DataVector> images;
	std::vector<uint8_t> labels;
};

struct Subset {
	DataSet *dataSet;
	std::vector<int> indices;
};

float ComputeLoss(NeuralNetwork &network, const Subset &subset) {
	assert(network.layers[0]->type == LayerType::IMAGE);

	NeuralLayer *finalLayer = network.layers.back();
	// Evaluate network on parts of training set, compute gradients, update weights, backpropagate.
	float totalLoss = 0.0f;
	auto &images = subset.dataSet->images;
	auto &labels = subset.dataSet->labels;
	for (int i = 0; i < images.size(); i++) {
		int index = subset.indices[i];
		network.layers[0]->neurons = images[index].data;
		RunNetwork(network);
		float loss = network.lossFunction(finalLayer->neurons, finalLayer->numNeurons, labels[index]);
		totalLoss += loss;
	}
	totalLoss /= subset.indices.size();
	return totalLoss;
}

void ComputeGradientBruteForce(NeuralNetwork &network, const Subset &subset, int layerIndex, float *gradient) {
	const float diff = 0.0001f;
	const float inv2Diff = 1.0f / (2.0 * diff);
	NeuralLayer &layer = *network.layers[layerIndex];
	size_t size = layer.numWeights;
	for (int i = 0; i < size; i++) {
		float origWeight = layer.weights[i];
		// Tweak up and compute loss
		layer.weights[i] = origWeight + diff;
		float up = ComputeLoss(network, subset);
		// Tweak down and compute
		layer.weights[i] = origWeight - diff;
		float down = ComputeLoss(network, subset);
		// Restore and compute gradient.
		layer.weights[i] = origWeight;
		gradient[i] = up - down;
	}
}

void UpdateLayerBruteForce(NeuralNetwork &network, const Subset &subset, int layerIndex, float speed) {
	NeuralLayer &layer = *network.layers[layerIndex];
	size_t size = layer.numWeights;
	float *gradient = new float[size];
	ComputeGradientBruteForce(network, subset, layerIndex, gradient);
	// Simple gradient descent.
	// Can be expressed as an axpy
	for (int i = 0; i < size; i++) {
		layer.weights[i] -= gradient[i] * speed;
	}
}

// simple architectures:

// INPUT -> FC -> RELU -> FC -> RELU -> FC
// INPUT -> FC -> RELU -> FC
// INPUT -> FC

int main() {
	DataSet trainingSet;
	trainingSet.images = LoadMNISTImages("C:/dev/MNIST/train-images.idx3-ubyte");
	trainingSet.labels = LoadMNISTLabels("C:/dev/MNIST/train-labels.idx1-ubyte");
	assert(trainingSet.images.size() == trainingSet.images.size());

	DataSet testSet;
	testSet.images = LoadMNISTImages("C:/dev/MNIST/t10k-images.idx3-ubyte");
	testSet.labels = LoadMNISTLabels("C:/dev/MNIST/t10k-labels.idx1-ubyte");
	assert(testSet.images.size() == testSet.images.size());

	NeuralNetwork network;
	NeuralLayer imageLayer{ LayerType::IMAGE, ivec3{ 28, 28, 1 } };
	imageLayer.numInputs = 0;
	imageLayer.numNeurons = 28 * 28 + 1;  // + 1 for bias trick
	network.layers.push_back(&imageLayer);

	/*
	NeuralLayer hiddenLayer{ LayerType::FC, ivec3{100,1,1} };
	hiddenLayer.numInputs = 28 * 28;
	hiddenLayer.numNeurons = 100;
	network.layers.push_back(&hiddenLayer);

	NeuralLayer relu{ LayerType::RELU };
	relu.numInputs = 100;
	relu.numNeurons = 100;
	network.layers.push_back(&relu);

	NeuralLayer fcLayer{ LayerType::FC, ivec3{ 32,32,1 } };
	fcLayer.numInputs = 100;
	fcLayer.numNeurons = 10;
	network.layers.push_back(&fcLayer);
	*/
	NeuralLayer linearLayer{ LayerType::FC };
	linearLayer.numInputs = 28 * 28 + 1;
	linearLayer.numNeurons = 10;
	network.layers.push_back(&linearLayer);

	NeuralLayer *finalLayer = network.layers.back();
	network.lossFunction = &ComputeSVMLoss;

	InitializeNetwork(network);

	static const char *labelNames[10] = { "0", "1", "2", "3", "4", "5", "6", "7", "8", "9" };

	int subsetSize = 32;
	
	std::vector<std::vector<int>> subsets = GenerateRandomSubsets(trainingSet.images.size(), subsetSize);

	Subset subset;
	subset.dataSet = &trainingSet;
	subset.indices = subsets[0];

	float loss = ComputeLoss(network, subset);

	UpdateLayerBruteForce(network, subset, 1, 0.01f);

	float lossAfterTraining = ComputeLoss(network, subset);

	printf("Loss before: %0.3f\n", loss);
	printf("Loss after: %0.3f\n", lossAfterTraining);

	while (true);
	/*
	// Evaluate network on the test set.
	float totalLoss = 0.0f;
	for (int i = 0; i < testImages.size(); i++) {
		assert(imageLayer.numNeurons == testImages[i].size);
		imageLayer.neurons = testImages[i].data;
		RunNetwork(network);
		// int inferredLabel = Judge(fcLayer.neurons, 10);
		float loss = network.lossFunction(fcLayer.neurons, 10, testLabels[i]);
		totalLoss += loss;
	}
	totalLoss /= trainLabels.size();
	*/
	return 0;
}
