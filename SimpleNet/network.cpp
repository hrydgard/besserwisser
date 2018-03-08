#include "network.h"
#include "math_util.h"

void NeuralNetwork::ClearGradients() {
	for (int i = 1; i < layers.size(); i++) {
		layers[i]->ClearGradients();
	}
}

// Inference.
void NeuralNetwork::RunForwardPass() {
	for (int i = 1; i < layers.size(); i++) {
		layers[i]->Forward(layers[i - 1]->data);
	}
}

void NeuralNetwork::RunBackwardPass() {
	for (int i = (int)layers.size() - 1; i >= 0; i--) {
		layers[i]->Backward(
			i > 0 ? layers[i - 1]->data : 0,
			(i < layers.size() - 1) ? layers[i + 1]->gradient : nullptr);
	}
}

void NeuralNetwork::AccumulateGradientSum() {
	for (int i = 0; i < layers.size(); i++) {
		if (layers[i]->type != LayerType::FC)
			continue;
		if (!layers[i]->gradientSum) {
			layers[i]->gradientSum = new float[layers[i]->numGradients]{};
		}
		layers[i]->AccumulateGradientSum();
	}
}

void NeuralNetwork::ScaleGradientSum(float factor) {
	for (int i = 0; i < layers.size(); i++) {
		if (layers[i]->type != LayerType::FC)
			continue;
		assert(layers[i]->gradientSum);
		layers[i]->ScaleGradientSum(factor);
	}
}

void NeuralNetwork::InitializeNetwork() {
	for (int i = 1; i < layers.size(); i++) {
		Layer &layer = *layers[i];
		assert(layer.numInputs == layers[i - 1]->numData);
		switch (layer.type) {
		case LayerType::FC:
			layer.data = new float[layer.numData];
			layer.numWeights = layer.numData * layer.numInputs;
			layer.weights = new float[layer.numWeights]{};
			layer.numGradients = layer.numWeights;
			layer.gradient = new float[layer.numGradients]{};  // Here we'll accumulate gradients before we do the adjust.
			GaussianNoise(layer.weights, layer.numWeights, 0.05f);
			break;
		case LayerType::RELU:
			assert(layer.numData == layer.numInputs);
			layer.data = new float[layer.numData];
			layer.numGradients = layer.numData;
			layer.gradient = new float[layer.numGradients]{};
			break;
		case LayerType::IMAGE:
			break;
		case LayerType::SOFTMAX_LOSS:
		case LayerType::SVM_LOSS:
			assert(layer.numData == 1);
			assert(layer.numInputs >= 1);
			layer.data = new float[layer.numInputs];
			layer.numGradients = layer.numInputs;
			layer.gradient = new float[layer.numGradients]{};
			break;
		}
	}
}
