#include "network.h"
#include "math_util.h"

void NeuralNetwork::ClearDeltaWeightSum() {
	for (int i = 1; i < layers.size(); i++) {
		layers[i]->ClearDeltaWeightSum();
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

void NeuralNetwork::ScaleDeltaWeightSum(float factor) {
	for (int i = 0; i < layers.size(); i++) {
		layers[i]->ScaleDeltaWeightSum(factor);
	}
}

void NeuralNetwork::InitializeNetwork() {
	for (int i = 1; i < layers.size(); i++) {
		Layer &layer = *layers[i];
		assert(layer.numInputs == layers[i - 1]->numData);
		switch (layer.type) {
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
		default:
			layer.Initialize();
			break;
		}
	}
}
