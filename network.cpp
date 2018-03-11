#include "network.h"
#include "math_util.h"

void NeuralNetwork::ClearDeltaWeightSum() {
	for (int i = 0; i < layers.size(); i++) {
		layers[i]->ClearDeltaWeightSum();
	}
}

// Inference.
void NeuralNetwork::RunForwardPass() {
	for (int i = 0; i < layers.size(); i++) {
		layers[i]->Forward(i > 0 ? layers[i - 1]->data : nullptr);
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
	for (int i = 0; i < layers.size(); i++) {
		Layer &layer = *layers[i];
		layer.Initialize();
	}
}
