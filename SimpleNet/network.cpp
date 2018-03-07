#include "network.h"

// Inference.
void RunForwardPass(NeuralNetwork &network) {
	for (int i = 1; i < network.layers.size(); i++) {
		network.layers[i]->Forward(network.layers[i - 1]->neurons);
	}
}

void RunBackwardPass(NeuralNetwork &network) {
	for (int i = network.layers.size() - 1; i >= 0; i++) {
		network.layers[i]->Backward((i < network.layers.size() - 1) ? network.layers[i + 1]->gradient : nullptr);
	}
}

void InitializeNetwork(NeuralNetwork &network) {
	for (int i = 1; i < network.layers.size(); i++) {
		Layer &layer = *network.layers[i];
		assert(layer.numInputs == network.layers[i - 1]->numNeurons);
		switch (layer.type) {
		case LayerType::FC:
			layer.neurons = new float[layer.numNeurons];
			layer.numWeights = layer.numNeurons * layer.numInputs;
			layer.weights = new float[layer.numWeights]{};
			layer.gradient = new float[layer.numWeights]{};  // Here we'll accumulate gradients before we do the adjust.
			GaussianNoise(layer.weights, layer.numWeights, 0.05f);
			break;
		case LayerType::RELU:
			assert(layer.numNeurons == layer.numInputs);
			layer.neurons = new float[layer.numNeurons];
			layer.gradient = new float[layer.numNeurons]{};
			break;
		case LayerType::IMAGE:
			break;
		case LayerType::SOFTMAX_LOSS:
		case LayerType::SVM_LOSS:
			assert(layer.numNeurons == layer.numInputs);
			layer.neurons = new float[layer.numNeurons];
			layer.gradient = new float[layer.numNeurons]{};
			break;
		}
	}
}