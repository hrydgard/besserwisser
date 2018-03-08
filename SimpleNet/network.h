#pragma once

#include "layer.h"

class NeuralNetwork {
public:
	// TODO: Just own the layers here, but connect them in a graph.
	std::vector<Layer *> layers;

	struct HyperParams {
		float regStrength = 0.5f;
	};
	HyperParams hyperParams;

	void InitializeNetwork();

	// Inference
	void RunForwardPass();

	// Training
	void RunBackwardPass();
	void ClearGradients();
	void AccumulateGradientSum();
};
