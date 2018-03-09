#pragma once

#include "layer.h"

// Simple neural network implementation.
class NeuralNetwork {
public:
	// TODO: Just own the layers here, but connect them in a graph.
	std::vector<Layer *> layers;

	struct HyperParams {
		float regStrength = 0.01f;
		float weightInitScale = 0.05f;
		int miniBatchSize = 32;
	};
	HyperParams hyperParams;

	void InitializeNetwork();

	// Inference
	void RunForwardPass();

	// Training. Note that due to how accumulation of weights happen internally,
	// we can't easily multithread this currently, will need some reorganization
	// like having one accumulation buffer per thread.
	void RunBackwardPass();
	void ClearDeltaWeightSum();
	void ScaleDeltaWeightSum(float factor);
};
