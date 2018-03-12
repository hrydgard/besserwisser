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
		size_t maxMiniBatchSize = 32;  // Determines the size of a lot of buffers.
		float trainingSpeed = 0.015f;
		int trainingEpochsSlowdown = 10;
		float trainingSlowdownFactor = 0.75f;
	};
	HyperParams hyperParams;

	void InitializeNetwork();

	// Inference
	void RunForwardPass(size_t miniBatchSize);

	// Training. Note that due to how accumulation of weights happen internally,
	// we can't easily multithread this currently, will need some reorganization
	// like having one accumulation buffer per thread.
	void RunBackwardPass(size_t miniBatchSize);

	void ClearDeltaWeightSum();
	void ScaleDeltaWeightSum(float factor);
};
