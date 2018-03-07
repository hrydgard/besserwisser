#pragma once

#include "layer.h"

class NeuralNetwork {
public:
	// TODO: Just own the layers here, but connect them in a graph.
	std::vector<Layer *> layers;

	void InitializeNetwork();
	void RunForwardPass();
	void RunBackwardPass();
};
