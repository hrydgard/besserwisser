#pragma once

#include "layer.h"

struct NeuralNetwork {
	std::vector<Layer *> layers;
};

void InitializeNetwork(NeuralNetwork &network);

void RunForwardPass(NeuralNetwork &network);
void RunBackwardPass(NeuralNetwork &network);
