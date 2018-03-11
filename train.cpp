#include <vector>
#include <memory>

#include "train.h"
#include "layer.h"
#include "network.h"

// Runs the forward pass.
float ComputeDataLoss(NeuralNetwork &network, const DataSet &dataSet, int index, RunStats *stats = nullptr) {
	assert(network.layers[0]->type == LayerType::IMAGE);
	Layer *scoreLayer = network.layers[network.layers.size() - 2];
	Layer *finalLayer = network.layers.back();
	// Last layer must be a loss layer.
	assert(finalLayer->type == LayerType::SVM_LOSS || finalLayer->type == LayerType::SOFTMAX_LOSS);
	network.layers[0]->data = dataSet.images[index].data;
	((LossLayer *)finalLayer)->label = dataSet.labels[index];
	network.RunForwardPass();
	return finalLayer->data[0];
}

float ComputeRegularizationLoss(NeuralNetwork &network) {
	// Penalize with regularization term 0.5lambdaX^2 to discourage high volume noise in the matrix.
	// Note that its gradient will be simply lambdaX.
	// TODO: AVX!
	double regSum = 0.0;
	for (size_t i = 0; i < network.layers.size(); i++) {
		Layer &layer = *network.layers[i];
		regSum += layer.GetRegularizationLoss();
	}
	return 0.5f * network.hyperParams.regStrength * (float)regSum;
}

float ComputeLossOverMinibatch(NeuralNetwork &network, const MiniBatch &subset, RunStats *stats) {
	Layer *scoreLayer = network.layers[network.layers.size() - 2];
	Layer *finalLayer = network.layers.back();
	// Last layer must be a loss layer.
	assert(finalLayer->type == LayerType::SVM_LOSS || finalLayer->type == LayerType::SOFTMAX_LOSS);

	// Computes the total loss as a single number over a set of input images.
	// Should probably do it as a vector instead, it's a bit crazy that this works as is.
	float totalLoss = 0.0f;
	for (int i = 0; i < subset.indices.size(); i++) {
		int index = subset.indices[i];
		float loss = ComputeDataLoss(network, *subset.dataSet, index, stats);
		if (stats) {
			int label = FindMaxIndex(scoreLayer->data, scoreLayer->numData);
			assert(label >= 0);
			if (label == subset.dataSet->labels[index]) {
				stats->correct++;
			} else {
				stats->wrong++;
			}
		}
		totalLoss += loss;
	}
	totalLoss /= subset.indices.size();

	totalLoss += ComputeRegularizationLoss(network);
	return totalLoss;
}

// Utility used for validation of the real training code. Will simply do a brute force gradient calculation
// by disturbing and resetting every training weight of the chosen layer (finite difference method).
static void ComputeDeltaWeightSumBruteForce(NeuralNetwork &network, const MiniBatch &subset, Layer *layer, float *gradient) {
	assert(layer->type == LayerType::FC);
	FcLayer *fcLayer = dynamic_cast<FcLayer *>(layer);

	const float diff = 0.001f;
	const float inv2Diff = 1.0f / (2.0f * diff);
	size_t size = fcLayer->numWeights;
	for (int i = 0; i < size; i++) {
		float origWeight = fcLayer->weights[i];
		// Tweak up and compute loss
		fcLayer->weights[i] = origWeight + diff;
		float up = ComputeLossOverMinibatch(network, subset);
		// Tweak down and compute loss
		fcLayer->weights[i] = origWeight - diff;
		float down = ComputeLossOverMinibatch(network, subset);
		// Restore and compute gradient.
		fcLayer->weights[i] = origWeight;
		gradient[i] = (up - down) * inv2Diff;
	}
	PrintFloatVector("Weights", fcLayer->weights, size, 10);
	PrintFloatVector("Gradient", gradient, size, 10);
}

// Trains all the weights in a network by running both a forward and a backward pass for every item
// in the minibatch. Hyperparameters for training are configured directly on the network object.
static void TrainNetworkOnMinibatch(NeuralNetwork &network, const MiniBatch &subset, float speed) {
	network.ClearDeltaWeightSum();
	for (auto index : subset.indices) {
		network.layers[0]->data = subset.dataSet->images[index].data;
		LossLayer *finalLayer = (LossLayer *)network.layers.back();
		assert(finalLayer->type == LayerType::SOFTMAX_LOSS || finalLayer->type == LayerType::SVM_LOSS);
		finalLayer->label = subset.dataSet->labels[index];
		network.RunForwardPass();
		network.RunBackwardPass();  // Accumulates delta weights
	}
	network.ScaleDeltaWeightSum(1.0f / subset.indices.size());

	// Update all training weights.
	for (auto *layer : network.layers) {
		layer->UpdateWeights(speed);
	}
}

// Trains a single layer of a network using ComputeDeltaWeightSumBruteForce. This does work just fine
// but is incredibly slow. Not really useful once you've validated all your back propagation.
static void TrainLayerBruteForce(NeuralNetwork &network, const MiniBatch &subset, Layer *layer, float speed) {
	// Hacky
	FcLayer *fcLayer = (FcLayer *)layer;

	size_t size = fcLayer->numWeights;
	float *gradient = new float[size];
	ComputeDeltaWeightSumBruteForce(network, subset, layer, gradient);
	// Simple gradient descent.
	// Saxpy(size, -speed, gradient, layer->weights);
	for (int i = 0; i < size; i++) {
		fcLayer->weights[i] -= gradient[i] * speed;
	}
	delete[] gradient;
}


bool RunBruteForceTest(NeuralNetwork &network, FcLayer *testLayer, const DataSet &dataSet) {
	MiniBatch subset;
	subset.dataSet = &dataSet;
	subset.indices = { 1, 2 };

	// Run the network first forward then backwards, then compute the brute force gradient and compare.
	printf("Fast gradient (b)...\n");
	network.ClearDeltaWeightSum();
	for (auto index : subset.indices) {
		network.layers[0]->data = subset.dataSet->images[index].data;
		LossLayer *finalLayer = (LossLayer *)network.layers.back();
		assert(finalLayer->type == LayerType::SOFTMAX_LOSS || finalLayer->type == LayerType::SVM_LOSS);
		finalLayer->label = subset.dataSet->labels[index];
		network.RunForwardPass();
		network.RunBackwardPass();  // Accumulates delta weights.
	}
	network.ScaleDeltaWeightSum(1.0f / subset.indices.size());

	std::unique_ptr<float[]> deltaWeightSum(new float[testLayer->numWeights]{});
	printf("Computing test gradient over %d examples by brute force (a)...\n", (int)subset.indices.size());
	ComputeDeltaWeightSumBruteForce(network, subset, testLayer, deltaWeightSum.get());
	int diffCount = DiffVectors(deltaWeightSum.get(), testLayer->deltaWeightSum, testLayer->numWeights, 0.01f, 200);
	printf("Done with test.\n");
	if (diffCount > 1000) {
		return false;
	}
	return true;  // probably ok.
}

void TrainAndEvaluateNetworkStochastic(NeuralNetwork &network, const DataSet &trainingSet, const DataSet &testSet, int maxEpochs) {
	float trainingSpeed = network.hyperParams.trainingSpeed;

	int subsetSize = network.hyperParams.miniBatchSize;
	MiniBatch testSubset;
	testSubset.dataSet = &testSet;
	testSubset.indices = GetFullSet(testSubset.dataSet->images.size());

	RunStats stats;

	std::vector<std::vector<int>> subsets;
	for (int epoch = 0; epoch < maxEpochs; epoch++) {
		// Generate a new bunch of subsets for each epoch. We could also do fun things
		// like example augmentation (distorting images, etc).
		subsets = GenerateRandomSubsets(trainingSet.images.size(), subsetSize);

		// Decay training speed every N epochs. Tunable through network.hyperParams.
		if (epoch != 0 && (epoch % network.hyperParams.trainingEpochsSlowdown == 0))
			trainingSpeed *= network.hyperParams.trainingSlowdownFactor;

		printf("Epoch %d, trainingSpeed=%f\n", epoch + 1, trainingSpeed);
		for (int i = 0; i < (int)subsets.size(); i++) {
			MiniBatch subset;
			subset.dataSet = &trainingSet;
			// Train on different subsets each round (stochastic gradient descent)
			subset.indices = subsets[i];

			// printf("Round %d/%d (subset %d/%d)\n", i + 1, rounds, subsetIndex + 1, (int)subsets.size());
			//float loss = ComputeLossOverSubset(network, subset);

			// TrainLayerBruteForce(network, subset, &linearLayer, trainingSpeed);
			TrainNetworkOnMinibatch(network, subset, trainingSpeed);
			// UpdateLayerFast(network, subset, &linearLayer, trainingSpeed);
			/*
			stats = {};
			float lossAfterTraining = ComputeLossOverSubset(network, subset, &stats);

			PrintFloatVector("Neurons", network.layers.back()->data, network.layers.back()->numData);
			printf("Loss before: %0.3f\n", loss);
			printf("Loss after: %0.3f\n", lossAfterTraining);
			stats.Print();*/
		}
		printf("Running on testset (%d images)...\n", (int)testSubset.dataSet->images.size());
		stats = {};
		float lossOnTestset = ComputeLossOverMinibatch(network, testSubset, &stats);
		printf("Loss on testset: %f\n", lossOnTestset);
		stats.Print();
	}
}
