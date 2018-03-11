#include "layer.h"
#include "network.h"
#include "math_util.h"

void FcLayer::Initialize() {
	data = new float[numData];
	numWeights = numData * numInputs;
	weights = new float[numWeights]{};
	numGradients = numInputs;
	gradient = new float[numGradients]{};  // Here we'll accumulate gradients before we do the adjust.
	GaussianNoise(weights, numWeights, network_->hyperParams.weightInitScale);
}

void FcLayer::Forward(const float *input) {
	// Just a matrix*vector multiplication.
	for (int y = 0; y < numData; y++) {
		data[y] = DotAVX(input, &weights[y * numInputs], numInputs);
	}
}

void FcLayer::Backward(const float *prev_data, const float *next_gradient) {
	float regStrength = network_->hyperParams.regStrength;

	// Partial derivative dL/dx.
	for (int y = 0; y < numData; y++) {
		int offset = y * numInputs;

		// The derivative of a multiplication with respect to a variable is the other variable.
		// Then do the chain rule multiplication. Remember to add on the partial derivative
		// of the regularization function, which turns out to be very simple.
		// Also note that we regularize the biases if they've been baked into weights, we don't care.
		// The literature says that it really doesn't seem to matter but is unclear on why.

		//for (int x = 0; x < numInputs; x++)
		//	deltaWeightSum[offset + x] += prev_data[x] * next_gradient[y] + weights[offset + x] * regStrength;
		AccumulateScaledVectors(deltaWeightSum + offset, prev_data, next_gradient[y], weights + offset, regStrength, numInputs);
	}

	if (skipBackProp)
		return;

	// We also need to back propagate the gradient through.
	// NOTE: We should be able to skip this if the previous layer is an image (or first!)!
	for (int x = 0; x < numInputs; x++) {
		float sum = 0.0f;
		for (int y = 0; y < numData; y++) {
			float w = weights[y * numInputs + x];
			sum += w * next_gradient[y];
		}
		gradient[x] = sum;
	}
}

float FcLayer::GetRegularizationLoss() {
	// Simple L2 norm.
	// The derivative is used in Backward() so if you change this,
	// gotta change there too.
	// for (int i = 0; i < numWeights; i++)
	//   sum += sqr(weights[i]);
	return SumSquaresAVX(weights, numWeights);
}

void FcLayer::ClearDeltaWeightSum() {
	if (!deltaWeightSum) {
		// We must be training.
		deltaWeightSum = new float[numWeights];
	}
	memset(deltaWeightSum, 0, numWeights * sizeof(float));
}

void FcLayer::ScaleDeltaWeightSum(float factor) {
	ScaleInPlace(deltaWeightSum, factor, numWeights);
}

void FcLayer::UpdateWeights(float trainingSpeed) {
	// Simple gradient descent. Should try with momentum etc as well.
	// for (int i = 0; i < layer->numWeights; i++)
	//   weights[i] -= deltaWeightSum[i] * speed;
	SaxpyAVX(numWeights, -trainingSpeed, deltaWeightSum, weights);
}

void ConvLayer::Initialize() {
	assert(kernelDim.width == kernelDim.height);
	assert(inputDim.width - kernelDim.width + 1 == outputDim.width);
	assert(inputDim.height - kernelDim.width + 1 == outputDim.height);
	assert(inputDim.width - kernelDim.width + 1 == outputDim.width);
	assert(kernelDim.depth == inputDim.depth);

	// numData is the size of the input image (or previous layers).
	// color images are passed in with 3 channels.
	assert(numInputs == inputDim.width * inputDim.height * inputDim.depth);
	data = new float[numData];

	numWeights = numData * numInputs;
	weights = new float[numWeights] {};
	bias = 1.0f;  // ?

	numGradients = numInputs;
	gradient = new float[numGradients] {};  // Here we'll accumulate gradients before we do the adjust.

	// TODO: Do circular weighting in the initialization to try to
	// encourage nice Gabor-like filters.
	GaussianNoise(weights, numWeights, network_->hyperParams.weightInitScale);
}

void ConvLayer::Forward(const float *input) {
	int border2 = (kernelDim.width - 1) / 2;
	// 3D convolution. This is gonna be slow. And need a lot of vectorization.
	int kernelPitch = kernelDim.TotalSize();

	for (int z = 0; z < outputDim.depth; z++) {
		// For each neuron:
		// Do a 2D by 3D correlation. That is, convolve all layers
		// with the kernel for this neuron to produce an output layer.
		for (int y = 0; y < inputDim.height - kernelDim.height + 1; y++) {
			for (int x = 0; x < inputDim.width - kernelDim.width + 1; x++) {
				float sum = 0.0f;
				for (int c = 0; c < kernelDim.depth; c++) {
					for (int b = 0; b < kernelDim.height; b++) {
						for (int a = 0; a < kernelDim.width; a++) {
							int inputIndex = inputDim.GetIndex(x + a, y + b, z + c);
							int weightIndex = kernelDim.GetIndex(a, b, c);
							sum += weights[weightIndex] * input[inputIndex];
						}
					}
				}
				int outputIndex = outputDim.GetIndex(x, y, z);
				data[outputIndex] = sum;
			}
		}
	}
}

void ConvLayer::Backward(const float *prev_data, const float *next_gradient) {
	int border2 = (kernelDim.width - 1) / 2;
	// 3D convolution. This is gonna be slow. And need a lot of vectorization.
	int kernelPitch = kernelDim.TotalSize();

	// First, update the training gradients.
	for (int z = 0; z < outputDim.depth; z++) {
		// For each neuron:
		// Do a 2D by 3D correlation. That is, convolve all layers
		// with the kernel for this neuron to produce an output layer.
		for (int y = 0; y < inputDim.height - kernelDim.height + 1; y++) {
			for (int x = 0; x < inputDim.width - kernelDim.width + 1; x++) {
				int outputIndex = outputDim.GetIndex(x, y, z);

				for (int c = 0; c < kernelDim.depth; c++) {
					for (int b = 0; b < kernelDim.height; b++) {
						for (int a = 0; a < kernelDim.width; a++) {
							int inputIndex = inputDim.GetIndex(x + a, y + b, z + c);
							int weightIndex = kernelDim.GetIndex(a, b, c);
							//for (int x = 0; x < numInputs; x++)
							//	deltaWeightSum[offset + x] += prev_data[x] * next_gradient[y] + weights[offset + x] * regStrength;

							deltaWeightSum[weightIndex] += prev_data[inputIndex] * next_gradient[outputIndex];
						}
					}
				}
				//data[outputIndex] = sum;
			}
		}
	}

	// Then, backprop. OMG.
}

float ConvLayer::GetRegularizationLoss() {
	// Simple L2 norm.
	// The derivative is used in Backward() so if you change this,
	// gotta change there too.
	// for (int i = 0; i < numWeights; i++)
	//   sum += sqr(weights[i]);
	return SumSquaresAVX(weights, numWeights);
}

void ConvLayer::ClearDeltaWeightSum() {
	if (!deltaWeightSum) {
		// We must be training.
		deltaWeightSum = new float[numWeights];
	}
	memset(deltaWeightSum, 0, numWeights * sizeof(float));
}

void ConvLayer::ScaleDeltaWeightSum(float factor) {
	ScaleInPlace(deltaWeightSum, factor, numWeights);
}

void ConvLayer::UpdateWeights(float trainingSpeed) {
	// Simple gradient descent. Should try with momentum etc as well.
	// for (int i = 0; i < layer->numWeights; i++)
	//   weights[i] -= deltaWeightSum[i] * speed;
	SaxpyAVX(numWeights, -trainingSpeed, deltaWeightSum, weights);
}

void LossLayer::Initialize() {
	assert(numData == 1);
	assert(numInputs >= 1);
	data = new float[numInputs];
	numGradients = numInputs;
	gradient = new float[numGradients] {};
}

void SVMLossLayer::Forward(const float *input) {
	assert(label != -1);
	float sum = 0.0f;
	for (size_t i = 0; i < numInputs; i++) {
		if (i != label) {
			sum += std::max(0.0f, input[i] - input[label] + 1.0f);
		}
	}
	data[0] = sum;
}

void SVMLossLayer::Backward(const float *prev_data, const float *next_gradient) {
	// There's no input gradient, we compute the original dL/dz gradient.
	assert(!next_gradient);

	// Note! The function we are differentiating here is the complete loss,
	// that is the sum of all the "neurons". We're differentiating the entire sum
	// with respect to each row.
	// The "correct" level is involved in all the rows so it needs a summing loop,
	// while the others can be computed directly.
	int positive_count = 0;
	for (size_t i = 0; i < numInputs; i++) {
		if (i != label)
			positive_count += (prev_data[i] - prev_data[label] + 1.0f) > 0.0f;
	}

	for (size_t i = 0; i < numInputs; i++) {
		if (i == label) {
			gradient[i] = -(float)positive_count;
		} else {
			gradient[i] = (prev_data[i] - prev_data[label] + 1.0f) > 0.0f ? 1.0f : 0.0f;
		}
	}
}

void SoftMaxLossLayer::Forward(const float *input) {
	float expSum = 0.0f;
	for (size_t i = 0; i < numInputs; i++) {
		expSum += expf(input[i]);
	}
	data[0] = -logf(expf(input[label]) / expSum);
}

void SoftMaxLossLayer::Backward(const float *prev_data, const float *next_gradient) {
	// There's no input gradient, we compute the original dL/dz gradient.
	assert(!next_gradient);

	// http://cs231n.github.io/neural-networks-case-study/#together
	// For simplicity, partially recompute the forward pass. This code isn't the bottleneck.
	float expSum = 0.0f;
	for (size_t i = 0; i < numInputs; i++) {
		expSum += expf(prev_data[i]);
	}
	for (size_t i = 0; i < numInputs; i++) {
		gradient[i] = expf(prev_data[i])/expSum - (label == i ? 1.0f : 0.0f);
	}
}

void ActivationLayer::Initialize() {
	assert(numData == numInputs);
	data = new float[numData];
	numGradients = numData;
	gradient = new float[numGradients] {};
}

void SigmoidLayer::Forward(const float *input) {
	// TODO: This can be very easily SIMD'd.
	for (int i = 0; i < numData; i++) {
		data[i] = Sigmoid(input[i]);
	}
}

void SigmoidLayer::Backward(const float *prev_data, const float *next_gradient) {
	for (int i = 0; i < numData; i++) {
		gradient[i] = (1.0f - data[i]) * data[i] * next_gradient[i];
	}
}

void ReluLayer::Forward(const float *input) {
	// for (int i = 0; i < numData; i++)
	//   data[i] = std::max(0.0f, input[i]);
	ClampDownToZero(data, input, numData);
}

void ReluLayer::Backward(const float *prev_data, const float *next_gradient) {
	for (int i = 0; i < numData; i++) {
		gradient[i] = prev_data[i] > 0.0f ? next_gradient[i] : 0.0f;
	}
}

void LeakyReluLayer::Forward(const float *input) {
	for (int i = 0; i < numData; i++)
		data[i] = std::max(coef * input[i], input[i]);
}

void LeakyReluLayer::Backward(const float *prev_data, const float *next_gradient) {
	for (int i = 0; i < numData; i++) {
		gradient[i] = prev_data[i] > 0.0f ? next_gradient[i] : next_gradient[i] * coef;
	}
}

void Relu6Layer::Forward(const float *input) {
	// TODO: This can be very easily SIMD'd.
	for (int i = 0; i < numData; i++) {
		data[i] = std::max(0.0f, std::min(input[i], 6.0f));
	}
}

void Relu6Layer::Backward(const float *prev_data, const float *input) {
	for (int i = 0; i < numData; i++) {
		gradient[i] = data[i] > 6.0f ? 0.0f : (data[i] > 0.0f ? input[i] : 0.0f);
	}
}

void MaxPoolLayer::Initialize() {
	assert(inputDim.width % 2 == 0);
	assert(inputDim.height % 2 == 0);
	assert(inputDim.width / 2 == outputDim.width);
	assert(inputDim.height / 2 == outputDim.height);
	assert(inputDim.depth == outputDim.depth);
}

void MaxPoolLayer::Forward(const float *input) {
	// For batches, will have to add a fourth loop level. On the GPU, I guess
	// we'll merge that into depth.
	for (int z = 0; z < outputDim.depth; z++) {
		for (int y = 0; y < outputDim.height; y++) {
			for (int x = 0; x < outputDim.width; x++) {
				int inputIndex = inputDim.GetIndex(x * 2, y * 2, z * 2);
				int outputIndex = outputDim.GetIndex(x, y, z);
				int maxInd = 0;
				float maxValue = input[inputIndex];
				if (input[inputIndex + 1] > maxValue) {
					maxInd = 1;
					maxValue = input[inputIndex + 1];
				}
				if (input[inputIndex + inputDim.width] > maxValue) {
					maxInd = 2;
					maxValue = input[inputIndex + inputDim.width];
				}
				if (input[inputIndex + inputDim.width + 1] > maxValue) {
					maxInd = 3;
					maxValue = input[inputIndex + inputDim.width + 1];
				}
				maxIndex[outputIndex] = maxInd;
				data[outputIndex] = maxValue;
			}
		}
	}
}

void MaxPoolLayer::Backward(const float *prev_data, const float *next_gradient) {
	const int offsets[4] = {
		0, 1,
		inputDim.width, inputDim.width + 1
	};
	for (int z = 0; z < outputDim.depth; z++) {
		for (int y = 0; y < outputDim.height; y++) {
			for (int x = 0; x < outputDim.width; x++) {
				int outputIndex = outputDim.GetIndex(x, y, z);
				int inputIndex = inputDim.GetIndex(x * 2, y * 2, z * 2);
				int maxInd = maxIndex[outputIndex];
				for (int i = 0; i < 4; i++) {
					gradient[inputIndex + offsets[i]] = i == maxInd ? next_gradient[outputIndex] : 0.0f;
				}
			}
		}
	}
}