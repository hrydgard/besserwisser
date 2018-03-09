#pragma once

#include <cstdint>
#include <vector>

// Training and evaluation utilities.

struct DataSet {
	std::vector<DataVector> images;
	std::vector<uint8_t> labels;
};

struct Subset {
	DataSet *dataSet;
	std::vector<int> indices;
};

