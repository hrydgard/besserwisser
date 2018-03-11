#pragma once

struct Dim {
	int count = 1;
	int depth, height, width;

	int TotalSize() const {
		return count * depth * height * width;
	}
	int GetIndex(int x, int y, int z, int n = 0) const {
		return n * (depth * width * height) + (width * height * z) + (width * y) + x;
	}
};

enum class DataType {
	FLOAT32,
};

struct Blob {
	~Blob() {
		delete[] data;
	}
	DataType type = DataType::FLOAT32;
	float *data = nullptr;
	size_t size = 0;
	Dim dim;
};
