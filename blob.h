#pragma once

struct Dim {
	int depth, height, width;

	int TotalSize() const {
		return depth * height * width;
	}
	int GetIndex(int x, int y, int z) const {
		return (width * height * z) + (width * y) + x;
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
