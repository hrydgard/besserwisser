#pragma once

#include <cstdint>
#include <vector>
#include <string>

#include "math_util.h"
#include "blob.h"

std::vector<Blob> LoadMNISTImages(std::string path);
std::vector<uint8_t> LoadMNISTLabels(std::string path);