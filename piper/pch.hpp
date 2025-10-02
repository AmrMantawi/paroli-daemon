#pragma once

#include <vector>
#include <string>
#include <iostream>
#include <fstream>
#include <map>
#include <algorithm>

// Handle spdlog/fmt compatibility for cross-compilation
#define SPDLOG_FMT_EXTERNAL 1
#include <fmt/core.h>
#include <spdlog/spdlog.h>

#include <nlohmann/json.hpp>

#include <xtensor/xarray.hpp>
