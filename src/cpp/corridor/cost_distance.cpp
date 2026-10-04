// Cost-distance (Dijkstra, 8-neighbour) over a 10 m resistance raster.
//
// Usage:
//   cost_distance --resistance <file> --source <file> --output <file>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "cpl_string.h"
#include "gdal_priv.h"

namespace {

constexpr float kNoData = -9999.0F;
constexpr float kInf = std::numeric_limits<float>::infinity();
constexpr double kCellSize = 10.0;
constexpr double kGeoTolerance = 1e-9;
constexpr double kCellTolerance = 1e-6;
constexpr double kDiagonal = kCellSize * 1.4142135623730951;
constexpr int kSourceBlockRows = 256;
constexpr std::size_t kProgressInterval = 1'000'000;
constexpr std::uint32_t kNotInHeap = std::numeric_limits<std::uint32_t>::max();

// ---------------------------------------------------------------------------
// Small helpers
// ---------------------------------------------------------------------------

struct Options {
  std::string resistance_path;
  std::string source_path;
  std::string output_path;
};

struct Neighbor {
  int d_row;
  int d_col;
  float step; // metric distance to the neighbour
};

constexpr std::array<Neighbor, 8> kNeighbors{{
    {-1, -1, static_cast<float>(kDiagonal)},
    {-1, 0, static_cast<float>(kCellSize)},
    {-1, 1, static_cast<float>(kDiagonal)},
    {0, -1, static_cast<float>(kCellSize)},
    {0, 1, static_cast<float>(kCellSize)},
    {1, -1, static_cast<float>(kDiagonal)},
    {1, 0, static_cast<float>(kCellSize)},
    {1, 1, static_cast<float>(kDiagonal)},
}};

struct ResistanceStats {
  std::size_t valid_count = 0;
  std::size_t invalid_count = 0;
  float min = std::numeric_limits<float>::max();
  float max = std::numeric_limits<float>::lowest();
};

[[nodiscard]] inline bool IsValidResistance(float value) {
  return std::isfinite(value) && value >= 0.0F && value <= 1.0F;
}

void CheckGdal(CPLErr status, const std::string &message) {
  if (status != CE_None) {
    throw std::runtime_error(message);
  }
}

[[nodiscard]] Options ParseArgs(int argc, char **argv) {
  Options options;
  for (int i = 1; i < argc; i += 2) {
    const std::string flag = argv[i];
    if (i + 1 >= argc) {
      throw std::invalid_argument("Missing value for " + flag);
    }
    const std::string value = argv[i + 1];

    if (flag == "--resistance") {
      options.resistance_path = value;
    } else if (flag == "--source") {
      options.source_path = value;
    } else if (flag == "--output") {
      options.output_path = value;
    } else {
      throw std::invalid_argument("Unknown argument: " + flag);
    }
  }

  if (options.resistance_path.empty() || options.source_path.empty() ||
      options.output_path.empty()) {
    throw std::invalid_argument(
        "Usage: cost_distance --resistance <file> --source <file> "
        "--output <file>");
  }
  return options;
}

// ---------------------------------------------------------------------------
// Indexed min-heap: every pixel appears at most once, decrease-key is O(log n)
// ---------------------------------------------------------------------------

class IndexedMinHeap {
public:
  explicit IndexedMinHeap(std::size_t capacity)
      : position_(capacity, kNotInHeap) {
    if (capacity >= kNotInHeap) {
      throw std::length_error("Raster too large for 32-bit pixel indices.");
    }
    heap_.reserve(capacity);
  }

  [[nodiscard]] bool Empty() const { return heap_.empty(); }

  void PushOrDecrease(std::uint32_t index, float cost) {
    const std::uint32_t pos = position_[index];

    if (pos == kNotInHeap) {
      position_[index] = static_cast<std::uint32_t>(heap_.size());
      heap_.push_back({cost, index});
      SiftUp(heap_.size() - 1);
    } else if (cost < heap_[pos].cost) {
      heap_[pos].cost = cost;
      SiftUp(pos);
    }
  }

  struct Node {
    float cost;
    std::uint32_t index;
  };

  Node PopMin() {
    if (heap_.empty()) {
      throw std::runtime_error("PopMin() called on an empty heap.");
    }

    SwapNodes(0, heap_.size() - 1);
    const Node result = heap_.back();
    heap_.pop_back();
    position_[result.index] = kNotInHeap;

    if (!heap_.empty()) {
      SiftDown(0);
    }
    return result;
  }

private:
  std::vector<Node> heap_;
  std::vector<std::uint32_t> position_;

  void SwapNodes(std::size_t a, std::size_t b) {
    std::swap(heap_[a], heap_[b]);
    position_[heap_[a].index] = static_cast<std::uint32_t>(a);
    position_[heap_[b].index] = static_cast<std::uint32_t>(b);
  }

  void SiftUp(std::size_t i) {
    while (i > 0) {
      const std::size_t parent = (i - 1) / 2;
      if (heap_[i].cost >= heap_[parent].cost) {
        break;
      }
      SwapNodes(i, parent);
      i = parent;
    }
  }

  void SiftDown(std::size_t i) {
    const std::size_t size = heap_.size();
    while (true) {
      const std::size_t left = (i * 2) + 1;
      const std::size_t right = left + 1;
      std::size_t smallest = i;

      if (left < size && heap_[left].cost < heap_[smallest].cost) {
        smallest = left;
      }
      if (right < size && heap_[right].cost < heap_[smallest].cost) {
        smallest = right;
      }
      if (smallest == i) {
        break;
      }
      SwapNodes(i, smallest);
      i = smallest;
    }
  }
};

// ---------------------------------------------------------------------------
// Raster I/O
// ---------------------------------------------------------------------------

using GeoTransform = std::array<double, 6>;

[[nodiscard]] GDALDatasetUniquePtr OpenRaster(const std::string &path) {
  GDALDatasetUniquePtr dataset(GDALDataset::FromHandle(
      GDALOpenEx(path.c_str(), GDAL_OF_RASTER | GDAL_OF_READONLY, nullptr,
                 nullptr, nullptr)));
  if (!dataset) {
    throw std::runtime_error("Failed to open raster: " + path);
  }
  if (dataset->GetRasterCount() < 1) {
    throw std::runtime_error("Raster has no bands: " + path);
  }
  return dataset;
}

[[nodiscard]] GeoTransform ReadGeoTransform(GDALDataset &dataset) {
  GeoTransform gt{};
  if (dataset.GetGeoTransform(gt.data()) != CE_None) {
    throw std::runtime_error("Failed to read geotransform.");
  }
  return gt;
}

void ValidateInputs(GDALDataset &resistance, GDALDataset &source,
                    const GeoTransform &gt) {
  if (resistance.GetRasterXSize() != source.GetRasterXSize() ||
      resistance.GetRasterYSize() != source.GetRasterYSize()) {
    throw std::runtime_error("Raster dimensions differ.");
  }

  const GeoTransform source_gt = ReadGeoTransform(source);
  for (std::size_t i = 0; i < gt.size(); ++i) {
    if (std::abs(gt[i] - source_gt[i]) > kGeoTolerance) {
      throw std::runtime_error("Resistance/source geotransforms differ.");
    }
  }

  if (std::abs(std::abs(gt[1]) - kCellSize) > kCellTolerance ||
      std::abs(std::abs(gt[5]) - kCellSize) > kCellTolerance) {
    throw std::runtime_error(
        "Grid must be 10 m x 10 m (actual: " + std::to_string(std::abs(gt[1])) +
        " x " + std::to_string(std::abs(gt[5])) + ").");
  }
}

// Replace invalid values with kNoData and collect statistics.
[[nodiscard]] ResistanceStats
SanitizeResistance(std::vector<float> &resistance) {
  ResistanceStats stats;
  for (float &value : resistance) {
    if (!IsValidResistance(value)) {
      value = kNoData;
      ++stats.invalid_count;
      continue;
    }
    ++stats.valid_count;
    stats.min = std::min(stats.min, value);
    stats.max = std::max(stats.max, value);
  }
  return stats;
}

// The source raster is read block by block to avoid a second full-size array.
[[nodiscard]] std::size_t SeedSources(GDALRasterBand &band,
                                      const std::vector<float> &resistance,
                                      int width, int height,
                                      std::vector<float> &cost,
                                      IndexedMinHeap &heap) {
  const auto w = static_cast<std::size_t>(width);
  std::vector<std::uint8_t> block(w * kSourceBlockRows);
  std::size_t count = 0;

  for (int row = 0; row < height; row += kSourceBlockRows) {
    const int rows = std::min(kSourceBlockRows, height - row);

    CheckGdal(band.RasterIO(GF_Read, 0, row, width, rows, block.data(), width,
                            rows, GDT_Byte, 0, 0, nullptr),
              "Failed to read source raster.");

    const std::size_t first = static_cast<std::size_t>(row) * w;
    const std::size_t n = static_cast<std::size_t>(rows) * w;

    for (std::size_t i = 0; i < n; ++i) {
      const std::size_t index = first + i;
      if (block[i] == 1 && IsValidResistance(resistance[index])) {
        cost[index] = 0.0F;
        heap.PushOrDecrease(static_cast<std::uint32_t>(index), 0.0F);
        ++count;
      }
    }
  }
  return count;
}

// ---------------------------------------------------------------------------
// Dijkstra
// ---------------------------------------------------------------------------

void RunDijkstra(const std::vector<float> &resistance, int width, int height,
                 std::vector<float> &cost, IndexedMinHeap &heap) {
  const auto w = static_cast<std::uint32_t>(width);
  std::size_t processed = 0;

  while (!heap.Empty()) {
    const auto [current_cost, current_index] = heap.PopMin();

    const int row = static_cast<int>(current_index / w);
    const int col = static_cast<int>(current_index % w);
    const float current_resistance = resistance[current_index];

    if (++processed % kProgressInterval == 0) {
      std::cout << "Processed: " << processed << '\n';
    }

    for (const Neighbor &n : kNeighbors) {
      const int nr = row + n.d_row;
      const int nc = col + n.d_col;
      if (nr < 0 || nr >= height || nc < 0 || nc >= width) {
        continue;
      }

      const std::size_t neighbor_index =
          (static_cast<std::size_t>(nr) * static_cast<std::size_t>(width)) +
          static_cast<std::size_t>(nc);

      const float neighbor_resistance = resistance[neighbor_index];
      if (!IsValidResistance(neighbor_resistance)) {
        continue;
      }

      const float edge_cost =
          (current_resistance + neighbor_resistance) * 0.5F * n.step;
      const float new_cost = current_cost + edge_cost;

      if (new_cost < cost[neighbor_index]) {
        cost[neighbor_index] = new_cost;
        heap.PushOrDecrease(static_cast<std::uint32_t>(neighbor_index),
                            new_cost);
      }
    }
  }
}

// ---------------------------------------------------------------------------
// Output
// ---------------------------------------------------------------------------

// Converts `cost` in place to the output convention (kNoData) and returns the
// number of reachable / unreachable valid pixels.
[[nodiscard]] std::pair<std::size_t, std::size_t>
FinalizeCost(const std::vector<float> &resistance, std::vector<float> &cost) {
  std::size_t reachable = 0;
  std::size_t unreachable = 0;

  for (std::size_t i = 0; i < cost.size(); ++i) {
    if (!IsValidResistance(resistance[i])) {
      cost[i] = kNoData;
    } else if (std::isfinite(cost[i])) {
      ++reachable;
    } else {
      ++unreachable;
      cost[i] = kNoData;
    }
  }
  return {reachable, unreachable};
}

void WriteOutput(const std::string &path, GDALDataset &reference, int width,
                 int height, const GeoTransform &gt,
                 const std::vector<float> &data) {
  GDALDriver *driver = GetGDALDriverManager()->GetDriverByName("GTiff");
  if (driver == nullptr) {
    throw std::runtime_error("GTiff driver is not available.");
  }

  CPLStringList options;
  options.SetNameValue("COMPRESS", "DEFLATE");
  options.SetNameValue("PREDICTOR", "3"); // floating-point predictor
  options.SetNameValue("TILED", "YES");
  options.SetNameValue("BIGTIFF", "IF_SAFER");

  GDALDatasetUniquePtr output(driver->Create(path.c_str(), width, height, 1,
                                             GDT_Float32, options.List()));
  if (!output) {
    throw std::runtime_error("Failed to create output: " + path);
  }

  output->SetGeoTransform(gt.data());
  if (const char *projection = reference.GetProjectionRef()) {
    output->SetProjection(projection);
  }

  GDALRasterBand *band = output->GetRasterBand(1);
  band->SetNoDataValue(static_cast<double>(kNoData));

  CheckGdal(band->RasterIO(GF_Write, 0, 0, width, height,
                           const_cast<float *>(data.data()), width, height,
                           GDT_Float32, 0, 0, nullptr),
            "Failed to write output raster.");

  output->FlushCache();
}

} // namespace

// ---------------------------------------------------------------------------

int main(int argc, char **argv) {
  try {
    const Options options = ParseArgs(argc, argv);

    GDALAllRegister();

    GDALDatasetUniquePtr resistance_ds = OpenRaster(options.resistance_path);
    GDALDatasetUniquePtr source_ds = OpenRaster(options.source_path);

    const GeoTransform gt = ReadGeoTransform(*resistance_ds);
    ValidateInputs(*resistance_ds, *source_ds, gt);

    const int width = resistance_ds->GetRasterXSize();
    const int height = resistance_ds->GetRasterYSize();
    const std::size_t pixel_count =
        static_cast<std::size_t>(width) * static_cast<std::size_t>(height);

    std::cout << "Raster dimensions: " << width << " x " << height << '\n'
              << "Total pixels: " << pixel_count << '\n';

    // Memory: resistance (4 B) + cost (4 B) + heap position (4 B) + heap
    // nodes (8 B, worst case) per pixel.
    std::vector<float> resistance(pixel_count);
    CheckGdal(resistance_ds->GetRasterBand(1)->RasterIO(
                  GF_Read, 0, 0, width, height, resistance.data(), width,
                  height, GDT_Float32, 0, 0, nullptr),
              "Failed to read resistance raster.");

    const ResistanceStats stats = SanitizeResistance(resistance);
    std::cout << "Valid resistance pixels: " << stats.valid_count << '\n'
              << "Invalid resistance pixels: " << stats.invalid_count << '\n';
    if (stats.valid_count == 0) {
      throw std::runtime_error("No valid resistance pixels.");
    }
    std::cout << "Resistance range: " << stats.min << " - " << stats.max
              << '\n';

    std::vector<float> cost(pixel_count, kInf);
    IndexedMinHeap heap(pixel_count);

    const std::size_t source_count = SeedSources(
        *source_ds->GetRasterBand(1), resistance, width, height, cost, heap);
    std::cout << "Source pixels: " << source_count << '\n';
    if (source_count == 0) {
      throw std::runtime_error("No valid source pixels.");
    }

    RunDijkstra(resistance, width, height, cost, heap);

    const auto [reachable, unreachable] = FinalizeCost(resistance, cost);
    std::cout << "Reachable valid pixels: " << reachable << '\n'
              << "Unreachable valid pixels: " << unreachable << '\n';

    WriteOutput(options.output_path, *resistance_ds, width, height, gt, cost);

    std::cout << "Dijkstra completed.\nOutput: " << options.output_path << '\n';
    return 0;
  } catch (const std::exception &e) {
    std::cerr << "Error: " << e.what() << '\n';
    return 1;
  }
}
