// Circuit Theory (optimized)
//
// Build:
//   g++ -O3 -march=native -fopenmp -std=c++17 circuit_theory.cpp \
//       $(gdal-config --cflags --libs) -o circuit_theory
//
// Optimizations vs. the original version:
//   * FIX: initial residual is r0 = b (x_f = 0), not b - A(V_fixed) (was 2x).
//   * 1-cell halo (padded grid) -> stencil without bounds / validity checks.
//   * Edge conductance precomputed once (4 directions, symmetric graph):
//         G = 0.5 * (1/Ra + 1/Rb) / d      (identical to harmonic mean form)
//   * Jacobi inverse diagonal precomputed (no recomputation per iteration).
//   * Fused PCG: 3 memory passes per iteration, z vector eliminated,
//     p.Ap fused into the operator.
//   * OpenMP on operator, vector updates and post-processing.
//   * Free components that cannot reach any source are pruned (V = 0).
//   * Input rasters are released right after the grid is built.

#include <cpl_conv.h>
#include <gdal_priv.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace {

constexpr double SQRT2 = 1.4142135623730950488;
constexpr double INV_SQRT2 = 1.0 / SQRT2;

struct Config {
  std::string resistance_path;
  std::string source_path;
  std::string current_output;
  std::string connectivity_output;

  int max_iterations = 5000;
  double tolerance = 1e-6; // relative residual ||r|| / ||r0||
  int threads = 0;         // 0 = OpenMP default
  int log_interval = 100;  // print solver progress every N iterations

  bool verbose = true;
};

struct RasterInfo {
  int width = 0;
  int height = 0;
  double geotransform[6] = {0.0, 1.0, 0.0, 0.0, 0.0, -1.0};
  std::string projection;
};

struct MemoryStats {
  std::size_t rss_bytes = 0;
};

struct SolverStats {
  int iterations = 0;
  double initial_residual = 0.0;
  double final_residual = 0.0; // absolute
  double relative_residual = 0.0;
  bool converged = false;
};

// Cell states on the padded grid.
enum : uint8_t {
  ST_INVALID = 0,  // NoData / halo
  ST_FREE = 1,     // unknown voltage, part of the linear system
  ST_SOURCE = 2,   // Dirichlet V = 1
  ST_BOUNDARY = 3, // Dirichlet V = 0
  ST_INACTIVE = 4  // free cell with no path to a source: V = 0, not solved
};

// -----------------------------------------------------------------------------
// Utility
// -----------------------------------------------------------------------------

void print_usage(const char *program) {
  std::cerr << "\nUsage:\n"
            << "  " << program << " \\\n"
            << "    --resistance <path> \\\n"
            << "    --source <path> \\\n"
            << "    --current-output <path> \\\n"
            << "    --connectivity-output <path> \\\n"
            << "    [--tolerance <value>] \\\n"
            << "    [--max-iterations <integer>] \\\n"
            << "    [--threads <integer>] \\\n"
            << "    [--log-interval <integer>] \\\n"
            << "    [--quiet]\n\n"
            << "Boundary formulation:\n"
            << "  source cells  : V = 1\n"
            << "  outer boundary: V = 0\n\n";
}

Config parse_arguments(int argc, char **argv) {
  Config config;

  for (int i = 1; i < argc; ++i) {
    const std::string arg(argv[i]);

    auto require_value = [&](const std::string &name) -> std::string {
      if (i + 1 >= argc) {
        throw std::runtime_error("Missing value for " + name);
      }
      return argv[++i];
    };

    if (arg == "--resistance") {
      config.resistance_path = require_value(arg);
    } else if (arg == "--source") {
      config.source_path = require_value(arg);
    } else if (arg == "--current-output") {
      config.current_output = require_value(arg);
    } else if (arg == "--connectivity-output") {
      config.connectivity_output = require_value(arg);
    } else if (arg == "--tolerance") {
      config.tolerance = std::stod(require_value(arg));
    } else if (arg == "--max-iterations") {
      config.max_iterations = std::stoi(require_value(arg));
    } else if (arg == "--threads") {
      config.threads = std::stoi(require_value(arg));
    } else if (arg == "--log-interval") {
      config.log_interval = std::stoi(require_value(arg));
    } else if (arg == "--quiet") {
      config.verbose = false;
    } else if (arg == "--help" || arg == "-h") {
      print_usage(argv[0]);
      std::exit(EXIT_SUCCESS);
    } else {
      throw std::runtime_error("Unknown argument: " + arg);
    }
  }

  if (config.resistance_path.empty() || config.source_path.empty() ||
      config.current_output.empty() || config.connectivity_output.empty()) {
    print_usage(argv[0]);
    throw std::runtime_error("Required arguments are missing.");
  }

  if (config.max_iterations <= 0) {
    throw std::runtime_error("--max-iterations must be > 0.");
  }

  if (config.tolerance <= 0.0) {
    throw std::runtime_error("--tolerance must be > 0.");
  }

  if (config.log_interval <= 0) {
    throw std::runtime_error("--log-interval must be > 0.");
  }

  if (config.threads < 0) {
    throw std::runtime_error("--threads must be >= 0.");
  }

  return config;
}

// -----------------------------------------------------------------------------
// Memory statistics
// -----------------------------------------------------------------------------

MemoryStats get_memory_stats() {
  MemoryStats stats;

#ifdef __linux__
  std::ifstream status("/proc/self/status");
  if (!status.is_open()) {
    return stats;
  }

  std::string line;
  while (std::getline(status, line)) {
    if (line.rfind("VmRSS:", 0) == 0) {
      std::istringstream iss(line);
      std::string label;
      std::size_t kb = 0;
      iss >> label >> kb;
      stats.rss_bytes = kb * 1024;
      break;
    }
  }
#endif

  return stats;
}

std::string format_bytes(std::size_t bytes) {
  constexpr double KB = 1024.0;
  constexpr double MB = KB * 1024.0;
  constexpr double GB = MB * 1024.0;

  std::ostringstream out;
  out << std::fixed << std::setprecision(2);

  if (bytes >= static_cast<std::size_t>(GB)) {
    out << bytes / GB << " GB";
  } else if (bytes >= static_cast<std::size_t>(MB)) {
    out << bytes / MB << " MB";
  } else if (bytes >= static_cast<std::size_t>(KB)) {
    out << bytes / KB << " KB";
  } else {
    out << bytes << " B";
  }

  return out.str();
}

void log_memory(const std::string &label) {
  const MemoryStats stats = get_memory_stats();

  std::cout << "[MEM] " << std::left << std::setw(28) << label << " RSS=";

  if (stats.rss_bytes > 0) {
    std::cout << format_bytes(stats.rss_bytes);
  } else {
    std::cout << "unavailable";
  }

  std::cout << '\n';
}

template <typename T> void release(std::vector<T> &v) {
  std::vector<T>().swap(v);
}

// -----------------------------------------------------------------------------
// Raster helpers
// -----------------------------------------------------------------------------

class Raster {
public:
  RasterInfo info;
  std::vector<float> data;

  // Reads band 1 fully into memory and closes the dataset immediately.
  void open(const std::string &path) {
    std::unique_ptr<GDALDataset, decltype(&GDALClose)> dataset(
        static_cast<GDALDataset *>(GDALOpen(path.c_str(), GA_ReadOnly)),
        GDALClose);

    if (!dataset) {
      throw std::runtime_error("Failed to open raster: " + path);
    }

    if (dataset->GetRasterCount() < 1) {
      throw std::runtime_error("Raster has no bands: " + path);
    }

    GDALRasterBand *band = dataset->GetRasterBand(1);

    info.width = dataset->GetRasterXSize();
    info.height = dataset->GetRasterYSize();

    if (dataset->GetGeoTransform(info.geotransform) != CE_None) {
      throw std::runtime_error("Failed to read geotransform: " + path);
    }

    const char *projection = dataset->GetProjectionRef();
    if (projection != nullptr) {
      info.projection = projection;
    }

    const std::size_t cells = static_cast<std::size_t>(info.width) *
                              static_cast<std::size_t>(info.height);

    data.resize(cells);

    if (band->RasterIO(GF_Read, 0, 0, info.width, info.height, data.data(),
                       info.width, info.height, GDT_Float32, 0, 0,
                       nullptr) != CE_None) {
      throw std::runtime_error("Failed to read raster data: " + path);
    }
  }
};

bool same_grid(const RasterInfo &a, const RasterInfo &b) {
  if (a.width != b.width || a.height != b.height) {
    return false;
  }

  constexpr double tolerance = 1e-9;

  for (int i = 0; i < 6; ++i) {
    if (std::abs(a.geotransform[i] - b.geotransform[i]) > tolerance) {
      return false;
    }
  }

  return true;
}

// -----------------------------------------------------------------------------
// Circuit Theory model
// -----------------------------------------------------------------------------
//
// Grid layout: padded with a 1-cell halo of ST_INVALID cells.
//   W = width + 2, H = height + 2
//   padded index of raster cell (row, col) = (row + 1) * W + (col + 1)
//
// Conductances are stored per cell for 4 "forward" directions:
//   ge [i] : i <-> i + 1       (E)   d = 1
//   gs [i] : i <-> i + W       (S)   d = 1
//   gse[i] : i <-> i + W + 1   (SE)  d = sqrt(2)
//   gsw[i] : i <-> i + W - 1   (SW)  d = sqrt(2)
// The opposite direction is read from the neighbor (e.g. W link = ge[i - 1]).
// A link to an invalid cell has conductance 0.

class CircuitTheoryEngine {
public:
  CircuitTheoryEngine(Raster &resistance, Raster &source, const Config &config)
      : resistance_(resistance), source_(source), config_(config) {
    width_ = resistance_.info.width;
    height_ = resistance_.info.height;

    cells_ =
        static_cast<std::size_t>(width_) * static_cast<std::size_t>(height_);

    W_ = static_cast<std::size_t>(width_) + 2;
    H_ = static_cast<std::size_t>(height_) + 2;
    N_ = W_ * H_;
  }

  void initialize() {
    std::cout << "\n";
    std::cout << "=============================================\n";
    std::cout << " Circuit Theory Engine (optimized)\n";
    std::cout << "=============================================\n";
    std::cout << "Grid       : " << width_ << " x " << height_ << "\n";
    std::cout << "Cells      : " << cells_ << "\n";
    std::cout << "Neighbors  : 8\n";
    std::cout << "Solver     : Fused Jacobi-PCG (matrix-free, precomputed G)\n";
    std::cout << "Tolerance  : " << config_.tolerance << " (relative)\n";
    std::cout << "Max iter   : " << config_.max_iterations << "\n";
#ifdef _OPENMP
    std::cout << "Threads    : " << omp_get_max_threads() << "\n";
#else
    std::cout << "Threads    : 1 (OpenMP disabled)\n";
#endif
    std::cout << "Boundary   : outer boundary = 0\n";
    std::cout << "Source     : KHL source = 1\n";

    build_grid();
    precompute_conductance();

    log_memory("after initialization");
  }

  void solve() {
    const auto start = std::chrono::steady_clock::now();

    std::cout << "\n[1/3] Solving electrical potential...\n";

    const SolverStats stats = conjugate_gradient();

    std::cout << "\nSolver result:\n";
    std::cout << "  Converged         : " << (stats.converged ? "YES" : "NO")
              << "\n";
    std::cout << "  Iterations        : " << stats.iterations << "\n";
    std::cout << std::scientific
              << "  Initial residual  : " << stats.initial_residual << "\n"
              << "  Final residual    : " << stats.final_residual << "\n"
              << "  Relative residual : " << stats.relative_residual << "\n"
              << std::defaultfloat;

    const double seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - start)
            .count();

    std::cout << "  Solver time       : " << seconds << " s\n";

    log_memory("after solver");
  }

  void calculate_current_density() {
    std::cout << "\n[2/3] Calculating current density...\n";

    // current_density_.assign(cells_, 0.0f);
    current_density_.assign(cells_, std::numeric_limits<float>::quiet_NaN());

    const double *v = voltage_.data();
    const std::size_t W = W_;

#pragma omp parallel for schedule(static)
    for (int row = 0; row < height_; ++row) {
      for (int col = 0; col < width_; ++col) {
        const std::size_t i = (static_cast<std::size_t>(row) + 1) * W +
                              static_cast<std::size_t>(col) + 1;

        if (state_[i] == ST_INVALID) {
          continue;
        }

        const double vi = v[i];

        const double density = ge_[i] * std::abs(vi - v[i + 1]) +
                               ge_[i - 1] * std::abs(vi - v[i - 1]) +
                               gs_[i] * std::abs(vi - v[i + W]) +
                               gs_[i - W] * std::abs(vi - v[i - W]) +
                               gse_[i] * std::abs(vi - v[i + W + 1]) +
                               gse_[i - W - 1] * std::abs(vi - v[i - W - 1]) +
                               gsw_[i] * std::abs(vi - v[i + W - 1]) +
                               gsw_[i - W + 1] * std::abs(vi - v[i - W + 1]);

        current_density_[static_cast<std::size_t>(row) *
                             static_cast<std::size_t>(width_) +
                         static_cast<std::size_t>(col)] =
            static_cast<float>(density);
      }
    }

    // Conductances are no longer needed.
    release(ge_);
    release(gs_);
    release(gse_);
    release(gsw_);

    log_memory("after current density");
  }

  void calculate_connectivity() {
    std::cout << "\n[3/3] Preparing connectivity raster...\n";

    connectivity_.assign(cells_, std::numeric_limits<float>::quiet_NaN());

    /*
     * Baseline definition: connectivity = normalized voltage
     * (source = 1, ground = 0).
     */
#pragma omp parallel for schedule(static)
    for (int row = 0; row < height_; ++row) {
      for (int col = 0; col < width_; ++col) {
        const std::size_t i = (static_cast<std::size_t>(row) + 1) * W_ +
                              static_cast<std::size_t>(col) + 1;

        if (state_[i] == ST_INVALID) {
          continue;
        }

        connectivity_[static_cast<std::size_t>(row) *
                          static_cast<std::size_t>(width_) +
                      static_cast<std::size_t>(col)] =
            static_cast<float>(std::clamp(voltage_[i], 0.0, 1.0));
      }
    }

    release(voltage_);
    release(state_);

    log_memory("after connectivity");
  }

  void write_outputs() {
    std::cout << "\nWriting GeoTIFF outputs...\n";

    write_raster(config_.current_output, current_density_);
    write_raster(config_.connectivity_output, connectivity_);

    std::cout << "Outputs written successfully.\n";
  }

private:
  Raster &resistance_;
  Raster &source_;
  const Config &config_;

  int width_ = 0;
  int height_ = 0;
  std::size_t cells_ = 0; // unpadded

  std::size_t W_ = 0, H_ = 0, N_ = 0; // padded

  std::vector<uint8_t> state_;

  // Precomputed link conductances (padded indexing).
  std::vector<double> ge_, gs_, gse_, gsw_;

  // 1 / diag(A) for free cells, 0 elsewhere.
  std::vector<double> inv_diag_;

  // Padded voltage field.
  std::vector<double> voltage_;

  // Outputs (unpadded).
  std::vector<float> current_density_;
  std::vector<float> connectivity_;

  // ---------------------------------------------------------------------------
  // Grid construction
  // ---------------------------------------------------------------------------

  void build_grid() {
    std::cout << "\nBuilding padded grid and masks...\n";

    state_.assign(N_, ST_INVALID);

    std::size_t valid_count = 0;
    std::size_t source_count = 0;

    // Pass 1: valid + source.
    for (int row = 0; row < height_; ++row) {
      for (int col = 0; col < width_; ++col) {
        const std::size_t ri =
            static_cast<std::size_t>(row) * static_cast<std::size_t>(width_) +
            static_cast<std::size_t>(col);
        const std::size_t pi = (static_cast<std::size_t>(row) + 1) * W_ +
                               static_cast<std::size_t>(col) + 1;

        const float r = resistance_.data[ri];

        if (std::isfinite(r) && r > 0.0f) {
          ++valid_count;

          const float s = source_.data[ri];
          if (std::isfinite(s) && s > 0.0f) {
            state_[pi] = ST_SOURCE;
            ++source_count;
          } else {
            state_[pi] = ST_FREE;
          }
        }
      }
    }

    // Pass 2: boundary = free cell with any of its 8 neighbors invalid
    // (outside raster = halo = invalid). Source keeps priority.
    // Changing FREE -> BOUNDARY does not affect the "is invalid?" test.
    std::size_t boundary_count = 0;
    const std::ptrdiff_t W = static_cast<std::ptrdiff_t>(W_);
    const std::ptrdiff_t offs[8] = {-W - 1, -W, -W + 1, -1, 1, W - 1, W, W + 1};

    for (int row = 0; row < height_; ++row) {
      for (int col = 0; col < width_; ++col) {
        const std::size_t i = (static_cast<std::size_t>(row) + 1) * W_ +
                              static_cast<std::size_t>(col) + 1;

        if (state_[i] != ST_FREE) {
          continue;
        }

        for (int k = 0; k < 8; ++k) {
          if (state_[static_cast<std::size_t>(static_cast<std::ptrdiff_t>(i) +
                                              offs[k])] == ST_INVALID) {
            state_[i] = ST_BOUNDARY;
            ++boundary_count;
            break;
          }
        }
      }
    }

    std::cout << "  Valid cells    : " << valid_count << "\n";
    std::cout << "  Source cells   : " << source_count << "\n";
    std::cout << "  Boundary cells : " << boundary_count << "\n";

    if (source_count == 0) {
      throw std::runtime_error("Source raster contains no valid source cells.");
    }

    if (boundary_count == 0) {
      throw std::runtime_error("No valid landscape boundary cells found.");
    }

    // Pass 3: prune free components that cannot reach any source.
    // Their solution is exactly 0 (all fixed neighbors are ground).
    std::vector<uint8_t> reached(N_, 0);
    std::vector<std::size_t> stack;
    stack.reserve(1 << 16);

    for (std::size_t i = 0; i < N_; ++i) {
      if (state_[i] == ST_SOURCE) {
        stack.push_back(i);
      }
    }

    while (!stack.empty()) {
      const std::size_t i = stack.back();
      stack.pop_back();

      for (int k = 0; k < 8; ++k) {
        const std::size_t j =
            static_cast<std::size_t>(static_cast<std::ptrdiff_t>(i) + offs[k]);

        if (state_[j] == ST_FREE && !reached[j]) {
          reached[j] = 1;
          stack.push_back(j);
        }
      }
    }

    std::size_t active = 0;
    std::size_t pruned = 0;

    for (std::size_t i = 0; i < N_; ++i) {
      if (state_[i] == ST_FREE) {
        if (reached[i]) {
          ++active;
        } else {
          state_[i] = ST_INACTIVE;
          ++pruned;
        }
      }
    }

    std::cout << "  Active unknowns: " << active << "\n";
    std::cout << "  Pruned (no src): " << pruned << "\n";

    voltage_.assign(N_, 0.0);

    log_memory("after masks");
  }

  // ---------------------------------------------------------------------------
  // Conductance precompute
  // ---------------------------------------------------------------------------
  //
  // Harmonic-mean edge resistance R_e = 2RaRb/(Ra+Rb), G = 1/(R_e * d)
  //   =>  G = 0.5 * (1/Ra + 1/Rb) / d

  void precompute_conductance() {
    std::vector<double> icond(N_, 0.0);

    for (int row = 0; row < height_; ++row) {
      for (int col = 0; col < width_; ++col) {
        const std::size_t ri =
            static_cast<std::size_t>(row) * static_cast<std::size_t>(width_) +
            static_cast<std::size_t>(col);
        const std::size_t pi = (static_cast<std::size_t>(row) + 1) * W_ +
                               static_cast<std::size_t>(col) + 1;

        if (state_[pi] != ST_INVALID) {
          icond[pi] = 1.0 / static_cast<double>(resistance_.data[ri]);
        }
      }
    }

    // Input rasters are no longer needed (only geotransform/projection).
    release(resistance_.data);
    release(source_.data);

    ge_.assign(N_, 0.0);
    gs_.assign(N_, 0.0);
    gse_.assign(N_, 0.0);
    gsw_.assign(N_, 0.0);
    inv_diag_.assign(N_, 0.0);

    const std::size_t W = W_;

#pragma omp parallel for schedule(static)
    for (int row = 1; row <= height_; ++row) {
      for (std::size_t c = 1; c <= static_cast<std::size_t>(width_); ++c) {
        const std::size_t i = static_cast<std::size_t>(row) * W + c;

        if (state_[i] == ST_INVALID) {
          continue;
        }

        const double a = icond[i];

        if (state_[i + 1] != ST_INVALID) {
          ge_[i] = 0.5 * (a + icond[i + 1]);
        }
        if (state_[i + W] != ST_INVALID) {
          gs_[i] = 0.5 * (a + icond[i + W]);
        }
        if (state_[i + W + 1] != ST_INVALID) {
          gse_[i] = 0.5 * (a + icond[i + W + 1]) * INV_SQRT2;
        }
        if (state_[i + W - 1] != ST_INVALID) {
          gsw_[i] = 0.5 * (a + icond[i + W - 1]) * INV_SQRT2;
        }
      }
    }

    release(icond);

    // Inverse Jacobi diagonal (sum of all 8 link conductances).
#pragma omp parallel for schedule(static)
    for (int row = 1; row <= height_; ++row) {
      for (std::size_t c = 1; c <= static_cast<std::size_t>(width_); ++c) {
        const std::size_t i = static_cast<std::size_t>(row) * W + c;

        if (state_[i] != ST_FREE) {
          continue;
        }

        const double d = ge_[i] + ge_[i - 1] + gs_[i] + gs_[i - W] + gse_[i] +
                         gse_[i - W - 1] + gsw_[i] + gsw_[i - W + 1];

        inv_diag_[i] = d > 0.0 ? 1.0 / d : 0.0;
      }
    }
  }

  // ---------------------------------------------------------------------------
  // Solver kernels
  // ---------------------------------------------------------------------------

  // y = A_ff x  (x must be zero on every non-free cell),
  // returns x . y. y is zero on non-free cells.
  double apply_operator_dot(const double *x, double *y) const {
    const std::size_t W = W_;
    const double *ge = ge_.data();
    const double *gs = gs_.data();
    const double *gse = gse_.data();
    const double *gsw = gsw_.data();
    const uint8_t *st = state_.data();

    double dot = 0.0;

#pragma omp parallel for reduction(+ : dot) schedule(static)
    for (int row = 1; row <= height_; ++row) {
      const std::size_t base = static_cast<std::size_t>(row) * W;

      for (std::size_t c = 1; c <= static_cast<std::size_t>(width_); ++c) {
        const std::size_t i = base + c;

        if (st[i] != ST_FREE) {
          y[i] = 0.0;
          continue;
        }

        const double xi = x[i];

        const double v = ge[i] * (xi - x[i + 1]) + ge[i - 1] * (xi - x[i - 1]) +
                         gs[i] * (xi - x[i + W]) + gs[i - W] * (xi - x[i - W]) +
                         gse[i] * (xi - x[i + W + 1]) +
                         gse[i - W - 1] * (xi - x[i - W - 1]) +
                         gsw[i] * (xi - x[i + W - 1]) +
                         gsw[i - W + 1] * (xi - x[i - W + 1]);

        y[i] = v;
        dot += xi * v;
      }
    }

    return dot;
  }

  SolverStats conjugate_gradient() {
    SolverStats stats;

    const std::size_t n = N_;

    std::vector<double> r(n, 0.0);
    std::vector<double> p(n, 0.0);
    std::vector<double> Ap(n, 0.0);
    double *x = voltage_.data(); // zero everywhere at this point

    /*
     * Reduced system on free cells (Dirichlet nodes moved to the RHS):
     *
     *     A_ff x_f = b,   b_i = sum_{j in source} G_ij * 1
     *
     * With x_f = 0 the initial residual is simply r0 = b.
     */
    {
      const std::size_t W = W_;
      const uint8_t *st = state_.data();

#pragma omp parallel for schedule(static)
      for (int row = 1; row <= height_; ++row) {
        for (std::size_t c = 1; c <= static_cast<std::size_t>(width_); ++c) {
          const std::size_t i = static_cast<std::size_t>(row) * W + c;

          if (st[i] != ST_FREE) {
            continue;
          }

          auto s = [&](std::size_t j) {
            return st[j] == ST_SOURCE ? 1.0 : 0.0;
          };

          r[i] = ge_[i] * s(i + 1) + ge_[i - 1] * s(i - 1) + gs_[i] * s(i + W) +
                 gs_[i - W] * s(i - W) + gse_[i] * s(i + W + 1) +
                 gse_[i - W - 1] * s(i - W - 1) + gsw_[i] * s(i + W - 1) +
                 gsw_[i - W + 1] * s(i - W + 1);
        }
      }
    }

    // z = M^-1 r, p = z, rz = r.z, ||r0||^2
    double rz = 0.0;
    double rr0 = 0.0;

#pragma omp parallel for reduction(+ : rz, rr0) schedule(static)
    for (std::int64_t i = 0; i < static_cast<std::int64_t>(n); ++i) {
      const double ri = r[i];
      const double zi = inv_diag_[i] * ri;
      p[i] = zi;
      rz += ri * zi;
      rr0 += ri * ri;
    }

    const double initial_norm = std::sqrt(rr0);

    stats.initial_residual = initial_norm;
    stats.final_residual = initial_norm;

    if (!(initial_norm > 0.0)) {
      // No free cell touches a source: solution is trivial.
      stats.converged = true;
      stats.relative_residual = 0.0;
      finalize_voltage();
      return stats;
    }

    for (int it = 1; it <= config_.max_iterations; ++it) {
      // Pass 1: Ap = A p, pAp = p . Ap
      const double pAp = apply_operator_dot(p.data(), Ap.data());

      if (!std::isfinite(pAp) || !(pAp > 0.0)) {
        std::cerr << "\nERROR: CG breakdown at iteration " << it << ".\n";
        stats.iterations = it;
        stats.relative_residual = stats.final_residual / initial_norm;
        finalize_voltage();
        return stats;
      }

      const double alpha = rz / pAp;

      if (!std::isfinite(alpha)) {
        throw std::runtime_error("CG produced non-finite alpha.");
      }

      // Pass 2: x += alpha p; r -= alpha Ap; rr = r.r; rz_new = r.M^-1 r
      double rr = 0.0;
      double rz_new = 0.0;

#pragma omp parallel for reduction(+ : rr, rz_new) schedule(static)
      for (std::int64_t i = 0; i < static_cast<std::int64_t>(n); ++i) {
        x[i] += alpha * p[i];
        const double ri = r[i] - alpha * Ap[i];
        r[i] = ri;
        rr += ri * ri;
        rz_new += ri * ri * inv_diag_[i];
      }

      const double residual_norm = std::sqrt(rr);
      const double relative = residual_norm / initial_norm;

      stats.iterations = it;
      stats.final_residual = residual_norm;
      stats.relative_residual = relative;

      if (config_.verbose &&
          (it % config_.log_interval == 0 || relative <= config_.tolerance ||
           it == config_.max_iterations)) {
        std::cout << "  Iteration " << std::setw(6) << it
                  << " | residual=" << std::scientific << std::setprecision(6)
                  << relative << std::defaultfloat << "\n"
                  << std::flush;
      }

      if (!std::isfinite(rz_new)) {
        throw std::runtime_error("CG produced non-finite rz.");
      }

      if (relative <= config_.tolerance) {
        stats.converged = true;
        finalize_voltage();
        return stats;
      }

      const double beta = rz_new / rz;
      rz = rz_new;

      // Pass 3: p = M^-1 r + beta p
#pragma omp parallel for schedule(static)
      for (std::int64_t i = 0; i < static_cast<std::int64_t>(n); ++i) {
        p[i] = inv_diag_[i] * r[i] + beta * p[i];
      }
    }

    stats.converged = false;
    finalize_voltage();
    return stats;
  }

  // Apply Dirichlet values to the full field (free cells already solved).
  void finalize_voltage() {
#pragma omp parallel for schedule(static)
    for (std::int64_t i = 0; i < static_cast<std::int64_t>(N_); ++i) {
      if (state_[i] == ST_SOURCE) {
        voltage_[i] = 1.0;
      }
    }
    release(inv_diag_);
  }

  // ---------------------------------------------------------------------------
  // Output
  // ---------------------------------------------------------------------------

  void write_raster(const std::string &path, const std::vector<float> &values) {
    GDALDriver *driver = GetGDALDriverManager()->GetDriverByName("GTiff");

    if (!driver) {
      throw std::runtime_error("GTiff driver is unavailable.");
    }

    char **options = nullptr;
    options = CSLSetNameValue(options, "TILED", "YES");
    options = CSLSetNameValue(options, "COMPRESS", "DEFLATE");
    options = CSLSetNameValue(options, "PREDICTOR", "3");
    options = CSLSetNameValue(options, "BIGTIFF", "IF_SAFER");

    std::unique_ptr<GDALDataset, decltype(&GDALClose)> output(
        driver->Create(path.c_str(), width_, height_, 1, GDT_Float32, options),
        GDALClose);

    CSLDestroy(options);

    if (!output) {
      throw std::runtime_error("Failed to create output raster: " + path);
    }

    output->SetGeoTransform(
        const_cast<double *>(resistance_.info.geotransform));
    output->SetProjection(resistance_.info.projection.c_str());

    GDALRasterBand *band = output->GetRasterBand(1);

    if (band->SetNoDataValue(std::numeric_limits<float>::quiet_NaN()) !=
        CE_None) {
      std::cerr << "Warning: failed to set NaN NoData.\n";
    }

    const int block_rows = 256;

    for (int row = 0; row < height_; row += block_rows) {
      const int rows = std::min(block_rows, height_ - row);

      const std::size_t offset =
          static_cast<std::size_t>(row) * static_cast<std::size_t>(width_);

      if (band->RasterIO(GF_Write, 0, row, width_, rows,
                         const_cast<float *>(values.data() + offset), width_,
                         rows, GDT_Float32, 0, 0, nullptr) != CE_None) {
        throw std::runtime_error("Failed writing raster: " + path);
      }
    }

    band->FlushCache();
    output->FlushCache();

    std::cout << "  " << path << "\n";
  }
};

} // namespace

// -----------------------------------------------------------------------------
// Main
// -----------------------------------------------------------------------------

int main(int argc, char **argv) {
  try {
    GDALAllRegister();

    const Config config = parse_arguments(argc, argv);

#ifdef _OPENMP
    if (config.threads > 0) {
      omp_set_num_threads(config.threads);
    }
#endif

    const auto program_start = std::chrono::steady_clock::now();

    std::cout << "Opening resistance raster...\n";

    Raster resistance;
    resistance.open(config.resistance_path);

    log_memory("after resistance input");

    std::cout << "Opening source raster...\n";

    Raster source;
    source.open(config.source_path);

    log_memory("after source input");

    if (!same_grid(resistance.info, source.info)) {
      throw std::runtime_error("Resistance and source rasters do not "
                               "have identical grids.");
    }

    CircuitTheoryEngine engine(resistance, source, config);

    engine.initialize();
    engine.solve();
    engine.calculate_current_density();
    engine.calculate_connectivity();
    engine.write_outputs();

    const double total_seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                      program_start)
            .count();

    std::cout << "\n=============================================\n"
              << " Circuit Theory completed\n"
              << "=============================================\n"
              << "Total time : " << total_seconds << " s\n";

    log_memory("final");

    return EXIT_SUCCESS;

  } catch (const std::exception &e) {
    std::cerr << "\nFATAL ERROR:\n" << e.what() << "\n";
    return EXIT_FAILURE;
  }
}
