#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <queue>
#include <stdexcept>
#include <string>
#include <vector>

#include "gdal_priv.h"
#include "cpl_conv.h"

namespace
{

constexpr float NODATA = -9999.0f;
constexpr double CELL_SIZE = 10.0;
constexpr double DIAGONAL = CELL_SIZE * 1.4142135623730951;
constexpr float INF = std::numeric_limits<float>::infinity();

struct HeapNode
{
    float cost;
    std::uint32_t index;
};

class IndexedMinHeap
{
public:
    explicit IndexedMinHeap(std::size_t capacity)
        : heap_(),
          position_(capacity, -1)
    {
        heap_.reserve(capacity);
    }

    bool empty() const
    {
        return heap_.empty();
    }

    void push_or_decrease(std::uint32_t index, float cost)
    {
        const int pos = position_[index];

        if (pos == -1)
        {
            position_[index] = static_cast<int>(heap_.size());
            heap_.push_back({cost, index});
            sift_up(heap_.size() - 1);
        }
        else if (cost < heap_[pos].cost)
        {
            heap_[pos].cost = cost;
            sift_up(static_cast<std::size_t>(pos));
        }
    }

    HeapNode pop_min()
    {
        if (heap_.empty())
        {
            throw std::runtime_error("Heap kosong.");
        }

        HeapNode result = heap_[0];

        position_[result.index] = -1;

        if (heap_.size() == 1)
        {
            heap_.pop_back();
            return result;
        }

        heap_[0] = heap_.back();
        heap_.pop_back();

        position_[heap_[0].index] = 0;

        sift_down(0);

        return result;
    }

private:
    std::vector<HeapNode> heap_;
    std::vector<int> position_;

    static bool less_than(
        const HeapNode& a,
        const HeapNode& b)
    {
        return a.cost < b.cost;
    }

    void sift_up(std::size_t index)
    {
        while (index > 0)
        {
            const std::size_t parent = (index - 1) / 2;

            if (!less_than(heap_[index], heap_[parent]))
            {
                break;
            }

            std::swap(heap_[index], heap_[parent]);

            position_[heap_[index].index] =
                static_cast<int>(index);

            position_[heap_[parent].index] =
                static_cast<int>(parent);

            index = parent;
        }
    }

    void sift_down(std::size_t index)
    {
        const std::size_t size = heap_.size();

        while (true)
        {
            const std::size_t left = index * 2 + 1;
            const std::size_t right = left + 1;

            std::size_t smallest = index;

            if (left < size &&
                less_than(heap_[left], heap_[smallest]))
            {
                smallest = left;
            }

            if (right < size &&
                less_than(heap_[right], heap_[smallest]))
            {
                smallest = right;
            }

            if (smallest == index)
            {
                break;
            }

            std::swap(heap_[index], heap_[smallest]);

            position_[heap_[index].index] =
                static_cast<int>(index);

            position_[heap_[smallest].index] =
                static_cast<int>(smallest);

            index = smallest;
        }
    }
};

inline bool valid_resistance(float value)
{
    return std::isfinite(value) &&
           value >= 0.0f &&
           value <= 1.0f;
}

} // namespace


int main(int argc, char** argv)
{
    if (argc != 7)
    {
        std::cerr
            << "Usage:\n"
            << "  cost_distance "
            << "--resistance <file> "
            << "--source <file> "
            << "--output <file>\n";

        return 1;
    }

    std::string resistance_path;
    std::string source_path;
    std::string output_path;

    for (int i = 1; i < argc; i += 2)
    {
        const std::string argument = argv[i];
        const std::string value = argv[i + 1];

        if (argument == "--resistance")
        {
            resistance_path = value;
        }
        else if (argument == "--source")
        {
            source_path = value;
        }
        else if (argument == "--output")
        {
            output_path = value;
        }
        else
        {
            std::cerr
                << "Unknown argument: "
                << argument << "\n";

            return 1;
        }
    }

    GDALAllRegister();

    GDALDataset* resistance_ds =
        static_cast<GDALDataset*>(
            GDALOpen(
                resistance_path.c_str(),
                GA_ReadOnly));

    GDALDataset* source_ds =
        static_cast<GDALDataset*>(
            GDALOpen(
                source_path.c_str(),
                GA_ReadOnly));

    if (!resistance_ds)
    {
        std::cerr
            << "Failed to open resistance raster:\n"
            << resistance_path << "\n";

        return 1;
    }

    if (!source_ds)
    {
        std::cerr
            << "Failed to open source raster:\n"
            << source_path << "\n";

        GDALClose(resistance_ds);
        return 1;
    }

    GDALRasterBand* resistance_band =
        resistance_ds->GetRasterBand(1);

    GDALRasterBand* source_band =
        source_ds->GetRasterBand(1);

    if (!resistance_band || !source_band)
    {
        std::cerr
            << "Raster band tidak ditemukan.\n";

        GDALClose(resistance_ds);
        GDALClose(source_ds);

        return 1;
    }

    const int width =
        resistance_ds->GetRasterXSize();

    const int height =
        resistance_ds->GetRasterYSize();

    if (width != source_ds->GetRasterXSize() ||
        height != source_ds->GetRasterYSize())
    {
        std::cerr
            << "Raster dimensions berbeda.\n";

        GDALClose(resistance_ds);
        GDALClose(source_ds);

        return 1;
    }

    double resistance_gt[6];
    double source_gt[6];

    if (resistance_ds->GetGeoTransform(resistance_gt) != CE_None ||
        source_ds->GetGeoTransform(source_gt) != CE_None)
    {
        std::cerr
            << "Gagal membaca geotransform.\n";

        GDALClose(resistance_ds);
        GDALClose(source_ds);

        return 1;
    }

    for (int i = 0; i < 6; ++i)
    {
        if (std::abs(resistance_gt[i] - source_gt[i]) > 1e-9)
        {
            std::cerr
                << "Geotransform resistance/source berbeda.\n";

            GDALClose(resistance_ds);
            GDALClose(source_ds);

            return 1;
        }
    }

    const double pixel_width =
        std::abs(resistance_gt[1]);

    const double pixel_height =
        std::abs(resistance_gt[5]);

    if (std::abs(pixel_width - CELL_SIZE) > 1e-6 ||
        std::abs(pixel_height - CELL_SIZE) > 1e-6)
    {
        std::cerr
            << "Grid harus 10m x 10m.\n"
            << "Actual: "
            << pixel_width << " x "
            << pixel_height << "\n";

        GDALClose(resistance_ds);
        GDALClose(source_ds);

        return 1;
    }

    const std::size_t pixel_count =
        static_cast<std::size_t>(width) *
        static_cast<std::size_t>(height);

    std::cout
        << "Raster dimensions: "
        << width << " x "
        << height << "\n";

    std::cout
        << "Total pixels: "
        << pixel_count << "\n";

    /*
     * Memory layout:
     *
     * resistance : float      ~4 bytes/pixel
     * cost       : float      ~4 bytes/pixel
     * heap pos   : int        ~4 bytes/pixel
     * heap       : up to ~8 bytes/node
     *
     * Source raster is processed in chunks and is NOT
     * retained as a second full-size array.
     */

    std::vector<float> resistance(pixel_count);
    std::vector<float> cost(pixel_count, INF);

    const std::size_t invalid_index =
        std::numeric_limits<std::size_t>::max();

    std::size_t valid_count = 0;
    std::size_t invalid_count = 0;

    float resistance_min =
        std::numeric_limits<float>::max();

    float resistance_max =
        std::numeric_limits<float>::lowest();

    /*
     * Read resistance raster.
     */
    if (resistance_band->RasterIO(
            GF_Read,
            0,
            0,
            width,
            height,
            resistance.data(),
            width,
            height,
            GDT_Float32,
            0,
            0,
            nullptr) != CE_None)
    {
        std::cerr
            << "Gagal membaca resistance raster.\n";

        GDALClose(resistance_ds);
        GDALClose(source_ds);

        return 1;
    }

    for (std::size_t i = 0; i < pixel_count; ++i)
    {
        const float value = resistance[i];

        if (!valid_resistance(value))
        {
            resistance[i] = NODATA;
            ++invalid_count;
            continue;
        }

        ++valid_count;

        resistance_min =
            std::min(resistance_min, value);

        resistance_max =
            std::max(resistance_max, value);
    }

    std::cout
        << "Valid resistance pixels: "
        << valid_count << "\n";

    std::cout
        << "Invalid resistance pixels: "
        << invalid_count << "\n";

    std::cout
        << "Resistance range: "
        << resistance_min
        << " - "
        << resistance_max
        << "\n";

    if (valid_count == 0)
    {
        std::cerr
            << "Tidak ada resistance pixel valid.\n";

        GDALClose(resistance_ds);
        GDALClose(source_ds);

        return 1;
    }

    /*
     * Indexed heap.
     *
     * position[index] == -1 means pixel is not currently
     * present in the heap.
     */
    IndexedMinHeap heap(pixel_count);

    /*
     * Source raster is read in row blocks.
     * This avoids keeping a complete source array in RAM.
     */
    const int block_rows = 256;

    std::vector<std::uint8_t> source_block(
        static_cast<std::size_t>(width) *
        block_rows);

    std::size_t source_count = 0;

    for (int row = 0;
         row < height;
         row += block_rows)
    {
        const int rows =
            std::min(block_rows, height - row);

        if (source_band->RasterIO(
                GF_Read,
                0,
                row,
                width,
                rows,
                source_block.data(),
                width,
                rows,
                GDT_Byte,
                0,
                0,
                nullptr) != CE_None)
        {
            std::cerr
                << "Gagal membaca source raster.\n";

            GDALClose(resistance_ds);
            GDALClose(source_ds);

            return 1;
        }

        for (int local_row = 0;
             local_row < rows;
             ++local_row)
        {
            const std::size_t global_row =
                static_cast<std::size_t>(row + local_row);

            for (int col = 0;
                 col < width;
                 ++col)
            {
                const std::size_t index =
                    global_row *
                    static_cast<std::size_t>(width) +
                    static_cast<std::size_t>(col);

                const std::size_t block_index =
                    static_cast<std::size_t>(local_row) *
                    static_cast<std::size_t>(width) +
                    static_cast<std::size_t>(col);

                if (source_block[block_index] == 1 &&
                    valid_resistance(resistance[index]))
                {
                    if (cost[index] != 0.0f)
                    {
                        cost[index] = 0.0f;
                        heap.push_or_decrease(
                            static_cast<std::uint32_t>(index),
                            0.0f);

                        ++source_count;
                    }
                }
            }
        }
    }

    std::cout
        << "Source pixels: "
        << source_count
        << "\n";

    if (source_count == 0)
    {
        std::cerr
            << "Tidak ada source pixel valid.\n";

        GDALClose(resistance_ds);
        GDALClose(source_ds);

        return 1;
    }

    /*
     * 8-neighbor offsets.
     */
    constexpr int dr[8] =
    {
        -1, -1, -1,
         0,  0,
         1,  1,  1
    };

    constexpr int dc[8] =
    {
        -1,  0,  1,
        -1,  1,
        -1,  0,  1
    };

    constexpr double distance[8] =
    {
        DIAGONAL,
        CELL_SIZE,
        DIAGONAL,
        CELL_SIZE,
        CELL_SIZE,
        DIAGONAL,
        CELL_SIZE,
        DIAGONAL
    };

    std::size_t processed = 0;

    while (!heap.empty())
    {
        const HeapNode current =
            heap.pop_min();

        const std::uint32_t current_index =
            current.index;

        const float current_cost =
            current.cost;

        /*
         * Heap contains each node at most once,
         * so no lazy duplicate check is required.
         */

        const int row =
            static_cast<int>(
                current_index /
                static_cast<std::uint32_t>(width));

        const int col =
            static_cast<int>(
                current_index %
                static_cast<std::uint32_t>(width));

        const float current_resistance =
            resistance[current_index];

        if (!valid_resistance(current_resistance))
        {
            continue;
        }

        ++processed;

        if (processed % 1000000 == 0)
        {
            std::cout
                << "Processed: "
                << processed
                << "\n";
        }

        for (int n = 0; n < 8; ++n)
        {
            const int nr = row + dr[n];
            const int nc = col + dc[n];

            if (nr < 0 ||
                nr >= height ||
                nc < 0 ||
                nc >= width)
            {
                continue;
            }

            const std::size_t neighbor_index =
                static_cast<std::size_t>(nr) *
                static_cast<std::size_t>(width) +
                static_cast<std::size_t>(nc);

            const float neighbor_resistance =
                resistance[neighbor_index];

            if (!valid_resistance(neighbor_resistance))
            {
                continue;
            }

            const float edge_cost =
                (
                    current_resistance +
                    neighbor_resistance
                ) *
                0.5f *
                static_cast<float>(distance[n]);

            const float new_cost =
                current_cost + edge_cost;

            if (new_cost < cost[neighbor_index])
            {
                cost[neighbor_index] = new_cost;

                heap.push_or_decrease(
                    static_cast<std::uint32_t>(
                        neighbor_index),
                    new_cost);
            }
        }
    }

    std::size_t reachable = 0;
    std::size_t unreachable = 0;

    for (std::size_t i = 0;
         i < pixel_count;
         ++i)
    {
        if (valid_resistance(resistance[i]))
        {
            if (std::isfinite(cost[i]))
            {
                ++reachable;
            }
            else
            {
                ++unreachable;
            }
        }
    }

    std::cout
        << "Reachable valid pixels: "
        << reachable
        << "\n";

    std::cout
        << "Unreachable valid pixels: "
        << unreachable
        << "\n";

    /*
     * Create output.
     */
    GDALDriver* driver =
        GetGDALDriverManager()->GetDriverByName("GTiff");

    if (!driver)
    {
        std::cerr
            << "GTiff driver tidak tersedia.\n";

        GDALClose(resistance_ds);
        GDALClose(source_ds);

        return 1;
    }

    char** creation_options = nullptr;

    creation_options =
        CSLSetNameValue(
            creation_options,
            "COMPRESS",
            "DEFLATE");

    creation_options =
        CSLSetNameValue(
            creation_options,
            "PREDICTOR",
            "2");

    creation_options =
        CSLSetNameValue(
            creation_options,
            "TILED",
            "YES");

    GDALDataset* output_ds =
        driver->Create(
            output_path.c_str(),
            width,
            height,
            1,
            GDT_Float32,
            creation_options);

    CSLDestroy(creation_options);

    if (!output_ds)
    {
        std::cerr
            << "Gagal membuat output:\n"
            << output_path << "\n";

        GDALClose(resistance_ds);
        GDALClose(source_ds);

        return 1;
    }

    output_ds->SetGeoTransform(resistance_gt);

    const char* projection =
        resistance_ds->GetProjectionRef();

    if (projection)
    {
        output_ds->SetProjection(projection);
    }

    GDALRasterBand* output_band =
        output_ds->GetRasterBand(1);

    output_band->SetNoDataValue(
        static_cast<double>(NODATA));

    std::vector<float> output_buffer(
        pixel_count);

    for (std::size_t i = 0;
         i < pixel_count;
         ++i)
    {
        if (valid_resistance(resistance[i]) &&
            std::isfinite(cost[i]))
        {
            output_buffer[i] = cost[i];
        }
        else
        {
            output_buffer[i] = NODATA;
        }
    }

    if (output_band->RasterIO(
            GF_Write,
            0,
            0,
            width,
            height,
            output_buffer.data(),
            width,
            height,
            GDT_Float32,
            0,
            0,
            nullptr) != CE_None)
    {
        std::cerr
            << "Gagal menulis output raster.\n";

        GDALClose(output_ds);
        GDALClose(resistance_ds);
        GDALClose(source_ds);

        return 1;
    }

    output_ds->FlushCache();

    GDALClose(output_ds);
    GDALClose(resistance_ds);
    GDALClose(source_ds);

    std::cout
        << "Dijkstra completed.\n";

    std::cout
        << "Output: "
        << output_path
        << "\n";

    return 0;
}
