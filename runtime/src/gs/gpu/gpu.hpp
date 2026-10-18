#pragma once

#include "../gs.hpp"

#include <cstdint>
#include <vector>

namespace ps2::gs::gpu {

enum class Mode { off, mirror, primary };

struct DisplaySource {
    std::uint32_t fbp, fbw, psm, dbx, dby, width, height;
    bool halved;
};

struct SurfaceTarget {
    enum Kind { X11, Wayland, Canvas } kind;
    void *display;
    std::uint64_t window;
    void *surface;
    const char *selector;
    std::uint32_t width, height;
};

Mode mode();
bool init(Mode m);
void draw(const State &s, const Vertex *v, int count);
void image_begin(const State &s);
void image_data(const std::uint8_t *data, std::size_t bytes);
void local_copy(const State &s);
void load_clut(std::uint64_t tex0, std::uint64_t texclut);
void flush();
void texflush();
void wait_idle();
void download(std::uint8_t *memory, std::uint32_t offset = 0, std::uint32_t bytes = kMemorySize);
bool download_async(std::uint8_t *memory, std::uint32_t offset, std::uint32_t bytes, std::uint32_t serial);
bool finish_download(std::uint32_t serial);
void read_state(std::vector<std::uint32_t> &vram, std::uint32_t &scale, std::vector<std::uint32_t> &clut);
void write_state(const std::vector<std::uint32_t> &vram, std::uint32_t scale, const std::vector<std::uint32_t> &clut);
bool display(const DisplaySource &src, std::vector<std::uint32_t> &rgba, std::uint32_t &width, std::uint32_t &height);
bool display_async(const DisplaySource &src, std::vector<std::uint32_t> &rgba, std::uint32_t &width, std::uint32_t &height);
bool attach_surface(const SurfaceTarget &target);
void resize_surface(std::uint32_t width, std::uint32_t height);
bool has_surface();
bool present(const DisplaySource &src);
double busy_ms();
// synchronous reads of GS memory back from the GPU, and the milliseconds the page thread waited for them
std::uint64_t readbacks();
// batches ended so far for one reason: 0 render target change, 1 unknown texel area, 2 texture read from the frame being
// drawn, 3 from the z buffer being written, 4 batch full, 5 upload or CLUT load over its pages, 6 flush, 7 CLUT slots full
std::uint64_t batch_ends(int reason);
double readback_ms();
void stats();

}
