#pragma once

#include "cpu.hpp"

#include <cstdint>
#include <ctime>
#include <functional>
#include <string>
#include <vector>

namespace ps2 {

inline constexpr std::uint64_t kEeClock = 294912000ull;
inline constexpr std::uint32_t kHandlerReturn = 0x00001000u;
inline constexpr std::uint32_t kIdle = 0x00001010u;

extern Cpu g_cpu;

struct Press {
    std::uint64_t frame;
    std::uint32_t frames;
    std::uint16_t buttons;
    std::uint8_t stick = 0;
};

struct Patch {
    std::uint8_t place;
    std::uint8_t bytes;
    std::uint32_t address;
    std::uint64_t value;
};

struct AxisPress {
    std::uint64_t frame;
    std::uint32_t frames;
    int axis;
    std::uint8_t value;
};

struct Options {
    bool trace_syscalls = false;
    bool trace_io = false;
    bool trace_threads = false;
    bool pal = true;
    bool widescreen = false;
    bool rate_forced = false;
    std::uint64_t max_frames = 0;
    std::string misses_path;
    std::string disc_path;
    bool trace_iop_modules = false;
    bool trace_iop_hle = false;
    std::string frames_dir;
    std::uint32_t frame_every = 60;
    std::vector<Press> presses;
    std::vector<AxisPress> axis_presses;
    std::vector<Patch> patches;
    std::function<void()> on_vblank;
    std::function<void(const std::string &name, const std::uint8_t *code, std::size_t size)> on_vu_program;
    std::string memcard_path;
    std::function<void()> on_memcard_saved;
    std::string save_state_path;
    std::uint64_t save_state_frame = 0;
    // every change of a controller port, one line per change, taken at the vblank once all input sources are applied
    std::string input_record_path;
    // a file written through input_record_path, applied at the same vblanks; frames are absolute, as for presses
    std::string input_replay_path;
    // a hash of the EE memory (RAM and scratchpad) and of the IOP RAM at every vblank, after the inputs are applied;
    // two runs with the same inputs must write the same file
    std::string hash_log_path;
    // within these frames the hash log also gets a "B frame ..." line with one hash per 64 KB block: 512 for the EE
    // RAM, 1 for the scratchpad and 32 for the IOP RAM, to tell where two runs first differ
    std::uint64_t hash_blocks_from = 0, hash_blocks_to = 0;
    // seconds since 1970 that the console's clock shows at boot; it then advances with the emulated time, so two
    // runs read the same clock. Negative leaves the host's clock.
    std::int64_t rtc_epoch = -1;
    // at that vblank, right after the hash, the EE RAM, the scratchpad and the IOP RAM go raw into dump_ram_path,
    // to find the addresses where two runs differ
    std::string dump_ram_path;
    std::uint64_t dump_ram_frame = 0;
};

extern Options g_options;

struct ThreadContext {
    Reg128 r[32];
    Reg128 hi, lo;
    std::uint32_t sa, pc;
    std::uint32_t f[32];
    std::uint32_t facc, fcr31;
};

void save_context(const Cpu &c, ThreadContext &t);
void load_context(Cpu &c, const ThreadContext &t);

std::uint64_t now();
// one hash of the EE RAM, the scratchpad and the IOP RAM; two runs in the same state give the same value
std::uint64_t state_hash();
// the time the console's clock shows, from Options::rtc_epoch or from the host
std::time_t guest_clock();
std::uint64_t hblank_cycles();
void set_budget(Cpu &c, std::int64_t cycles);
void schedule(int event, std::uint64_t at, std::function<void()> action);
std::uint64_t scheduled_at(int event);
bool preempt(Cpu &c);

bool has_code(std::uint32_t pc);
void raise_intc(unsigned cause);
void request_reschedule();
void request_iop_sync();
void sync_iop();

bool load_elf(const std::string &path, std::uint32_t &entry);
void boot(std::uint32_t entry, const std::vector<std::string> &args);
int run();
void halt(const std::string &reason);
bool save_state(const std::string &path);
bool load_state(const std::string &path);
bool load_state(const std::uint8_t *data, std::size_t size);
bool halted();
const std::string &halt_reason();
std::uint64_t frame_count();

bool display_frame(std::vector<std::uint32_t> &rgba, std::uint32_t &width, std::uint32_t &height);

struct WindowTarget {
    enum Kind { X11, Wayland, Canvas } kind = X11;
    void *display = nullptr;
    std::uint64_t window = 0;
    void *surface = nullptr;
    const char *selector = nullptr;
    std::uint32_t width = 0, height = 0;
};

// Lets the WebGPU renderer present straight to a window or canvas instead of handing frames back.
bool attach_window(const WindowTarget &target);
void resize_window(std::uint32_t width, std::uint32_t height);
bool present_frame();
void init_graphics();
void start_render_thread();
void start_vu_thread();
void run_vu_thread();
double vu_thread_idle_ms();
std::uint64_t vu1_interpreted_calls();
// VU1 programs compiled on the fly (the web JIT) and the milliseconds spent compiling them
std::uint64_t vu1_jit_programs();
double vu1_jit_ms();
// time spent blocked on a full GS command ring: 0 is the ring to the render thread, 1 the one from the EE to the VU thread
double gs_ring_wait_ms(int ring);
void post_present();
enum class Rendered { idle, presented, busy };
// before_present runs ahead of each frame's presentation; returning false drops that presentation, while the frame
// itself is still drawn into GS memory and counts as presented for the emulation
Rendered render_pending(bool surface, const std::function<void(const std::uint32_t *rgba, std::uint32_t width, std::uint32_t height)> &copy,
                        const std::function<bool()> &before_present, double budget_ms);
void wait_render_work(int timeout_ms);
void wake_render_thread();
bool copy_frame(std::vector<std::uint32_t> &rgba, std::uint32_t &width, std::uint32_t &height);
bool parse_press(const std::string &spec, Press &press);
bool parse_axis(const std::string &spec, AxisPress &press);
bool load_patches(const std::string &path);
double gpu_busy_ms();
// synchronous reads of GS memory back from the GPU so far, and the milliseconds spent waiting for them
double gpu_readback_ms(std::uint64_t &count);
// GPU batches ended so far for one reason (see gs/gpu/gpu.hpp, batch_ends)
std::uint64_t gpu_batch_ends(int reason);
void browser_yield();
std::uint64_t frames_run();
std::uint64_t primitives_drawn();
// pad 0 is the first controller port, pad 1 the second, which starts unplugged
void set_pad_buttons(std::uint16_t pressed, int pad = 0);
void set_pad_axis(int axis, std::uint8_t value, int pad = 0);
void set_pad_pressure(int button, std::uint8_t value, int pad = 0);
void set_pad_connected(int pad, bool connected);
std::uint16_t pad_rumble(int pad = 0);
void set_audio_sink(std::function<void(const std::int16_t *samples, std::size_t frames)> sink);

}
