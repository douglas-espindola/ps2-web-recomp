#include "dmac.hpp"
#include "gs/gif.hpp"
#include "vif.hpp"
#include "vu_micro.hpp"
#include "gs/gs.hpp"
#include "gs/queue.hpp"
#include "hw.hpp"
#include "iop/iop.hpp"
#include "iop/sif.hpp"
#include "iop/sio2.hpp"
#include "iop/spu2.hpp"
#include "ipu/ipu.hpp"
#include "ps2/disc.hpp"
#include "kernel.hpp"
#include "ps2/generated.hpp"
#include "ps2/memory.hpp"
#include "ps2/state.hpp"
#include "ps2/system.hpp"

#ifdef __EMSCRIPTEN__
#include <emscripten.h>
#endif

#include <algorithm>
#include <atomic>
#include <cctype>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <cstdlib>
#include <iterator>
#include <stdexcept>

namespace ps2 {

Cpu g_cpu;
Options g_options;

namespace {

constexpr std::int64_t kMaxSlice = 20000;

struct Event {
    int id;
    std::uint64_t at;
    std::function<void()> action;
};

std::uint64_t g_base = 0;
std::int64_t g_given = 0;
std::vector<Event> g_events;
bool g_halted = false;
std::string g_halt_reason;
std::uint64_t g_frames = 0;
std::uint64_t g_iop_synced = 0;
bool g_state_request = false;
bool g_resumed = false;

std::uint64_t next_event_time() {
    std::uint64_t t = UINT64_MAX;
    for (const Event &e : g_events) t = std::min(t, e.at);
    return t;
}

void run_due_events() {
    sync_iop();
    for (;;) {
        const std::uint64_t t = now();
        auto it = std::min_element(g_events.begin(), g_events.end(), [](const Event &a, const Event &b) { return a.at < b.at; });
        if (it == g_events.end() || it->at > t) return;
        auto action = std::move(it->action);
        g_events.erase(it);
        action();
    }
}

void refill(Cpu &c) {
    const std::uint64_t t = now(), next = next_event_time();
    std::int64_t slice = kMaxSlice;
    if (next != UINT64_MAX) slice = std::clamp<std::int64_t>(static_cast<std::int64_t>(next - std::min(next, t)), 1, kMaxSlice);
    set_budget(c, slice);
}

std::uint64_t frame_cycles() { return kEeClock / (g_options.pal ? 50u : 60u); }

void dump_frame() {
    std::vector<std::uint32_t> rgba;
    std::uint32_t w = 0, h = 0;
    if (!gs::display(rgba, w, h)) return;
    char name[512];
    std::snprintf(name, sizeof name, "%s/frame_%05llu.ppm", g_options.frames_dir.c_str(), static_cast<unsigned long long>(g_frames));
    std::FILE *f = std::fopen(name, "wb");
    if (!f) return;
    std::fprintf(f, "P6\n%u %u\n255\n", w, h);
    for (std::uint32_t p : rgba) {
        const std::uint8_t px[3] = {static_cast<std::uint8_t>(p), static_cast<std::uint8_t>(p >> 8), static_cast<std::uint8_t>(p >> 16)};
        std::fwrite(px, 1, 3, f);
    }
    std::fclose(f);
}

void apply_presses() {
    if (!g_options.presses.empty()) {
        std::uint16_t mask = 0;
        std::uint8_t stick = 0;
        for (const Press &p : g_options.presses)
            if (g_frames >= p.frame && g_frames < p.frame + p.frames) {
                mask |= p.buttons;
                stick |= p.stick;
            }
        iop::sio2::set_buttons(0, mask);
        iop::sio2::set_axis(0, 2, (stick & 4u) ? 0x00 : (stick & 8u) ? 0xFF : 0x80);
        iop::sio2::set_axis(0, 3, (stick & 1u) ? 0x00 : (stick & 2u) ? 0xFF : 0x80);
    }
    if (!g_options.axis_presses.empty()) {
        // Outside a hold the stick rests; within one it takes the given tilt.
        std::uint8_t axes[4] = {0x80, 0x80, 0x80, 0x80};
        for (const AxisPress &p : g_options.axis_presses)
            if (g_frames >= p.frame && g_frames < p.frame + p.frames) axes[p.axis] = p.value;
        for (int a = 0; a < 4; ++a) iop::sio2::set_axis(0, a, axes[a]);
    }
}

// One port's state at a vblank. The text form is "frame pad connected buttons axes pressure", in hex past the
// first two fields, with the axes in SIO2 order (right x, right y, left x, left y).
struct InputLine {
    std::uint64_t frame = 0;
    int pad = 0;
    bool connected = false;
    std::uint16_t buttons = 0;
    std::uint8_t axes[4] = {0x80, 0x80, 0x80, 0x80};
    std::uint8_t pressure[16] = {};
};

std::FILE *g_input_record = nullptr;
InputLine g_input_recorded[2];
bool g_input_record_started = false;
std::vector<InputLine> g_input_replay;
std::size_t g_input_replay_next = 0;
bool g_input_replay_loaded = false;

bool same_input(const InputLine &a, const InputLine &b) {
    return a.connected == b.connected && a.buttons == b.buttons && std::equal(std::begin(a.axes), std::end(a.axes), b.axes) &&
           std::equal(std::begin(a.pressure), std::end(a.pressure), b.pressure);
}

bool parse_input(const std::string &text, InputLine &l) {
    unsigned long long frame = 0;
    int pad = 0, connected = 0;
    unsigned buttons = 0, axes = 0;
    char pressure[33] = {};
    if (std::sscanf(text.c_str(), "%llu %d %d %x %x %32s", &frame, &pad, &connected, &buttons, &axes, pressure) != 6) return false;
    if (pad < 0 || pad > 1 || std::strlen(pressure) != 32) return false;
    l.frame = frame;
    l.pad = pad;
    l.connected = connected != 0;
    l.buttons = static_cast<std::uint16_t>(buttons);
    for (int i = 0; i < 4; ++i) l.axes[i] = static_cast<std::uint8_t>(axes >> (24 - i * 8));
    for (int i = 0; i < 16; ++i) {
        const char hex[3] = {pressure[i * 2], pressure[i * 2 + 1], 0};
        l.pressure[i] = static_cast<std::uint8_t>(std::strtoul(hex, nullptr, 16));
    }
    return true;
}

void apply_input(const InputLine &l) {
    iop::sio2::set_connected(l.pad, l.connected);
    iop::sio2::set_buttons(l.pad, l.buttons);
    for (int i = 0; i < 4; ++i) iop::sio2::set_axis(l.pad, i, l.axes[i]);
    for (int i = 0; i < 16; ++i) iop::sio2::set_pressure(l.pad, i, l.pressure[i]);
}

void replay_inputs() {
    if (g_options.input_replay_path.empty()) return;
    if (!g_input_replay_loaded) {
        g_input_replay_loaded = true;
        std::ifstream in(g_options.input_replay_path);
        if (!in) std::fprintf(stderr, "[input] cannot read %s\n", g_options.input_replay_path.c_str());
        std::string text;
        InputLine l;
        while (std::getline(in, text))
            if (!text.empty() && text[0] != '#' && parse_input(text, l)) g_input_replay.push_back(l);
        std::stable_sort(g_input_replay.begin(), g_input_replay.end(), [](const InputLine &a, const InputLine &b) { return a.frame < b.frame; });
        std::fprintf(stderr, "[input] replaying %zu changes from %s\n", g_input_replay.size(), g_options.input_replay_path.c_str());
    }
    // a replay started from a later state catches up by applying every earlier change in order
    while (g_input_replay_next < g_input_replay.size() && g_input_replay[g_input_replay_next].frame <= g_frames) apply_input(g_input_replay[g_input_replay_next++]);
}

void record_inputs() {
    if (g_options.input_record_path.empty()) return;
    if (!g_input_record) {
        g_input_record = std::fopen(g_options.input_record_path.c_str(), "w");
        if (!g_input_record) {
            std::fprintf(stderr, "[input] cannot write %s\n", g_options.input_record_path.c_str());
            g_options.input_record_path.clear();
            return;
        }
        std::fprintf(g_input_record, "# ps2 input record v1: frame pad connected buttons axes(rx ry lx ly) pressure[16]\n");
    }
    for (int pad = 0; pad < 2; ++pad) {
        InputLine l;
        l.frame = g_frames;
        l.pad = pad;
        iop::sio2::get_input(pad, l.connected, l.buttons, l.axes, l.pressure);
        if (g_input_record_started && same_input(l, g_input_recorded[pad])) continue;
        g_input_recorded[pad] = l;
        char pressure[33];
        for (int i = 0; i < 16; ++i) std::snprintf(pressure + i * 2, 3, "%02x", l.pressure[i]);
        std::fprintf(g_input_record, "%llu %d %d %04x %02x%02x%02x%02x %s\n", static_cast<unsigned long long>(l.frame), pad, l.connected ? 1 : 0, l.buttons,
                     l.axes[0], l.axes[1], l.axes[2], l.axes[3], pressure);
    }
    g_input_record_started = true;
    // flushed about once a second: on the web each write is a call to the page thread
    if (g_frames % 50 == 0) std::fflush(g_input_record);
}

// Four lanes of multiply and rotate over 8-byte words, folded together at the end; fast enough to run every frame
// over the 32 MB of EE RAM. The sizes hashed here are all multiples of 32 bytes.
std::uint64_t hash_memory(const std::uint8_t *data, std::size_t size) {
    constexpr std::uint64_t kPrime1 = 0x9E3779B185EBCA87ull, kPrime2 = 0xC2B2AE3D27D4EB4Full;
    std::uint64_t lane[4] = {kPrime1, kPrime2, ~kPrime1, ~kPrime2};
    for (std::size_t at = 0; at + 32 <= size; at += 32)
        for (int i = 0; i < 4; ++i) {
            std::uint64_t word;
            std::memcpy(&word, data + at + i * 8, sizeof word);
            const std::uint64_t mixed = lane[i] + word * kPrime2;
            lane[i] = ((mixed << 31) | (mixed >> 33)) * kPrime1;
        }
    std::uint64_t h = size;
    for (const std::uint64_t l : lane) h = (h ^ l) * kPrime1 + kPrime2;
    return h ^ (h >> 29);
}

std::FILE *g_hash_log = nullptr;

void log_hashes() {
    if (g_options.hash_log_path.empty()) return;
    if (!g_hash_log) {
        g_hash_log = std::fopen(g_options.hash_log_path.c_str(), "w");
        if (!g_hash_log) {
            std::fprintf(stderr, "[hash] cannot write %s\n", g_options.hash_log_path.c_str());
            g_options.hash_log_path.clear();
            return;
        }
        std::fprintf(g_hash_log, "# ps2 state hash v1: frame ee(ram+scratchpad) iop(ram)\n");
    }
    const std::uint64_t ee = hash_memory(g_ram, kRamSize) ^ (hash_memory(g_spr, kScratchpadSize) * 0x100000001B3ull);
    std::fprintf(g_hash_log, "%llu %016llx %016llx\n", static_cast<unsigned long long>(g_frames), static_cast<unsigned long long>(ee),
                 static_cast<unsigned long long>(hash_memory(iop::g_ram, iop::kRamSize)));
    if (g_frames >= g_options.hash_blocks_from && g_frames <= g_options.hash_blocks_to) {
        constexpr std::size_t kBlock = 64u << 10;
        std::fprintf(g_hash_log, "B %llu", static_cast<unsigned long long>(g_frames));
        for (std::size_t at = 0; at < kRamSize; at += kBlock) std::fprintf(g_hash_log, " %016llx", static_cast<unsigned long long>(hash_memory(g_ram + at, kBlock)));
        std::fprintf(g_hash_log, " %016llx", static_cast<unsigned long long>(hash_memory(g_spr, kScratchpadSize)));
        for (std::size_t at = 0; at < iop::kRamSize; at += kBlock) std::fprintf(g_hash_log, " %016llx", static_cast<unsigned long long>(hash_memory(iop::g_ram + at, kBlock)));
        std::fprintf(g_hash_log, "\n");
    }
    if (g_frames % 50 == 0) std::fflush(g_hash_log);
}

void dump_ram() {
    if (g_options.dump_ram_path.empty() || g_frames != g_options.dump_ram_frame) return;
    std::FILE *f = std::fopen(g_options.dump_ram_path.c_str(), "wb");
    if (!f) {
        std::fprintf(stderr, "[hash] cannot write %s\n", g_options.dump_ram_path.c_str());
        return;
    }
    std::fwrite(g_ram, 1, kRamSize, f);
    std::fwrite(g_spr, 1, kScratchpadSize, f);
    std::fwrite(iop::g_ram, 1, iop::kRamSize, f);
    std::fclose(f);
    std::fprintf(stderr, "[hash] memory of frame %llu written to %s\n", static_cast<unsigned long long>(g_frames), g_options.dump_ram_path.c_str());
}

void time_frames() {
    static const char *spec = std::getenv("PS2_TIME");
    if (!spec) return;
    static const std::uint64_t from = std::strtoull(spec, nullptr, 10);
    static const std::uint64_t count = std::strchr(spec, ',') ? std::strtoull(std::strchr(spec, ',') + 1, nullptr, 10) : 500;
    static std::chrono::steady_clock::time_point start;
    if (g_frames == from) {
        gs::flush_and_wait();
        start = std::chrono::steady_clock::now();
    } else if (g_frames == from + count) {
        gs::flush_and_wait();
        const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
        std::fprintf(stderr, "[time] frames %llu-%llu: %.2f ms per frame\n", static_cast<unsigned long long>(from), static_cast<unsigned long long>(from + count), ms / static_cast<double>(count));
    }
}

void apply_patches(bool boot) {
    for (const Patch &p : g_options.patches) {
        if (boot ? p.place == 1 : p.place == 0) continue;
        switch (p.bytes) {
        case 1: st8(p.address, static_cast<std::uint8_t>(p.value)); break;
        case 2: st16(p.address, static_cast<std::uint16_t>(p.value)); break;
        case 4: st32(p.address, static_cast<std::uint32_t>(p.value)); break;
        default: st64(p.address, p.value); break;
        }
    }
}

void vblank() {
    ++g_frames;
    time_frames();
    apply_presses();
    apply_patches(false);
    if (g_options.on_vblank) g_options.on_vblank();
    replay_inputs();
    record_inputs();
    log_hashes();
    dump_ram();
    iop::sio2::flush_cards();
    if (g_options.trace_iop_modules && g_frames % 25 == 0) std::fprintf(stderr, "[frame] %llu prims %llu\n", static_cast<unsigned long long>(g_frames), static_cast<unsigned long long>(gs::primitives_drawn()));
    if (!g_options.frames_dir.empty() && g_options.frame_every && g_frames % g_options.frame_every == 0) dump_frame();
    hw::vblank_start();
    iop::vblank(true);
    schedule(hw::kEventVblankEnd + 100, now() + hblank_cycles() * 22u, [] { iop::vblank(false); });
    kernel::on_vblank_start();
    if (g_options.max_frames && g_frames >= g_options.max_frames) halt("frame limit reached");
    if (g_options.save_state_frame && g_frames == g_options.save_state_frame) g_state_request = true;
    schedule(hw::kEventVblank, now() + frame_cycles(), vblank);
}

void idle(Cpu &c) {
    std::uint64_t next = next_event_time();
    if (next == UINT64_MAX && iop::idle()) {
        halt("deadlock: every thread waits and no event is pending");
        return;
    }
    next = std::min(next, now() + kEeClock / 2000);
    g_base = std::max(now(), next);
    g_given = 0;
    c.budget = 0;
    run_due_events();
}

}

void save_context(const Cpu &c, ThreadContext &t) {
    std::memcpy(t.r, c.r, sizeof t.r);
    t.hi = c.hi;
    t.lo = c.lo;
    t.sa = c.sa;
    t.pc = c.pc & ~kUnwind;
    std::memcpy(t.f, c.f, sizeof t.f);
    t.facc = c.facc;
    t.fcr31 = c.fcr31;
}

void load_context(Cpu &c, const ThreadContext &t) {
    std::memcpy(c.r, t.r, sizeof t.r);
    c.hi = t.hi;
    c.lo = t.lo;
    c.sa = t.sa;
    c.pc = t.pc;
    std::memcpy(c.f, t.f, sizeof t.f);
    c.facc = t.facc;
    c.fcr31 = t.fcr31;
}

std::uint64_t now() { return g_base + static_cast<std::uint64_t>(g_given - g_cpu.budget); }

std::uint64_t hblank_cycles() { return frame_cycles() * 2u / (g_options.pal ? 625u : 525u); }

void set_budget(Cpu &c, std::int64_t cycles) {
    g_base = now();
    g_given = cycles;
    c.budget = static_cast<std::int32_t>(cycles);
}

void schedule(int event, std::uint64_t at, std::function<void()> action) {
    g_events.erase(std::remove_if(g_events.begin(), g_events.end(), [&](const Event &e) { return e.id == event; }), g_events.end());
    g_events.push_back({event, at, std::move(action)});
}

std::uint64_t scheduled_at(int event) {
    for (const Event &e : g_events)
        if (e.id == event) return e.at;
    return UINT64_MAX;
}

bool preempt(Cpu &c) {
    run_due_events();
    refill(c);
    return g_halted || g_state_request || kernel::must_unwind();
}

void halt(const std::string &reason) {
    if (!g_halted) std::fprintf(stderr, "[ee] halted: %s\n", reason.c_str());
    if (g_input_record) std::fflush(g_input_record);
    if (g_hash_log) std::fflush(g_hash_log);
    g_halted = true;
    g_halt_reason = reason;
}

bool halted() { return g_halted; }
const std::string &halt_reason() { return g_halt_reason; }

namespace {

constexpr char kStateMagic[8] = {'P', 'S', '2', 'S', 'T', 'A', 'T', 'E'};
constexpr std::uint32_t kStateVersion = 3;

std::function<void()> event_action(int id, std::uint64_t at) {
    if (id == hw::kEventVblank) return vblank;
    if (id == hw::kEventVblankEnd + 100) return [] { iop::vblank(false); };
    if (id >= hw::kEventCount) return kernel::alarm_action(id - hw::kEventCount);
    return hw::event_action(id, at);
}

std::atomic<std::uint32_t> g_vu_fence{0};

void sync_vu_thread() {
    if (!gs::queue::vu_active()) return;
    const std::uint32_t seen = g_vu_fence.load();
    gs::queue::vu().post(gs::queue::kVuFence, 0, 0, nullptr, 0);
    while (g_vu_fence.load() == seen) gs::queue::wait(g_vu_fence, seen);
}

void machine_state(state::Stream &s) {
    char magic[8];
    std::memcpy(magic, kStateMagic, sizeof magic);
    s.bytes(magic, sizeof magic);
    std::uint32_t version = kStateVersion;
    s.pod(version);
    if (std::memcmp(magic, kStateMagic, sizeof magic) != 0) throw std::runtime_error("not a state file");
    if (version != kStateVersion && !(s.loading() && version == 2)) throw std::runtime_error("state file from another version");
    s.set_version(version);
    s.tag("ee");
    s.pod(g_cpu);
    s.bytes(g_ram, kRamSize);
    s.bytes(g_spr, kScratchpadSize);
    s.pod(g_base);
    s.pod(g_given);
    s.pod(g_frames);
    s.pod(g_iop_synced);
    s.pod(g_options.pal);
    s.vec(g_events, [&s](Event &e) {
        s.pod(e.id);
        s.pod(e.at);
        if (s.loading()) e.action = event_action(e.id, e.at);
    });
    kernel::state(s);
    hw::state(s);
    dmac::state(s);
    sync_vu_thread();
    vif::state(s);
    vu::state(s);
    ipu::state(s);
    gs::state(s);
    gif::state(s);
    iop::state(s);
    sif::state(s);
    s.tag("end");
}

}

bool save_state(const std::string &path) {
    try {
        state::Writer w;
        machine_state(w);
        std::ofstream out(path, std::ios::binary);
        out.write(reinterpret_cast<const char *>(w.out.data()), static_cast<std::streamsize>(w.out.size()));
        if (!out) throw std::runtime_error("cannot write " + path);
        std::fprintf(stderr, "[state] saved frame %llu to %s (%zu KB)\n", static_cast<unsigned long long>(g_frames), path.c_str(), w.out.size() >> 10);
        return true;
    } catch (const std::exception &e) {
        std::fprintf(stderr, "[state] cannot save: %s\n", e.what());
        return false;
    }
}

bool load_state(const std::uint8_t *data, std::size_t size) {
    try {
        state::Reader r(data, size);
        machine_state(r);
        g_resumed = true;
        std::fprintf(stderr, "[state] loaded frame %llu\n", static_cast<unsigned long long>(g_frames));
        return true;
    } catch (const std::exception &e) {
        std::fprintf(stderr, "[state] cannot load: %s\n", e.what());
        return false;
    }
}

bool load_state(const std::string &path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        std::fprintf(stderr, "[state] cannot open %s\n", path.c_str());
        return false;
    }
    const std::vector<std::uint8_t> data{std::istreambuf_iterator<char>(in), {}};
    return load_state(data.data(), data.size());
}

std::uint64_t frame_count() { return g_frames; }

bool load_elf(const std::string &path, std::uint32_t &entry) {
    std::ifstream file(path, std::ios::binary);
    if (!file) return false;
    const std::vector<std::uint8_t> data{std::istreambuf_iterator<char>(file), {}};
    return load_elf(data.data(), data.size(), entry);
}

bool load_elf(const std::uint8_t *data, std::size_t size, std::uint32_t &entry) {
    if (size < 52 || std::memcmp(data, "\x7f" "ELF", 4) != 0) return false;
    const auto u32 = [&](std::size_t o) { std::uint32_t v; std::memcpy(&v, data + o, 4); return v; };
    const auto u16 = [&](std::size_t o) { std::uint16_t v; std::memcpy(&v, data + o, 2); return v; };
    entry = u32(24);
    const std::uint32_t phoff = u32(28);
    for (std::uint32_t i = 0; i < u16(44); ++i) {
        const std::size_t ph = phoff + i * u16(42);
        if (u32(ph) != 1) continue;
        const std::uint32_t offset = u32(ph + 4), vaddr = physical(u32(ph + 8)), filesz = u32(ph + 16), memsz = u32(ph + 20);
        if (vaddr + memsz > kRamSize || offset + filesz > size) return false;
        std::memcpy(g_ram + vaddr, data + offset, filesz);
        watch_span(vaddr, filesz, "boot:load_exec");
        std::memset(g_ram + vaddr + filesz, 0, memsz - filesz);
    }
    return true;
}

void sync_iop() {
    const std::uint64_t t = now();
    if (t > g_iop_synced) {
        iop::run(t - g_iop_synced);
        g_iop_synced = t;
    }
    sif::pump();
}

void request_iop_sync() {
    if (g_cpu.budget > 64) set_budget(g_cpu, 64);
}

void boot(std::uint32_t entry, const std::vector<std::string> &args) {
    init_watch();
    hw::reset();
    kernel::reset();
    if (!g_options.disc_path.empty() && !disc::open(g_options.disc_path))
        std::fprintf(stderr, "[ee] cannot open disc image %s\n", g_options.disc_path.c_str());
    iop::set_trace(g_options.trace_iop_modules, g_options.trace_iop_hle);
    iop::reset();
    std::vector<std::string> guest_args = args;
    if (!guest_args.empty() && guest_args[0].find(':') == std::string::npos) {
        std::string name = guest_args[0].substr(guest_args[0].find_last_of("/\\") + 1);
        for (char &ch : name) ch = static_cast<char>(std::toupper(static_cast<unsigned char>(ch)));
        guest_args[0] = "cdrom0:\\" + name + ";1";
    }
    std::uint32_t argv = 0x00000800u, payload = argv + 4u * static_cast<std::uint32_t>(guest_args.size() + 1);
    for (std::size_t i = 0; i < guest_args.size(); ++i) {
        store<std::uint32_t>(g_ram + argv + 4 * i, payload);
        std::memcpy(g_ram + payload, guest_args[i].c_str(), guest_args[i].size() + 1);
        payload += static_cast<std::uint32_t>(guest_args[i].size() + 1);
    }
    store<std::uint32_t>(g_ram + argv + 4 * guest_args.size(), 0);
    kernel::start_main_thread(entry, 0, static_cast<std::uint32_t>(guest_args.size()), argv);
    apply_patches(true);
    schedule(hw::kEventVblank, frame_cycles(), vblank);
}

int run() {
    Cpu &c = g_cpu;
    if (!g_resumed) set_budget(c, kMaxSlice);
    while (!g_halted) {
        if (g_state_request) {
            g_state_request = false;
            save_state(g_options.save_state_path);
        }
        c.pc &= ~kUnwind;
        run_due_events();
        kernel::service(c);
        if (g_halted) break;
        if (c.pc == kIdle) {
            idle(c);
            continue;
        }
        if (c.budget <= 0) refill(c);
        dispatch(c, 0, 0);
    }
    kernel::dump();
    iop::dump();
    sif::dump();
    if (const char *peek = std::getenv("PS2_PEEK")) {
        std::uint32_t at = 0, len = 0;
        if (std::sscanf(peek, "%x,%x", &at, &len) == 2)
            for (std::uint32_t i = 0; i < len; i += 16) {
                std::fprintf(stderr, "[peek] %08X:", at + i);
                for (std::uint32_t j = 0; j < 16 && i + j < len; j += 4) std::fprintf(stderr, " %08X", ld32(at + i + j));
                std::fprintf(stderr, "\n");
            }
    }
    if (const char *path = std::getenv("PS2_DUMP")) {
        if (std::FILE *f = std::fopen(path, "wb")) {
            std::fwrite(g_ram, 1, kRamSize, f);
            std::fwrite(iop::ram_ptr(0), 1, iop::kRamSize, f);
            std::fwrite(iop::spu2::ram(), 2, 1u << 20, f);
            std::fclose(f);
        }
    }
    gs::stats();
    ipu::stats();
    if (const char *path = std::getenv("PS2_DUMP_AUDIO")) iop::spu2::dump_audio(path);
    if (std::getenv("PS2_TRACE_SPU2")) iop::spu2::dump_voices();
    dmac::stats();
    gif::stats();
    vif::stats();
    std::fprintf(stderr, "[ee] stopped after %llu frames, %llu cycles: %s\n",
                 static_cast<unsigned long long>(g_frames), static_cast<unsigned long long>(now()), g_halt_reason.c_str());
    return 0;
}

bool display_frame(std::vector<std::uint32_t> &rgba, std::uint32_t &width, std::uint32_t &height) { return gs::display(rgba, width, height); }
bool attach_window(const WindowTarget &t) { return gs::attach_window(static_cast<int>(t.kind), t.display, t.window, t.surface, t.selector, t.width, t.height); }
void resize_window(std::uint32_t width, std::uint32_t height) { gs::resize_window(width, height); }
bool present_frame() { return gs::present(); }
void init_graphics() { gs::init(); }
void start_render_thread() { gs::start_thread(); }
void start_vu_thread() { gs::queue::vu().start(32u << 20); }

std::atomic<double> g_vu_idle_ms{0.0};

double vu_thread_idle_ms() { return g_vu_idle_ms.load(); }
std::uint64_t vu1_interpreted_calls() { return vu::interpreted_calls(); }
std::uint64_t vu1_jit_programs() { return vu::jit_compiled(); }
double vu1_jit_ms() { return vu::jit_ms(); }
double gs_ring_wait_ms(int ring) { return ring ? gs::queue::vu().blocked_ms() : gs::queue::render().blocked_ms(); }

void run_vu_thread() {
    gs::queue::role() = gs::queue::Role::Vu;
    gs::queue::Ring &in = gs::queue::vu();
    for (;;) {
        gs::queue::Header h;
        const std::uint8_t *payload = nullptr;
        while (in.peek(h, payload)) {
            switch (h.type) {
            case gs::queue::kVif1: vif::transfer(1, payload, h.arg0); break;
            case gs::queue::kVif1Write: vif::write(1, h.arg0, h.arg1); break;
            case gs::queue::kVuReset:
                vif::reset_vu_side();
                vu::reset();
                break;
            case gs::queue::kVuFence:
                g_vu_fence.fetch_add(1);
                gs::queue::notify(g_vu_fence);
                break;
            default: gs::queue::render().post(static_cast<gs::queue::Type>(h.type), h.arg0, h.arg1, payload, h.bytes); break;
            }
            in.pop(h);
        }
        const auto start = std::chrono::steady_clock::now();
        in.wait_for_work(50);
        g_vu_idle_ms.store(g_vu_idle_ms.load() + std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count());
    }
}
void post_present() { gs::post_present(); }

Rendered render_pending(bool surface, const std::function<void(const std::uint32_t *, std::uint32_t, std::uint32_t)> &copy,
                        const std::function<bool()> &before_present, double budget_ms) {
    static std::vector<std::uint32_t> frame;
    return gs::render_pending([&](const gs::DisplayRegs &regs) {
        if (before_present && !before_present()) return;
        if (surface) {
            gs::present(regs);
            return;
        }
        std::uint32_t width = 0, height = 0;
        if (copy && gs::copy_frame(regs, frame, width, height)) copy(frame.data(), width, height);
    }, budget_ms);
}

void wait_render_work(int timeout_ms) { gs::queue::render().wait_for_work(timeout_ms); }
void wake_render_thread() { gs::queue::render().wake(); }
bool copy_frame(std::vector<std::uint32_t> &rgba, std::uint32_t &width, std::uint32_t &height) { return gs::copy_frame(rgba, width, height); }

bool parse_press(const std::string &spec, Press &press) {
    static const char *const kNames[16] = {"select", "l3", "r3", "start", "up", "right", "down", "left",
                                           "l2", "r2", "l1", "r1", "triangle", "circle", "cross", "square"};
    const auto colon = spec.find(':');
    if (colon == std::string::npos) return false;
    press.frame = std::strtoull(spec.c_str(), nullptr, 10);
    std::string rest = spec.substr(colon + 1);
    press.frames = 6;
    const auto second = rest.find(':');
    if (second != std::string::npos) {
        press.frames = static_cast<std::uint32_t>(std::strtoul(rest.c_str() + second + 1, nullptr, 10));
        rest = rest.substr(0, second);
    }
    press.buttons = 0;
    press.stick = 0;
    static const char *const kStick[4] = {"lup", "ldown", "lleft", "lright"};
    std::size_t start = 0;
    while (start <= rest.size()) {
        const auto plus = rest.find('+', start);
        const std::string name = rest.substr(start, plus == std::string::npos ? std::string::npos : plus - start);
        int bit = -1;
        for (int i = 0; i < 16; ++i)
            if (name == kNames[i]) bit = i;
        int axis = -1;
        for (int i = 0; i < 4; ++i)
            if (name == kStick[i]) axis = i;
        if (bit < 0 && axis < 0) return false;
        if (bit >= 0) press.buttons |= static_cast<std::uint16_t>(1u << bit);
        else press.stick |= static_cast<std::uint8_t>(1u << axis);
        if (plus == std::string::npos) break;
        start = plus + 1;
    }
    return true;
}

bool load_patches(const std::string &path) {
    std::FILE *f = std::fopen(path.c_str(), "r");
    if (!f) return false;
    unsigned place = 0, bytes = 0;
    unsigned long address = 0;
    unsigned long long value = 0;
    while (std::fscanf(f, "%u %lx %u %llx", &place, &address, &bytes, &value) == 4)
        g_options.patches.push_back({static_cast<std::uint8_t>(place), static_cast<std::uint8_t>(bytes), static_cast<std::uint32_t>(address), value});
    std::fclose(f);
    return true;
}

bool parse_axis(const std::string &spec, AxisPress &press) {
    static const char *const kNames[4] = {"rx", "ry", "lx", "ly"};
    std::size_t at = 0;
    const auto field = [&](std::string &out) {
        const auto colon = spec.find(':', at);
        if (colon == std::string::npos) return false;
        out = spec.substr(at, colon - at);
        at = colon + 1;
        return true;
    };
    std::string frame, axis, value;
    if (!field(frame) || !field(axis)) return false;
    const auto tail = spec.find(':', at);
    value = spec.substr(at, tail == std::string::npos ? std::string::npos : tail - at);
    press.frame = std::strtoull(frame.c_str(), nullptr, 10);
    press.axis = -1;
    for (int i = 0; i < 4; ++i)
        if (axis == kNames[i]) press.axis = i;
    if (press.axis < 0) return false;
    press.value = static_cast<std::uint8_t>(std::strtoul(value.c_str(), nullptr, 0));
    press.frames = 6;
    if (tail != std::string::npos) press.frames = static_cast<std::uint32_t>(std::strtoul(spec.c_str() + tail + 1, nullptr, 10));
    return press.frame || press.frames;
}

double gpu_busy_ms() { return gs::gpu_busy_ms(); }
double gpu_readback_ms(std::uint64_t &count) { return gs::gpu_readback_ms(count); }
std::uint64_t gpu_batch_ends(int reason) { return gs::gpu_batch_ends(reason); }

std::uint64_t state_hash() {
    const std::uint64_t ee = hash_memory(g_ram, kRamSize) ^ (hash_memory(g_spr, kScratchpadSize) * 0x100000001B3ull);
    return ee ^ (hash_memory(iop::g_ram, iop::kRamSize) * 0x9E3779B97F4A7C15ull);
}

std::time_t guest_clock() {
    if (g_options.rtc_epoch < 0) return std::time(nullptr);
    return static_cast<std::time_t>(g_options.rtc_epoch + static_cast<std::int64_t>(now() / kEeClock));
}

#ifdef __EMSCRIPTEN__
EM_ASYNC_JS(void, ps2_browser_yield, (), {
    if (!Module.yieldPort) {
        const channel = new MessageChannel();
        Module.yieldWaiters = [];
        channel.port1.onmessage = () => Module.yieldWaiters.shift()();
        Module.yieldPort = channel.port2;
    }
    await new Promise((resolve) => { Module.yieldWaiters.push(resolve); Module.yieldPort.postMessage(0); });
});
void browser_yield() { ps2_browser_yield(); }
#else
void browser_yield() {}
#endif

std::uint64_t frames_run() { return g_frames; }
std::uint64_t primitives_drawn() { return gs::primitives_drawn(); }
void set_pad_buttons(std::uint16_t pressed, int pad) { iop::sio2::set_buttons(pad, pressed); }
void set_pad_axis(int axis, std::uint8_t value, int pad) { iop::sio2::set_axis(pad, axis, value); }
void set_pad_pressure(int button, std::uint8_t value, int pad) { iop::sio2::set_pressure(pad, button, value); }
void set_pad_connected(int pad, bool connected) { iop::sio2::set_connected(pad, connected); }
std::uint16_t pad_rumble(int pad) { return iop::sio2::rumble(pad); }
void set_audio_sink(std::function<void(const std::int16_t *, std::size_t)> sink) { iop::spu2::set_sink(std::move(sink)); }

}
