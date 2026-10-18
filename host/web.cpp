#include "ps2/disc.hpp"
#include "ps2/system.hpp"

#include <emscripten.h>
#include <emscripten/threading.h>
#include <pthread.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

void register_generated_code();
void register_generated_vu();

EM_JS_DEPS(ps2web, "$stringToNewUTF8,$UTF8ToString");

EM_JS(void, ps2web_fetch, (const char *url, double offset, int size, void *out, std::atomic<std::uint32_t> *state), {
    const target = UTF8ToString(url);
    const finish = (value) => {
        Atomics.store(HEAP32, state >> 2, value);
        Atomics.notify(HEAP32, state >> 2);
    };
    fetch(target, {headers: {Range: 'bytes=' + offset + '-' + (offset + size - 1)}})
        .then((response) => {
            if (response.status !== 206 && response.status !== 200) throw new Error('status ' + response.status);
            return response.arrayBuffer().then((buffer) => {
                let bytes = new Uint8Array(buffer);
                if (response.status === 200) bytes = bytes.subarray(offset, offset + size);
                HEAPU8.set(bytes.subarray(0, Math.min(bytes.length, size)), out);
                finish(3);
            });
        })
        .catch((e) => {
            console.warn('disc read failed', e);
            finish(4);
        });
});

EM_JS(int, ps2web_save_program, (const char *name, const std::uint8_t *code, int size), {
    try {
        const xhr = new XMLHttpRequest();
        xhr.open('PUT', 'vu1/' + UTF8ToString(name), false);
        xhr.send(HEAPU8.slice(code, code + size));
        return xhr.status;
    } catch (e) {
        return 0;
    }
});

EM_JS(double, ps2web_disc_size, (), { return Module.discSize || 0; });
EM_JS(char *, ps2web_state_url, (), { return Module.stateUrl ? stringToNewUTF8(new URL(Module.stateUrl, location.href).href) : 0; });
EM_JS(double, ps2web_url_size, (const char *url), {
    try {
        const xhr = new XMLHttpRequest();
        xhr.open('HEAD', UTF8ToString(url), false);
        xhr.send();
        return xhr.status === 200 ? Number(xhr.getResponseHeader('Content-Length') || 0) : 0;
    } catch (e) {
        return 0;
    }
});
EM_JS(double, ps2web_url_read, (const char *url, std::uint8_t *out, double size), {
    try {
        const xhr = new XMLHttpRequest();
        xhr.open('GET', UTF8ToString(url), false);
        xhr.responseType = 'arraybuffer';
        xhr.send();
        if (xhr.status !== 200) return 0;
        const bytes = new Uint8Array(xhr.response);
        HEAPU8.set(bytes.subarray(0, size), out);
        return bytes.length;
    } catch (e) {
        return 0;
    }
});
EM_JS(char *, ps2web_disc_url, (), { return stringToNewUTF8(new URL(Module.discUrl, location.href).href); });
EM_JS(void, ps2web_frame_done, (), {
    if (!Module.shownChannel) {
        Module.shownChannel = new MessageChannel();
        Module.shownWaiters = [];
        Module.shownChannel.port1.onmessage = () => {
            Module.shownPending = false;
            const waiters = Module.shownWaiters;
            Module.shownWaiters = [];
            for (const resolve of waiters) resolve();
        };
    }
    Module.shownPending = true;
    requestAnimationFrame(() => Module.shownChannel.port2.postMessage(0));
    if (Module.onFrame) Module.onFrame();
});
EM_JS(int, ps2web_shown_pending, (), { return Module.shownPending ? 1 : 0; });
// a frame drawn but not shown: the page still polls input and counts it
EM_JS(void, ps2web_frame_dropped, (), { if (Module.onFrame) Module.onFrame(true); });
EM_JS(void, ps2web_blit, (const std::uint32_t *rgba, int width, int height, int wide), {
    const canvas = Module.canvas;
    if (!Module.blitContext) Module.blitContext = canvas.getContext('2d');
    if (!Module.blitImage || Module.blitImage.width !== width || Module.blitImage.height !== height) {
        Module.blitImage = new ImageData(width, height);
        Module.blitSource = new OffscreenCanvas(width, height);
    }
    Module.blitImage.data.set(HEAPU8.subarray(rgba, rgba + width * height * 4));
    Module.blitSource.getContext('2d').putImageData(Module.blitImage, 0, 0);
    const ctx = Module.blitContext;
    const across = wide ? 16 : 4, down = wide ? 9 : 3;
    const scale = Math.min(canvas.width / across, canvas.height / down);
    const w = scale * across, h = scale * down;
    const sx = Math.max(1, Math.floor(w / width)), sy = Math.max(1, Math.floor(h / height));
    let source = Module.blitSource;
    if (sx > 1 || sy > 1) {
        if (!Module.blitSharp || Module.blitSharp.width !== width * sx || Module.blitSharp.height !== height * sy)
            Module.blitSharp = new OffscreenCanvas(width * sx, height * sy);
        const sharp = Module.blitSharp.getContext('2d');
        sharp.imageSmoothingEnabled = false;
        sharp.drawImage(Module.blitSource, 0, 0, width * sx, height * sy);
        source = Module.blitSharp;
    }
    ctx.fillStyle = '#000';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    ctx.imageSmoothingEnabled = true;
    ctx.drawImage(source, (canvas.width - w) / 2, (canvas.height - h) / 2, w, h);
});

namespace {

constexpr std::size_t kChunk = 256u << 10;
constexpr std::uint32_t kSlots = 256;
constexpr std::uint64_t kReadAhead = 16;
constexpr std::uint32_t kAudioFrames = 1u << 14;

enum : std::uint32_t { kEmpty, kWanted, kLoading, kReady, kFailed };

struct Slot {
    std::atomic<std::uint32_t> state{kEmpty};
    std::uint64_t index = 0, used = 0;
    std::uint32_t size = 0;
    std::uint8_t *data = nullptr;
};

struct AudioRing {
    std::atomic<std::uint32_t> write{0}, read{0};
    std::uint32_t capacity = kAudioFrames;
    std::atomic<std::uint32_t> active{0};
    std::int16_t samples[kAudioFrames * 2];
};

// What the page last reported for each port. The page writes it whenever it polls; the emulation thread only
// hands it to the SIO2 at the vblank, so an input never changes in the middle of a frame. state packs the
// buttons in bits 0-15, the axes in bits 16-47 (right x highest, left y lowest) and the connected flag in bit 48.
struct LivePad {
    std::atomic<std::uint64_t> state;
    std::atomic<std::uint32_t> pressure[4];
};

constexpr std::uint64_t kConnected = 1ull << 48;
constexpr std::uint64_t kCentred = 0x80808080ull << 16;
constexpr std::uint32_t kNoPressure = 0xFFFFFFFFu;

// port 1 starts with a controller in it and port 2 empty, as the SIO2 does
LivePad g_live[2] = {{{kCentred | kConnected}, {kNoPressure, kNoPressure, kNoPressure, kNoPressure}},
                     {{kCentred}, {kNoPressure, kNoPressure, kNoPressure, kNoPressure}}};
// what the emulation thread last handed over; only a change is applied, so scripted presses keep working
std::uint64_t g_latched_state[2] = {kCentred | kConnected, kCentred};
std::uint32_t g_latched_pressure[2][4] = {{kNoPressure, kNoPressure, kNoPressure, kNoPressure}, {kNoPressure, kNoPressure, kNoPressure, kNoPressure}};

void latch_input() {
    // a replay owns the controllers; live input would make it diverge
    if (!ps2::g_options.input_replay_path.empty()) return;
    for (int pad = 0; pad < 2; ++pad) {
        const std::uint64_t state = g_live[pad].state.load(std::memory_order_acquire);
        if (state != g_latched_state[pad]) {
            if ((state ^ g_latched_state[pad]) & kConnected) ps2::set_pad_connected(pad, (state & kConnected) != 0);
            g_latched_state[pad] = state;
            ps2::set_pad_buttons(static_cast<std::uint16_t>(state), pad);
            for (int axis = 0; axis < 4; ++axis) ps2::set_pad_axis(axis, static_cast<std::uint8_t>(state >> (40 - axis * 8)), pad);
        }
        for (int word = 0; word < 4; ++word) {
            const std::uint32_t packed = g_live[pad].pressure[word].load(std::memory_order_relaxed);
            if (packed == g_latched_pressure[pad][word]) continue;
            g_latched_pressure[pad][word] = packed;
            for (int i = 0; i < 4; ++i) ps2::set_pad_pressure(word * 4 + i, static_cast<std::uint8_t>(packed >> (i * 8)), pad);
        }
    }
}

AudioRing g_audio;
Slot g_slots[kSlots];
// milliseconds the emulation spent blocked on a disc block that had not arrived yet
std::atomic<double> g_disc_wait{0.0};
std::uint64_t g_disc_size = 0, g_use_clock = 0, g_last_chunk = ~0ull, g_run = 0;
char *g_disc_url = nullptr;
char *g_state_url = nullptr;

Slot *find_slot(std::uint64_t index) {
    for (Slot &s : g_slots)
        if (s.state.load(std::memory_order_acquire) != kEmpty && s.index == index) return &s;
    return nullptr;
}

Slot *want(std::uint64_t index) {
    if (index * kChunk >= g_disc_size) return nullptr;
    if (Slot *s = find_slot(index)) return s;
    Slot *victim = nullptr;
    for (Slot &s : g_slots) {
        const std::uint32_t state = s.state.load(std::memory_order_acquire);
        if (state == kEmpty) {
            victim = &s;
            break;
        }
        if ((state == kReady || state == kFailed) && (!victim || s.used < victim->used)) victim = &s;
    }
    if (!victim) return nullptr;
    if (!victim->data) victim->data = new std::uint8_t[kChunk];
    victim->index = index;
    victim->size = static_cast<std::uint32_t>(std::min<std::uint64_t>(kChunk, g_disc_size - index * kChunk));
    victim->used = ++g_use_clock;
    victim->state.store(kWanted, std::memory_order_release);
    ps2::wake_render_thread();
    return victim;
}

void advance(std::uint64_t index) {
    if (index <= g_last_chunk && g_last_chunk - index <= kReadAhead * 4) return;
    g_run = index == g_last_chunk + 1 ? g_run + 1 : 0;
    g_last_chunk = index;
}

void read_ahead(std::uint64_t index) {
    for (std::uint64_t k = 1; k <= std::min(g_run * 2, kReadAhead); ++k) want(index + k);
}

const Slot *chunk(std::uint64_t index) {
    Slot *s = want(index);
    advance(index);
    read_ahead(std::max(index, g_last_chunk));
    for (int attempt = 0; s && attempt < 4; ++attempt) {
        const std::uint32_t first = s->state.load();
        const double start = first != kReady && first != kFailed ? emscripten_get_now() : 0.0;
        for (std::uint32_t state = s->state.load(); state != kReady && state != kFailed; state = s->state.load())
            emscripten_futex_wait(&s->state, state, INFINITY);
        if (start > 0.0) g_disc_wait.store(g_disc_wait.load() + emscripten_get_now() - start);
        if (s->state.load() == kReady) {
            s->used = ++g_use_clock;
            return s;
        }
        usleep(100000);
        s->state.store(kWanted, std::memory_order_release);
        ps2::wake_render_thread();
    }
    return nullptr;
}

bool disc_ready(std::uint64_t offset, std::size_t size) {
    if (!size) return true;
    const std::uint64_t last = (offset + size - 1) / kChunk;
    bool ready = true;
    for (std::uint64_t index = offset / kChunk; index <= last; ++index) {
        Slot *s = want(index);
        if (s && s->state.load(std::memory_order_acquire) == kFailed) {
            s->state.store(kWanted, std::memory_order_release);
            ps2::wake_render_thread();
        }
        ready = ready && s && s->state.load(std::memory_order_acquire) == kReady;
        advance(index);
    }
    read_ahead(std::max(last, g_last_chunk));
    return ready;
}

std::size_t read_disc(std::uint64_t offset, void *out, std::size_t size) {
    std::size_t done = 0;
    auto *to = static_cast<std::uint8_t *>(out);
    while (done < size && offset + done < g_disc_size) {
        const std::uint64_t at = offset + done;
        const Slot *c = chunk(at / kChunk);
        if (!c) break;
        const std::size_t within = static_cast<std::size_t>(at % kChunk);
        const std::size_t take = std::min<std::size_t>(size - done, c->size - within);
        std::memcpy(to + done, c->data + within, take);
        done += take;
    }
    return done;
}

void service_disc() {
    for (Slot &s : g_slots) {
        std::uint32_t expected = kWanted;
        if (s.state.compare_exchange_strong(expected, kLoading, std::memory_order_acq_rel))
            ps2web_fetch(g_disc_url, static_cast<double>(s.index * kChunk), static_cast<int>(s.size), s.data, &s.state);
    }
}

void push_audio(const std::int16_t *samples, std::size_t frames) {
    std::uint32_t write = g_audio.write.load(std::memory_order_relaxed);
    const std::uint32_t read = g_audio.read.load(std::memory_order_acquire);
    for (std::size_t i = 0; i < frames && write - read < kAudioFrames; ++i, ++write) {
        const std::uint32_t at = (write & (kAudioFrames - 1)) * 2;
        g_audio.samples[at] = samples[i * 2];
        g_audio.samples[at + 1] = samples[i * 2 + 1];
    }
    g_audio.write.store(write, std::memory_order_release);
}

double audio_queued_ms() { return static_cast<double>(g_audio.write.load() - g_audio.read.load()) / 48.0; }

double g_next_frame = 0.0;
std::atomic<double> g_emulation_idle{0.0}, g_render_idle{0.0};

// waits at the vblank that are not frame work: for the page to show earlier frames, and sleeps that
// overran what was asked by more than 4 ms
std::atomic<double> g_present_wait{0.0}, g_oversleep{0.0};
std::atomic<std::uint64_t> g_audio_low{0}, g_dropped_frames{0};

void sleep_ms(double ms) {
    const double before = emscripten_get_now();
    usleep(static_cast<useconds_t>(ms * 1000.0));
    const double over = emscripten_get_now() - before - ms;
    if (over > 4.0) g_oversleep.store(g_oversleep.load() + over);
}

void pace() {
    const double start = emscripten_get_now();
    ps2::post_present();
    g_present_wait.store(g_present_wait.load() + emscripten_get_now() - start);
    if (g_audio.active.load()) {
        while (audio_queued_ms() > 80.0) sleep_ms(1.0);
        g_next_frame = 0.0;
        return;
    }
    const double period = ps2::g_options.pal ? 20.0 : 1000.0 / 59.94;
    const double now = emscripten_get_now();
    if (g_next_frame == 0.0 || now - g_next_frame > 200.0) g_next_frame = now;
    g_next_frame += period;
    if (g_next_frame > now) sleep_ms(g_next_frame - now);
}

// A frame whose emulation took more than one and a half frame periods. With Module.perf set, each one is
// logged with what the emulation thread waited on meanwhile.
bool g_perf = false;
double g_frame_started = 0.0, g_mark_jit_ms = 0.0, g_mark_disc_ms = 0.0, g_mark_ring_ms[2] = {};
std::uint64_t g_mark_jit_programs = 0;
std::atomic<std::uint64_t> g_slow_frames{0};

void check_frame(double now) {
    const double period = ps2::g_options.pal ? 20.0 : 1000.0 / 59.94;
    const std::uint64_t programs = ps2::vu1_jit_programs();
    const double jit = ps2::vu1_jit_ms(), disc = g_disc_wait.load();
    const double ring[2] = {ps2::gs_ring_wait_ms(0), ps2::gs_ring_wait_ms(1)};
    const double work = g_frame_started > 0.0 ? now - g_frame_started : 0.0;
    if (work > period * 1.5) {
        g_slow_frames.fetch_add(1, std::memory_order_relaxed);
        if (g_perf)
            std::fprintf(stderr, "[perf] frame %llu: %.1f ms of emulation for a %.1f ms frame; VU1 JIT %llu programs %.1f ms, disc wait %.1f ms, "
                                 "full GS ring %.1f ms, full VU ring %.1f ms, audio %.0f ms queued\n",
                         static_cast<unsigned long long>(ps2::frames_run()), work, period, static_cast<unsigned long long>(programs - g_mark_jit_programs),
                         jit - g_mark_jit_ms, disc - g_mark_disc_ms, ring[0] - g_mark_ring_ms[0], ring[1] - g_mark_ring_ms[1], audio_queued_ms());
    }
    g_mark_jit_programs = programs;
    g_mark_jit_ms = jit;
    g_mark_disc_ms = disc;
    g_mark_ring_ms[0] = ring[0];
    g_mark_ring_ms[1] = ring[1];
}

// The audio ring close to empty once the vblank wait is over: the next stall, of any kind, is heard.
void check_audio(double present_before, double oversleep_before) {
    if (!g_audio.active.load()) return;
    const double queued = audio_queued_ms();
    if (queued >= 20.0) return;
    g_audio_low.fetch_add(1, std::memory_order_relaxed);
    if (g_perf)
        std::fprintf(stderr, "[perf] frame %llu: audio down to %.0f ms after the vblank wait; it waited %.1f ms for the page to show frames, sleeps overran by %.1f ms\n",
                     static_cast<unsigned long long>(ps2::frames_run()), queued, g_present_wait.load() - present_before, g_oversleep.load() - oversleep_before);
}

void on_vblank() {
    const double start = emscripten_get_now();
    check_frame(start);
    const double present_before = g_present_wait.load(), oversleep_before = g_oversleep.load();
    pace();
    check_audio(present_before, oversleep_before);
    g_frame_started = emscripten_get_now();
    g_emulation_idle.store(g_emulation_idle.load() + g_frame_started - start);
    // after the wait, so the frame starts with the freshest input
    latch_input();
}

struct Launch {
    std::vector<std::string> args;
};

void *emulate(void *data) {
    const Launch &launch = *static_cast<const Launch *>(data);
    if (g_disc_size && !ps2::disc::open("disc", g_disc_size, read_disc, disc_ready)) std::fprintf(stderr, "[web] cannot read the disc image\n");
    std::uint32_t entry = 0;
    ps2::load_patches("/game/patches.txt");
    if (!ps2::load_elf("/game/core.elf", entry)) {
        std::fprintf(stderr, "cannot load the game executable\n");
        return nullptr;
    }
    ps2::boot(entry, launch.args);
    if (g_state_url) {
        const double size = ps2web_url_size(g_state_url);
        std::vector<std::uint8_t> data(static_cast<std::size_t>(size));
        if (size <= 0 || ps2web_url_read(g_state_url, data.data(), size) != size || !ps2::load_state(data.data(), data.size())) {
            std::fprintf(stderr, "[web] cannot load the state %s\n", g_state_url);
            return nullptr;
        }
    }
    ps2::run();
    EM_ASM({
        if (typeof Module.onHalt === 'function')
            Module.onHalt(UTF8ToString($0));
    }, ps2::halt_reason().c_str());
    return nullptr;
}

}

extern "C" EMSCRIPTEN_KEEPALIVE void ps2web_pad(int pad, int buttons, int lx, int ly, int rx, int ry) {
    if (pad < 0 || pad > 1) return;
    const std::uint64_t axes = static_cast<std::uint64_t>(rx & 0xFF) << 40 | static_cast<std::uint64_t>(ry & 0xFF) << 32 |
                               static_cast<std::uint64_t>(lx & 0xFF) << 24 | static_cast<std::uint64_t>(ly & 0xFF) << 16;
    std::uint64_t old = g_live[pad].state.load(std::memory_order_relaxed);
    while (!g_live[pad].state.compare_exchange_weak(old, (old & kConnected) | axes | static_cast<std::uint16_t>(buttons), std::memory_order_release)) {}
}

extern "C" EMSCRIPTEN_KEEPALIVE void ps2web_pad_connected(int pad, int connected) {
    if (pad < 0 || pad > 1) return;
    if (connected) g_live[pad].state.fetch_or(kConnected, std::memory_order_release);
    else g_live[pad].state.fetch_and(~kConnected, std::memory_order_release);
}

extern "C" EMSCRIPTEN_KEEPALIVE void ps2web_pressure(int pad, int a, int b, int c, int d) {
    if (pad < 0 || pad > 1) return;
    const int packed[4] = {a, b, c, d};
    for (int i = 0; i < 4; ++i) g_live[pad].pressure[i].store(static_cast<std::uint32_t>(packed[i]), std::memory_order_relaxed);
}

extern "C" EMSCRIPTEN_KEEPALIVE int ps2web_rumble(int pad) { return ps2::pad_rumble(pad); }

extern "C" EMSCRIPTEN_KEEPALIVE void ps2web_resize(int width, int height) {
    ps2::resize_window(static_cast<std::uint32_t>(width), static_cast<std::uint32_t>(height));
}

extern "C" EMSCRIPTEN_KEEPALIVE double ps2web_gpu_ms() { return ps2::gpu_busy_ms(); }
extern "C" EMSCRIPTEN_KEEPALIVE double ps2web_frames() { return static_cast<double>(ps2::frames_run()); }
extern "C" EMSCRIPTEN_KEEPALIVE double ps2web_primitives() { return static_cast<double>(ps2::primitives_drawn()); }
extern "C" EMSCRIPTEN_KEEPALIVE void *ps2web_audio_ring() { return &g_audio; }
extern "C" EMSCRIPTEN_KEEPALIVE double ps2web_audio_queued_ms() { return audio_queued_ms(); }
extern "C" EMSCRIPTEN_KEEPALIVE double ps2web_vu_interpreted() { return static_cast<double>(ps2::vu1_interpreted_calls()); }
extern "C" EMSCRIPTEN_KEEPALIVE double ps2web_vu_jit_programs() { return static_cast<double>(ps2::vu1_jit_programs()); }
extern "C" EMSCRIPTEN_KEEPALIVE double ps2web_vu_jit_ms() { return ps2::vu1_jit_ms(); }
extern "C" EMSCRIPTEN_KEEPALIVE double ps2web_disc_wait_ms() { return g_disc_wait.load(); }
extern "C" EMSCRIPTEN_KEEPALIVE double ps2web_slow_frames() { return static_cast<double>(g_slow_frames.load(std::memory_order_relaxed)); }
extern "C" EMSCRIPTEN_KEEPALIVE double ps2web_present_wait_ms() { return g_present_wait.load(); }
extern "C" EMSCRIPTEN_KEEPALIVE double ps2web_oversleep_ms() { return g_oversleep.load(); }
extern "C" EMSCRIPTEN_KEEPALIVE double ps2web_audio_low_frames() { return static_cast<double>(g_audio_low.load(std::memory_order_relaxed)); }
extern "C" EMSCRIPTEN_KEEPALIVE double ps2web_dropped_frames() { return static_cast<double>(g_dropped_frames.load(std::memory_order_relaxed)); }
extern "C" EMSCRIPTEN_KEEPALIVE double ps2web_ring_wait_ms(int ring) { return ps2::gs_ring_wait_ms(ring); }
extern "C" EMSCRIPTEN_KEEPALIVE double ps2web_idle_ms(int which) {
    if (which == 2) return ps2::vu_thread_idle_ms();
    return which ? g_render_idle.load() : g_emulation_idle.load();
}

int main(int argc, char **argv) {
    static Launch launch;
    for (int i = 2; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--arg" && i + 1 < argc) launch.args.push_back(argv[++i]);
        else if (a == "--frames" && i + 1 < argc) ps2::g_options.max_frames = std::strtoull(argv[++i], nullptr, 10);
        else if (a == "--widescreen") ps2::g_options.widescreen = true;
        else if (a == "--press" && i + 1 < argc) {
            ps2::Press press;
            if (ps2::parse_press(argv[++i], press)) ps2::g_options.presses.push_back(press);
        }
        else if (a == "--record-input" && i + 1 < argc) ps2::g_options.input_record_path = argv[++i];
        else if (a == "--replay-input" && i + 1 < argc) ps2::g_options.input_replay_path = argv[++i];
        else if (a == "--hash-log" && i + 1 < argc) ps2::g_options.hash_log_path = argv[++i];
    }
    if (launch.args.empty()) {
        std::string boot = "/game/core.elf";
        if (std::FILE *f = std::fopen("/game/boot.txt", "rb")) {
            char text[256] = {};
            const std::size_t n = std::fread(text, 1, sizeof text - 1, f);
            std::fclose(f);
            std::string line(text, n);
            line.erase(line.find_last_not_of(" \r\n\t") + 1);
            if (!line.empty()) boot = line;
        }
        launch.args.push_back(boot);
    }
    g_disc_size = static_cast<std::uint64_t>(ps2web_disc_size());
    g_disc_url = ps2web_disc_url();
    g_state_url = ps2web_state_url();
    g_perf = EM_ASM_INT({ return Module.perf ? 1 : 0; }) != 0;
    ps2::g_options.on_vblank = on_vblank;
    ps2::g_options.on_vu_program = [](const std::string &name, const std::uint8_t *code, std::size_t size) {
        static bool unsupported = false;
        if (unsupported) return;
        const int status = ps2web_save_program(name.c_str(), code, static_cast<int>(size));
        if (status == 200 || status == 201) std::fprintf(stderr, "[web] saved VU1 program %s\n", name.c_str());
        else unsupported = true;
    };
    if (!EM_ASM_INT({ return Module.noCard ? 1 : 0; })) {
        ps2::g_options.memcard_path = "/memcard/Mcd001.ps2";
        ps2::g_options.on_memcard_saved = [] { MAIN_THREAD_ASYNC_EM_ASM({ if (Module.ps2CardSaved) Module.ps2CardSaved(); }); };
    }
    ps2::set_audio_sink(push_audio);
    register_generated_code();
    register_generated_vu();

    ps2::init_graphics();
    const bool no_surface = EM_ASM_INT({ return Module.noSurface ? 1 : 0; });
    bool copy = EM_ASM_INT({ return Module.presentCopy ? 1 : 0; });
    bool surface = false;
    if (!copy && !no_surface) {
        ps2::WindowTarget target;
        target.kind = ps2::WindowTarget::Canvas;
        target.selector = "#canvas";
        target.width = static_cast<std::uint32_t>(EM_ASM_INT({ return Module.canvas.width; }));
        target.height = static_cast<std::uint32_t>(EM_ASM_INT({ return Module.canvas.height; }));
        surface = ps2::attach_window(target);
    }
    copy = copy || (!surface && !no_surface);
    std::fprintf(stderr, "[web] %s\n", surface ? "presenting through WebGPU" : copy ? "presenting through a 2D canvas copy" : "not presenting");

    ps2::start_render_thread();
    pthread_attr_t attr;
    pthread_attr_init(&attr);
    pthread_attr_setstacksize(&attr, 16u << 20);
    if (!std::getenv("PS2_NO_VU_THREAD")) {
        ps2::start_vu_thread();
        pthread_t vu_thread;
        if (pthread_create(&vu_thread, &attr, [](void *) -> void * {
                ps2::run_vu_thread();
                return nullptr;
            }, nullptr) != 0) {
            std::fprintf(stderr, "[web] cannot start the VU thread\n");
            return 1;
        }
    }
    pthread_t thread;
    if (pthread_create(&thread, &attr, emulate, &launch) != 0) {
        std::fprintf(stderr, "[web] cannot start the emulation thread\n");
        return 1;
    }
    const std::function<void(const std::uint32_t *, std::uint32_t, std::uint32_t)> blit = [](const std::uint32_t *rgba, std::uint32_t width, std::uint32_t height) {
        ps2web_blit(rgba, static_cast<int>(width), static_cast<int>(height), ps2::g_options.widescreen ? 1 : 0);
    };
    const std::function<void(const std::uint32_t *, std::uint32_t, std::uint32_t)> none;
    // While the page has not shown the previous frame, a new one is drawn but not presented. Waiting instead
    // would hold the emulation to the display: a slow or covered window would slow the game and starve the audio.
    static bool dropped = false;
    const std::function<bool()> show_unless_pending = [] {
        dropped = ps2web_shown_pending() != 0;
        if (dropped) g_dropped_frames.fetch_add(1, std::memory_order_relaxed);
        return !dropped;
    };
    for (;;) {
        service_disc();
        const ps2::Rendered rendered = ps2::render_pending(surface, copy ? blit : none, show_unless_pending, 6.0);
        if (rendered == ps2::Rendered::presented) {
            if (dropped) ps2web_frame_dropped();
            else ps2web_frame_done();
        }
        if (rendered != ps2::Rendered::idle) {
            ps2::browser_yield();
        } else {
            const double start = emscripten_get_now();
            ps2::wait_render_work(50);
            g_render_idle.store(g_render_idle.load() + emscripten_get_now() - start);
        }
    }
}
