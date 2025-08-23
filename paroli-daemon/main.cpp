#include <atomic>
#include <bit>
#include <csignal>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <mutex>
#include <condition_variable>
#include <optional>
#include <queue>
#include <span>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include <nlohmann/json.hpp>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/spdlog.h>

#include "piper/piper.hpp"
#include "paroli_daemon.hpp"
#include "OggOpusEncoder.hpp"

#ifdef __APPLE__
#include <mach-o/dyld.h>
#endif

using json = nlohmann::json;
using namespace std;

struct RunConfig {
    filesystem::path encoderPath;
    filesystem::path decoderPath;
    filesystem::path modelConfigPath;
    optional<filesystem::path> eSpeakDataPath;
    string accelerator = "";
    bool jsonl = false;
    bool stream = false;
    int maxConcurrency = 1;
    optional<filesystem::path> outputFile;
    bool playAudio = false;
    float volume = 1.0f;
};

static unique_ptr<ParoliSynthesizer> gSynth;
static atomic<bool> gShuttingDown{false};

static void printError(const string &msg) {
    json e;
    e["error"] = msg;
    cerr << e.dump() << '\n';
    cerr.flush();
}

static void printUsage(const char *argv0) {
    cerr << "\nusage: " << argv0 << " [options]\n\n";
    cerr << "options:\n";
    cerr << "   --encoder FILE            path to encoder model file\n";
    cerr << "   --decoder FILE            path to decoder model file\n";
    cerr << "   -c, --config FILE         path to model config file\n";
    cerr << "   --espeak_data DIR         path to espeak-ng data directory\n";
    cerr << "   --accelerator STR         accelerator for ONNX (e.g., cuda|tensorrt)\n";
    cerr << "   --jsonl                   JSON-in/JSON-out only (no logs to stdout)\n";
    cerr << "   --max-concurrency N       number of concurrent jobs (default 1)\n";
    cerr << "   --stream                  enable length-prefixed chunked streaming\n";
    cerr << "   --output FILE             write output to file instead of stdout\n";
    cerr << "   --play                    play audio directly to speakers (PCM format only)\n";
    cerr << "   --volume FLOAT            volume level for audio playback (0.0 to 1.0)\n";
}

static void parseArgs(int argc, char *argv[], RunConfig &cfg) {
    optional<filesystem::path> modelConfigPath;
    
    for (int i = 1; i < argc; i++) {
        string arg = argv[i];
        
        if (arg == "--encoder" && i + 1 < argc) {
            cfg.encoderPath = filesystem::path(argv[++i]);
        } else if (arg == "--decoder" && i + 1 < argc) {
            cfg.decoderPath = filesystem::path(argv[++i]);
        } else if ((arg == "-c" || arg == "--config") && i + 1 < argc) {
            modelConfigPath = filesystem::path(argv[++i]);
        } else if ((arg == "--espeak_data" || arg == "--espeak-data") && i + 1 < argc) {
            cfg.eSpeakDataPath = filesystem::path(argv[++i]);
        } else if (arg == "--accelerator" && i + 1 < argc) {
            cfg.accelerator = argv[++i];
        } else if (arg == "--jsonl") {
            cfg.jsonl = true;
        } else if (arg == "--max-concurrency" && i + 1 < argc) {
            cfg.maxConcurrency = max(1, stoi(argv[++i]));
        } else if (arg == "--stream") {
            cfg.stream = true;
        } else if (arg == "--output" && i + 1 < argc) {
            cfg.outputFile = filesystem::path(argv[++i]);
        } else if (arg == "--play") {
            cfg.playAudio = true;
        } else if (arg == "--volume" && i + 1 < argc) {
            cfg.volume = stof(argv[++i]);
            if (cfg.volume < 0.0f || cfg.volume > 1.0f) {
                throw runtime_error("Volume must be between 0.0 and 1.0");
            }
        } else if (arg == "--debug") {
            spdlog::set_level(spdlog::level::debug);
        } else if (arg == "-q" || arg == "--quiet") {
            spdlog::set_level(spdlog::level::off);
        } else if (arg == "-h" || arg == "--help") {
            printUsage(argv[0]);
            exit(0);
        }
    }

    // Validate required files
    if (!filesystem::exists(cfg.encoderPath)) {
        throw runtime_error("Encoder model file doesn't exist");
    }
    if (!filesystem::exists(cfg.decoderPath)) {
        throw runtime_error("Decoder model file doesn't exist");
    }
    if (!modelConfigPath) {
        throw runtime_error("Model config file must be provided");
    }
    cfg.modelConfigPath = *modelConfigPath;
    if (!filesystem::exists(cfg.modelConfigPath)) {
        throw runtime_error("Model config doesn't exist");
    }
}

static void setupPiper(const RunConfig &cfg) {
    ParoliSynthesizer::InitOptions opts;
    opts.encoderPath = cfg.encoderPath;
    opts.decoderPath = cfg.decoderPath;
    opts.modelConfigPath = cfg.modelConfigPath;
    opts.eSpeakDataPath = cfg.eSpeakDataPath;
    opts.accelerator = cfg.accelerator;
    gSynth = std::make_unique<ParoliSynthesizer>(opts);
    gSynth->setVolume(cfg.volume);
}

struct Request {
    string text;
    string format; // opus|wav|pcm
    optional<int> sampleRate;
    size_t id;
};

static vector<uint8_t> toLittleEndian4(uint32_t v) {
    vector<uint8_t> b(4);
    b[0] = static_cast<uint8_t>(v & 0xFF);
    b[1] = static_cast<uint8_t>((v >> 8) & 0xFF);
    b[2] = static_cast<uint8_t>((v >> 16) & 0xFF);
    b[3] = static_cast<uint8_t>((v >> 24) & 0xFF);
    return b;
}

static void writeAll(ostream &os, const char *data, size_t n) {
    os.write(data, n);
    os.flush();
}

static vector<int16_t> resample(span<const int16_t> input, size_t orig_sr, size_t out_sr, int channels) {
    return ParoliSynthesizer::resample(input, orig_sr, out_sr, channels);
}

static bool synthesizeOne(const RunConfig &cfg, const Request &req, ostream &outStream) {
    try {
        if (req.format != "opus" && req.format != "wav" && req.format != "pcm") {
            throw runtime_error("Unsupported format (opus|wav|pcm)");
        }

        // Setup output stream
        unique_ptr<ostream> fileOut;
        ostream *dst = &outStream;
        if (cfg.outputFile) {
            fileOut = make_unique<ofstream>(cfg.outputFile->string(), ios::binary);
            if (!fileOut->good()) throw runtime_error("Failed to open output file");
            dst = fileOut.get();
        }

        // Handle PCM format (native sample rate)
        if (req.format == "pcm") {
            if (cfg.playAudio) {
                if (!gSynth->speak(req.text)) {
                    cerr << "Failed to speak: " << gSynth->getLastError() << endl;
                }
            } else if (cfg.stream) {
                            gSynth->synthesizeStreamPcm(req.text, [&](std::span<const int16_t> view) {
                if (!view.empty() && view.data() != nullptr) {
                    uint32_t bytes = static_cast<uint32_t>(view.size() * sizeof(int16_t));
                    auto hdr = toLittleEndian4(bytes);
                    writeAll(*dst, reinterpret_cast<const char *>(hdr.data()), hdr.size());
                    writeAll(*dst, reinterpret_cast<const char *>(view.data()), bytes);
                }
            });
            } else {
                auto audio = gSynth->synthesizePcm(req.text);
                writeAll(*dst, reinterpret_cast<const char *>(audio.data()), audio.size() * sizeof(int16_t));
            }
            return true;
        }

        // Handle WAV/OPUS formats
        const int nativeSr = gSynth->nativeSampleRate();
        const int outSr = req.sampleRate.value_or(req.format == "opus" ? 24000 : nativeSr);

        if (cfg.stream) {
            vector<int16_t> chunk;
            unique_ptr<StreamingOggOpusEncoder> opusEnc;
            if (req.format == "opus") opusEnc = make_unique<StreamingOggOpusEncoder>(outSr, 1);
            
            auto processChunk = [&]() {
                if (chunk.empty()) return;
                
                vector<int16_t> pcm = chunk;
                if (outSr != nativeSr) {
                    pcm = resample(std::span<const short>(chunk.data(), chunk.size()), nativeSr, outSr, 1);
                }
                
                if (req.format == "wav") {
                    uint32_t bytes = static_cast<uint32_t>(pcm.size() * sizeof(int16_t));
                    auto hdr = toLittleEndian4(bytes);
                    writeAll(*dst, reinterpret_cast<const char *>(hdr.data()), hdr.size());
                    writeAll(*dst, reinterpret_cast<const char *>(pcm.data()), bytes);
                } else if (req.format == "opus") {
                    auto ogg = opusEnc->encode(pcm);
                    if (!ogg.empty()) {
                        uint32_t bytes = static_cast<uint32_t>(ogg.size());
                        auto hdr = toLittleEndian4(bytes);
                        writeAll(*dst, reinterpret_cast<const char *>(hdr.data()), hdr.size());
                        writeAll(*dst, reinterpret_cast<const char *>(ogg.data()), ogg.size());
                    }
                }
                
                // Clear chunk after processing to prevent reuse
                chunk.clear();
            };

            gSynth->synthesizeStreamPcm(req.text, [&](std::span<const int16_t> view){
                chunk.assign(view.begin(), view.end());
                processChunk();
            });
            
            if (req.format == "opus") {
                auto tail = opusEnc->finish();
                if (!tail.empty()) {
                    uint32_t bytes = static_cast<uint32_t>(tail.size());
                    auto hdr = toLittleEndian4(bytes);
                    writeAll(*dst, reinterpret_cast<const char *>(hdr.data()), hdr.size());
                    writeAll(*dst, reinterpret_cast<const char *>(tail.data()), tail.size());
                }
            }
            return true;
        }

        // Non-streaming WAV/OPUS
        if (req.format == "wav") {
            auto wav = gSynth->synthesizeWav(req.text);
            writeAll(*dst, reinterpret_cast<const char*>(wav.data()), wav.size());
        } else if (req.format == "opus") {
            auto audio = gSynth->synthesizePcm(req.text);
            vector<int16_t> pcm = audio;
            if (outSr != nativeSr) {
                pcm = resample(std::span<const short>(audio.data(), audio.size()), nativeSr, outSr, 1);
            }
            auto ogg = encodeOgg(pcm, outSr, 1);
            writeAll(*dst, reinterpret_cast<const char *>(ogg.data()), ogg.size());
        }
        return true;
    } catch (const exception &e) {
        printError(e.what());
        return false;
    }
}

struct WorkItem { Request req; };

int main(int argc, char *argv[]) {
    spdlog::set_default_logger(spdlog::stderr_color_st("paroli"));

    RunConfig cfg;
    try {
        parseArgs(argc, argv, cfg);
        setupPiper(cfg);
    } catch (const exception &e) {
        printError(e.what());
        return 1;
    }

    atomic<size_t> nextId{0};
    mutex qMutex;
    condition_variable qCv;
    queue<WorkItem> q;

    // Signal handling for graceful shutdown
    auto handler = +[](int) { gShuttingDown.store(true); };
    signal(SIGINT, handler);
    signal(SIGTERM, handler);

    // Worker threads
    vector<thread> workers;
    workers.reserve(cfg.maxConcurrency);
    for (int i = 0; i < cfg.maxConcurrency; i++) {
        workers.emplace_back([&]() {
            while (true) {
                WorkItem item;
                {
                    unique_lock<mutex> lk(qMutex);
                    qCv.wait(lk, [&]() { return gShuttingDown.load() || !q.empty(); });
                    if (gShuttingDown.load() && q.empty()) return;
                    if (q.empty()) continue;
                    item = q.front();
                    q.pop();
                }
                synthesizeOne(cfg, item.req, cout);
            }
        });
    }

    // Read requests from stdin (one JSON per line)
    string line;
    while (!gShuttingDown.load() && getline(cin, line)) {
        if (line.empty()) continue;
        try {
            auto j = json::parse(line);
            Request r;
            if (!j.contains("text")) throw runtime_error("Missing text");
            r.text = j["text"].get<string>();
            r.format = j.value<string>("format", "wav");
            if (j.contains("sample_rate") && !j["sample_rate"].is_null()) r.sampleRate = j["sample_rate"].get<int>();
            r.id = nextId.fetch_add(1);

            if (gShuttingDown.load()) break;
            {
                unique_lock<mutex> lk(qMutex);
                q.push(WorkItem{r});
            }
            qCv.notify_one();
        } catch (const exception &e) {
            printError(e.what());
            continue;
        }
    }

    // Begin shutdown: reject new, finish in-flight
    gShuttingDown.store(true);
    qCv.notify_all();
    for (auto &t : workers) t.join();
    gSynth.reset();
    return 0;
}


