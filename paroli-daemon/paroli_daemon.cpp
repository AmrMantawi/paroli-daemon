#include "paroli_daemon.hpp"

#include <sstream>
#include <stdexcept>
#include <soxr.h>
#include <fstream>
#include <alsa/asoundlib.h>

#include "OggOpusEncoder.hpp"

using namespace std;

namespace {
static std::filesystem::path getExePath() {
    return std::filesystem::canonical("/proc/self/exe");
}
}

ParoliSynthesizer::ParoliSynthesizer(const InitOptions& opts) {
    try {
        // Load voice/models
        std::optional<piper::SpeakerId> speakerId = std::nullopt;
        loadVoice(cfg_, "", opts.encoderPath.string(), opts.decoderPath.string(),
                  opts.modelConfigPath.string(), voice_, speakerId, opts.accelerator);

        // Configure espeak
        if (voice_.phonemizeConfig.phonemeType == piper::eSpeakPhonemes) {
            if (opts.eSpeakDataPath) {
                cfg_.eSpeakDataPath = opts.eSpeakDataPath->string();
            } else {
                auto exePath = std::filesystem::canonical("/proc/self/exe");
                cfg_.eSpeakDataPath = std::filesystem::absolute(exePath.parent_path().append("espeak-ng-data")).string();
            }
        } else {
            cfg_.useESpeak = false;
        }

        piper::initialize(cfg_);
        initialized_ = true;
        lastError_.clear();
        

    } catch (const std::exception& e) {
        lastError_ = e.what();
        initialized_ = false;
        throw;
    }
}

ParoliSynthesizer::~ParoliSynthesizer() {
    piper::terminate(cfg_);
}

vector<uint8_t> ParoliSynthesizer::synthesizeWav(const std::string& text) {
    piper::SynthesisResult result;
    stringstream ss;
    piper::textToWavFile(cfg_, voice_, text, ss, result);
    auto s = ss.str();
    return vector<uint8_t>(s.begin(), s.end());
}

vector<int16_t> ParoliSynthesizer::synthesizePcm(const std::string& text) {
    vector<int16_t> audio;
    audio.clear(); // Ensure buffer starts clean
    piper::SynthesisResult result;
    auto cb = [&]() {};
    piper::textToAudio(cfg_, voice_, text, audio, result, cb);
    
    // If audio is empty, try using the streaming approach to collect all audio
    if (audio.empty()) {
        vector<int16_t> allAudio;
        allAudio.clear(); // Ensure buffer starts clean
        auto streamCb = [&]() {
            if (!audio.empty()) {
                allAudio.insert(allAudio.end(), audio.begin(), audio.end());
                audio.clear(); // Clear temp buffer after copying
            }
        };
        piper::textToAudio(cfg_, voice_, text, audio, result, streamCb);
        return allAudio;
    }
    
    return audio;
}

void ParoliSynthesizer::synthesizeStreamPcm(const std::string& text,
                                            const function<void(std::span<const int16_t>)>& onChunk) {
    vector<int16_t> chunk;
    piper::SynthesisResult result;
    auto cb = [&]() {
        if (!chunk.empty()) {
            onChunk(std::span<const int16_t>(chunk.data(), chunk.size()));
            chunk.clear(); // Clear chunk after processing to prevent reuse of stale data
        }
    };
    piper::textToAudio(cfg_, voice_, text, chunk, result, cb);
}

static vector<int16_t> soxrResample(span<const int16_t> input, size_t orig_sr, size_t out_sr, int channels) {
    soxr_io_spec_t io_spec = soxr_io_spec(SOXR_INT16_I, SOXR_INT16_I);
    soxr_quality_spec_t q_spec = soxr_quality_spec(SOXR_MQ, 0);
    soxr_error_t error;
    soxr_t soxr = soxr_create(orig_sr, out_sr, channels, &error, &io_spec, &q_spec, NULL);
    if (error != NULL) throw runtime_error("soxr_create failed");
    vector<int16_t> output(input.size() * out_sr / orig_sr);
    size_t idone = 0, odone = 0;
    error = soxr_process(soxr, input.data(), input.size(), &idone, output.data(), output.size(), &odone);
    if (error != NULL) { soxr_delete(soxr); throw runtime_error("soxr_process failed"); }
    output.resize(odone);
    soxr_delete(soxr);
    return output;
}

vector<uint8_t> ParoliSynthesizer::synthesizeOpus(const std::string& text, int outSampleRate) {
    auto audio = synthesizePcm(text);
    auto pcm = (outSampleRate == nativeSampleRate())
                   ? audio
                   : soxrResample(std::span<const int16_t>(audio.data(), audio.size()), nativeSampleRate(), outSampleRate, 1);
    auto ogg = encodeOgg(pcm, outSampleRate, 1);
    return ogg;
}

void ParoliSynthesizer::synthesizeStreamOpus(const std::string& text,
                                             const function<void(const uint8_t*, size_t)>& onChunk,
                                             int outSampleRate) {
    StreamingOggOpusEncoder enc(outSampleRate, 1);
    vector<int16_t> chunk;
    piper::SynthesisResult result;
    auto cb = [&]() {
        if (chunk.empty()) return;
        auto pcm = (outSampleRate == nativeSampleRate())
                       ? chunk
                       : soxrResample(std::span<const int16_t>(chunk.data(), chunk.size()), nativeSampleRate(), outSampleRate, 1);
        auto ogg = enc.encode(pcm);
        if (!ogg.empty()) onChunk(ogg.data(), ogg.size());
    };
    piper::textToAudio(cfg_, voice_, text, chunk, result, cb);
    auto tail = enc.finish();
    if (!tail.empty()) onChunk(tail.data(), tail.size());
}

std::vector<int16_t> ParoliSynthesizer::resample(std::span<const int16_t> input, size_t orig_sr, size_t out_sr, int channels) {
    return soxrResample(input, orig_sr, out_sr, channels);
}



// Direct speak method with volume control
bool ParoliSynthesizer::speak(const std::string& text) {
    if (!isModelLoaded()) {
        return false;
    }
    
    try {
        auto audio = synthesizePcm(text);
        if (audio.empty()) {
            return false;
        }
        
        // Apply volume with overflow protection
        if (volume_ != 1.0f) {
            for (auto& sample : audio) {
                int32_t scaled = static_cast<int32_t>(sample * volume_);
                // Clamp to int16_t range to prevent overflow
                sample = static_cast<int16_t>(std::clamp(scaled, 
                    static_cast<int32_t>(std::numeric_limits<int16_t>::min()),
                    static_cast<int32_t>(std::numeric_limits<int16_t>::max())));
            }
        }
        
        // Play audio using ALSA
        snd_pcm_t *handle;
        int err;
        
        if ((err = snd_pcm_open(&handle, "default", SND_PCM_STREAM_PLAYBACK, 0)) < 0) {
            lastError_ = "Cannot open audio device: " + std::string(snd_strerror(err));
            return false;
        }
        
        // Set hardware parameters
        snd_pcm_hw_params_t *params;
        snd_pcm_hw_params_alloca(&params);
        snd_pcm_hw_params_any(handle, params);
        snd_pcm_hw_params_set_access(handle, params, SND_PCM_ACCESS_RW_INTERLEAVED);
        snd_pcm_hw_params_set_format(handle, params, SND_PCM_FORMAT_S16_LE);
        snd_pcm_hw_params_set_channels(handle, params, 1);
        
        // Set sample rate more safely
        unsigned int rate = nativeSampleRate();
        if ((err = snd_pcm_hw_params_set_rate_near(handle, params, &rate, 0)) < 0) {
            lastError_ = "Cannot set sample rate: " + std::string(snd_strerror(err));
            snd_pcm_close(handle);
            return false;
        }
        
        // Set buffer size to prevent underruns
        snd_pcm_uframes_t buffer_size = audio.size();
        if ((err = snd_pcm_hw_params_set_buffer_size_near(handle, params, &buffer_size)) < 0) {
            lastError_ = "Cannot set buffer size: " + std::string(snd_strerror(err));
            snd_pcm_close(handle);
            return false;
        }
        
        if ((err = snd_pcm_hw_params(handle, params)) < 0) {
            lastError_ = "Cannot set parameters: " + std::string(snd_strerror(err));
            snd_pcm_close(handle);
            return false;
        }
        
        // Prepare the PCM device
        if ((err = snd_pcm_prepare(handle)) < 0) {
            lastError_ = "Cannot prepare audio device: " + std::string(snd_strerror(err));
            snd_pcm_close(handle);
            return false;
        }
        
        // Write audio data
        snd_pcm_sframes_t frames = snd_pcm_writei(handle, audio.data(), audio.size());
        if (frames < 0) {
            lastError_ = "Write error: " + std::string(snd_strerror(frames));
            snd_pcm_close(handle);
            return false;
        }
        
        // Wait for playback to complete
        snd_pcm_drain(handle);
        snd_pcm_close(handle);
        
        lastError_.clear();
        return true;
    } catch (const std::exception& e) {
        lastError_ = e.what();
        return false;
    }
}

// Speak to file method
bool ParoliSynthesizer::speakToFile(const std::string& text, const std::string& filename, const std::string& format) {
    if (!isModelLoaded()) {
        return false;
    }
    
    try {
        std::ofstream file(filename, std::ios::binary);
        if (!file.is_open()) {
            lastError_ = "Cannot open file: " + filename;
            return false;
        }
        
        if (format == "wav") {
            auto wav = synthesizeWav(text);
            file.write(reinterpret_cast<const char*>(wav.data()), wav.size());
        } else if (format == "pcm") {
            auto pcm = synthesizePcm(text);
            file.write(reinterpret_cast<const char*>(pcm.data()), pcm.size() * sizeof(int16_t));
        } else if (format == "opus") {
            auto opus = synthesizeOpus(text);
            file.write(reinterpret_cast<const char*>(opus.data()), opus.size());
        } else {
            lastError_ = "Unsupported format: " + format;
            return false;
        }
        
        file.close();
        lastError_.clear();
        return true;
    } catch (const std::exception& e) {
        lastError_ = e.what();
        return false;
    }
}

// Speak to buffer method
std::vector<int16_t> ParoliSynthesizer::speakToBuffer(const std::string& text, int sampleRate) {
    if (!isModelLoaded()) {
        return {};
    }
    
    try {
        auto audio = synthesizePcm(text);
        
        // Resample if needed
        if (sampleRate > 0 && sampleRate != nativeSampleRate()) {
            audio = resample(std::span<const int16_t>(audio.data(), audio.size()), 
                           nativeSampleRate(), sampleRate, 1);
        }
        
        lastError_.clear();
        return audio;
    } catch (const std::exception& e) {
        lastError_ = e.what();
        return {};
    }
}


