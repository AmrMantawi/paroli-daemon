#pragma once

#include <filesystem>
#include <functional>
#include <optional>
#include <span>
#include <string>
#include <vector>
#include <algorithm>

#include "piper/piper.hpp"

class ParoliSynthesizer {
public:
    struct InitOptions {
        std::filesystem::path encoderPath;
        std::filesystem::path decoderPath;
        std::filesystem::path modelConfigPath;
        std::optional<std::filesystem::path> eSpeakDataPath;
        std::string accelerator = ""; // e.g., "cuda", "tensorrt"
    };

    explicit ParoliSynthesizer(const InitOptions& opts);
    ~ParoliSynthesizer();

    const piper::Voice& voice() const { return voice_; }
    int nativeSampleRate() const { return voice_.synthesisConfig.sampleRate; }

    std::vector<uint8_t> synthesizeWav(const std::string& text);
    std::vector<int16_t> synthesizePcm(const std::string& text);
    std::vector<uint8_t> synthesizeOpus(const std::string& text, int outSampleRate = 24000);

    void synthesizeStreamPcm(const std::string& text,
                             const std::function<void(std::span<const int16_t>)>& onChunk);
    void synthesizeStreamOpus(const std::string& text,
                              const std::function<void(const uint8_t*, size_t)>& onChunk,
                              int outSampleRate = 24000);

    static std::vector<int16_t> resample(std::span<const int16_t> input, size_t orig_sr, size_t out_sr, int channels);

    // Status and error handling
    bool isInitialized() const { return initialized_; }
    std::string getLastError() const { return lastError_; }
    void clearError() { lastError_.clear(); }

    // Volume control
    void setVolume(float volume) { volume_ = std::clamp(volume, 0.0f, 1.0f); }
    float getVolume() const { return volume_; }

    // Model loading status
    bool isModelLoaded() const { return initialized_ && lastError_.empty(); }

    // Convenience methods for integration
    bool speak(const std::string& text);
    bool speakToFile(const std::string& text, const std::string& filename, const std::string& format = "wav");
    std::vector<int16_t> speakToBuffer(const std::string& text, int sampleRate = -1);

private:
    piper::PiperConfig cfg_;
    piper::Voice voice_;
    bool initialized_ = false;
    std::string lastError_;
    float volume_ = 1.0f;
};


