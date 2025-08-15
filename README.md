# Paroli

Streaming mode implementation of the Piper TTS system in C++ with (optional) RK3588 NPU acceleration support. Named after "speaking" in Esperanto.

## How to use

Before building, you will need to fulfill the following dependencies

* xtensor
* spdlog
* libfmt
* piper-phoenomize
* onnxruntime (1.14 or 1.15)
* A C++20 capable compiler

* libsoxr
* libopusenc
    * You'll need to build this from source if on Ubuntu 22.04. Package available starting on 23.04

(RKNN support)
* [rknnrt >= 1.6.0](https://github.com/rockchip-linux/rknn-toolkit2/tree/v1.6.0/rknpu2/runtime/Linux/librknn_api)

In which `piper-phoenomize` and `onnxruntime` binary (not the source! Unless you want to build yourselves!) likely needs to be downloaded and decompressed manually. Afterwards run CMake and point to the folders you recompressed them.

```bash
mkdir build
cd build
cmake .. -DORT_ROOT=/path/to/your/onnxruntime-linux-aarch64-1.14.1 -DPIPER_PHONEMIZE_ROOT=/path/to/your/piper-phonemize-2023-11-14 -DCMAKE_BUILD_TYPE=Release
make -j
# IMPORTANT! Copy espeak-ng-data or pass `--espeak_data` CLI flag
cp -r /path/to/your/piper-phonemize-2023-11-14/share/espeak-ng-data .
```

Afterwards run `paroli-cli` and type into the console to synthesize speech. Please refer to later sections for generating the models.

```plaintext
./paroli-cli --encoder /path/to/your/encoder.onnx --decoder /path/to/your/decoder.onnx -c /path/to/your/model.json
...
[2023-12-23 03:13:12.452] [paroli] [info] Wrote /home/marty/Documents/rkpiper/build/./1703301190238261389.wav
[2023-12-23 03:13:12.452] [paroli] [info] Real-time factor: 0.16085024956315996 (infer=2.201744556427002 sec, audio=13.688163757324219 sec)
```



## Obtaining models

To obtain the encoder and decoder models, you'll either need to download them or creating one from checkpoints. Checkpoints are the trained raw model piper generates. Please refer to [piper's TRAINING.md](https://github.com/rhasspy/piper/blob/master/TRAINING.md) for details. To convert checkpoints into ONNX file pairs, you'll need [mush42's piper fork and the streaming branch](https://github.com/mush42/piper/tree/streaming). Run

```bash
python3 -m piper_train.export_onnx_streaming /path/to/your/traning/lighting_logs/version_0/checkpoints/blablablas.ckpt /path/to/output/directory
```

### Downloading models

Some 100% legal models are provided on [HuggingFace](https://huggingface.co/marty1885/streaming-piper/tree/main).

## Accelerators

By default the models run on the CPU and could be power hungry and slow. If you'd like to use a GPU and, etc.. You can pass the `--accelerator cuda` flag in the CLI to enable it. For now the only supported accelerator is CUDA. But ROCm can be easily supported, just I don't have the hardware to test it. Feel free to contribute.

This is the list of supported accelerators:
* `cuda` - NVIDIA CUDA
* `tensorrt` - NVIDIA TensorRT


### Rockchip NPU (RK3588)

Additionally, on RK3588 based systems, the NPU support can be enabled by passing `-DUSE_RKNN=ON` into CMake and passing an RKNN model instead of ONNX as the decoder. Resulting in ~4.3x speedup compare to running on the RK3588 CPU cores. Note that the `accelerator` flag has no effect when the a RKNN model is used and only the decoder can run on the RK3588 NPU.

Rockchip does not provide any package of some sort to install the libraries and headers. This has to be done manually.

```bash
git clone https://github.com/rockchip-linux/rknn-toolkit2
cd rknn-toolkit2/rknpu2/runtime/Linux/librknn_api
sudo cp aarch64/librknnrt.so /usr/lib/
sudo cp include/* /usr/include/
```

Also, converting ONNX to RKNN has to be done on an x64 computer. As of writing this document, you likely want to install the version for Python 3.10 as this is the same version that works with upstream piper. rknn-toolkit2 version 1.6.0 is required.

```bash
# Install rknn-toolkit2
git clone https://github.com/rockchip-linux/rknn-toolkit2
cd rknn-toolkit2/tree/master/rknn-toolkit2/packages
pip install rknn_toolkit2-1.6.0+81f21f4d-cp310-cp310-linux_x86_64.whl

# Run the conversion script
python tools/decoder2rknn.py /path/to/model/decoder.onnx /path/to/model/decoder.rknn
```

To use RKNN for inference, simply pass the RKNN model in the CLI. An error will appear if RKNN is passed in but RKNN support not enabled during compiling.

```bash
./paroli-cli --encoder /path/to/your/encoder.rknn --decoder /path/to/your/decoder.onnx -c /path/to/your/model.json
#                                           ^^^^
#                                      The only change
```

## paroli-daemon (persistent CLI)

`paroli-daemon` is a resident, low-latency, non-network TTS process that keeps models hot in memory and communicates over stdin/stdout using line-delimited JSON (JSONL). It provides both streaming and non-streaming synthesis with support for direct audio playback.

### Build

`paroli-daemon` builds by default. You can disable via CMake option:

```bash
cmake .. -DBUILD_DAEMON=OFF
```

### Quickstart

Basic usage:
```bash
echo '{"text":"Hello world","format":"opus"}' | ./paroli-daemon \
  --encoder ENC.onnx --decoder DEC.onnx -c model.json --espeak_data ./espeak-ng-data > out.opus
```

Direct audio playback:
```bash
echo '{"text":"Hello world","format":"pcm"}' | ./paroli-daemon \
  --encoder ENC.onnx --decoder DEC.onnx -c model.json --play --volume 0.8
```

### Features

- **Multiple output formats**: PCM (raw), WAV, and Opus
- **Direct audio playback**: Play synthesized speech directly through speakers
- **Volume control**: Adjust playback volume (0.0 to 1.0)
- **Streaming support**: Low-latency chunked audio output
- **Concurrent processing**: Multiple synthesis jobs with configurable concurrency
- **Robust error handling**: Graceful error reporting and recovery
- **Graceful shutdown**: Handles SIGINT/SIGTERM properly

### Command Line Options

**Model Configuration:**
- `--encoder FILE` - Path to encoder model file
- `--decoder FILE` - Path to decoder model file  
- `-c, --config FILE` - Path to model config file
- `--espeak_data DIR` - Path to espeak-ng data directory
- `--accelerator STR` - Accelerator for ONNX (e.g., cuda, tensorrt)

**Output Control:**
- `--play` - Play audio directly to speakers (PCM format only)
- `--volume FLOAT` - Volume level for audio playback (0.0 to 1.0)
- `--output FILE` - Write output to file instead of stdout
- `--stream` - Enable length-prefixed chunked streaming

**Processing:**
- `--max-concurrency N` - Number of concurrent jobs (default 1)
- `--jsonl` - JSON-in/JSON-out only (no logs to stdout)

**Debugging:**
- `--debug` - Enable debug logging
- `-q, --quiet` - Suppress all logging

### Input Protocol

Send JSON objects via stdin, one per line:

```json
{"text": "Hello world", "format": "pcm", "sample_rate": 22050}
```

**Fields:**
- `text` (required) - Text to synthesize
- `format` (optional) - Output format: `"pcm"`, `"wav"`, or `"opus"` (default: `"wav"`)
- `sample_rate` (optional) - Target sample rate for container formats

### Output Protocol

**Non-streaming mode:**
- `pcm`: Raw 16-bit little-endian mono PCM samples
- `wav`: Complete WAV file bytes
- `opus`: Complete Opus file bytes

**Streaming mode (`--stream`):**
- Audio chunks prefixed with 4-byte little-endian length headers
- Each chunk contains audio data in the specified format

**Error output (stderr):**
```json
{"error": "Error message"}
```

### Examples

**Basic synthesis:**
```bash
echo '{"text":"Hello world","format":"wav"}' | ./paroli-daemon \
  --encoder model/encoder.onnx --decoder model/decoder.onnx -c model/config.json > output.wav
```

**Direct playback with volume control:**
```bash
echo '{"text":"Hello world","format":"pcm"}' | ./paroli-daemon \
  --encoder model/encoder.onnx --decoder model/decoder.onnx -c model/config.json \
  --play --volume 0.7
```

**Streaming synthesis:**
```bash
echo '{"text":"Hello world","format":"opus","sample_rate":24000}"' | ./paroli-daemon \
  --encoder model/encoder.onnx --decoder model/decoder.onnx -c model/config.json \
  --stream > output.opus
```

**Multiple concurrent requests:**
```bash
echo '{"text":"Request 1","format":"pcm"}' > requests.txt
echo '{"text":"Request 2","format":"pcm"}' >> requests.txt
cat requests.txt | ./paroli-daemon \
  --encoder model/encoder.onnx --decoder model/decoder.onnx -c model/config.json \
  --max-concurrency 2 --play
```

### Lifecycle and Robustness

- **Graceful shutdown**: Handles SIGINT/SIGTERM signals
- **Request management**: Rejects new requests during shutdown, finishes in-flight ones
- **Error recovery**: Continues processing after individual request failures
- **Resource cleanup**: Properly releases audio devices and model resources

### Security

The daemon communicates strictly over stdin/stdout and is not network exposed. No external network connections are made.

### Testing

Run the smoke test with your models:

```bash
export PAROLI_TEST_ENCODER=/path/enc.onnx
export PAROLI_TEST_DECODER=/path/dec.onnx
export PAROLI_TEST_CONFIG=/path/model.json
export PAROLI_TEST_ESPEAK=/path/espeak-ng-data
ctest -R paroli-daemon-smoke --output-on-failure
```