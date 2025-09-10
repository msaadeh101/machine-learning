# Tensorflow

This guide explains the life-cycle of **Tensorflow** - Repository layout and governance, build toolchain, packaging and release, tests and test strategy, CI/presubmit, how custom ops/kernels and multi-language bindings fit in, performance benchmarking, and practical dev tips.

## Overview

TensorFlow is a large multi-language, multi-platform codebase whose development is organized around the public repo plus companion repositories pulled in (Java, Serving, Lite, TF Addons).
- Built mostly with Bazel
- Validated via Github Actions and Google Kokoro CI for platform and hardware builds.
- Packaged as pip wheels/manyLinux/OS artifacts.
- Governed by CONTRIBUTING rules, mandatory tests and extensive presubmit process.

## Repository Overview

The repository can be thought of generally as frontend (Python, intuitive APIs, `tensorflow/python`) and backend (C++, executes computations, memory, devices, `tensorflow/core/`, `tensorflow/stream_executor/`).

- The build system used is **Bazel**, and build instructions are located in `BUILD` files throughout the repo.

|Directory|Purpose|
|---------|-------|
|`tensorflow/`| Python and C++ Source Code for TF|
|`third_party/`| External dependencies and wrappers (cuDNN, Eigen, XLA)|
|`tools/`| Developer utilities, codegen scripts, build helpers|
|`ci/`| Continuous Integration configs and scripts|
|`.github`|Github-specific workflows and issue templates|

### Key Files

- `WORKSPACE`: Bazel workspace definitions. Declares external dependencies.
- `BUILD`, `*BUILD`: Bazel build targets (libs, binaries, tests) for various components.
- `configure.py`: System detection (CUDA, MKL) and build config generator.
- `.bazelrc`, `.bazelversion`: Bazel tuning and version pinning.
- `README.md`, `CONTRIBUTING.md`, `LICENSE`: Docs and Governance.
- `requirements_lock_*.txt`: Python dependency locks for various versions.

### Directory Breakdown

A general breakdown of the major directories (non-exhaustive).

**tensorflow/**: Contains submodules for every major component.
- Low-level Operations and Kernels
    - `core/`: C++ implementation of TF's runtime, graph execution, and memory management.
    - `c/`: C API bindings for TF.
    - `lite/`: TF Lite, optimized for mobile and embedded devices.
    - `compiler/`: MLIR and XLA compiler infrastructure
- APIs and Python Wrappers
    - `python/`: Python API definitions, decorators, high-level abstractions.
    - `python/eager/`: Eager execution engine.
    - `python/keras/`: Keras integration.
    - `python/framework/`: Core python framework utilities (tensors, ops, sessions).
    - `python/ops/`: Python wrappers for ops (math, image, nn).
- Utilities and Extensions
    - `tools/`: includes `pip_package/` for building the pip wheel, and `ci_build/` for Docker and CI.
    - `examples/`: Sample models and training scripts (**less emphasized**).

**third_party/**: Isolates external dependencies and custom build rules. Ensures reproducability.
- Select Third party dirs
    - `xla/`: XLA is a Just-in-Time compiler for TensorFlow graphs.
    - `boringssl/`: Contains BoringSSL, google developed for of OpenSSL.
    - `googleapis/`: contains client library and proto buffer definitions for cloud integration.

**ci/**: Contains the scripts and configurations for CI and build/test automation.
- Subfolders
    - `devinfra/`: managed by TF DevInfra team but not officially part of build/test/release.
    - `official/`: Offical build/test scripts.

## Development Processes

The CONTRIBUTING.md contains the process to contribute code to the TensorFlow project.

1. **New PR**
- PRs are inspected and labels added such as size:, comp:
2. **Valid?**
- If PR passes quality checks, its assigned a reviewer. If not, sent back with changes.
3. **Review**
- Cycle repeats, at end, if looks good it gets approved.
4. **Approved**
- Once aproved, label of `koroko:force-run` gets applied, initiates CI/CD.
- If all tests pass, code is brought in using job called `copybara`
5. **Copy to Google Internal codevase and run internal CI**
- Make sure it integrates with rest of system.

## Bazel Build System

**Bazel** is Google's scalable, multi-language build system that covers any size codebase. Bazel was chosen for:
- Hermetic builds: Reproducible builds with explicit dependencies.
- Incremental Compilation: Only rebuilds what is changed.
- Cross platform: Unified builds across Linux, MacOS, Windows.
- Language agnostic: Handles C++, Python, CUDA, proto buffers.
- Distributed builds: Can leverage remote execution.

### BUILD Files Architecture

Every dir with compatible code contains a BUILD file (or BUILD.bazel).

- Example snippet that defines a C++ library target `tf_cc_library`

```python
# Example BUILD file structure
load("//tensorflow:tensorflow.bzl", "tf_cc_library")

tf_cc_library(
    name = "framework",
    srcs = ["framework.cc"],
    hdrs = ["framework.h"],
    deps = [
        "//tensorflow/core:lib",
        "@com_google_absl//absl/strings",
    ],
    visibility = ["//tensorflow:internal"],
)
```
|Field	|Description|
|-------|------------|
|`name`	|The name of the Bazel target (`//tensorflow:framework`)|
|`srcs`	|Source files for the library (`framework.cc`)|
|`hdrs`	|Public headers (`framework.h`) exposed to dependents|
|`deps`	|Dependencies needed to compile this library:<br>• `//tensorflow/core:lib`: internal TensorFlow core library<br>• `@com_google_absl//absl/strings`: Abseil string utilities|
|`visibility`	|Restricts who can depend on this target. Only targets in `//tensorflow:internal` can use it|

### WORKSPACE Internals

The WORKSPACE file defines:
1. External repos via http_archive, git_repository, etc.
1. Repository rules for dynamic dependency resolution.
1. Toolchain registration for different platforms.
1. Custom repository rules like @local_config_cuda (populated by running custom repo rules invoked in WORKSPACE file).

### Build Configuration System

TensorFlow now uses **LLVM/Clang 17** as the default compiler starting with TensorFlow 2.13. The configuration process:

1. Run `./configure` to detect system capabilities.
1. Generates platform-specific `.bazelrc` settings.
1. Detects `CUDA`, `Python` versions, optimization flags.
1. Creates symbolic links for external dependencies.

## Packaging and Distribution

TensorFlow's **Python wheel generation** centers around the `build_pip_package` script, which orchestrates the process of creating distributable packages from the repo.

```bash
# Simplified build command
bazel build //tensorflow/tools/pip_package:build_pip_package
./bazel-bin/tensorflow/tools/pip_package/build_pip_package /tmp/tensorflow_pkg
```

**Build pipeline stages**:

1. **Bazel Build Phase**: Compiles all C++/CUDA sources, Python extensions, and proto definitions.
2. **Wheel Assembly**: Collects built artifacts, Python source files, and metadata into wheel structure.
3. **Native Library Bundling**: Includes compiled shared libraries (`.so` files) with proper `RPATH` configuration.
4. **Metadata Generation**: Creates `WHEEL`, `METADATA`, and dependency specifications.

**manylinux** Compatibility is maintained by leveraging docker-based build environments.
- glibc version constraints: Must target oldest supported glibc.
- Symbol versioning: Use compatible symbol versioning.
- Library building: Third-party libs are statically linked or bundled.
- Audit tools: `auditwheel` validates and repairs wheels post-build.

```dockerfile
# Example build container setup
FROM tensorflow/build:latest-python3.9
RUN /tensorflow/tools/ci_build/builds/build_pip_packages.sh
```

### Nightly Build Infrastructure

The `tf-nightly` distribution performs continuous validation, feature previews, and build system testing/validation.

### Java Ecosystem

- Native libraries (JNI):

```cpp
// Generated JNI bindings
JNIEXPORT jlong JNICALL Java_org_tensorflow_TensorFlow_allocate
  (JNIEnv *, jclass, jint, jlongArray);
```
1. C++ API Compilation
1. JNI Wrapper generation
1. JAR assembly
1. Maven deployment

```xml
<dependency>
    <groupId>org.tensorflow</groupId>
    <artifactId>tensorflow</artifactId>
    <version>2.x.x</version>
</dependency>
```

### C API Distribution

The libtensorflow C API enables integration with non-Python environments.

```txt
libtensorflow/
├── lib/
│   ├── libtensorflow.so       # Core library
│   └── libtensorflow_framework.so
├── include/
│   └── tensorflow/c/
│       ├── c_api.h            # Primary C interface
│       └── c_api_experimental.h
└── LICENSE
```

### Mobile Platforms

Android Archive AAR Builds require specialized toolchain configuration.

```python
# android_configure.bzl
android_ndk_repository(
    name = "androidndk",
    path = "/opt/android-ndk-r21e",
)
```

### iOS Framework

iOS builds face unique constraints around dynamic linking and App Store policies.

```objc
// TensorFlowLiteC framework integration
#import <TensorFlowLiteC/TensorFlowLiteC.h>
TfLiteInterpreter* interpreter = TfLiteInterpreterCreate(model, options);
```

### GPU Build Complexity

CUDA/cuDNN Integration challenges: GPU-enabled TensorFlow wheels face real packaging constraints.

```bash
# Required CUDA runtime libraries
libcudart.so.11.0
libcublas.so.11
libcurand.so.10
libcusolver.so.11
libcusparse.so.11
libcudnn.so.8
```

### Ahead of Time Compilation

**AOT**, provided by TensorFlow via `XLA`. Build process includes graph freezing, XLA compilation, Static linking and Header generation.

```cpp
// Generated AOT interface
#include "my_model.h"
MyModel model;
model.set_arg0_data(input_data);
model.Run();
float* output = model.result0_data();
```

### Trade-offs and Engineering Decisions

**Distribution Size vs. Compatibility**
| Approach | Wheel Size | Compatibility | Maintenance | 
|-----------| ---------| -------------| -------------| 
| System deps| ~500MB| Fragile| Low| 
| Bundled runtime| ~2GB| Robust| High| 
| Multiple variants| ~500MB each|  Targeted| Very High| 

Build Options:
- **Full matrix**: `~20 platforms × 4 Python versions × 2 GPU variants = 160 combinations`
- **Selective building**: Strategic subset based on usage analytics
- **Incremental builds**: Sophisticated caching to reduce CI times from hours to minutes

The packaging and distribution system represents one of TensorFlow's most complex engineering challenges, requiring careful balance between user convenience, build system maintainability, and distribution costs.

## Testing Strategy

**Test Types**
- **Unit Tests**: small, fast, cover individual modules.
- **Integration Tests**: Larger texts that exercise multiple components (training loop + saved model + serving).
- **Hardware Tests**: CPU/GPU/TPU tests that validate kernels on devices - slower, running in specialized CI. Cuda Runtime validation, memory management, multi-gpu coordination, mixed precision (FP16/BF16 numerical accuracy verification), TPU sharding behavior.
- **Perf and Benchmark Tests**: Measure throughput/latency, regression checks for performance drift.
- **End-to-end tests**: build wheel, install, run example training, export SavedModel and load in TF-Serving.

**Note**: Do not import tf for tiny unit test, utilize bazel test to run on individual test targets and only depend on submodules.