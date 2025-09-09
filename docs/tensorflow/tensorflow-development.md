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

## Processes

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