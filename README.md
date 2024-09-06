# woolyrust

A Rust wrapper around the [llama.cpp library](https://github.com/ggerganov/llama.cpp), aiming for a high-level
API that provides enough functionality to be versatile and useful. The basic, higher-level C functions that this
library builds upon are are provided by the [woolycore](https://github.com/tbogdala/woolycore) library.

At present, it is in development and the API is unstable, though no breaking changes are envisioned.  

Supported Operating Systems: Windows, MacOS, Linux


## License

MIT licensed, like the core upstream `llama.cpp` it wraps. See `LICENSE` for details.


## Features

* Simple high-level Rust interface to use for text generation (`Llama`).
* Basic samplers of llama.cpp, including: temp, top-k, top-p, min-p, tail free sampling, locally typical sampling, mirostat.
* Support for llama.cpp's BNF-like grammar rules for sampling.
* Ability to cache the processed prompt data in memory so that it can be reused to speed up regeneration using the exact same prompt.
* Tokenize text or just get the number of tokens for a given text string.
* Generate embeddings using models such as `nomic-ai/nomic-embed-text-v1.5-GGUF` on HuggingFace in a batched process.


## Build notes

The upstream `llama.cpp` code is built through the `build.rs` build script with the `cmake` crate as it builds
the API bindings code - all automatically, so a simple `cargo build` suffices. 

```bash
cargo build --release
```

This should automatically include Metal support and embed the shaders if the library is being built on MacOS.

For CUDA systems, a feature called `cuda` has been added, which needs to be supplied for CUDA acceleration. This
will greatly increase the compile time of the project. An example build command to enable CUDA would be:

```bash
cargo build --release features cuda
```

NOTE: Upstream `llamacpp` makes heavy use of cmake build files and `woolycore` has adopted them to avoid
duplication of effort and to greatly ease maintenance. This unfortunately means cmake is a **required 
dependency** to build the library.


## Git updates

This project uses submodules for upstream projects so make sure to update with appropriate parameters:

```bash
git pull --recurse-submodules
```


## Tests

The unit tests require an environment variable (`WOOLY_TEST_MODEL_FILE`) to be set with the 
path to the GGUF file for the model to use during testing. Passing `--nocapture` as a parameter to
cargo allows for the predicted text to show up on stdout for your viewing pleasure.

```bash
export WOOLY_TEST_MODEL_FILE=models/example-llama-3-8b.gguf
cargo test --release -- --nocapture --test-threads 1
```

On windows, setting the environment may look something like this:

```bash
set WOOLY_TEST_MODEL_FILE=models/example-llama-3-8b.gguf
```

Don't forget to add `--features cuda` for CUDA acceleration on Windows/Linux platforms if that
is desired.

To run the embeddings unit test, you have to set the environment variable (`WOOLY_TEST_EMB_MODEL_FILE`)
as well. This should be set to a GGUF file for the embedding model to use for those tests. At present,
the tests are designed for `nomic-ai/nomic-embed-text-v1.5-GGUF`.


## Final Notes

* Was unsuccessful getting `woolycore` to build with cmake and get statically linked to upstream `llama.cpp` code...