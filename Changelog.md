# Change log

## v0.4.0

* Sync'd changes with upstream woolycore.

* Added the `dataset_generator.rs` example as a command-line tool to generate datasets.


## v0.3.0

* Sync'd changes with upstream woolycore.

* Documentation updates. Generating docs with `cargo` now shows better examples of how to use the library.

* Added `processAdditionalPrompt` to add more prompt tokens before generating text.

## v0.2.0

* Added `ManagedGptParams` which encapsulates the autogenerated `wooly_gpt_params` structure
  so that lifetimes for buffers and pointers made to convert Rust strings and data to C
  compatible CStrings, etc. Objects like `CStrings` need to stay alive for their pointers
  to be valid and this seemed like the cleanest way to make it happen.

* Changed `predict_text()` signature to use the above `ManagedGptParams` instead and removed
  the need to pass the other data separately.