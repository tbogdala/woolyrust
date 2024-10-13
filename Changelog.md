# v0.2.0

* Added `ManagedGptParams` which encapsulates the autogenerated `wooly_gpt_params` structure
  so that lifetimes for buffers and pointers made to convert Rust strings and data to C
  compatible CStrings, etc. Objects like `CStrings` need to stay alive for their pointers
  to be valid and this seemed like the cleanest way to make it happen.

* Changed `predict_text()` signature to use the above `ManagedGptParams` instead and removed
  the need to pass the other data separately.