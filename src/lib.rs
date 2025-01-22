#![allow(non_upper_case_globals)]
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]

//! # WoolyRust: A Rust Library for LLM Text Inference
//!
//! `woolyrust` is a Rust library designed to facilitate the loading and inference of
//! LLMs using the llama.cpp library. It provides a high-level interface for interacting
//! with language models, allowing users to perform tasks such as text prediction,
//! tokenization, and detokenization. This high level interface simplifies the process of
//! integrating language models into Rust applications.
//!
//! ## Features
//!
//! - **Model Loading**: Load LLMs from GGUF files.
//! - **Text Prediction**: Generate text based on prompts using the loaded model.
//! - **Tokenization**: Convert text into tokens and vice versa.
//! - **Embedding Generation**: Create embeddings for tokenized prompts.
//!
//! ## Example Usage
//!
//! Below is a simple example demonstrating how to use `woolyrust` to predict text.
//! This example includes setting up the model, defining parameters, and generating a response to a prompt.
//!
//! ```rust
//! use woolyrust::{Llama, ManagedGptParams};
//!
//! fn main() {
//!     // Define model and context parameters
//!     let mut model_params = woolyrust::get_default_model_params();
//!     let mut context_params = woolyrust::get_default_context_params();
//!     context_params.n_ctx = 2048;
//!
//!     // Load the model from a file
//!     let model_filepath = get_test_model_path(); // Replace with your model file path
//!     let mut llama = Llama::new();
//!     llama.load_model(model_filepath.as_str(), model_params, context_params, false);
//!
//!     // Define prediction parameters
//!     let mut params = ManagedGptParams::defaults();
//!     params.params.seed = 42;
//!     params.params.n_threads = 4;
//!     params.params.n_predict = 100;
//!     params.params.temp = 0.7;
//!     params.params.top_k = 50;
//!     params.params.top_p = 1.0;
//!     params.params.min_p = 0.05;
//!     params.params.penalty_repeat = 1.1;
//!     params.params.penalty_last_n = 64;
//!     params.params.ignore_eos = false;
//!     params.params.flash_attn = true;
//!     params.params.n_batch = 8;
//!     params.params.prompt_cache_all = true;
//!
//!     // Set the prompt and antiprompts
//!     let antiprompts = vec!["<|end|>"];
//!     let prompt = "<|user|>\nExplain the concept of artificial intelligence in a few sentences.<|end|>\n<|assistant|>\n";
//!     params.set_prompt(prompt);
//!     params.set_antiprompts(&antiprompts);
//!
//!     // Predict text
//!     match llama.predict_text(&mut params, None) {
//!         Ok((_, prediction)) => {
//!             println!("{}", prediction);
//!         }
//!         Err(e) => {
//!             eprintln!("Prediction failed: {}", e);
//!         }
//!     }
//! }
//!
//! // helper function to get the model path from an environment variable (so unit testing won't fail)
//! pub fn get_test_model_path() -> String {
//!     std::env::var("WOOLY_TEST_MODEL_FILE")
//!         .expect("Set WOOLY_TEST_MODEL_FILE environment variable to the gguf file to use for testing")
//! }
//! ```
//!
//! ## Testing
//!
//! For a more comprehensive example, refer to the `basic_test.rs` and `step_prediction_tests.rs` files
//! in the `tests` directory. These test demonstrates loading a model, setting parameters,
//! and generating text with prompt caching and grammar rules.
//!
//! ```bash
//! cargo test --release -- --nocapture --test-threads 1
//! ```

use std::{
    ffi::{CStr, CString},
    ptr::{null, null_mut},
};
include!(concat!(env!("OUT_DIR"), "/bindings.rs"));

/// Represents a token in the context of the language model.
pub type Token = i32;

/// A list of tokens used for various operations within the language model.
pub type TokenList = Vec<Token>;

/// Represents an embedding, which is a vector of floating-point numbers representing
/// the semantic meaning of a token or a sequence of tokens.
pub type Embedding = Vec<f32>;

/// Represents the different pooling types available for embeddings.
#[derive(Debug)]
pub enum LlamaPoolingType {
    Unspecified = -1,
    None = 0,
    Mean = 1,
    Cls = 2,
    Last = 3,
}

/// Represents the different normalization methods available for embeddings.
#[derive(Debug)]
pub enum EmbeddingNormalization {
    None = -1,
    MaxAbsoluteInt16 = 0,
    Taxicab = 1,
    Euclidean = 2,
    PNorm = 3,
}

/// This represents one turn in the prompt template. A whole prompt for the LLM can be
/// generated from a vector of these.
pub struct ChatMessage {
    pub role: String, // Roles can be 'user', 'assistant' or 'system'
    pub content: String,
}

impl ChatMessage {
    pub fn new(role: String, content: String) -> Self {
        Self { role, content }
    }
}

/// This is a wrapper struct to encapsulate the raw C pointer used in this FFI library.
/// It adds some convenience functions for converting to the C pointer needed as well
/// as making sure to free itself when it's dropped from scope.
#[derive(Debug)]
pub struct FrozenState {
    pub is_alive: bool,
    wrapped_ptr: *mut wooly_prompt_cache_t,
}
impl Into<*mut wooly_prompt_cache_t> for &mut FrozenState {
    fn into(self) -> *mut wooly_prompt_cache_t {
        self.wrapped_ptr
    }
}
impl Drop for FrozenState {
    fn drop(&mut self) {
        unsafe {
            wooly_free_prompt_cache(self.wrapped_ptr);
            self.wrapped_ptr = null_mut();
        }
        self.is_alive = false;
    }
}

/// This is a wrapper struct to encapsulate the raw C pointer used in this FFI library.
/// It adds some convenience functions for converting to the C pointer needed as well
/// as making sure to free itself when it's dropped from scope.
#[derive(Debug)]
pub struct GptSampler {
    pub is_alive: bool,
    wrapped_ptr: *mut wooly_sampler_t,
}
impl Into<*mut wooly_sampler_t> for &mut GptSampler {
    fn into(self) -> *mut wooly_sampler_t {
        self.wrapped_ptr
    }
}
impl Drop for GptSampler {
    fn drop(&mut self) {
        unsafe {
            wooly_free_sampler(self.wrapped_ptr);
            self.wrapped_ptr = null_mut();
        }
        self.is_alive = false;
    }
}

/// Parameters for managing the LLM's behavior during text generation and prompt processing.
///
/// The `ManagedGptParams` struct encapsulates various settings that control the model's operation,
/// including prompt handling, token generation, and other configuration options. This struct is used
/// throughout the text generation process to ensure consistent behavior and to allow fine-grained control
/// over the model's output.
///
/// # Notes:
/// Rust's lifetime management makes it necessary to make sure the CStrings
/// and native pointers stay alive as long as the parameters do, so this
/// structure was made to contain them as hidden members. The structure's
/// functions should be used to set prompt, antiprompt and grammar strings.
pub struct ManagedGptParams {
    pub params: wooly_gpt_params,
    native_prompt: Option<CString>,
    native_grammar: Option<CString>,
    native_antis: Option<Vec<CString>>,
    native_antips: Option<Vec<*const i8>>,
    native_breakers: Option<Vec<CString>>,
    native_breakersp: Option<Vec<*const i8>>,
}
impl ManagedGptParams {
    pub fn defaults() -> Self {
        Self {
            params: new_text_gen_params(),
            native_prompt: None,
            native_grammar: None,
            native_antis: None,
            native_antips: None,
            native_breakers: None,
            native_breakersp: None,
        }
    }

    pub fn set_prompt(&mut self, prompt: &str) {
        // allocate the prompt
        let native = CString::new(prompt).expect("Invalid prompt string");
        self.params.prompt = native.as_ptr();
        self.native_prompt = Some(native);
    }

    pub fn set_grammar(&mut self, grammar: &str) {
        // allocate the grammar string
        let native = CString::new(grammar).expect("Invalid grammar string");
        self.params.grammar = native.as_ptr();
        self.native_grammar = Some(native);
    }

    pub fn set_antiprompts(&mut self, antiprompts: &Vec<&str>) {
        // allocate the antipompts
        // we store the CStrings for the antiprompts as well as building a vector of pointers to send to the library
        let mut native_anti_strings: Vec<CString>;
        let mut native_anti_pointers: Vec<*const i8>;
        let count = antiprompts.len();
        native_anti_strings = Vec::with_capacity(count);
        native_anti_pointers = Vec::with_capacity(count);
        for antiprompt in antiprompts {
            let native_anti = CString::new(*antiprompt).expect("Invalid antiprompt string");
            native_anti_pointers.push(native_anti.as_ptr());
            native_anti_strings.push(native_anti);
        }
        self.params.antiprompts = native_anti_pointers.as_mut_ptr();
        self.params.antiprompt_count = count as i32;
        self.native_antis = Some(native_anti_strings);
        self.native_antips = Some(native_anti_pointers);
    }

    pub fn set_dry_sequence_breakers(&mut self, sequence_breakers: &Vec<&str>) {
        // allocate the sequence breaker strings
        // we store the CStrings for them as well as building a vector of pointers to send to the library
        let mut native_strings: Vec<CString>;
        let mut native_pointers: Vec<*const i8>;
        let count = sequence_breakers.len();
        native_strings = Vec::with_capacity(count);
        native_pointers = Vec::with_capacity(count);
        for breaker in sequence_breakers {
            let native_breaker = CString::new(*breaker).expect("Invalid antiprompt string");
            native_pointers.push(native_breaker.as_ptr());
            native_strings.push(native_breaker);
        }
        self.params.dry_sequence_breakers = native_pointers.as_mut_ptr();
        self.params.dry_sequence_breakers_count = count as i32;
        self.native_breakers = Some(native_strings);
        self.native_breakersp = Some(native_pointers);
    }
}

/// Returns the default model parameters for loading models.
///
/// This function calls the underlying C function `wooly_get_default_llama_model_params`
/// to retrieve the default parameters used to load the LLaMA model.
pub fn get_default_model_params() -> wooly_llama_model_params {
    unsafe {
        return wooly_get_default_llama_model_params();
    }
}

/// Returns the default context parameters for the LLaMA model.
///
/// This function retrieves the default settings for the context parameters used by the LLaMA model.
/// The returned `wooly_llama_context_params` struct can be modified as needed before being used to
/// load the model and generate text.
pub fn get_default_context_params() -> wooly_llama_context_params {
    unsafe {
        return wooly_get_default_llama_context_params();
    }
}

/// Creates a new instance of `wooly_gpt_params` with default values.
///
/// This function is used to initialize the parameters needed for text generation
/// with the GPT model. It returns a `wooly_gpt_params` struct that can be further
/// customized using the `ManagedGptParams` struct methods.
pub fn new_text_gen_params() -> wooly_gpt_params {
    unsafe {
        return wooly_new_gpt_params();
    }
}

use std::f32;

/// Computes the cosine similarity between two embeddings.
///
/// # Parameters
/// - `embd1` - A slice of f32 representing the first embedding.
/// - `embd2` - A slice of f32 representing the second embedding.
///
/// # Returns
/// - The cosine similarity between the two embeddings as an f32. If one or both vectors are zero vectors,
/// the function returns 0.0 unless both are zero vectors, in which case it returns 1.0.
pub fn embeddings_similarity_cos(embd1: &[f32], embd2: &[f32]) -> f32 {
    let n = embd1.len();

    let mut sum = 0.0;
    let mut sum1 = 0.0;
    let mut sum2 = 0.0;

    for i in 0..n {
        sum += embd1[i] * embd2[i];
        sum1 += embd1[i] * embd1[i];
        sum2 += embd2[i] * embd2[i];
    }

    // Handle the case where one or both vectors are zero vectors
    if sum1 == 0.0 || sum2 == 0.0 {
        if sum1 == 0.0 && sum2 == 0.0 {
            return 1.0; // two zero vectors are similar
        }
        return 0.0;
    }

    sum as f32 / ((sum1.sqrt() * sum2.sqrt()) as f32)
}

/// Represents a LLM (Large Language Model) instance, encapsulating the model, context, and related resources
/// needed to do text prediction and embedding generation.
///
/// The `Llama` struct is designed to manage the lifecycle of a LLM, including loading, processing prompts,
/// generating text, and handling embeddings. It provides methods to load and free the model, process prompts,
/// generate text based on prompts, and manage the model's state through freezing and defrosting.
///
/// Once a model is loaded, `predict_text()` can be called to generate text based on the given options. For
/// finer control over text generation, `process_prompt()` can be used to ingest prompt text and then
/// `sample_next_token()` and then `process_next_token()` can be called iteratively to generate text token by token.
///
/// # Fields
/// - `ctx`: A raw pointer to the context of the LLaMA model.
/// - `model`: A raw pointer to the LLaMA model.
/// - `prompt_cache`: A raw pointer to the prompt cache, used to store intermediate results for efficient re-use.
/// - `loaded_context_len`: The length of the loaded context, representing the number of tokens in the context.
/// - `loaded_context_params`: An optional struct containing parameters used for creating the context. This is set when a model is successfully loaded.
/// - `loaded_model_params`: An optional struct containing parameters used for loading the model. This is set when a model is successfully loaded.
#[derive(Debug, Clone)]
pub struct Llama {
    ctx: *mut wooly_llama_context_t,
    model: *mut wooly_llama_model_t,
    prompt_cache: *mut wooly_prompt_cache_t,
    loaded_context_len: u32,
    loaded_context_params: Option<wooly_llama_context_params>,
    loaded_model_params: Option<wooly_llama_model_params>,
}

impl Drop for Llama {
    fn drop(&mut self) {
        self.free_model();
    }
}

impl Llama {
    pub fn new() -> Self {
        return Self {
            ctx: null_mut(),
            model: null_mut(),
            prompt_cache: null_mut(),
            loaded_context_len: 0,
            loaded_context_params: None,
            loaded_model_params: None,
        };
    }

    /// Loads a model from the specified file path using the provided model and context parameters.
    ///
    /// # Parameters
    /// - `model_file` - A string slice representing the file path to the model.
    /// - `model_params` - Parameters used for loading the model.
    /// - `context_params` - Parameters used for creating a new context.
    /// - `silence` - A boolean flag that, when set to `true`, suppresses all output from upstream `llama.cpp`.
    ///
    /// # Returns
    /// - `true` if the model is successfully loaded and the context is created and `false` if the loading process fails.
    pub fn load_model(
        &mut self,
        model_file: &str,
        model_params: wooly_llama_model_params,
        context_params: wooly_llama_context_params,
        silence: bool,
    ) -> bool {
        // free the model and prompt cache if there's one set already
        self.free_model();

        // attempt to load the model
        let result: wooly_load_model_result;
        let native_model_path = CString::new(model_file).unwrap();
        unsafe {
            result = wooly_load_model(
                native_model_path.as_ptr(),
                model_params,
                context_params,
                silence,
            );
        }

        // return false if we don't get the right set of pointers back
        if result.ctx.is_null() || result.model.is_null() {
            return false;
        }

        // set our internal state and return
        self.model = result.model;
        self.ctx = result.ctx;
        self.loaded_context_len = result.context_length;
        self.loaded_context_params = Some(context_params);
        self.loaded_model_params = Some(model_params);
        return true;
    }

    /// Free the model and prompt cache if they are set, and reset the internal state.
    pub fn free_model(&mut self) {
        unsafe {
            wooly_free_model(self.ctx, self.model);
        }
        if !self.prompt_cache.is_null() {
            unsafe {
                wooly_free_prompt_cache(self.prompt_cache);
            }
        }
        self.ctx = null_mut();
        self.model = null_mut();
        self.prompt_cache = null_mut();
        self.loaded_context_len = 0;
        self.loaded_context_params = None;
        self.loaded_model_params = None;
    }

    /// Checks if both the model and the context have been successfully loaded.
    pub fn is_loaded(&self) -> bool {
        return !self.ctx.is_null() && !self.model.is_null();
    }

    /// Constructs a prompt from a list of chat messages and applies a chat template.
    /// The constructed prompt is returned and the total number of characters procssed.
    pub fn makePromptFromMessages(
        &self,
        messages: Vec<ChatMessage>,
        template_override: Option<String>,
    ) -> (String, i64) {
        // handle empty lists
        if messages.is_empty() {
            return ("".to_string(), 0);
        }

        // turn the template override into a native string
        let tmpl_override_native = if let Some(ovr) = template_override {
            Some(CString::new(ovr).expect("Invalid template override string"))
        } else {
            None
        };

        // build our data structures for the message log conversion
        let message_count = messages.len();
        let mut message_log: Vec<wooly_chat_message> = Vec::with_capacity((message_count) as usize);

        let mut native_strings = vec![];
        for i in 0..message_count {
            let role_string =
                CString::new(messages[i].role.clone()).expect("Invalid role string in ChatMessage");
            let content_string = CString::new(messages[i].content.clone())
                .expect("Invalid content string in ChatMessage");

            let mlog = wooly_chat_message {
                role: role_string.as_ptr(),
                content: content_string.as_ptr(),
            };
            message_log.push(mlog);

            native_strings.push(role_string);
            native_strings.push(content_string);
        }

        // try and build the prompt based on the default chat template for the model, or the override
        // if specified.
        let output_text_size = (self.loaded_context_len * 4 * 10) as i64;
        let mut output_text = Vec::with_capacity((output_text_size) as usize);

        let maybe_template_override = match tmpl_override_native {
            Some(ref cstr) => cstr.as_ptr(),
            None => null(),
        };
        unsafe {
            let num_processed = wooly_apply_chat_template(
                self.model,
                maybe_template_override,
                message_log.as_ptr(),
                message_count as i64,
                output_text.as_mut_ptr(),
                output_text_size,
            );

            let c_str_result: &CStr = CStr::from_ptr(output_text.as_mut_ptr());
            let result_string: String = c_str_result.to_str().unwrap().to_owned();
            (result_string, num_processed)
        }
    }

    /// Processes the given prompt using the parameters provided within the context of the loaded model.
    ///
    /// This function performs the prefill operation, which involves preparing the model to generate a response
    /// based on the input prompt. Upon completion of the prefill, it returns the number of tokens processed
    /// and a `GptSampler` instance that should be used to continue generating text.
    ///
    /// This function does not do any token generation - it only prepares the model's context for generation.
    ///
    /// # Parameters
    /// - `params`: A mutable reference to `ManagedGptParams` containing the parameters for processing the prompt.
    ///
    /// # Returns
    /// - A tuple containing:
    ///   - The number of tokens processed as an `i32`.
    ///   - A `GptSampler` instance to use in the subsequent generation steps.
    pub fn process_prompt(&mut self, params: &mut ManagedGptParams) -> (i32, GptSampler) {
        let results: wooly_process_prompt_results;
        unsafe {
            results = wooly_process_prompt(params.params, self.ctx, self.model);
        }
        (
            results.result,
            GptSampler {
                is_alive: true,
                wrapped_ptr: results.gpt_sampler,
            },
        )
    }

    /// Takes a sampler that was returned from a previous `process_prompt()` call
    /// and applies additional prompt text to it, updating the sampler's state.
    ///
    /// # Parameters
    /// - `sampler` - A mutable reference to the `GptSampler` that was previously returned from a call to `process_prompt()`.
    /// - `additional_prompt` - A string slice containing the additional prompt text to be processed.
    ///
    /// # Returns
    /// - the number of tokens added to the sampler's state, or a negative number if an error occurred.
    pub fn process_additional_prompt(
        &mut self,
        sampler: &mut GptSampler,
        additional_prompt: &str,
    ) -> i32 {
        let native_text =
            CString::new(additional_prompt).expect("Invalid additional prompt string");

        unsafe {
            wooly_process_additional_prompt(
                self.ctx,
                self.model,
                sampler.wrapped_ptr,
                native_text.as_ptr(),
            )
        }
    }

    /// Samples the next token based on the provided sampler.
    ///
    /// This function does not perform a forward pass through the model. Call
    /// `process_next_token()` after sampling in order to advance the model's state.
    ///
    /// # Parameters
    /// - `sampler`: A mutable reference to the `GptSampler` used for sampling the next token.
    ///
    /// # Returns
    /// - the next sampled `Token`.
    pub fn sample_next_token(&mut self, sampler: &mut GptSampler) -> Token {
        unsafe { wooly_sample_next(self.ctx, sampler.into()) }
    }

    /// Processes the provided `next_token` through the forward pass of the model.
    ///
    /// This operation is computationally intensive. Once completed, `sample_next_token()`
    /// can be called to obtain the next token.
    ///
    /// # Parameters
    ///
    /// - `next_token`: The token to be processed through the model's forward pass.
    ///
    /// # Returns
    /// - The function returns `true` if the operation was successful, and `false` otherwise.
    pub fn process_next_token(&mut self, next_token: Token) -> bool {
        unsafe {
            let result = wooly_process_next_token(self.ctx, next_token);
            result == 0
        }
    }

    /// Freezes the current state of the model and returns it encapsulated in a `FrozenState`.
    ///
    /// This method accounts for the prompt tokens provided in the `params`. If token prediction
    /// has been performed after calling `process_prompt()`, the newly predicted tokens can be
    /// included by passing them via `tokens_opt`. The frozen state will then encapsulate both
    /// the prompt and any predicted tokens.
    ///
    /// # Parameters
    /// - `params`: A mutable reference to `ManagedGptParams` containing the prompt tokens.
    /// - `tokens_opt`: An optional mutable reference to `TokenList` containing newly predicted tokens.
    ///
    /// # Returns
    /// - a `FrozenState` object that holds the frozen state of the model.
    pub fn freeze(
        &mut self,
        params: &mut ManagedGptParams,
        tokens_opt: Option<&mut TokenList>,
    ) -> FrozenState {
        let ptr: *mut wooly_prompt_cache_t;
        unsafe {
            if let Some(tokens) = tokens_opt {
                ptr = wooly_freeze_prediction_state(
                    params.params,
                    self.ctx,
                    self.model,
                    tokens.as_mut_ptr(),
                    tokens.len() as i64,
                );
            } else {
                ptr = wooly_freeze_prediction_state(
                    params.params,
                    self.ctx,
                    self.model,
                    null_mut(),
                    0,
                );
            }
        }

        FrozenState {
            is_alive: true,
            wrapped_ptr: ptr,
        }
    }

    /// Restores the model's state from the provided `frozen_state` and resets the loaded context.
    ///
    /// # Parameters
    /// - `params`: A mutable reference to the parameters managing the GPT model.
    /// - `frozen_state`: A reference to the frozen state from which the model's state will be restored.
    ///
    /// # Returns
    /// - `(i32, GptSampler)`: A tuple where the first element is the number of frozen tokens processed,
    ///   and the second element is a new `GptSampler` instance which should be used for future predictions.
    pub fn defrost(
        &mut self,
        params: &mut ManagedGptParams,
        frozen_state: &FrozenState,
    ) -> (i32, GptSampler) {
        let results: wooly_process_prompt_results;
        unsafe {
            results = wooly_defrost_prediction_state(
                params.params,
                self.ctx,
                self.model,
                frozen_state.wrapped_ptr.into(),
            );
        }

        (
            results.result,
            GptSampler {
                is_alive: true,
                wrapped_ptr: results.gpt_sampler,
            },
        )
    }

    /// Predicts text using the loaded language model (LLM).
    ///
    /// This function generates text based on the provided parameters and a callback function
    /// can be optionally supplied to handle token updates during the prediction process.
    ///
    /// If `params.params.prompt_cache_all` is set to `true`, the state of the LLM will be cached
    /// after processing the prompt. This allows for efficient re-use of the LLM state in subsequent
    /// predictions when the prompt remains exactly the same.
    ///
    /// # Parameters
    /// - `params`: A mutable reference to `ManagedGptParams` containing the parameters for the prediction.
    /// - `callback`: A callback function of type `wooly_token_update_callback` that is invoked during token updates.
    ///
    /// # Returns
    /// - `Ok((wooly_predict_result, String))`: A tuple containing the result of the prediction and the predicted text as a `String`.
    /// - `Err(anyhow::Error)`: An error if the prediction fails, including the return value from the prediction function.
    pub fn predict_text(
        &mut self,
        params: &mut ManagedGptParams,
        callback: wooly_token_update_callback,
    ) -> anyhow::Result<(wooly_predict_result, String)> {
        // allocate the output string towards a maximum of 4-bytes per utf-8 worst case for the whole context
        // and a generous estimate of 10 characters per token.
        let predicted_text_size = (self.loaded_context_len * 4 * 10) as i64;
        let mut predicted_text = Vec::with_capacity((predicted_text_size) as usize);
        let prediction_result = unsafe {
            wooly_predict(
                params.params,
                self.ctx,
                self.model,
                false,
                predicted_text.as_mut_ptr(),
                predicted_text_size,
                self.prompt_cache,
                callback,
            )
        };

        if prediction_result.result != 0 {
            return Err(anyhow::anyhow!(
                "Failed to predict the text; return value: {}",
                prediction_result.result
            ));
        }

        // handle freeing any previous prompt_cache and store the new one, if requested to by the caller
        if (!self.prompt_cache.is_null() && self.prompt_cache != prediction_result.prompt_cache)
            || !params.params.prompt_cache_all
        {
            unsafe { wooly_free_prompt_cache(self.prompt_cache) };
            self.prompt_cache = null_mut();
        }
        if params.params.prompt_cache_all {
            self.prompt_cache = prediction_result.prompt_cache;
        }

        let c_str_result: &CStr = unsafe { CStr::from_ptr(predicted_text.as_mut_ptr()) };
        let result_string: String = c_str_result.to_str().unwrap().to_owned();

        return Ok((prediction_result, result_string));
    }

    /// Returns the token count for the given `text_prompt` when processed by the loaded model's tokenizer.
    ///
    /// # Parameters
    /// - `text_prompt`: The input text string to be tokenized.
    /// - `add_special`: A boolean flag indicating whether to include special tokens (e.g., 'bos' tokens) during tokenization.
    /// - `parse_special`: A boolean flag indicating whether to parse for additional special tokens defined for the model (e.g., '<|begin_of_text|>' for Llama-3).
    ///
    /// # Returns
    /// - the number of tokens in the `text_prompt` as an `i64`.
    pub fn get_token_count(
        &mut self,
        text_prompt: &str,
        add_special: bool,
        parse_special: bool,
    ) -> i64 {
        // make the native string of our text prompt
        let native_text = CString::new(text_prompt).expect("Invalid prompt string");

        // run the library call to get the tokens
        unsafe {
            let tokens_count = wooly_llama_tokenize(
                self.ctx,
                native_text.as_ptr(),
                add_special,
                parse_special,
                null_mut(),
                0,
            );
            return tokens_count;
        }
    }

    /// Tokenizes a given text prompt into a list of integers representing tokens.
    ///
    /// # Parameters
    /// - `text_prompt` - A string slice containing the text to be tokenized.
    /// - `add_special` - A boolean indicating whether to add special tokens like 'bos' (beginning of sequence).
    /// - `parse_special` - A boolean indicating whether to parse and tokenize additional special tokens configured for the model.
    ///
    /// # Returns
    /// - a `TokenList` containing the tokens generated by the loaded model.
    pub fn tokenize_text(
        &mut self,
        text_prompt: &str,
        add_special: bool,
        parse_special: bool,
    ) -> TokenList {
        // make the native string of our text prompt
        let native_text = CString::new(text_prompt).expect("Invalid prompt string");

        // create the output token buffer with a worst case size of one token per character
        let token_buffer_size = text_prompt.len();
        let mut token_buff = Vec::with_capacity((token_buffer_size) as usize);

        // run the library call to get the tokens
        unsafe {
            let tokens_count = wooly_llama_tokenize(
                self.ctx,
                native_text.as_ptr(),
                add_special,
                parse_special,
                token_buff.as_mut_ptr(),
                token_buffer_size as i64,
            );
            token_buff.set_len(tokens_count as usize);
        }

        return token_buff;
    }

    /// Converts a list of tokens into a human-readable string.
    ///
    /// This function takes a mutable reference to a `TokenList` and a boolean flag `render_specials`.
    /// It detokenizes the tokens into a string. If `render_specials` is set to `true`, special tokens
    /// are included in the output; otherwise, they are skipped.
    ///
    /// # Parameters
    /// - `tokens`: A mutable reference to a `TokenList` containing the tokens to be detokenized.
    /// - `render_specials`: A boolean flag indicating whether special tokens should be rendered in the output.
    ///
    /// # Returns
    /// - a `String` representing the detokenized text.
    pub fn detokenize_text(&mut self, tokens: &mut TokenList, render_specials: bool) -> String {
        // allocate the output string towards a maximum of 4-bytes per utf-8 worst case
        // for the whole context and a generous estimate of 10 characters per token.
        let output_text_size = (self.loaded_context_len * 4 * 10) as i64;
        let mut output_text = Vec::with_capacity((output_text_size) as usize);

        // run the library call to get the tokens
        unsafe {
            let _char_count = wooly_llama_detokenize(
                self.ctx,
                render_specials,
                tokens.as_mut_ptr(),
                tokens.len() as i64,
                output_text.as_mut_ptr(),
                output_text_size,
            );
            let c_str_result: &CStr = CStr::from_ptr(output_text.as_mut_ptr());
            let result_string: String = c_str_result.to_str().unwrap().to_owned();
            return result_string;
        }
    }

    /// Generates embeddings for the given tokenized prompts.
    ///
    /// # Parameters
    /// - `embd_normalize` - Specifies how the embeddings should be normalized.
    /// - `tokenized_prompts` - A vector of tokenized prompts for which embeddings are to be generated.
    ///
    /// # Returns
    /// - `Option<Vec<Embedding>>` - Returns a vector of embeddings if successful, otherwise `None`.
    ///
    /// # Notes
    /// If no context is loaded (`loaded_context_params` is `None`), the function will return `None`.
    /// The number of embeddings generated depends on the pooling type:
    /// - If pooling type is `LlamaPoolingType::None`, an embedding vector is generated for every token.
    /// - Otherwise, one embedding vector is generated per prompt.
    pub fn make_embeddings(
        &mut self,
        embd_normalize: EmbeddingNormalization,
        tokenized_prompts: &mut Vec<TokenList>,
    ) -> Option<Vec<Embedding>> {
        // if we dont have a loaded context, just return here without an answer
        if self.loaded_context_params.is_none() {
            return None;
        }
        let context_params = self.loaded_context_params.unwrap();

        // count the number of embeddings needed.
        let embd_needed = if context_params.pooling_type == LlamaPoolingType::None as i32 {
            // no pooling means we get a full embedding vector for every token, so
            // go through all the prompts and figure out the total number of tokens
            // and change the needed float count accordingly
            tokenized_prompts.iter().fold(0, |acc, p| acc + p.len())
        } else {
            tokenized_prompts.len()
        };

        // get the size of the embedding vectors and scale the floats needed
        // by that size and make the output buffer
        let n_embd = unsafe { wooly_llama_n_embd(self.model) };
        let total_floats_needed = embd_needed * n_embd as usize;
        let mut output_embeddings: Vec<f32> = Vec::with_capacity((total_floats_needed) as usize);
        let mut tp_sizes: Vec<i64> = vec![];
        let mut tp_ptrs = vec![];
        for tp in tokenized_prompts {
            let tp_ptr = tp.as_mut_ptr();
            tp_ptrs.push(tp_ptr);
            tp_sizes.push(tp.len() as i64);
        }

        unsafe {
            let success = wooly_llama_make_embeddings(
                self.model,
                self.ctx,
                context_params.n_batch as i32,
                context_params.pooling_type as i32,
                embd_normalize as i32,
                tp_ptrs.len() as i64,
                tp_ptrs.as_mut_ptr(),
                tp_sizes.as_mut_ptr(),
                output_embeddings.as_mut_ptr(),
                total_floats_needed as i64,
            );

            if success < 0 {
                None
            } else {
                // populating the vec in C doesn't change the length, so if we've succeeded
                // then we can assume we got the number of embeddings we requested, so
                // force the length of the collection and then chunk them out into separate vectors.
                output_embeddings.set_len(total_floats_needed);
                Some(
                    output_embeddings
                        .chunks(n_embd as usize)
                        .map(|c| c.to_vec())
                        .collect(),
                )
            }
        }
    }
}
