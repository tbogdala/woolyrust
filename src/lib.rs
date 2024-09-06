#![allow(non_upper_case_globals)]
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]

use std::{ffi::{c_void, CStr, CString}, ptr::{null, null_mut}};
include!(concat!(env!("OUT_DIR"), "/bindings.rs"));

pub type Token = i32;
pub type TokenList = Vec<Token>;
pub type Embedding = Vec<f32>;

#[derive(Debug)]
pub enum LlamaPoolingType {
    Unspecified = -1,
    None = 0,
    Mean = 1,
    Cls = 2,
    Last = 3,
}

#[derive(Debug)]
pub enum EmbeddingNormalization {
    None = -1,
    MaxAbsoluteInt16 = 0,
    Taxicab = 1,
    Euclidean = 2,
    PNorm = 3,
}


pub fn get_default_model_params() -> wooly_llama_model_params {
    unsafe {
        return wooly_get_default_llama_model_params();
    } 
}

pub fn get_default_context_params() -> wooly_llama_context_params {
    unsafe {
        return wooly_get_default_llama_context_params();
    }
}

pub fn new_text_gen_params() -> wooly_gpt_params {
    unsafe {
        return wooly_new_gpt_params();
    }
}

use std::f32;

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

#[derive(Debug, Clone)]
pub struct Llama {
    ctx: *mut c_void,
    model: *mut c_void,
    prompt_cache: *mut c_void,
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
        return Self{
            ctx: null_mut(),
            model: null_mut(),
            prompt_cache: null_mut(),
            loaded_context_len: 0,
            loaded_context_params: None,
            loaded_model_params: None,
        }
    }

    // Load a model at the filepath specified in model_file. The model and context
    // parameters are used when loading the model and creating a new context to
    // operate from. The silence boolean will allow the client code to
    // disable all of the information that upstream llama.cpp writes to output streams.
    //
    // Should the process fail, false is returned.
    pub fn load_model(&mut self, model_file: &str, model_params: wooly_llama_model_params, 
        context_params: wooly_llama_context_params, silence: bool) -> bool {
        // free the model and prompt cache if there's one set already 
        self.free_model();

        // attempt to load the model
        let result: wooly_load_model_result;
        let native_model_path = CString::new(model_file).unwrap();
        unsafe {
            result = wooly_load_model(native_model_path.as_ptr(), model_params, context_params, silence);
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

    pub fn is_loaded(&self) -> bool {
        return !self.ctx.is_null() && !self.model.is_null();
    }

    // This function does the text prediction using the loaded LLM. The parameters to the function take
    // precedence over the similarly named fields in `params` as a convenience so that the lifetimes
    // of the C-compatible strings are managed for the caller.
    pub fn predict_text(&mut self, 
        params: &mut wooly_gpt_params, 
        prompt: &str, 
        antiprompts_opt: Option<&Vec<&str>>,
        grammar_opt: Option<&str>,
        callback: wooly_token_update_callback,
    ) -> anyhow::Result<(wooly_predict_result, String)> {
        // allocate the prompt
        let native_prompt = CString::new(prompt).expect("Invalid prompt string");
        params.prompt = native_prompt.as_ptr();

        // allocate the grammar string
        let native_grammar: CString;
        if let Some(grammar) = grammar_opt {
            native_grammar = CString::new(grammar).expect("Invalid grammar string");
            params.grammar = native_grammar.as_ptr();
        }

        // allocate the antipompts
        // we store the CStrings for the antiprompts as well as building a vector of pointers to send to the library
        let mut native_anti_strings: Vec<CString>;
        let mut native_anti_pointers: Vec<*const i8>;
        if let Some(antiprompts) = antiprompts_opt {
            let count = antiprompts.len();
            native_anti_strings = Vec::with_capacity(count);
            native_anti_pointers = Vec::with_capacity(count);
            for antiprompt in antiprompts {
                let native_anti = CString::new(*antiprompt).expect("Invalid antiprompt string");
                native_anti_pointers.push(native_anti.as_ptr());
                native_anti_strings.push(native_anti);
            }
            params.antiprompts = native_anti_pointers.as_mut_ptr();
            params.antiprompt_count = count as i32;
        }

        // allocate the output string towards a maximum of 4-bytes per utf-8 worst case for the whole context
        // and a generous estimate of 10 characters per token.
        let predicted_text_size = (self.loaded_context_len * 4 * 10) as i64 ;
        let mut predicted_text = Vec::with_capacity((predicted_text_size) as usize);
        let prediction_result = unsafe { 
            wooly_predict(*params, self.ctx, self.model, false, predicted_text.as_mut_ptr(), predicted_text_size, self.prompt_cache, callback) 
        };

        if prediction_result.result != 0 {
            return Err(anyhow::anyhow!("Failed to predict the text; return value: {}", prediction_result.result));
        }

        // handle freeing any previous prompt_cache and store the new one, if requested to by the caller
        if (!self.prompt_cache.is_null() && self.prompt_cache != prediction_result.prompt_cache) || !params.prompt_cache_all {
            unsafe { 
                wooly_free_prompt_cache(self.prompt_cache) 
            };
            self.prompt_cache = null_mut();
        }
        if params.prompt_cache_all {
            self.prompt_cache = prediction_result.prompt_cache;
        }

        let c_str_result: &CStr = unsafe { CStr::from_ptr(predicted_text.as_mut_ptr()) };
        let result_string: String = c_str_result.to_str().unwrap().to_owned();

        return Ok((prediction_result, result_string));
    }

    // returns the token count for the `text_prompt` when processed by the loaded
    // model's tokenizer. `add_special` controls whether or not to add special tokens
    // when encoding sequences, such as 'bos' tokens. `parse_special`
    // controls whether or not to parse for additional 'special' tokens defined for
    // the model, such as '<|begin_of_text|>' for Llama-3.
    pub fn get_token_count(&mut self, text_prompt: &str, add_special: bool, parse_special: bool) -> i64 {
        // make the native string of our text prompt
        let native_text = CString::new(text_prompt).expect("Invalid prompt string");

        // run the library call to get the tokens
        unsafe {
            let tokens_count = wooly_llama_tokenize(
                self.model, 
                native_text.as_ptr(), 
                add_special, 
                parse_special, 
                null_mut(), 
                0);
            return tokens_count;
        }
    }

    // returns a Vec of ints representing the tokens generated by the loaded
    // model for a given `text_prompt`. If `add_special` is true, the special
    // tokens like 'bos' are added. If `parse_special is true, the tokenizer
    // will look for the additional special tokens configured for the model
    // and tokenize them accordingly.
    pub fn tokenize_text(&mut self, text_prompt: &str, add_special: bool, parse_special: bool) -> TokenList
    {
        // make the native string of our text prompt
        let native_text = CString::new(text_prompt).expect("Invalid prompt string");
        
        // create the output token buffer with a worst case size of one token per character
        let token_buffer_size = text_prompt.len();
        let mut token_buff = Vec::with_capacity((token_buffer_size) as usize);
        
        // run the library call to get the tokens
        unsafe {
            let tokens_count = wooly_llama_tokenize(
                self.model, 
                native_text.as_ptr(), 
                add_special, 
                parse_special, 
                token_buff.as_mut_ptr(), 
                token_buffer_size as i64);
            token_buff.set_len(tokens_count as usize);
        }

        return token_buff
    }

    // returns a string based on the TokenList passed in. if `render_specials` is set to true,
    // then the special tokens will get rendered to text; otherwise they are skipped over.
    pub fn detokenize_text(&mut self, tokens: &mut TokenList, render_specials: bool) -> String
    {
        // allocate the output string towards a maximum of 4-bytes per utf-8 worst case 
        // for the whole context and a generous estimate of 10 characters per token.
        let output_text_size = (self.loaded_context_len * 4 * 10) as i64 ;
        let mut output_text = Vec::with_capacity((output_text_size) as usize);

        // run the library call to get the tokens
        unsafe {
            let _char_count = wooly_llama_detokenize(
                self.ctx, 
                render_specials, 
                tokens.as_mut_ptr(), 
                tokens.len() as i64, 
                output_text.as_mut_ptr(), 
                output_text_size);
            let c_str_result: &CStr =CStr::from_ptr(output_text.as_mut_ptr());
            let result_string: String = c_str_result.to_str().unwrap().to_owned();
            return result_string;
        }
    }

    pub fn make_embeddings(&mut self, embd_normalize: EmbeddingNormalization, tokenized_prompts: &mut Vec<TokenList>) -> Option<Vec<Embedding>>
    {
        // if we dont have a loaded context, just return here without an answer
        if self.loaded_context_params.is_none() {
            return None;
        }
        let context_params = self.loaded_context_params.unwrap();

        // count the number of embeddings needed.
        let embd_needed = 
            if context_params.pooling_type == LlamaPoolingType::None as i32 {
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
                total_floats_needed as i64);
            
            if success < 0 {
                None
            } else {
                // populating the vec in C doesn't change the length, so if we've succeeded
                // then we can assume we got the number of embeddings we requested, so
                // force the length of the collection and then chunk them out into separate vectors.
                output_embeddings.set_len(total_floats_needed);
                Some(output_embeddings.chunks(n_embd as usize).map(|c| c.to_vec()).collect())
            }
        }
    }
}
