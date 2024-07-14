#![allow(non_upper_case_globals)]
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]

use std::{ffi::{c_void, CStr, CString}, ptr::null_mut};
include!(concat!(env!("OUT_DIR"), "/bindings.rs"));

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

#[derive(Debug, Clone)]
pub struct Llama {
    ctx: *mut c_void,
    model: *mut c_void,
    prompt_cache: *mut c_void,
    loadedContextLen: u32,
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
            loadedContextLen: 0,
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
        self.loadedContextLen = result.context_length;
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
        self.loadedContextLen = 0;
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
        let mut predicted_text = Vec::with_capacity((self.loadedContextLen * 4) as usize);
        let prediction_result = unsafe { 
            wooly_predict(*params, self.ctx, self.model, false, predicted_text.as_mut_ptr(), self.prompt_cache, callback) 
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

}
