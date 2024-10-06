use std::{ffi::CStr, fs::read_to_string, io::Write, process::exit};

use woolyrust::Llama;

#[test]
pub fn basic_test() {
    let mut model_params = woolyrust::get_default_model_params();
    model_params.n_gpu_layers = 100;
    let mut context_params = woolyrust::get_default_context_params();
    context_params.n_ctx = 1024 * 2;

    let model_filepath = get_test_model_path();
    let mut llama = Llama::new();
    let load_success = llama.load_model(model_filepath.as_str(), model_params, context_params, true);
    assert_eq!(load_success, true);
    assert_eq!(llama.is_loaded(), true);

    let mut params = woolyrust::new_text_gen_params();
    params.seed = 42;
    params.n_threads = 4;
    params.n_predict = 100;
    params.temp = 0.1;
    params.top_k = 1;
    params.top_p = 1.0;
    params.min_p = 0.1;
    params.penalty_repeat = 1.1;
    params.penalty_last_n = 512;
    params.ignore_eos = false;
    params.flash_attn = true;
    params.n_batch = 128;
    params.prompt_cache_all = true;
    
    let antiprompts = vec!["<|end|>"];
    let prompt =  "<|user|>\nWrite the start to the next movie collaboration between Quentin Tarantino and Robert Rodriguez.<|end|>\n<|assistant|>\n";

    let (results, prediction) = llama.predict_text(&mut params, prompt, Some(&antiprompts), None, None).expect("Failed to run the prediction");
    assert_eq!(results.result, 0);

    // the first prediction is done without the callback for tokens, so we make sure to print out the result
    println!(
        "\n{}\n\nTiming Data: {} tokens total in {:.2} ms ({:.2} T/s) ; {} prompt tokens in {:.2} ms ({:.2} T/s)\n\n",
        prediction,
        results.n_eval,
        (results.t_end_ms - results.t_start_ms),
        1e3 / (results.t_end_ms - results.t_start_ms) * results.n_eval as f64,
        results.n_p_eval,
        results.t_p_eval_ms,
        1e3 / results.t_p_eval_ms * results.n_p_eval as f64);

    // change the seed and try another prediction to see if prompt caching is working.
    params.seed = 1337;
    let (results, _prediction) = llama.predict_text(&mut params, prompt, Some(&antiprompts), None, Some(predict_callback)).expect("Failed to run the prediction");
    assert_eq!(results.result, 0);
    assert_eq!(results.n_p_eval, 1); // test to see if prompt processing was successfully skipped (min value upstream is now 1)

    println!(
        "\n\nTiming Data: {} tokens total in {:.2} ms ({:.2} T/s) ; {} prompt tokens in {:.2} ms ({:.2} T/s)\n\n",
        results.n_eval,
        (results.t_end_ms - results.t_start_ms),
        1e3 / (results.t_end_ms - results.t_start_ms) * results.n_eval as f64,
        results.n_p_eval,
        results.t_p_eval_ms,
        1e3 / results.t_p_eval_ms * results.n_p_eval as f64);

    // now see if grammar works
    let gbnf_string = read_to_string("woolycore/llama.cpp/grammars/json.gbnf").expect("Couldn't read the GBNF grammar file from upstream llama.cpp");
    params.n_predict = -1;
    let prompt = "<|user|>\nReturn a JSON object that describes an object in a fictional Dark Souls game. The returned JSON object should have 'Title' and 'Description' fields that define the item in the game. Make sure to write the item lore in the style of Fromsoft and thier Dark Souls series of games: there should be over-the-top naming of fantastically gross monsters and tragic historical events from the world, all with a very nihilistic feel.<|end|>\n<|assistant|>\n";
    let (results, _prediction) = llama.predict_text(&mut params, prompt, Some(&antiprompts), Some(gbnf_string.as_str()), Some(predict_callback)).expect("Failed to run the prediction");
    assert_eq!(results.result, 0);
    assert_ne!(results.n_p_eval, 0); // test to see if prompt processing was successfully resumed

    println!(
        "\n\nTiming Data: {} tokens total in {:.2} ms ({:.2} T/s) ; {} prompt tokens in {:.2} ms ({:.2} T/s)\n\n",
        results.n_eval,
        (results.t_end_ms - results.t_start_ms),
        1e3 / (results.t_end_ms - results.t_start_ms) * results.n_eval as f64,
        results.n_p_eval,
        results.t_p_eval_ms,
        1e3 / results.t_p_eval_ms * results.n_p_eval as f64);
}

// the relative path to the model to load for the tests
pub fn get_test_model_path() -> String {
    let model_filepath = std::env::var("WOOLY_TEST_MODEL_FILE");
    if let Ok(fp) = model_filepath {
        return fp;
    } else {
        println!("Set WOOLY_TEST_MODEL_FILE environment variable to the gguf file to use for testing");
        exit(1);
    }
}


#[no_mangle]
extern "C" fn predict_callback(token_str: *const ::std::os::raw::c_char) -> bool {
    // for the test we just play fast and loose.
    unsafe {
        let c_string = CStr::from_ptr(token_str);
        let token = String::from_utf8_lossy(c_string.to_bytes()).to_string();
        print!("{}", token);
        std::io::stdout().flush().expect("Error flushing stdout");
    }

    return true;
}