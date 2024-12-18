use std::process::exit;

use woolyrust::{Llama, ManagedGptParams, TokenList};

#[test]
pub fn step_prediction_test() {
    // load up our test model
    let mut model_params = woolyrust::get_default_model_params();
    model_params.n_gpu_layers = 100;
    let mut context_params = woolyrust::get_default_context_params();
    context_params.n_ctx = 1024 * 2;

    let model_filepath = get_test_model_path();
    let mut llama = Llama::new();
    let load_success =
        llama.load_model(model_filepath.as_str(), model_params, context_params, true);
    assert_eq!(load_success, true);
    assert_eq!(llama.is_loaded(), true);

    // set the text generation parameters up
    let mut params = ManagedGptParams::defaults();
    params.params.seed = 42;
    params.params.n_threads = -1;
    params.params.n_predict = 100;
    params.params.temp = 0.1;
    params.params.top_k = 1;
    params.params.top_p = 1.0;
    params.params.min_p = 0.1;
    params.params.penalty_repeat = 1.1;
    params.params.penalty_last_n = 512;
    params.params.ignore_eos = false;
    params.params.flash_attn = true;
    params.params.n_batch = 128;
    params.params.prompt_cache_all = false;
    let antiprompts = vec!["<|end|>"];
    let prompt = "<|user|>\nWrite the start to the next movie collaboration between Quentin Tarantino and Robert Rodriguez.<|end|>\n<|assistant|>\n";
    params.set_antiprompts(&antiprompts);
    params.set_prompt(prompt);

    params.params.dry_multiplier = 0.8;
    params.params.dry_base = 1.75;
    params.params.dry_allowed_length = 2;
    params.params.dry_penalty_last_n = -1;
    let seq_breakers = vec!["\n", ":", "\"", "*"];
    params.set_dry_sequence_breakers(&seq_breakers);

    // get the prompt ingested into the context and pull the sampler
    // used in the process so that repeat penalties and such are
    // accounted for.
    let (prompt_token_count, mut first_sampler) = llama.process_prompt(&mut params);
    assert_eq!(prompt_token_count > 0, true);

    // we freeze the state after processing the prompt so that we can generate
    // a second block of text after the first one without having to reprocess
    // the prompt. not as big of a deal when your prompt is 31 tokens, but it
    // IS a bigger deal when your prompt is 31,000 tokens.
    let frozen_prompt = llama.freeze(&mut params, None);

    // start our prediction loop now that the prompt has been processed
    let mut predictions: TokenList = vec![];
    while predictions.len() < params.params.n_predict as usize {
        let next_token = llama.sample_next_token(&mut first_sampler);

        if predictions.len() < params.params.n_predict as usize {
            let processed = llama.process_next_token(next_token);
            assert!(processed);
        }

        predictions.push(next_token);
    }

    // print out our first prediction
    let first_prediction_count = predictions.len() as i32;
    let first_prediction_str = llama.detokenize_text(&mut predictions, false);
    println!("Prompt token count: {}", prompt_token_count);
    println!(
        "Prediction (tokens: {})\n{}",
        predictions.len(),
        first_prediction_str
    );
    assert!(!first_prediction_str.is_empty());
    assert!(predictions.len() > 0);

    // freeze our prediction state too for further testing later
    let frozen_prediction = llama.freeze(&mut params, Some(&mut predictions));

    println!("\n~~~ ---- ~~~~\n\n");

    // defrost our frozen state from processing the prompt and generate something new
    let (frozen_prompt_token_count, mut second_sampler) =
        llama.defrost(&mut params, &frozen_prompt);
    assert!(prompt_token_count == frozen_prompt_token_count);

    // inject a little more prompt in just to make it different and test
    // the ability to add more prompt to ingest. This should produce a
    // distinctly different result than the first prediction.
    let new_prompt_text = "Do you have a suggestion for genre?<|end|>\n<|user|>\nMake it like a Pixar movie script, but with those two authors!<|end|>\n<|assistant|>\n";
    let new_prompt_tokens = llama.process_additional_prompt(&mut second_sampler, new_prompt_text);
    assert!(new_prompt_tokens > 0);

    // start our prediction loop and make something new with the frozen prompt
    let mut predictions: TokenList = vec![];
    while predictions.len() < params.params.n_predict as usize {
        let next_token = llama.sample_next_token(&mut second_sampler);

        if predictions.len() < params.params.n_predict as usize {
            let processed = llama.process_next_token(next_token);
            assert!(processed);
        }

        predictions.push(next_token);
    }

    // print out our second prediction
    let second_prediction_str = llama.detokenize_text(&mut predictions, false);
    println!(
        "Prediction (tokens: {})\n{}",
        predictions.len(),
        second_prediction_str
    );
    assert!(!second_prediction_str.is_empty());
    assert!(predictions.len() > 0);

    println!("\n~~~ ---- ~~~~\n\n");

    // defrost our frozen state from our first prediction and continue it
    let (frozen_pred_token_count, mut third_sampler) =
        llama.defrost(&mut params, &frozen_prediction);
    assert!(prompt_token_count + first_prediction_count == frozen_pred_token_count);

    // start our prediction loop and continue our frozen prediction from earlier
    let mut predictions: TokenList = vec![];
    while predictions.len() < params.params.n_predict as usize {
        let next_token = llama.sample_next_token(&mut third_sampler);

        // Note: it's important to account for the new token count from the
        // frozen prediction state or else the continuation won't make sense.
        if predictions.len() < params.params.n_predict as usize {
            let processed = llama.process_next_token(next_token);
            assert!(processed);
        }

        predictions.push(next_token);
    }

    // print out our second prediction
    let third_prediction_str = llama.detokenize_text(&mut predictions, false);
    println!(
        "Continued prediction (tokens: {})\n{}{}",
        predictions.len(),
        first_prediction_str,
        third_prediction_str
    );
    assert!(!third_prediction_str.is_empty());
    assert!(predictions.len() > 0);
}

// the relative path to the model to load for the tests
pub fn get_test_model_path() -> String {
    let model_filepath = std::env::var("WOOLY_TEST_MODEL_FILE");
    if let Ok(fp) = model_filepath {
        return fp;
    } else {
        println!(
            "Set WOOLY_TEST_MODEL_FILE environment variable to the gguf file to use for testing"
        );
        exit(1);
    }
}
