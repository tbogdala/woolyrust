
use std::process::exit;

use woolyrust::{embeddings_similarity_cos, Embedding, Llama, TokenList};

#[test]
pub fn embeddings_test() {
    let mut model_params = woolyrust::get_default_model_params();
    model_params.n_gpu_layers = 100;
    let mut context_params = woolyrust::get_default_context_params();
    context_params.seed = 42;
    context_params.n_ctx = 2048;

    // the test is designed for nomic-ai/nomic-embed-text-v1.5-GGUF which has 2048 context by default.
    context_params.n_batch = 2048;

    // setup embedding specific behaviors
    context_params.embeddings = true;
    context_params.n_ubatch = context_params.n_batch;
    context_params.pooling_type = 1; // mean pooling

    let model_filepath = get_test_model_path();
    let mut llama = Llama::new();
    let load_success = llama.load_model(model_filepath.as_str(), model_params, context_params, true);
    assert_eq!(load_success, true);
    assert_eq!(llama.is_loaded(), true);

    // setup some test sentences to test for similarity against the first prompt
    let prompts = [
        "That is a happy person.",
        "That's a very happy person.",
        "She is not happy about the news.",
        "Is that a happy dog?",
        "Behold an individual brimming with boundless joy and contentment.",
        "The weather is beautiful today.",
        "The sun is shining brightly.",
        "I've seen things you people wouldn't believe. Attack ships on fire off the shoulder of Orion.",
      ];

    // tokenize the prompts and do some extra testing of the API
    let mut tokenized_prompts: Vec<TokenList> = vec![];
    for i in 0..prompts.len() {
        let mut tokens = llama.tokenize_text(prompts[i], true, false);
        println!("Tokenizing prompt: \"{}\"\n\t{:?}", prompts[i], &tokens);
        let actual_token_count = tokens.len();
        tokenized_prompts.push(tokens.clone());

        // check it against the predicted token count
        let predicted_token_count = llama.get_token_count(prompts[i], true, false);
        assert_eq!(predicted_token_count, actual_token_count as i64);

        // detokenize the tokenized prompt to round trip it. we don't do
        // equality tests here because it's not guaranteed to be the same -
        // for example the nomic embedding models do not generate an exact
        // round-tripped string, but llama-3 will. We'll just make sure the
        // returned string isn't empty
        let round_tripped_prompt = llama.detokenize_text(&mut tokens, false);
        println!("\tDetokenized: \"{}\"", round_tripped_prompt);
        assert_eq!(round_tripped_prompt.is_empty(), false);
    }

    // generate the embeddings for the tokenized prompts
    let embeddings: Vec<Embedding> = llama.make_embeddings(
        woolyrust::EmbeddingNormalization::Euclidean,
        &mut tokenized_prompts).expect("creating vector embeddings for tokenized prompts");

    println!("Got a total of {} embeddings.", embeddings.len());

    println!("\nTesting similarity to \"{}\"\n", prompts[0]);
    for (i, embd) in embeddings.iter().enumerate() {
        let score = embeddings_similarity_cos(embeddings[0].as_slice(), embd.as_slice());
        println!("\t{}: {}", score, prompts[i]);
    }
}

// the relative path to the model to load for the tests
pub fn get_test_model_path() -> String {
    let model_filepath = std::env::var("WOOLY_TEST_EMB_MODEL_FILE");
    if let Ok(fp) = model_filepath {
        return fp;
    } else {
        println!("Set WOOLY_TEST_EMB_MODEL_FILE environment variable to the gguf embedding model to use for testing");
        exit(1);
    }
}
