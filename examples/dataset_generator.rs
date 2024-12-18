//! # Dataset Generator Example
//!
//! This example demonstrates how to generate a synthetic dataset using the woolyrust library.
//!
//! The program takes in several command-line arguments to customize the generation process, including:
//! - The path to the GGUF model file
//! - The system prompt and instruction prompt for the dataset
//! - The number of datasets to generate
//! - The output file path to save the generated predictions
//! - Various parameters to control the text generation, such as temperature, top-k sampling, and penalty for repeating tokens
//!
//! The program uses the woolyrust library to load the GGUF model, process the prompts, and generate text predictions.
//! The initial prompt is 'cached' so that it doesn't have to be processed on each loop iteration.
//! The generated predictions are saved to the specified output file in JSONL format.

use anyhow::Result;
use clap::Parser;
use serde::{Deserialize, Serialize};
use std::fs::OpenOptions;
use std::io::{BufWriter, Write};
use std::process::exit;
use woolyrust::{Llama, ManagedGptParams, TokenList};

#[derive(Serialize, Deserialize)]
struct Prediction {
    prompt: String,
    prediction: String,
}

fn main() -> Result<()> {
    let cli = DatasetGenArgs::parse();

    // Load up our test model
    let mut model_params = woolyrust::get_default_model_params();
    model_params.n_gpu_layers = cli.gpu_layers;
    let mut context_params = woolyrust::get_default_context_params();
    context_params.n_ctx = cli.context_size;

    let model_filepath = cli.model_filepath;
    let mut llama = Llama::new();
    let load_success =
        llama.load_model(model_filepath.as_str(), model_params, context_params, true);
    if !load_success {
        println!(
            "ERROR: couldn't load GGUF model: {}",
            model_filepath.as_str()
        );
        exit(1);
    }

    // Set the text generation parameters up
    let mut params = ManagedGptParams::defaults();
    params.params.seed = u32::MAX;
    params.params.n_threads = cli.threads;
    params.params.n_predict = cli.predict_length;
    params.params.temp = cli.temperature;
    params.params.top_k = cli.top_k;
    params.params.top_p = cli.top_p;
    params.params.min_p = cli.min_p;
    params.params.penalty_repeat = cli.penalty_repeat;
    params.params.penalty_last_n = cli.penalty_last_n;
    params.params.ignore_eos = false;
    params.params.flash_attn = true;
    params.params.n_batch = 128;
    params.params.prompt_cache_all = false;

    // specifically tune the prompt and antiprompts to the Llama 3 Instruct prompt syntax.
    let antiprompts = vec!["<|eot_id|>", "<|start_header_id|>"];
    let prompt = format!(
        "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{}<|eot_id|><|start_header_id|>assistant<|end_header_id|>",
        cli.system_prompt, cli.instruction_prompt
    );
    params.set_antiprompts(&antiprompts);
    params.set_prompt(prompt.as_str());

    // Get the prompt ingested into the context and pull the sampler
    // used in the process so that repeat penalties and such are
    // accounted for.
    let (_prompt_token_count, mut first_sampler) = llama.process_prompt(&mut params);

    // Freeze the state after processing the prompt so that we can restore it
    // during each iteration of the dataset generation loop
    let frozen_prompt = llama.freeze(&mut params, None);

    // Open the output file before the loop
    let file = OpenOptions::new()
        .append(true)
        .create(true)
        .open(&cli.output_filepath)?;
    let mut buf_writer = BufWriter::new(file);

    for i in 0..cli.count {
        // Start our prediction loop now that the prompt has been processed
        let mut prediction_tokens: TokenList = vec![];
        while prediction_tokens.len() < params.params.n_predict as usize {
            let next_token = llama.sample_next_token(&mut first_sampler);

            if prediction_tokens.len() < params.params.n_predict as usize {
                let processed = llama.process_next_token(next_token);
                assert!(processed);
            }

            prediction_tokens.push(next_token);
        }

        // Print out our prediction
        let prediction_str = llama
            .detokenize_text(&mut prediction_tokens, false)
            .trim()
            .to_string();

        println!(
            "\n\n<<=============================>>\nPrediction #{} (tokens: {})\n{}",
            i + 1,
            prediction_tokens.len(),
            prediction_str
        );
        if !prediction_str.is_empty() && prediction_tokens.len() > 0 {
            // Serialize and write the prediction immediately
            let prediction = Prediction {
                prompt: prompt.clone(),
                prediction: prediction_str,
            };
            let json = serde_json::to_string(&prediction)?;
            writeln!(buf_writer, "{}", json)?;

            // Flush the buffer to ensure all data is written to the file
            buf_writer.flush()?;
        }

        // Defrost our frozen state from processing the prompt and generate something new
        let (_, next_sampler) = llama.defrost(&mut params, &frozen_prompt);
        first_sampler = next_sampler;
    }

    println!("Predictions saved to {}", cli.output_filepath);
    Ok(())
}

#[derive(Parser)]
#[command(author, version, about = "Dataset Generator Example", long_about = None)]
struct DatasetGenArgs {
    #[arg(
        short('o'),
        long("output"),
        default_value = "predictions.jsonl",
        help = "Number of datasets to generate."
    )]
    output_filepath: String,

    #[arg(
        short('i'),
        long,
        default_value_t = 1,
        help = "Number of datasets to generate."
    )]
    count: u32,

    #[arg(
        long("system"),
        required = true,
        help = "System prompt for the dataset."
    )]
    system_prompt: String,

    #[arg(
        long("prompt"),
        required = true,
        help = "Instruction prompt for the dataset."
    )]
    instruction_prompt: String,

    #[arg(
        short('t'),
        long("temp"),
        default_value_t = 0.7,
        help = "Temperature setting for randomness in generation."
    )]
    temperature: f32,

    #[arg(short('k'), long, default_value_t = 50, help = "Top-k sampling value.")]
    top_k: i32,

    #[arg(
        short('p'),
        long,
        default_value_t = 0.9,
        help = "Top-p sampling value."
    )]
    top_p: f32,

    #[arg(
        short('m'),
        long,
        default_value_t = 0.05,
        help = "Minimum probability threshold."
    )]
    min_p: f32,

    #[arg(
        short('r'),
        long,
        default_value_t = 1.02,
        help = "Penalty for repeating tokens."
    )]
    penalty_repeat: f32,

    #[arg(
        long("rep-pen-range"),
        default_value_t = 512,
        help = "Last N tokens to consider for penalty."
    )]
    penalty_last_n: i32,

    #[arg(
        short('n'),
        long("predict"),
        default_value_t = 100,
        help = "Predicted length of the generated text."
    )]
    predict_length: i32,

    #[arg(
        long("ngl"),
        default_value_t = 100,
        help = "Number of layers to offload to the GPU."
    )]
    gpu_layers: i32,

    #[arg(
        long("threads"),
        default_value_t = -1,
        help = "Number of CPU threads used for processing."
    )]
    threads: i32,

    #[arg(
        short('c'),
        long("context-size"),
        default_value_t = 1024 * 2,
        help = "Number tokens to support in the context of the loaded model."
    )]
    context_size: u32,

    #[arg(
        short('f'),
        long("model"),
        required = true,
        help = "Path to the GGUF model file."
    )]
    model_filepath: String,
}
