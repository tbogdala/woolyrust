
use std::{env, path::PathBuf};

use cmake::Config;

fn main() {
    let out_path = PathBuf::from(env::var("OUT_DIR").expect("No out dir found"));
    compile_bindings(&out_path);

    let dst = 
        if cfg!(feature = "cuda") {
            Config::new("woolycore")
            .define("GGML_CUDA", "On")
                 .build()
        } else { 
            Config::new("woolycore")
                 .build()
        };
    let build_folder = dst.join("build");
    println!("cargo:rustc-link-search=native={}", build_folder.display());
    println!("cargo:rustc-link-search=native={}", dst.join("lib").display());
    println!("cargo:rustc-link-lib=dylib=woolycore");
}

fn compile_bindings(out_path: &PathBuf) {
    let bindings = bindgen::Builder::default()
        .header("./woolycore/bindings.h")
        .clang_arg("-Iwoolycore/")
        .parse_callbacks(Box::new(bindgen::CargoCallbacks::new()))
        .allowlist_function("wooly.*")
        .allowlist_type("wooly.*")
        .generate()
        .expect("Unable to generate bindings for woolycore's bindings.h file");

    bindings
        .write_to_file(&out_path.join("bindings.rs"))
        .expect("Couldn't write bindings!");
}