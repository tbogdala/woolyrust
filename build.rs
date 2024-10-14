use std::{env, path::PathBuf};

use cmake::Config;

fn main() {
    let is_release = env::var("OPT_LEVEL").map(|val| val == "3").unwrap_or(false);
    let out_path = PathBuf::from(env::var("OUT_DIR").expect("No out dir found"));
    compile_bindings(&out_path);

    // startup a new cmake config for our library
    let mut config: &mut Config = &mut Config::new("woolycore");
    config = config.define("WOOLY_TESTS", "OFF")
        .define("BUILD_SHARED_LIBS", "FALSE");

        // enable the CUDA flags if the feature is present
    // NOTE: metal is automatically enabled in the upstream library and needs
    // no special intervention.
    if cfg!(feature = "cuda") {
        config = config.define("GGML_CUDA", "On");
    }

    // a few more flags for Windows to make sure that it creates a dll that
    // exports all of the functions we want to export.
    if std::env::consts::OS == "windows" {
        config = config.define("CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS", "TRUE");
    }

    let dst = config.build();
    let build_folder = dst.join("build");
    println!("cargo:rustc-link-search=native={}", build_folder.display());
    println!("cargo:rustc-link-search=native={}", dst.join("lib").display());
    println!("cargo:rustc-link-lib=dylib=woolycore");

    // Windows is the special child, like always. Do some extra path and
    // file management to make things smoother.
    if std::env::consts::OS == "windows" {
        let dll_folder_str = if is_release { "Release" } else { "Debug" };
        let dll_folder = build_folder.join(dll_folder_str);
        println!("cargo:rustc-link-search=native={}", dll_folder.display());    
    }
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