
use std::{env, path::PathBuf};

use cmake::Config;

fn main() {
    let out_path = PathBuf::from(env::var("OUT_DIR").expect("No out dir found"));
    compile_bindings(&out_path);

    let dst = Config::new("woolycore")
                 .build();
    let build_folder = dst.join("build");
    println!("cargo:rustc-link-search=native={}", build_folder.display());
    println!("cargo:rustc-link-search=native={}", dst.join("lib").display());
    // println!("cargo:rustc-link-search=native={}", build_folder.join("llama.cpp").join("common").display());
    // println!("cargo:rustc-link-search=native={}", build_folder.join("llama.cpp").join("src").display());
    // println!("cargo:rustc-link-search=native={}", build_folder.join("llama.cpp").join("ggml").join("src").display());
    println!("cargo:rustc-link-lib=dylib=woolycore");
    // println!("cargo:rustc-link-lib=static=woolycore_static");
    // println!("cargo:rustc-link-lib=dylib=c++");
    // println!("cargo:rustc-link-lib=static=llama");
    // println!("cargo:rustc-link-lib=static=ggml");


    // println!("cargo:rustc-link-lib:dylib:+whole-archive=ggml");
    // println!("cargo:rustc-link-lib:dylib:+whole-archive=llama");

    //println!("cargo:rustc-link-search=build");
    // println!("cargo:rustc-link-search=build/llama.cpp/src");
    // println!("cargo:rustc-link-search=build/llama.cpp/ggml/src");
    // println!("cargo:rustc-link-lib=static:+whole-archive=woolycore_static");
    // println!("cargo:rustc-link-lib=static:+whole-archive=ggml");
    // println!("cargo:rustc-link-lib=static:+whole-archive=llama");

    //println!("cargo:rustc-link-lib=dylib=woolycore");
    // println!("cargo:rustc-link-lib=framework=Accelerate");

    // println!("cargo:rustc-link-lib=framework=Metal");
    // println!("cargo:rustc-link-lib=framework=Foundation");
    // println!("cargo:rustc-link-lib=framework=MetalPerformanceShaders");
    // println!("cargo:rustc-link-lib=framework=MetalKit");
    // println!("cargo:rustc-link-lib=framework=Accelerate");

    //println!("cargo:rustc-link-arg=-lstdc++");
    //println!("cargo:rustc-cfg=target-feature=\"crt-static\"");
    
    
    //  /out/build/llama.cpp/ggml/src   ;   /out/build/llama.cpp/src   ; /out/build/llama.cpp/common

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