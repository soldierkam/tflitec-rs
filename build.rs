extern crate bindgen;

use std::env;
use std::fmt::Debug;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::str::FromStr;
use strum_macros::{EnumString, IntoStaticStr};
use users::{get_effective_gid, get_effective_uid};

const TAG: &str = "2.11";
const BAZEL_COPTS_ENV_VAR: &str = "TFLITEC_BAZEL_COPTS";
const PREBUILT_PATH_ENV_VAR: &str = "TFLITEC_PREBUILT_PATH";
const HEADER_DIR_ENV_VAR: &str = "TFLITEC_HEADER_DIR";

#[derive(Debug, PartialEq, EnumString, IntoStaticStr)]
enum TargetOs {
    #[strum(serialize = "windows")]
    Windows,
    #[strum(serialize = "macos")]
    MacOS,
    #[strum(serialize = "ios")]
    iOS,
    #[strum(serialize = "linux")]
    Linux,
    #[strum(serialize = "android")]
    Android,
    #[strum(serialize = "freebsd")]
    FreeBSD,
    #[strum(serialize = "dragonfly")]
    Dragonfly,
    #[strum(serialize = "openbsd")]
    OpenBSD,
    #[strum(serialize = "netbsd")]
    NetBSD,
}

fn target_os() -> TargetOs {
    let v = env::var("CARGO_CFG_TARGET_OS").expect("Unable to get TARGET_OS");
    TargetOs::from_str(&v).expect(format!("Unsupported target OS: {}", v).as_ref())
}

#[derive(Debug, PartialEq, EnumString, IntoStaticStr)]
enum TargetArch {
    #[strum(serialize = "x86")]
    x86,
    #[strum(serialize = "x86_64")]
    x86_64,
    #[strum(serialize = "mips")]
    mips,
    #[strum(serialize = "powerpc")]
    powerpc,
    #[strum(serialize = "powerpc64")]
    powerpc64,
    #[strum(serialize = "arm")]
    arm,
    #[strum(serialize = "aarch64")]
    aarch64,
}

fn target_arch() -> TargetArch {
    let v = env::var("CARGO_CFG_TARGET_ARCH").expect("Unable to get TARGET_ARCH");
    TargetArch::from_str(&v).expect(format!("Unsupported target ARCH: {}", v).as_ref())
}

fn dll_extension() -> &'static str {
    match target_os() {
        TargetOs::MacOS => "dylib",
        TargetOs::Windows => "dll",
        _ => "so",
    }
}

fn dll_prefix() -> &'static str {
    match target_os() {
        TargetOs::Windows => "",
        _ => "lib",
    }
}

fn copy_or_overwrite<P: AsRef<Path> + Debug, Q: AsRef<Path> + Debug>(src: P, dest: Q) {
    println!("Copy {:?} -> {:?}", src, dest);
    let src_path: &Path = src.as_ref();
    let dest_path: &Path = dest.as_ref();
    if dest_path.exists() {
        if dest_path.is_file() {
            std::fs::remove_file(dest_path).expect("Cannot remove file");
        } else {
            std::fs::remove_dir_all(dest_path).expect("Cannot remove directory");
        }
    }
    if src_path.is_dir() {
        let options = fs_extra::dir::CopyOptions {
            copy_inside: true,
            ..fs_extra::dir::CopyOptions::new()
        };
        fs_extra::dir::copy(src_path, dest_path, &options).unwrap_or_else(|e| {
            panic!(
                "Cannot copy directory from {:?} to {:?}. Error: {}",
                src, dest, e
            )
        });
    } else {
        std::fs::copy(src_path, dest_path).unwrap_or_else(|e| {
            panic!(
                "Cannot copy file from {:?} to {:?}. Error: {}",
                src, dest, e
            )
        });
    }
}

fn normalized_target() -> Option<String> {
    env::var("TARGET")
        .ok()
        .map(|t| t.to_uppercase().replace('-', "_"))
}

/// Looks for the env var `var_${NORMALIZED_TARGET}`, and falls back to just `var` when
/// it is not found.
///
/// `NORMALIZED_TARGET` is the target triple which is converted to uppercase and underscores.
fn get_target_dependent_env_var(var: &str) -> Option<String> {
    if let Some(target) = normalized_target() {
        if let Ok(v) = env::var(format!("{var}_{target}")) {
            return Some(v);
        }
    }
    env::var(var).ok()
}

fn test_python_bin(python_bin_path: &str) -> bool {
    println!("Testing Python at {}", python_bin_path);
    let success = std::process::Command::new(python_bin_path)
        .args(["-c", "import numpy, importlib.util"])
        .status()
        .map(|s| s.success())
        .unwrap_or_default();
    if success {
        println!("Using Python at {}", python_bin_path);
    }
    success
}

fn get_python_bin_path() -> Option<PathBuf> {
    if let Ok(val) = env::var("PYTHON_BIN_PATH") {
        if !test_python_bin(&val) {
            panic!("Given Python binary failed in test!")
        }
        Some(PathBuf::from(val))
    } else {
        let bin = if target_os() == TargetOs::Windows {
            "where"
        } else {
            "which"
        };
        if let Ok(x) = std::process::Command::new(bin).arg("python3").output() {
            for path in String::from_utf8(x.stdout).unwrap().lines() {
                if test_python_bin(path) {
                    return Some(PathBuf::from(path));
                }
                println!("cargo:warning={:?} failed import test", path)
            }
        }
        if let Ok(x) = std::process::Command::new(bin).arg("python").output() {
            for path in String::from_utf8(x.stdout).unwrap().lines() {
                if test_python_bin(path) {
                    return Some(PathBuf::from(path));
                }
                println!("cargo:warning={:?} failed import test", path)
            }
            None
        } else {
            None
        }
    }
}

fn check_and_set_envs() {
    let python_bin_path = get_python_bin_path().expect(
        "Cannot find Python binary having required packages. \
        Make sure that `which python3` or `which python` points to a Python3 binary having numpy \
        installed. Or set PYTHON_BIN_PATH to the path of that binary.",
    );
    let os = target_os();
    let default_envs = [
        ["PYTHON_BIN_PATH", python_bin_path.to_str().unwrap()],
        ["USE_DEFAULT_PYTHON_LIB_PATH", "1"],
        ["TF_NEED_OPENCL", "0"],
        ["TF_CUDA_CLANG", "0"],
        ["TF_NEED_TENSORRT", "0"],
        ["TF_DOWNLOAD_CLANG", "0"],
        ["TF_NEED_MPI", "0"],
        ["TF_NEED_ROCM", "0"],
        ["TF_NEED_CUDA", "0"],
        ["TF_OVERRIDE_EIGEN_STRONG_INLINE", "1"], // Windows only
        ["CC_OPT_FLAGS", "-Wno-sign-compare"],
        [
            "TF_SET_ANDROID_WORKSPACE",
            if os == TargetOs::Android { "1" } else { "0" },
        ],
        [
            "TF_CONFIGURE_IOS",
            if os == TargetOs::iOS { "1" } else { "0" },
        ],
    ];
    for kv in default_envs {
        let name = kv[0];
        let val = kv[1];
        if env::var(name).is_err() {
            env::set_var(name, val);
        }
    }
    let true_vals = ["1", "t", "true", "y", "yes"];
    if true_vals.contains(&env::var("TF_SET_ANDROID_WORKSPACE").unwrap().as_str()) {
        let android_env_vars = [
            "ANDROID_NDK_HOME",
            "ANDROID_NDK_API_LEVEL",
            "ANDROID_SDK_HOME",
            "ANDROID_API_LEVEL",
            "ANDROID_BUILD_TOOLS_VERSION",
        ];
        for name in android_env_vars {
            env::var(name)
                .unwrap_or_else(|_| panic!("{} should be set to build for Android", name));
        }
    }
}

fn lib_output_path(name: &str, ios_name: &str) -> PathBuf {
    if target_os() != TargetOs::iOS {
        let ext = dll_extension();
        let lib_prefix = dll_prefix();
        out_dir().join(format!("{}{}.{}", lib_prefix, name, ext))
    } else {
        out_dir().join(format!("{}.framework", ios_name))
    }
}

fn build_tensorflow_with_docker(
    tf_src_path: &Path,
    lib_output_path: &Path,
    arch: &TargetArch,
    os: &TargetOs,
) {
    let target_os = target_os();
    let ext = dll_extension();
    let lib_prefix = dll_prefix();

    let bazel_target = "//tensorflow/lite/c:tensorflowlite_c";
    println!("Target OS: {}", Into::<&str>::into(os));
    println!("Target Arch: {}", Into::<&str>::into(arch));
    let mut bazel_cmd = "bazel build".to_owned();

    // Configure XNNPACK flags
    // In r2.6, it is enabled for some OS such as Windows by default.
    // To enable it by feature flag, we disable it by default on all platforms.
    #[cfg(not(feature = "xnnpack"))]
    bazel_cmd.push_str(" --define tflite_with_xnnpack=false");
    #[cfg(any(feature = "xnnpack_qu8", feature = "xnnpack_qs8"))]
    bazel_cmd.push_str(" --define tflite_with_xnnpack=true");
    #[cfg(feature = "xnnpack_qs8")]
    bazel_cmd.push_str(" --define xnn_enable_qs8=true");
    #[cfg(feature = "xnnpack_qu8")]
    bazel_cmd.push_str(" --define xnn_enable_qu8=true");

    if let Some(copts) = get_target_dependent_env_var(BAZEL_COPTS_ENV_VAR) {
        let copts = copts.split_ascii_whitespace();
        for opt in copts {
            bazel_cmd.push_str(format!(" --copt={}", opt).as_str());
        }
    }

    if target_os == TargetOs::iOS {
        bazel_cmd.push_str(" --apple_bitcode=embedded --copt=-fembed-bitcode");
    }
    bazel_cmd.push_str(" --config=linux ");
    bazel_cmd.push_str(bazel_target);
    bazel_cmd.push_str(" && cp bazel-bin/tensorflow/lite/c/libtensorflowlite_c.so /tensorflow");
    let mut docker = std::process::Command::new("docker");
    docker.args([
        "run",
        "-w",
        "/tensorflow",
        "-v",
        format!("{}:/tensorflow", tf_src_path.to_str().unwrap()).as_ref(),
        "--rm",
        "-e",
        format!("HOST_PERMS={}:{}", get_effective_uid(), get_effective_gid()).as_ref(),
        "tensorflow/tensorflow:devel",
        "bash",
        "-c",
        bazel_cmd.as_str(),
    ]);

    println!("Bazel Build Command: {:?}", docker);
    if !docker.status().expect("Cannot execute bazel").success() {
        panic!("Cannot build TensorFlowLiteC");
    }
    let bazel_output_path_buf =
        PathBuf::from(tf_src_path).join(format!("{}tensorflowlite_c.{}", lib_prefix, ext));
    if !bazel_output_path_buf.exists() {
        panic!(
            "Library/Framework not found in {}",
            bazel_output_path_buf.display()
        )
    }
    if target_os != TargetOs::iOS {
        copy_or_overwrite(&bazel_output_path_buf, lib_output_path);
        if target_os == TargetOs::Windows {
            let mut bazel_output_winlib_path_buf = bazel_output_path_buf;
            bazel_output_winlib_path_buf.set_extension("dll.if.lib");
            let winlib_output_path_buf = out_dir().join("tensorflowlite_c.lib");
            copy_or_overwrite(bazel_output_winlib_path_buf, winlib_output_path_buf);
        }
    } else {
        if lib_output_path.exists() {
            std::fs::remove_dir_all(lib_output_path).unwrap();
        }
        let mut unzip = std::process::Command::new("unzip");
        unzip.args([
            "-q",
            bazel_output_path_buf.to_str().unwrap(),
            "-d",
            out_dir().to_str().unwrap(),
        ]);
        unzip.status().expect("Cannot execute unzip");
    }
}

#[derive(Debug, PartialEq, EnumString)]
enum TensorflowCPU {
    #[strum(serialize = "k8")]
    LINUX_K8,
    #[strum(serialize = "armv7a")]
    LINUX_ARM7,
    #[strum(serialize = "aarch64")]
    LINUX_AARCH64,
    #[strum(serialize = "darwin_arm64")]
    DARWIN_ARM64,
    #[strum(serialize = "darwin_x86_64")]
    DARWIN_64,
    #[strum(serialize = "x64_windows")]
    WINDOWS_64,
}

fn tf_cpu(arch: TargetArch, os: TargetOs) -> TensorflowCPU {
    match os {
        TargetOs::Linux => match arch {
            TargetArch::x86_64 => TensorflowCPU::LINUX_K8,
            TargetArch::arm => TensorflowCPU::LINUX_ARM7,
            TargetArch::aarch64 => TensorflowCPU::LINUX_AARCH64,
            _ => panic!(format!("Unsupported arch {:?} for os {:?}", arch, os)),
        },
        _ => panic!(format!("Unsupported arch {:?} for os {:?}", arch, os)),
    }
}

fn build_libedgetpu_with_docker(
    edgetpu_src_path: &str,
    lib_output_path: &Path,
    arch: &TargetArch,
    os: &TargetOs,
) {
    //$ DOCKER_CPUS="k8" DOCKER_IMAGE="ubuntu:18.04" DOCKER_TARGETS=libedgetpu make docker-build
    //$ DOCKER_CPUS="armv7a aarch64" DOCKER_IMAGE="debian:stretch" DOCKER_TARGETS=libedgetpu make docker-build

    let mut make = std::process::Command::new("make");
    make.arg("docker-build");
    make.env("DOCKER_CPUS", "k8");
    make.env("DOCKER_IMAGE", "ubuntu:18.04");
    make.env("DOCKER_TARGETS", "libedgetpu");

    make.current_dir(edgetpu_src_path);

    println!("Build Command: {:?}", make);
    if !make.status().expect("Cannot execute make").success() {
        panic!("Cannot build libedgetpu");
    }
    let make_output_path_buf = PathBuf::from(edgetpu_src_path)
        .join("out")
        .join("direct")
        .join("k8")
        .join("libedgetpu.so.1.0");
    if !make_output_path_buf.exists() {
        panic!(
            "Library/Framework not found in {}",
            make_output_path_buf.display()
        )
    }
    copy_or_overwrite(make_output_path_buf, lib_output_path);
}

fn out_dir() -> PathBuf {
    PathBuf::from(env::var("OUT_DIR").unwrap())
}

fn prepare_for_docsrs() {
    // Docs.rs cannot access to network, use resource files
    let library_path = out_dir().join("libtensorflowlite_c.so");
    let bindings_path = out_dir().join("bindings.rs");

    let mut unzip = std::process::Command::new("unzip");
    let root = std::path::PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    unzip
        .arg(root.join("build-res/docsrs_res.zip"))
        .arg("-d")
        .arg(out_dir());
    if !(unzip
        .status()
        .unwrap_or_else(|_| panic!("Cannot execute unzip"))
        .success()
        && library_path.exists()
        && bindings_path.exists())
    {
        panic!("Cannot extract docs.rs resources")
    }
}

fn generate_bindings(tf_src_path: PathBuf, edgetpu_src_path: PathBuf) {
    let mut builder = bindgen::Builder::default().header(
        tf_src_path
            .join("tensorflow/lite/c/c_api.h")
            .to_str()
            .unwrap(),
    );
    if cfg!(feature = "xnnpack") {
        builder = builder.header(
            tf_src_path
                .join("tensorflow/lite/delegates/xnnpack/xnnpack_delegate.h")
                .to_str()
                .unwrap(),
        );
    }
    if cfg!(feature = "edgetpu") {
        // https://github.com/google-coral/libedgetpu.git
        builder = builder.header(
            edgetpu_src_path
                .join("tflite/public/edgetpu_c.h")
                .to_str()
                .unwrap(),
        );
    }

    let bindings = builder
        .clang_arg(format!("-I{}", tf_src_path.to_str().unwrap()))
        .clang_arg(format!("-I{}", edgetpu_src_path.to_str().unwrap()))
        // Tell cargo to invalidate the built crate whenever any of the
        // included header files changed.
        .parse_callbacks(Box::new(bindgen::CargoCallbacks))
        // Finish the builder and generate the bindings.
        .generate()
        // Unwrap the Result and panic on failure.
        .expect("Unable to generate bindings");

    // Write the bindings to the $OUT_DIR/bindings.rs file.
    bindings
        .write_to_file(out_dir().join("bindings.rs"))
        .expect("Couldn't write bindings!");
}

fn install_prebuilt(prebuilt_tflitec_path: &str, tf_src_path: &Path, lib_output_path: &PathBuf) {
    // Copy prebuilt library to given path
    {
        let prebuilt_tflitec_path = PathBuf::from(prebuilt_tflitec_path);
        // Copy .{so,dylib,dll,Framework} file
        copy_or_overwrite(&prebuilt_tflitec_path, lib_output_path);

        if target_os() == TargetOs::Windows {
            // Copy .lib file
            let mut prebuilt_lib_path = prebuilt_tflitec_path;
            prebuilt_lib_path.set_extension("lib");
            if !prebuilt_lib_path.exists() {
                panic!("A prebuilt windows .dll file must have the corresponding .lib file under the same directory!")
            }
            let mut lib_file_path = lib_output_path.clone();
            lib_file_path.set_extension("lib");
            copy_or_overwrite(prebuilt_lib_path, lib_file_path);
        }
    }

    copy_or_download_headers(
        tf_src_path,
        &[
            "tensorflow/lite/c/c_api.h",
            "tensorflow/lite/c/c_api_types.h",
        ],
    );
    if cfg!(feature = "xnnpack") {
        copy_or_download_headers(
            tf_src_path,
            &[
                "tensorflow/lite/delegates/xnnpack/xnnpack_delegate.h",
                "tensorflow/lite/c/common.h",
            ],
        );
    }
}

fn copy_or_download_headers(tf_src_path: &Path, file_paths: &[&str]) {
    if let Some(header_src_dir) = get_target_dependent_env_var(HEADER_DIR_ENV_VAR) {
        copy_headers(Path::new(&header_src_dir), tf_src_path, file_paths)
    } else {
        download_headers(tf_src_path, file_paths)
    }
}

fn copy_headers(header_src_dir: &Path, tf_src_path: &Path, file_paths: &[&str]) {
    // Download header files from Github
    for file_path in file_paths {
        let dst_path = tf_src_path.join(file_path);
        if dst_path.exists() {
            continue;
        }
        if let Some(p) = dst_path.parent() {
            std::fs::create_dir_all(p).expect("Cannot generate header dir");
        }
        copy_or_overwrite(header_src_dir.join(file_path), dst_path);
    }
}

fn download_headers(tf_src_path: &Path, file_paths: &[&str]) {
    // Download header files from Github
    for file_path in file_paths {
        let download_path = tf_src_path.join(file_path);
        if download_path.exists() {
            continue;
        }
        if let Some(p) = download_path.parent() {
            std::fs::create_dir_all(p).expect("Cannot generate header dir");
        }
        let url = format!(
            "https://raw.githubusercontent.com/tensorflow/tensorflow/{}/{}",
            TAG, file_path
        );
        download_file(&url, download_path.as_path());
    }
}

fn download_file(url: &str, path: &Path) {
    let mut easy = curl::easy::Easy::new();
    let output_file = std::fs::File::create(path).unwrap();
    let mut writer = std::io::BufWriter::new(output_file);
    easy.url(url).unwrap();
    easy.write_function(move |data| Ok(writer.write(data).unwrap()))
        .unwrap();
    easy.perform().unwrap_or_else(|e| {
        std::fs::remove_file(path).unwrap(); // Delete corrupted or empty file
        panic!("Error occurred while downloading from {}: {:?}", url, e);
    });
}

fn main() {
    {
        let env_vars = [
            BAZEL_COPTS_ENV_VAR,
            PREBUILT_PATH_ENV_VAR,
            HEADER_DIR_ENV_VAR,
        ];
        for env_var in env_vars {
            println!("cargo:rerun-if-env-changed={env_var}");
            if let Some(target) = normalized_target() {
                println!("cargo:rerun-if-env-changed={env_var}_{target}");
            }
        }
    }

    let out_path = out_dir();
    let src_dir = PathBuf::from(env::var_os("CARGO_MANIFEST_DIR").unwrap());
    let os = target_os();
    let arch = target_arch();
    if os != TargetOs::iOS {
        println!("cargo:rustc-link-search=native={}", out_path.display());
        println!("cargo:rustc-link-lib=dylib=tensorflowlite_c");
        println!("cargo:rustc-link-lib=dylib=edgetpu_c");
    } else {
        println!("cargo:rustc-link-search=framework={}", out_path.display());
        println!("cargo:rustc-link-lib=framework=TensorFlowLiteC");
        println!("cargo:rustc-link-lib=framework=EdgeTpuC");
    }
    if env::var("DOCS_RS") == Ok(String::from("1")) {
        // docs.rs cannot access to network, use resource files
        prepare_for_docsrs();
    } else {
        let tf_src_path = src_dir.join("tensorflow");
        let edgetpu_src_path = src_dir.join("libedgetpu");
        let tensorflow_lib_output_path = lib_output_path("tensorflowlite_c", "TensorFlowLiteC");
        let edgetpu_lib_output_path = lib_output_path("edgetpu_c", "EdgeTpuC");

        if let Some(prebuilt_tflitec_path) = get_target_dependent_env_var(PREBUILT_PATH_ENV_VAR) {
            install_prebuilt(
                &prebuilt_tflitec_path,
                &tf_src_path,
                &tensorflow_lib_output_path,
            );
        } else {
            // Build from source
            check_and_set_envs();
            build_libedgetpu_with_docker(
                edgetpu_src_path.to_str().unwrap(),
                edgetpu_lib_output_path.as_path(),
                &arch,
                &os,
            );
            build_tensorflow_with_docker(
                tf_src_path.as_path(),
                tensorflow_lib_output_path.as_path(),
                &arch,
                &os,
            );
        }

        // Generate bindings using headers
        generate_bindings(tf_src_path, edgetpu_src_path);
    }
}
