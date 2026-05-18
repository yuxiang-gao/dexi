//! Test: Python comparison (skipped if maturin/Python not available).

/// Full comparison requires Python with dex-retargeting and the dexi_rs extension
/// installed, which is not guaranteed in a Rust test environment. The executable
/// comparison script is kept in scripts/compare_python_rust.py and can be run from
/// an appropriately provisioned Python environment.
#[test]
fn test_python_comparison_script_exists() {
    let script = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .join("scripts/compare_python_rust.py");
    assert!(
        script.exists(),
        "Python/Rust comparison script is missing: {}",
        script.display()
    );
}
