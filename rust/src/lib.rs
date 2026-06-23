use pyo3::prelude::*;
mod universe_mapping;

use universe_mapping::{PySecurityAxis, PyUniverseMapping};

#[pyfunction]
fn sum_as_string(a: usize, b: usize) -> PyResult<String> {
    Ok((a + b).to_string())
}

/// A Python module implemented in Rust.
#[pymodule]
mod _rs {
    #[pymodule_export]
    use super::{sum_as_string, PySecurityAxis, PyUniverseMapping};
}
