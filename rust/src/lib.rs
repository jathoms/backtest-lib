use pyo3::prelude::*;
mod universe_mapping;

use universe_mapping::{PySecurityAxis, PyUniverseMapping};

/// A Python module implemented in Rust.
#[pymodule]
mod _rs {
    #[pymodule_export]
    use super::{PySecurityAxis, PyUniverseMapping};
}
