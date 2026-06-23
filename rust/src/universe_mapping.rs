use coalign::{
    errors::CoalignError,
    vector_mapping::VectorMapping,
    vector_ops::{VectorDiv, VectorOp, VectorSub},
};
use polars_arrow::array::PrimitiveArray;
use polars_buffer::Buffer;
use polars_core::prelude::{CompatLevel, DataType, Float64Chunked, IntoSeries, PlSmallStr, Series};
use pyo3::{
    IntoPyObjectExt,
    exceptions::{PyIndexError, PyKeyError, PyValueError},
    prelude::*,
    types::PyAny,
};
use pyo3_polars::types::PySeries;
use std::sync::Arc;

type Keys = Arc<[Arc<str>]>;
type Values = Buffer<f64>;

#[derive(FromPyObject)]
enum MappingOrF64<'py> {
    Mapping(PyRef<'py, PyUniverseMapping>),
    F64(f64),
}

#[pyclass(skip_from_py_object)]
#[derive(Debug, Clone)]
pub struct PySecurityAxis {
    keys: Keys,
}

impl PySecurityAxis {
    fn from_keys(keys: Keys) -> Self {
        Self { keys }
    }

    fn keys_arc(&self) -> Keys {
        Arc::clone(&self.keys)
    }
}

#[pymethods]
impl PySecurityAxis {
    #[new]
    fn new(keys: Vec<String>) -> Self {
        let keys = keys
            .into_iter()
            .map(Arc::<str>::from)
            .collect::<Vec<_>>()
            .into();
        Self { keys }
    }

    fn __len__(&self) -> usize {
        self.keys.len()
    }

    fn __repr__(&self) -> String {
        format!("PySecurityAxis(len={})", self.keys.len())
    }

    fn keys(&self) -> Vec<String> {
        self.keys.iter().map(|s| s.to_string()).collect()
    }

    fn take(&self, indices: Vec<usize>) -> PyResult<Self> {
        let mut keys = Vec::with_capacity(indices.len());
        for idx in indices {
            let key = self
                .keys
                .get(idx)
                .ok_or_else(|| PyIndexError::new_err(idx.to_string()))?;
            keys.push(Arc::clone(key));
        }
        Ok(Self::from_keys(keys.into()))
    }
}

#[pyclass(skip_from_py_object)]
#[derive(Debug, Clone)]
pub struct PyUniverseMapping {
    inner: VectorMapping<Arc<str>, f64, Keys, Values>,
}

impl PyUniverseMapping {
    fn from_vec_result(res: VectorMapping<Arc<str>, f64, Keys, Vec<f64>>) -> Self {
        let (ordering, values) = res.into_parts();
        let inner = VectorMapping::<_, _, _, Buffer<f64>>::from_parts(ordering, values.into());
        Self { inner }
    }

    fn coalign_err(err: CoalignError) -> PyErr {
        PyValueError::new_err(err.to_string())
    }

    fn apply_scalar_left<O: VectorOp>(&self, scalar: f64) -> Self {
        let values = self
            .inner
            .values()
            .iter()
            .map(|value| O::apply_scalar_scalar(scalar, *value))
            .collect::<Vec<_>>();
        let (ordering, _) = self.inner.clone().into_parts();
        let inner = VectorMapping::<_, _, _, Buffer<f64>>::from_parts(ordering, values.into());
        Self { inner }
    }

    fn make_series(values: Values) -> PySeries {
        let arr = PrimitiveArray::new(
            DataType::Float64.to_arrow(CompatLevel::newest()),
            values,
            None,
        );
        PySeries(Float64Chunked::with_chunk(PlSmallStr::from_static("series"), arr).into_series())
    }

    fn from_axis_and_values(axis: &PySecurityAxis, values: Values) -> PyResult<Self> {
        if axis.keys.len() != values.len() {
            return Err(PyValueError::new_err(
                "keys and values must have the same length",
            ));
        }

        Ok(Self {
            inner: VectorMapping::new(axis.keys_arc(), values),
        })
    }

    fn values_from_float64_chunked(ca: &Float64Chunked) -> PyResult<Values> {
        if ca.null_count() != 0 {
            return Err(PyValueError::new_err(
                "Polars series with nulls cannot back a native universe mapping",
            ));
        }
        if ca.len() == 0 {
            return Ok(Vec::<f64>::new().into());
        }

        let rechunked = ca.rechunk();
        let arr = rechunked.downcast_iter().next().ok_or_else(|| {
            PyValueError::new_err("Polars series unexpectedly had no backing chunks")
        })?;
        Ok(arr.values().clone())
    }

    fn values_from_series(series: Series) -> PyResult<Values> {
        let series = if series.dtype() == &DataType::Float64 {
            series
        } else {
            series
                .cast(&DataType::Float64)
                .map_err(|err| PyValueError::new_err(err.to_string()))?
        };
        let ca = series.try_f64().ok_or_else(|| {
            PyValueError::new_err("Polars series could not be converted to Float64")
        })?;
        Self::values_from_float64_chunked(ca)
    }
}

#[pymethods]
impl PyUniverseMapping {
    #[new]
    fn new(keys: Vec<String>, values: Vec<f64>) -> PyResult<Self> {
        let axis = PySecurityAxis::new(keys);
        Self::from_axis_and_values(&axis, values.into())
    }

    #[staticmethod]
    fn from_axis(axis: PyRef<'_, PySecurityAxis>, values: Vec<f64>) -> PyResult<Self> {
        Self::from_axis_and_values(&axis, values.into())
    }

    #[staticmethod]
    fn from_series(axis: PyRef<'_, PySecurityAxis>, series: PySeries) -> PyResult<Self> {
        let values = Self::values_from_series(series.0)?;
        Self::from_axis_and_values(&axis, values)
    }

    fn __len__(&self) -> usize {
        self.inner.len()
    }

    fn __getitem__(&self, py: Python<'_>, key: Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        if let Ok(key) = key.extract::<String>() {
            let value = self
                .inner
                .get(key.as_str())
                .copied()
                .ok_or_else(|| PyKeyError::new_err(key))?;
            return value.into_py_any(py);
        }

        let mut selected = Vec::new();
        for item in key.try_iter()? {
            let item = item?;
            let name = item.extract::<String>()?;
            let value = self
                .inner
                .get(name.as_str())
                .copied()
                .ok_or_else(|| PyKeyError::new_err(name))?;
            selected.push(value);
        }
        Ok(Self::make_series(selected.into()).into_py_any(py)?)
    }

    fn __repr__(&self) -> String {
        format!(
            "PyUniverseMapping(len={}, keys={:?})",
            self.inner.values().len(),
            self.inner.keys()
        )
    }

    fn keys(&self) -> Vec<String> {
        self.inner
            .keys()
            .iter()
            .map(|s| s.to_string())
            .collect::<Vec<_>>()
    }

    fn values(&self) -> PySeries {
        Self::make_series(self.inner.values().clone())
    }

    fn sum(&self) -> f64 {
        self.inner.values().iter().sum()
    }

    fn __add__(&self, rhs: MappingOrF64<'_>) -> PyResult<Self> {
        match rhs {
            MappingOrF64::Mapping(rhs) => self
                .inner
                .add(&rhs.inner)
                .map(Self::from_vec_result)
                .map_err(Self::coalign_err),
            MappingOrF64::F64(rhs) => Ok(Self::from_vec_result(self.inner.add_scalar(rhs))),
        }
    }

    fn __radd__(&self, rhs: MappingOrF64<'_>) -> PyResult<Self> {
        match rhs {
            MappingOrF64::Mapping(rhs) => rhs
                .inner
                .add(&self.inner)
                .map(Self::from_vec_result)
                .map_err(Self::coalign_err),
            MappingOrF64::F64(rhs) => Ok(Self::from_vec_result(self.inner.add_scalar(rhs))),
        }
    }

    fn __sub__(&self, rhs: MappingOrF64<'_>) -> PyResult<Self> {
        match rhs {
            MappingOrF64::Mapping(rhs) => self
                .inner
                .sub(&rhs.inner)
                .map(Self::from_vec_result)
                .map_err(Self::coalign_err),
            MappingOrF64::F64(rhs) => Ok(Self::from_vec_result(self.inner.sub_scalar(rhs))),
        }
    }

    fn __rsub__(&self, rhs: MappingOrF64<'_>) -> PyResult<Self> {
        match rhs {
            MappingOrF64::Mapping(rhs) => rhs
                .inner
                .sub(&self.inner)
                .map(Self::from_vec_result)
                .map_err(Self::coalign_err),
            MappingOrF64::F64(rhs) => Ok(self.apply_scalar_left::<VectorSub>(rhs)),
        }
    }

    fn __mul__(&self, rhs: MappingOrF64<'_>) -> PyResult<Self> {
        match rhs {
            MappingOrF64::Mapping(rhs) => self
                .inner
                .mul(&rhs.inner)
                .map(Self::from_vec_result)
                .map_err(Self::coalign_err),
            MappingOrF64::F64(rhs) => Ok(Self::from_vec_result(self.inner.mul_scalar(rhs))),
        }
    }

    fn __rmul__(&self, rhs: MappingOrF64<'_>) -> PyResult<Self> {
        match rhs {
            MappingOrF64::Mapping(rhs) => rhs
                .inner
                .mul(&self.inner)
                .map(Self::from_vec_result)
                .map_err(Self::coalign_err),
            MappingOrF64::F64(rhs) => Ok(Self::from_vec_result(self.inner.mul_scalar(rhs))),
        }
    }

    fn __truediv__(&self, rhs: MappingOrF64<'_>) -> PyResult<Self> {
        match rhs {
            MappingOrF64::Mapping(rhs) => self
                .inner
                .div(&rhs.inner)
                .map(Self::from_vec_result)
                .map_err(Self::coalign_err),
            MappingOrF64::F64(rhs) => Ok(Self::from_vec_result(self.inner.div_scalar(rhs))),
        }
    }

    fn __rtruediv__(&self, rhs: MappingOrF64<'_>) -> PyResult<Self> {
        match rhs {
            MappingOrF64::Mapping(rhs) => rhs
                .inner
                .div(&self.inner)
                .map(Self::from_vec_result)
                .map_err(Self::coalign_err),
            MappingOrF64::F64(rhs) => Ok(self.apply_scalar_left::<VectorDiv>(rhs)),
        }
    }
}
