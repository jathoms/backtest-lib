use std::{fs::File, marker::PhantomData, path::Path};

use arrow::{
    array::{AsArray, Int32Array, Int64Array, RecordBatch, RecordBatchReader, UInt64Array},
    compute::concat_batches,
};
use coalign::vector_mapping::{KeyContainer, KeyDomain, ValueContainer, ValueDomain};
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;

struct SecurityAxis<'a, K: KeyDomain> {
    backing: &'a [K],
    ordering: coalign::OrderingContainer<K, &'a [K]>,
    selected_idxs: Option<Vec<usize>>,
}
struct PeriodAxis<'a> {
    backing: &'a [u64],
    selected_idxs: Option<Vec<usize>>,
}
struct ByPeriod<V: ValueDomain> {
    table: RecordBatch,
    _marker: PhantomData<V>,
}
struct BySecurity<V: ValueDomain> {
    table: RecordBatch,
    _marker: PhantomData<V>,
}

pub struct PastView<'a, K: KeyDomain, V: ValueDomain> {
    security_axis: SecurityAxis<'a, K>,
    period_axis: PeriodAxis<'a>,
    by_period: ByPeriod<V>,
    by_security: BySecurity<V>,
}
pub fn read_parquet(path: &Path) -> Result<RecordBatch, anyhow::Error> {
    let file = File::open(path)?;
    let reader = ParquetRecordBatchReaderBuilder::try_new(file)?.build()?;
    let schema = reader.schema();
    let batches = reader.collect::<Result<Vec<_>, _>>()?;
    Ok(concat_batches(&schema, &batches)?)
}

impl<'a, K: KeyDomain, V: ValueDomain> PastView<'a, K, V> {}
