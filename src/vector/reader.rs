//! Per-segment vector reader.
//!
//! Composes a [`FlatVecReader`] and an [`IvfVecReader`]. Callers —
//! primarily [`VectorBackend::for_segment`](super::backend::VectorBackend::for_segment)
//! and the flat-format merge routine — ask for a field's column via
//! [`VectorColumnReader::open_column`].

use std::collections::BTreeMap;

use super::flat::{FlatVecReader, FlatVectorColumn};
use super::ivf::{IvfVecReader, IvfVectorColumn};
use crate::schema::{Field, FieldType};
use crate::{DocId, SegmentReader, TantivyError};

pub trait VectorColumnReader {
    type Column;

    fn open_column(&self, field: Field) -> crate::Result<Self::Column>;

    fn count(&self, field: Field) -> crate::Result<usize>;

    fn dim(&self, field: Field) -> crate::Result<usize>;
}

pub struct VectorReader {
    flat: FlatVecReader,
    ivf: IvfVecReader,
    vector_dims: BTreeMap<Field, usize>,
}

impl VectorReader {
    pub(crate) fn open(segment_reader: &SegmentReader) -> crate::Result<Self> {
        let schema = segment_reader.schema();
        let mut vector_dims = BTreeMap::new();
        for (field, entry) in schema.fields() {
            if let FieldType::Vector(opts) = entry.field_type() {
                vector_dims.insert(field, opts.dim());
            }
        }
        Ok(Self {
            flat: FlatVecReader::open(segment_reader)?,
            ivf: IvfVecReader::stub(&schema),
            vector_dims,
        })
    }
}

impl VectorColumnReader for VectorReader {
    type Column = VectorColumn;

    fn open_column(&self, field: Field) -> crate::Result<VectorColumn> {
        if !self.vector_dims.contains_key(&field) {
            return Err(TantivyError::InvalidArgument(format!(
                "field {field:?} is not a vector field"
            )));
        }
        if self.ivf.has_column(field) {
            return self.ivf.open_column(field).map(VectorColumn::Ivf);
        }
        if self.flat.has_column(field) {
            return self.flat.open_column(field).map(VectorColumn::Flat);
        }
        Err(TantivyError::InternalError(format!(
            "no vector data for vector field {field:?} in segment"
        )))
    }

    fn count(&self, field: Field) -> crate::Result<usize> {
        if !self.vector_dims.contains_key(&field) {
            return Err(TantivyError::InvalidArgument(format!(
                "field {field:?} is not a vector field"
            )));
        }
        if self.ivf.has_column(field) {
            return self.ivf.count(field);
        }
        if self.flat.has_column(field) {
            return self.flat.count(field);
        }
        Err(TantivyError::InternalError(format!(
            "no vector data for vector field {field:?} in segment"
        )))
    }

    fn dim(&self, field: Field) -> crate::Result<usize> {
        self.vector_dims.get(&field).copied().ok_or_else(|| {
            TantivyError::InvalidArgument(format!("field {field:?} is not a vector field"))
        })
    }
}

pub enum VectorColumn {
    Flat(FlatVectorColumn),
    Ivf(IvfVectorColumn),
}

impl VectorColumn {
    pub fn dim(&self) -> usize {
        match self {
            Self::Flat(column) => column.dim(),
            Self::Ivf(column) => column.dim(),
        }
    }

    pub fn len(&self) -> usize {
        match self {
            Self::Flat(column) => column.len(),
            Self::Ivf(column) => column.len(),
        }
    }

    pub fn is_empty(&self) -> bool {
        match self {
            Self::Flat(column) => column.is_empty(),
            Self::Ivf(column) => column.is_empty(),
        }
    }

    pub fn contains(&self, doc_id: DocId) -> bool {
        match self {
            Self::Flat(column) => column.contains(doc_id),
            Self::Ivf(column) => column.contains(doc_id),
        }
    }

    pub fn vector_bytes_at(&self, doc_id: DocId) -> Option<&[u8]> {
        match self {
            Self::Flat(column) => column.vector_bytes_at(doc_id),
            Self::Ivf(column) => column.vector_bytes_at(doc_id),
        }
    }
}
