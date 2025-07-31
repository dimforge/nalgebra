use crate::CsrMatrix;
use serde::{Deserialize, Deserializer, Serialize, Serializer, de};

/// This is an intermediate type for (de)serializing `CsrMatrix`.
///
/// Deserialization requires using a `try_from_*` function for validation. We could have used
/// the `remote = "Self"` trick (https://github.com/serde-rs/serde/issues/1220) which allows
/// to directly serialize/deserialize the original fields and combine it with validation.
/// However, this would lead to nested serialization of the `CsMatrix` and `SparsityPattern`
/// types. Instead, we decided that we want a more human-readable serialization format using
/// field names like `row_offsets` and `cal_indices`. The easiest way to achieve this is to
/// introduce an intermediate type. It also allows the serialization format to stay constant
/// even if the internal layout in `nalgebra` changes.
///
/// We want to avoid unnecessary copies when serializing (i.e. cloning slices into owned
/// storage). Therefore, we use generic arguments to allow using slices during serialization and
/// owned storage (i.e. `Vec`) during deserialization. Without a major update of serde, slices
/// and `Vec`s should always (de)serialize identically.
#[derive(Serialize, Deserialize)]
struct CsrMatrixSerializationData<Indices, Values> {
    nrows: usize,
    ncols: usize,
    row_offsets: Indices,
    col_indices: Indices,
    values: Values,
}

impl<T> Serialize for CsrMatrix<T>
where
    T: Serialize + Clone,
{
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        CsrMatrixSerializationData::<&[usize], &[T]> {
            nrows: self.nrows(),
            ncols: self.ncols(),
            row_offsets: self.row_offsets(),
            col_indices: self.col_indices(),
            values: self.values(),
        }
        .serialize(serializer)
    }
}

impl<'de, T> Deserialize<'de> for CsrMatrix<T>
where
    T: Deserialize<'de> + Clone,
{
    fn deserialize<D>(deserializer: D) -> Result<CsrMatrix<T>, D::Error>
    where
        D: Deserializer<'de>,
    {
        let de = CsrMatrixSerializationData::<Vec<usize>, Vec<T>>::deserialize(deserializer)?;
        CsrMatrix::try_from_csr_data(
            de.nrows,
            de.ncols,
            de.row_offsets,
            de.col_indices,
            de.values,
        )
        .map_err(|e| de::Error::custom(e))
    }
}
