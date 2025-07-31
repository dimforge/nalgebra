use crate::pattern::SparsityPattern;
use serde::{Deserialize, Deserializer, Serialize, Serializer, de};

/// This is an intermediate type for (de)serializing `SparsityPattern`.
///
/// Deserialization requires using a `try_from_*` function for validation. We could have used
/// the `remote = "Self"` trick (https://github.com/serde-rs/serde/issues/1220) which allows
/// to directly serialize/deserialize the original fields and combine it with validation.
/// However, this would lead to nested serialization of the `CsMatrix` and `SparsityPattern`
/// types. Instead, we decided that we want a more human-readable serialization format using
/// field names like `major_offsets` and `minor_indices`. The easiest way to achieve this is to
/// introduce an intermediate type. It also allows the serialization format to stay constant
/// even when the internal layout in `nalgebra` changes.
///
/// We want to avoid unnecessary copies when serializing (i.e. cloning slices into owned
/// storage). Therefore, we use generic arguments to allow using slices during serialization and
/// owned storage (i.e. `Vec`) during deserialization. Without a major update of serde, slices
/// and `Vec`s should always (de)serialize identically.
#[derive(Serialize, Deserialize)]
struct SparsityPatternSerializationData<Indices> {
    major_dim: usize,
    minor_dim: usize,
    major_offsets: Indices,
    minor_indices: Indices,
}

impl Serialize for SparsityPattern {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        SparsityPatternSerializationData::<&[usize]> {
            major_dim: self.major_dim(),
            minor_dim: self.minor_dim(),
            major_offsets: self.major_offsets(),
            minor_indices: self.minor_indices(),
        }
        .serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for SparsityPattern {
    fn deserialize<D>(deserializer: D) -> Result<SparsityPattern, D::Error>
    where
        D: Deserializer<'de>,
    {
        let de = SparsityPatternSerializationData::<Vec<usize>>::deserialize(deserializer)?;
        SparsityPattern::try_from_offsets_and_indices(
            de.major_dim,
            de.minor_dim,
            de.major_offsets,
            de.minor_indices,
        )
        .map_err(|e| de::Error::custom(e))
    }
}
