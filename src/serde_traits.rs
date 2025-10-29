//! Traits for serializing and deserializing types to/from files and strings.
//!
//! These traits provide a common interface for serialization that can be implemented
//! using different serialization backends (e.g., serde_json).

use core::error::Error;
use std::{fs, path::Path};

/// Trait for types that can be serialized to strings and files.
pub trait Serialize {
    /// Serialize this value to a JSON string.
    fn to_string(&self) -> Result<String, Box<dyn Error>>;

    /// Serialize this value to a file at the given path.
    fn to_file<P: AsRef<Path>>(&self, path: P) -> Result<(), Box<dyn Error>> {
        fs::write(path, self.to_string()?)?;
        Ok(())
    }
}

/// Trait for types that can be deserialized from strings and files.
pub trait Deserialize: Sized {
    /// Deserialize this value from a JSON string.
    fn from_str(s: &str) -> Result<Self, Box<dyn Error>>;

    /// Deserialize this value from a file at the given path.
    fn from_file<P: AsRef<Path>>(path: P) -> Result<Self, Box<dyn Error>> {
        Self::from_str(&fs::read_to_string(path)?)
    }
}

// Generic implementation for types that implement serde::Serialize
#[cfg(feature = "serialize")]
impl<T> Serialize for T
where
    T: serde::Serialize,
{
    fn to_string(&self) -> Result<String, Box<dyn Error>> {
        Ok(serde_json::to_string(self)?)
    }
}

// Generic implementation for types that implement serde::Deserialize
#[cfg(feature = "serialize")]
impl<T> Deserialize for T
where
    T: for<'de> serde::Deserialize<'de>,
{
    fn from_str(s: &str) -> Result<Self, Box<dyn Error>> {
        serde_json::from_str(s).map_err(|op| op.into())
    }
}
