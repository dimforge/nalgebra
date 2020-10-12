/// Normalized device coordinates (NDC) system.
pub trait System: Send + Sync {}

/// Left handed with a depth range of -1 to 1.
#[derive(Default)]
pub struct LhNo {}

impl System for LhNo {}

/// Right handed with a depth range of 0 to 1.
#[derive(Default)]
pub struct RhZo {}

impl System for RhZo {}

// Aliases for commonly used NDCs.

/// OpenGL NDC (left handed with a depth range of -1 to 1).
pub type OpenGL = LhNo;

/// Vulkan NDC (right handed with a depth range of 0 to 1).
pub type Vulkan = RhZo;
