/// Normalized device coordinates (NDC) system.
pub trait System: Send + Sync {}

/// Left handed with a depth range of -1 to 1.
#[derive(Default)]
pub struct LhNo {}

impl System for LhNo {}

/// Left handed with a depth range of 0 to 1.
#[derive(Default)]
pub struct LhZo {}

impl System for LhZo {}

/// Right handed with a depth range of -1 to 1.
#[derive(Default)]
pub struct RhNo {}

impl System for RhNo {}

/// Right handed with a depth range of 0 to 1.
#[derive(Default)]
pub struct RhZo {}

impl System for RhZo {}

// Aliases for commonly used NDCs.

/// OpenGL NDC (right handed with a depth range of -1 to 1).
pub type OpenGL = RhNo;

/// Vulkan NDC (right handed with a depth range of 0 to 1).
pub type Vulkan = RhZo;
