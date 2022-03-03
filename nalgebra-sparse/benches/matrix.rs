pub const ALL_MATRICES: [&'static str; 13] = [
    "cant",
    "conf5_4-8x8-05",
    "consph",
    "cop20k_A",
    "mac_econ_fwd500",
    "mc2depi",
    "pdb1HYS",
    "pwtk",
    "rail4284",
    "rma10",
    "scircuit",
    "shipsec1",
    "webbase-1M",
];

pub const INT_MATRICES: [&'static str; 2] = ["mc2depi", "rail4284"];

pub const REAL_MATRICES: [&'static str; 10] = [
    "cant",
    "consph",
    "cop20k_A",
    "mac_econ_fwd500",
    "pdb1HYS",
    "pwtk",
    "rma10",
    "scircuit",
    "shipsec1",
    "webbase-1M",
];

// Allow dead code here because we don't use complex here right now
// Because rand crate does not support random numbers yet, so we can't generate random complex number and matrix
#[allow(dead_code)]
pub const COMPLEX_MATRICES: [&'static str; 1] = ["conf5_4-8x8-05"];

// TODO: add more matrix collections, such as SYMMETRIC_MATRICES, PSD_MATRICES
