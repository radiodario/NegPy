//! The stages of `DarkroomEngine.process()`, in its order: base (geometry + normalization),
//! exposure, clahe, lab, alt process, toning, crop, finish, output encode.

pub mod constants;
pub mod papers;
pub mod working;
