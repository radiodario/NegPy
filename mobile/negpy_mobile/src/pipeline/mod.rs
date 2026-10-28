//! The stages of `DarkroomEngine.process()`, in its order: base (geometry + normalization),
//! exposure, clahe, lab, alt process, toning, crop, finish, output encode.

pub mod altprocess;
pub mod constants;
pub mod exposure;
pub mod finish;
pub mod geometry;
pub mod lab;
pub mod normalization;
pub mod papers;
pub mod toning;
pub mod working;
