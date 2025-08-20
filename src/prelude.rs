// Copyright 2024 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

pub use anyhow::Result;
pub use anyhow::anyhow;
pub use anyhow::bail;
pub use tracing::debug;
pub use tracing::debug_span;
pub use tracing::error;
pub use tracing::error_span;
pub use tracing::field;
pub use tracing::info;
pub use tracing::info_span;
pub use tracing::instrument;
pub use tracing::span;
pub use tracing::trace;
pub use tracing::trace_span;
pub use tracing::warn;
pub use tracing::warn_span;

pub use crate::error_utils::Location;
pub use crate::error_utils::LocationContextExt;
pub use crate::error_utils::LogAndIgnoreExt;
pub use crate::error_utils::LogExt;
pub use crate::error_utils::fname;
pub use crate::error_utils::loc;
pub use crate::error_utils::log_and_continue;
pub use crate::error_utils::log_and_return;
pub use crate::error_utils::warn_and_return;
