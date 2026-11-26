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

use std::env;
use std::fmt;
use std::fmt::Debug;
use std::fs;
use std::path::Path;
use std::path::PathBuf;
use std::process;
use std::str::FromStr;
use std::sync::atomic::AtomicBool;
use std::sync::atomic::Ordering;

use bpaf::Parser;
use optional_struct::Applicable;
use ron::Options;
use ron::extensions::Extensions;
use serde::Deserialize;
use serde::Serialize;
use tracing::Level;
use tracing::metadata::ParseLevelError;

use crate::prelude::*;
use crate::serialization::framing::DEFAULT_JPEG_QUALITY;
use crate::serialization::framing::JpegSubsamp;
use crate::serialization::framing::MAX_JPEG_QUALITY;
use crate::serialization::framing::MIN_JPEG_QUALITY;

pub trait Config: Debug + Default + Serialize {
    fn config_file(&self) -> PathBuf;

    fn print_default_and_exit() {
        println!(
            "{}",
            ron::ser::to_string_pretty::<Self>(
                &Default::default(),
                ron::ser::PrettyConfig::default()
            )
            .unwrap()
        );
        process::exit(0);
    }
}

pub trait OptionalConfig<Conf: Config>:
    Debug + Default + Applicable<Base = Conf> + for<'a> Deserialize<'a>
{
    fn parse_args() -> Self;
    fn print_default_config_and_exit(&self) -> Option<bool>;
    fn config_file(&self) -> Option<PathBuf>;

    fn read_from_file(&self) -> Option<Self> {
        let config_file = self
            .config_file()
            .unwrap_or_else(|| Conf::default().config_file());
        if !config_file.exists() {
            eprintln!("config file does not exist at {config_file:?}");
            return None;
        }

        let config_str = fs::read_to_string(&config_file).unwrap_or_else(|_| {
            panic!("config file at path {config_file:?} exists but there was an error reading it")
        });
        let config: Self = Options::default()
            .with_default_extension(Extensions::IMPLICIT_SOME)
            .from_str(&config_str)
            .expect("error parsing config file");
        eprintln!("config from file {config_file:?}: {config:#?}");
        Some(config)
    }
}

pub fn init_config<Conf: Config, OptConf: OptionalConfig<Conf>>() -> Conf {
    let mut config = Conf::default();
    let args = OptConf::parse_args();
    eprintln!("config from args: {args:#?}");

    // Do this before parsing the config file so a broken config file doesn't
    // prevent printing a new one to replace it.
    if let Some(true) = args.print_default_config_and_exit() {
        Conf::print_default_and_exit();
    }

    if let Some(config_from_file) = args.read_from_file() {
        config_from_file.apply_to(&mut config);
    }
    args.apply_to(&mut config);
    eprintln!("running config: {config:#?}");
    config
}

pub fn default_print_default_config_and_exit() -> bool {
    false
}

pub fn print_default_config_and_exit() -> impl Parser<Option<bool>> {
    bpaf::long("print-default-config-and-exit")
        .argument::<bool>("BOOL")
        .help("Print a configuration file with default values to stdout. Convenient for seeing default values and for generating a new config file by redirecting stdout to the config file location.")
        .optional()
}

fn fallback_config_parent_dir() -> Result<PathBuf> {
    Ok(Path::join(
        &home::home_dir().ok_or(anyhow!("unable to determine home dir"))?,
        ".config",
    ))
}

pub fn default_config_file_dir() -> PathBuf {
    Path::join(
        &env::var("XDG_CONFIG_HOME")
            .log(loc!())
            .ok()
            .map(Into::into)
            .or(fallback_config_parent_dir().log(loc!()).ok())
            .unwrap_or_else(|| "/etc".into()),
        "wprs",
    )
}

pub fn default_config_file(name: &str) -> PathBuf {
    Path::join(&default_config_file_dir(), format!("{name}.ron"))
}

pub fn config_file() -> impl Parser<Option<PathBuf>> {
    bpaf::long("config-file")
        .argument::<PathBuf>("PATH")
        .help("The path to the config file to use. Defaults to $XDG_CONFIG_HOME/wprs/${bin}.ron if $XDG_CONFIG_HOME is set, ~/.config/wprs/${bin}.ron if it doesn't, and /etc/wprs/${bin}.ron if we can't determine the user's home directory.")
        .optional()
}

pub fn default_wayland_display() -> String {
    "wprs-0".to_string()
}

pub fn wayland_display() -> impl Parser<Option<String>> {
    bpaf::long("wayland-display")
        .argument::<String>("NAME")
        .optional()
}

fn socket_dir() -> PathBuf {
    env::var_os("XDG_RUNTIME_DIR")
        .map(Into::into)
        .unwrap_or_else(|| {
            // Fall back to the numeric UID when `getpwuid` has no entry so
            // `Conf::default()` never panics before CLI `--socket` overrides apply.
            let user =
                whoami::username().unwrap_or_else(|_| nix::unistd::Uid::effective().to_string());
            Path::join(&env::temp_dir(), user)
        })
}

pub fn default_socket_path() -> PathBuf {
    Path::join(&socket_dir(), "wprs.sock")
}

pub fn socket() -> impl Parser<Option<PathBuf>> {
    bpaf::long("socket").argument::<PathBuf>("PATH").optional()
}

pub fn default_control_socket_path(prefix: &str) -> PathBuf {
    Path::join(&socket_dir(), format!("{prefix}-ctrl.sock"))
}

pub fn control_socket() -> impl Parser<Option<PathBuf>> {
    bpaf::long("control-socket")
        .argument::<PathBuf>("PATH")
        .optional()
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SerializableLevel(pub Level);

impl FromStr for SerializableLevel {
    type Err = ParseLevelError;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Ok(Self(Level::from_str(s)?))
    }
}

impl Serialize for SerializableLevel {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.serialize_str(self.0.as_str())
    }
}

impl<'de> Deserialize<'de> for SerializableLevel {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let s = String::deserialize(deserializer)?;
        Ok(Self(Level::from_str(&s).map_err(serde::de::Error::custom)?))
    }
}

pub fn stderr_log_level() -> impl Parser<Option<SerializableLevel>> {
    bpaf::long("stderr-log-level")
        .argument::<String>("LEVEL")
        .parse(|s| FromStr::from_str(&s))
        .optional()
}

pub fn file_log_level() -> impl Parser<Option<SerializableLevel>> {
    bpaf::long("stderr-log-level")
        .argument::<String>("LEVEL")
        .parse(|s| FromStr::from_str(&s))
        .optional()
}

pub fn log_file() -> impl Parser<Option<Option<PathBuf>>> {
    // let argv0 = PathBuf::from(env::args().next().unwrap());
    // let argv0_basename = Path::new(argv0.components().last().unwrap().as_os_str());
    // let tmp_dir = env::temp_dir();
    // let default_log_file_path = Path::join(&tmp_dir, argv0_basename.with_extension("log"));

    bpaf::long("log-file")
        .argument::<PathBuf>("PATH")
        .optional()
        .map(|log_file| log_file.map(Some))
}

pub fn framerate() -> impl Parser<Option<u32>> {
    bpaf::long("framerate").argument::<u32>("FPS").optional()
}

pub fn log_priv_data() -> impl Parser<Option<bool>> {
    bpaf::long("log-priv-data")
        .argument::<bool>("BOOL")
        .optional()
}

pub fn title_prefix() -> impl Parser<Option<String>> {
    bpaf::long("title-prefix")
        .argument::<String>("STRING")
        .help("Prefix windows titles with a string.")
        .optional()
}

/// CLI/config selector for the frame compression mode (`lossless` vs `hybrid`).
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Deserialize, Serialize)]
pub enum CompressionKind {
    #[default]
    Lossless,
    Hybrid,
}

impl fmt::Display for CompressionKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Lossless => "lossless",
            Self::Hybrid => "hybrid",
        })
    }
}

impl FromStr for CompressionKind {
    type Err = anyhow::Error;

    fn from_str(s: &str) -> Result<Self> {
        match s {
            "lossless" => Ok(Self::Lossless),
            "hybrid" => Ok(Self::Hybrid),
            _ => bail!("invalid compression mode {s:?}: expected \"lossless\" or \"hybrid\""),
        }
    }
}

/// Validated JPEG quality (`1..=100`, default `90`) serialized as a plain `u8`
/// in RON config files so out-of-range values fail at both CLI and config parse time.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct JpegQuality(u8);

impl JpegQuality {
    pub fn new(value: u8) -> Result<Self> {
        if !(MIN_JPEG_QUALITY..=MAX_JPEG_QUALITY).contains(&value) {
            bail!("invalid JPEG quality {value}: expected {MIN_JPEG_QUALITY}..={MAX_JPEG_QUALITY}");
        }
        Ok(Self(value))
    }

    pub const fn get(self) -> u8 {
        self.0
    }
}

impl Default for JpegQuality {
    fn default() -> Self {
        Self(DEFAULT_JPEG_QUALITY)
    }
}

impl fmt::Display for JpegQuality {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl FromStr for JpegQuality {
    type Err = anyhow::Error;

    fn from_str(s: &str) -> Result<Self> {
        let value = u8::from_str(s)
            .with_context(loc!(), || format!("invalid JPEG quality {s:?}: expected integer {MIN_JPEG_QUALITY}..={MAX_JPEG_QUALITY}"))?;
        Self::new(value)
    }
}

impl Serialize for JpegQuality {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.serialize_u8(self.0)
    }
}

impl<'de> Deserialize<'de> for JpegQuality {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let value = u8::deserialize(deserializer)?;
        Self::new(value).map_err(serde::de::Error::custom)
    }
}

pub fn compression_mode() -> impl Parser<Option<CompressionKind>> {
    bpaf::long("compression-mode")
        .argument::<String>("MODE")
        .help("Frame compression mode: lossless (default) or hybrid (TextSafe TurboJPEG).")
        .parse(|s| CompressionKind::from_str(&s))
        .optional()
}

pub fn jpeg_quality() -> impl Parser<Option<JpegQuality>> {
    bpaf::long("jpeg-quality")
        .argument::<String>("QUALITY")
        .help("JPEG quality (1..=100, default 90) used in hybrid compression mode.")
        .parse(|s| JpegQuality::from_str(&s))
        .optional()
}

pub fn jpeg_subsamp() -> impl Parser<Option<JpegSubsamp>> {
    bpaf::long("jpeg-subsamp")
        .argument::<String>("SUBSAMP")
        .help("JPEG chroma subsampling (444, 422, or 420; default 444) used in hybrid compression mode.")
        .parse(|s| JpegSubsamp::from_str(&s))
        .optional()
}

pub static LOG_PRIV_DATA: AtomicBool = AtomicBool::new(false);

pub fn set_log_priv_data(val: bool) {
    LOG_PRIV_DATA.store(val, Ordering::Relaxed);
}

pub fn get_log_priv_data() -> bool {
    LOG_PRIV_DATA.load(Ordering::Relaxed)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compression_cli_and_ron_parsing() {
        // CompressionKind Display / FromStr / RON
        assert_eq!(CompressionKind::default(), CompressionKind::Lossless);
        for (s, expected) in [
            ("lossless", CompressionKind::Lossless),
            ("hybrid", CompressionKind::Hybrid),
        ] {
            assert_eq!(CompressionKind::from_str(s).unwrap(), expected);
            assert_eq!(expected.to_string(), s);
        }
        for bad in ["", "Lossless", "jpeg", "lossy"] {
            assert!(CompressionKind::from_str(bad).is_err());
        }

        // JpegQuality bounds (`1..=100`), CLI parser, and RON roundtrip / rejection
        assert_eq!(JpegQuality::default().get(), DEFAULT_JPEG_QUALITY);
        assert_eq!(JpegQuality::from_str("1").unwrap().get(), 1);
        assert_eq!(JpegQuality::from_str("90").unwrap().get(), 90);
        assert_eq!(JpegQuality::from_str("100").unwrap().get(), 100);
        assert_eq!(JpegQuality::new(85).unwrap().to_string(), "85");
        for bad in ["0", "101", "255", "-1", "abc"] {
            assert!(JpegQuality::from_str(bad).is_err());
        }
        let ron_q = ron::to_string(&JpegQuality::new(85).unwrap()).unwrap();
        assert_eq!(ron_q, "85");
        assert_eq!(
            ron::from_str::<JpegQuality>(&ron_q).unwrap(),
            JpegQuality::new(85).unwrap()
        );
        assert!(ron::from_str::<JpegQuality>("0").is_err());
        assert!(ron::from_str::<JpegQuality>("101").is_err());

        // bpaf parsers accept valid flags and reject out-of-range / invalid values
        let empty: &[&str] = &[];
        let mode_opts = compression_mode().to_options();
        assert_eq!(mode_opts.run_inner(empty).unwrap(), None);
        assert_eq!(
            mode_opts
                .run_inner(&["--compression-mode", "hybrid"][..])
                .unwrap(),
            Some(CompressionKind::Hybrid)
        );
        assert!(
            mode_opts
                .run_inner(&["--compression-mode", "turbo"][..])
                .is_err()
        );

        let qual_opts = jpeg_quality().to_options();
        assert_eq!(qual_opts.run_inner(empty).unwrap(), None);
        assert_eq!(
            qual_opts.run_inner(&["--jpeg-quality", "85"][..]).unwrap(),
            Some(JpegQuality::new(85).unwrap())
        );
        assert!(qual_opts.run_inner(&["--jpeg-quality", "0"][..]).is_err());
        assert!(qual_opts.run_inner(&["--jpeg-quality", "101"][..]).is_err());

        let sub_opts = jpeg_subsamp().to_options();
        assert_eq!(sub_opts.run_inner(empty).unwrap(), None);
        assert_eq!(
            sub_opts.run_inner(&["--jpeg-subsamp", "420"][..]).unwrap(),
            Some(JpegSubsamp::Sub420)
        );
        assert!(sub_opts.run_inner(&["--jpeg-subsamp", "411"][..]).is_err());
    }
}
