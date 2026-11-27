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

use std::fs;
use std::path::PathBuf;
use std::process::Command;
use std::sync::Arc;
use std::time::Duration;

use bpaf::Parser;
use optional_struct::optional_struct;
use serde_derive::Deserialize;
use serde_derive::Serialize;
use smithay::reexports::calloop::EventLoop;
use smithay::reexports::calloop::Interest;
use smithay::reexports::calloop::Mode;
use smithay::reexports::calloop::PostAction;
use smithay::reexports::calloop::channel::Event;
use smithay::reexports::calloop::generic::Generic;
use smithay::reexports::wayland_server::Display;
use smithay::wayland::socket::ListeningSocketSource;
use tracing::Level;
use wprs::args;
use wprs::args::Config;
use wprs::args::OptionalConfig;
use wprs::args::SerializableLevel;
use wprs::args::XwaylandBackend;
use wprs::prelude::*;
use wprs::serialization::Serializer;
use wprs::server::WprsServerState;
use wprs::server::smithay_handlers::ClientState;
use wprs::utils;

#[optional_struct]
#[derive(Clone, Debug, Eq, PartialEq, Deserialize, Serialize)]
pub struct WprsdConfig {
    // Skip serializing fields which aren't ever useful to put into a config
    // file.
    #[serde(skip_serializing)]
    print_default_config_and_exit: bool,
    #[serde(skip_serializing)]
    config_file: PathBuf,
    wayland_display: String,
    socket: PathBuf,
    framerate: u32,
    // Optional fields don't get wrapped unless we specify it ourselves
    #[optional_wrap]
    log_file: Option<PathBuf>,
    stderr_log_level: SerializableLevel,
    file_log_level: SerializableLevel,
    log_priv_data: bool,
    xwayland: XwaylandBackend,
    xwayland_display: u32,
    #[optional_wrap]
    xwayland_path: Option<String>,
    xwayland_wayland_debug: bool,
    xwayland_args: Vec<String>,
    kde_server_side_decorations: bool,
}

impl Default for WprsdConfig {
    fn default() -> Self {
        Self {
            print_default_config_and_exit: false,
            config_file: args::default_config_file("wprsd"),
            wayland_display: "wprs-0".to_string(),
            socket: args::default_socket_path(),
            framerate: 60,
            log_file: None,
            stderr_log_level: SerializableLevel(Level::INFO),
            file_log_level: SerializableLevel(Level::TRACE),
            log_priv_data: false,
            xwayland: XwaylandBackend::default(),
            xwayland_display: 100,
            xwayland_path: None,
            xwayland_wayland_debug: false,
            xwayland_args: Vec::new(),
            kde_server_side_decorations: false,
        }
    }
}

impl Config for WprsdConfig {
    fn config_file(&self) -> PathBuf {
        self.config_file.clone()
    }
}

fn xwayland_display() -> impl Parser<Option<u32>> {
    bpaf::long("xwayland-display")
        .argument::<u32>("NUM")
        .help("X11 display number for the XWayland backend (default 100, i.e., :100).")
        .optional()
}

fn xwayland_path() -> impl Parser<Option<Option<String>>> {
    bpaf::long("xwayland-path")
        .argument::<String>("PATH")
        .help("Override the XWayland backend executable path.")
        .optional()
        .map(|path| path.map(Some))
}

fn xwayland_wayland_debug() -> impl Parser<Option<bool>> {
    bpaf::long("xwayland-wayland-debug")
        .argument::<bool>("BOOL")
        .optional()
}

fn xwayland_args() -> impl Parser<Option<Vec<String>>> {
    bpaf::long("xwayland-args")
        .argument::<String>("ARG1,ARG2,...,ARGN")
        .map(|s| s.split(',').map(str::to_string).collect::<Vec<_>>())
        .many()
        .map(|nested| (!nested.is_empty()).then(|| nested.into_iter().flatten().collect()))
}

fn kde_server_side_decorations() -> impl Parser<Option<bool>> {
    bpaf::long("kde-server-side-decorations")
        .argument::<bool>("BOOL")
        .help("Whether to prefer server-side decorations for applications which still use the org_kde_kwin_server_decoration_manager protocol. GTK is the only major user of that protocol and it ignores polite suggestions from the compositor about whether server-side or client-side decorations should be used, so have to configure this preference at wprsd startup. Once GTK moves to the xdg-decoration protocol, this can be removed and we can auto-detect the preference of the client compositor.")
        .optional()
}

impl OptionalWprsdConfig {
    fn args_parser() -> impl Parser<Self> {
        let print_default_config_and_exit = args::print_default_config_and_exit();
        let config_file = args::config_file();
        let wayland_display = args::wayland_display();
        let socket = args::socket();
        let framerate = args::framerate();
        let log_file = args::log_file();
        let stderr_log_level = args::stderr_log_level();
        let file_log_level = args::file_log_level();
        let log_priv_data = args::log_priv_data();
        let xwayland = args::xwayland();
        let xwayland_display = xwayland_display();
        let xwayland_path = xwayland_path();
        let xwayland_wayland_debug = xwayland_wayland_debug();
        let xwayland_args = xwayland_args();
        let kde_server_side_decorations = kde_server_side_decorations();
        bpaf::construct!(Self {
            print_default_config_and_exit,
            config_file,
            wayland_display,
            socket,
            framerate,
            log_file,
            stderr_log_level,
            file_log_level,
            log_priv_data,
            xwayland,
            xwayland_display,
            xwayland_path,
            xwayland_wayland_debug,
            xwayland_args,
            kde_server_side_decorations,
        })
    }
}

impl OptionalConfig<WprsdConfig> for OptionalWprsdConfig {
    fn parse_args() -> Self {
        Self::args_parser().to_options().run()
    }

    fn print_default_config_and_exit(&self) -> Option<bool> {
        self.print_default_config_and_exit
    }

    fn config_file(&self) -> Option<PathBuf> {
        self.config_file.clone()
    }
}

fn init_wayland_listener(
    wayland_display: &str,
    mut display: Display<WprsServerState>,
    state: &mut WprsServerState,
    event_loop: &EventLoop<WprsServerState>,
) -> Result<()> {
    let listening_socket = ListeningSocketSource::with_name(wayland_display).location(loc!())?;
    let writer = state.serializer.writer();
    let mut dh = display.handle();

    event_loop
        .handle()
        .insert_source(listening_socket, move |stream, _, _| {
            dh.insert_client(stream, Arc::new(ClientState::new(writer.clone())))
                .unwrap();
        })
        .location(loc!())?;

    event_loop
        .handle()
        .insert_source(
            Generic::new(
                display.backend().poll_fd().try_clone_to_owned().unwrap(),
                Interest::READ,
                Mode::Level,
            ),
            move |_, _, state| {
                display.dispatch_clients(state).unwrap();
                Ok(PostAction::Continue)
            },
        )
        .location(loc!())?;

    Ok(())
}

fn build_xwayland_command(config: &WprsdConfig) -> Option<Command> {
    let default_binary = config.xwayland.default_binary()?;
    let program = config.xwayland_path.as_deref().unwrap_or(default_binary);
    let mut cmd = Command::new(program);
    cmd.env("WAYLAND_DISPLAY", &config.wayland_display).env(
        "WAYLAND_DEBUG",
        if config.xwayland_wayland_debug {
            "1"
        } else {
            "0"
        },
    );
    match config.xwayland {
        XwaylandBackend::Disabled => unreachable!("default_binary returned Some"),
        XwaylandBackend::XwaylandXdgShell => {
            cmd.args(["--display", &config.xwayland_display.to_string()]);
        },
        XwaylandBackend::XwaylandSatellite => {
            cmd.arg(format!(":{}", config.xwayland_display));
        },
    }
    cmd.args(&config.xwayland_args);
    Some(cmd)
}

fn start_xwayland(config: &WprsdConfig) {
    let Some(mut cmd) = build_xwayland_command(config) else {
        return;
    };
    let backend = config.xwayland;
    let mut child = cmd
        .spawn()
        .unwrap_or_else(|err| panic!("failed executing {backend}: {err}"));

    std::thread::spawn(move || {
        child
            .wait()
            .unwrap_or_else(|err| panic!("failed waiting {backend}: {err}"));
    });
}

#[allow(clippy::missing_panics_doc)]
pub fn main() -> Result<()> {
    let config = args::init_config::<WprsdConfig, OptionalWprsdConfig>();
    args::set_log_priv_data(config.log_priv_data);
    utils::configure_tracing(
        config.stderr_log_level.0,
        config.log_file.clone(),
        config.file_log_level.0,
    )
    .location(loc!())?;
    utils::exit_on_thread_panic();

    fs::create_dir_all(config.socket.parent().location(loc!())?).location(loc!())?;
    let mut serializer = Serializer::new_server(&config.socket).location(loc!())?;
    let reader = serializer.reader().location(loc!())?;

    let mut event_loop = EventLoop::try_new().location(loc!())?;
    let display: Display<WprsServerState> = Display::new().location(loc!())?;

    let frame_interval = Duration::from_secs_f64(1.0 / (config.framerate as f64));

    let mut state = WprsServerState::new(
        display.handle(),
        event_loop.handle(),
        serializer,
        config.xwayland.is_enabled(),
        frame_interval,
        config.kde_server_side_decorations,
    );

    init_wayland_listener(&config.wayland_display, display, &mut state, &event_loop)
        .location(loc!())?;

    start_xwayland(&config);

    // TODO: do this in WprsServerState::new;
    let _keyboard = state
        .seat
        .add_keyboard(Default::default(), 200, 200)
        .location(loc!())?;
    let _pointer = state.seat.add_pointer();

    event_loop
        .handle()
        .insert_source(reader, |event, _metadata, state| {
            match event {
                Event::Msg((conn_id, msg)) => state.handle_event(conn_id, msg),
                Event::Closed => {
                    unreachable!("reader is an in-memory channel whose write end has the same lifetime as serializer: the lifetime of the program.")
                },
            }
        }).unwrap();

    event_loop
        .run(None, &mut state, move |state| {
            state.dh.flush_clients().unwrap();
        })
        .location(loc!())?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use std::ffi::OsStr;

    use optional_struct::Applicable;
    use ron::Options;
    use ron::extensions::Extensions;

    use super::*;

    fn command_envs(cmd: &Command) -> Vec<(&OsStr, Option<&OsStr>)> {
        cmd.get_envs().collect()
    }

    fn command_args(cmd: &Command) -> Vec<&OsStr> {
        cmd.get_args().collect()
    }

    #[test]
    fn test_wprsd_config_xwayland_ron_and_cli() {
        let default_cfg = WprsdConfig::default();
        assert_eq!(default_cfg.xwayland, XwaylandBackend::XwaylandXdgShell);
        assert_eq!(default_cfg.xwayland_display, 100);
        assert_eq!(default_cfg.xwayland_path, None);
        assert!(!default_cfg.xwayland_wayland_debug);
        assert!(default_cfg.xwayland_args.is_empty());
        assert!(default_cfg.xwayland.is_enabled());

        // `--print-default-config-and-exit` output must roundtrip through `OptionalWprsdConfig`.
        let printed =
            ron::ser::to_string_pretty(&default_cfg, ron::ser::PrettyConfig::default()).unwrap();
        let parsed_opt: OptionalWprsdConfig = Options::default()
            .with_default_extension(Extensions::IMPLICIT_SOME)
            .from_str(&printed)
            .unwrap();
        let mut applied = WprsdConfig::default();
        parsed_opt.apply_to(&mut applied);
        assert_eq!(applied, default_cfg);

        // RON overrides for all three XWayland variants, including `Disabled`
        // under `Extensions::IMPLICIT_SOME`, plus preserving `xwayland_args`
        // when the CLI omits `--xwayland-args`.
        let sat_ron = r#"(
            xwayland: XwaylandSatellite,
            xwayland_display: 105,
            xwayland_path: "/opt/bin/xwayland-satellite",
            xwayland_wayland_debug: true,
            xwayland_args: ["--extra", "foo"],
        )"#;
        let sat_opt: OptionalWprsdConfig = Options::default()
            .with_default_extension(Extensions::IMPLICIT_SOME)
            .from_str(sat_ron)
            .unwrap();
        let mut sat_cfg = WprsdConfig::default();
        sat_opt.apply_to(&mut sat_cfg);
        let empty: &[&str] = &[];
        let empty_cli = OptionalWprsdConfig::args_parser()
            .to_options()
            .run_inner(empty)
            .unwrap();
        assert_eq!(empty_cli.xwayland_args, None);
        empty_cli.apply_to(&mut sat_cfg);
        assert_eq!(sat_cfg.xwayland, XwaylandBackend::XwaylandSatellite);
        assert_eq!(sat_cfg.xwayland_display, 105);
        assert_eq!(
            sat_cfg.xwayland_path.as_deref(),
            Some("/opt/bin/xwayland-satellite")
        );
        assert!(sat_cfg.xwayland_wayland_debug);
        assert_eq!(sat_cfg.xwayland_args, vec!["--extra", "foo"]);
        assert!(sat_cfg.xwayland.is_enabled());

        let disabled_opt: OptionalWprsdConfig = Options::default()
            .with_default_extension(Extensions::IMPLICIT_SOME)
            .from_str("(xwayland: Disabled)")
            .unwrap();
        let mut disabled_cfg = WprsdConfig::default();
        disabled_opt.apply_to(&mut disabled_cfg);
        assert_eq!(disabled_cfg.xwayland, XwaylandBackend::Disabled);
        assert!(!disabled_cfg.xwayland.is_enabled());

        assert!(
            Options::default()
                .with_default_extension(Extensions::IMPLICIT_SOME)
                .from_str::<OptionalWprsdConfig>("(xwayland: Unknown)")
                .is_err()
        );

        // CLI flag parsing via `OptionalWprsdConfig::args_parser`.
        let cli_opt = OptionalWprsdConfig::args_parser()
            .to_options()
            .run_inner(
                &[
                    "--xwayland",
                    "xwayland-satellite",
                    "--xwayland-display",
                    "102",
                    "--xwayland-path",
                    "/usr/local/bin/xwayland-satellite",
                    "--xwayland-wayland-debug",
                    "true",
                    "--xwayland-args=--arg1,--arg2",
                    "--xwayland-args=--arg3",
                ][..],
            )
            .unwrap();
        let mut cli_cfg = WprsdConfig::default();
        cli_opt.apply_to(&mut cli_cfg);
        assert_eq!(cli_cfg.xwayland, XwaylandBackend::XwaylandSatellite);
        assert_eq!(cli_cfg.xwayland_display, 102);
        assert_eq!(
            cli_cfg.xwayland_path.as_deref(),
            Some("/usr/local/bin/xwayland-satellite")
        );
        assert!(cli_cfg.xwayland_wayland_debug);
        assert_eq!(cli_cfg.xwayland_args, vec!["--arg1", "--arg2", "--arg3"]);

        let cli_disabled = OptionalWprsdConfig::args_parser()
            .to_options()
            .run_inner(&["--xwayland", "disabled"][..])
            .unwrap();
        let mut cli_disabled_cfg = WprsdConfig::default();
        cli_disabled.apply_to(&mut cli_disabled_cfg);
        assert_eq!(cli_disabled_cfg.xwayland, XwaylandBackend::Disabled);
        assert!(!cli_disabled_cfg.xwayland.is_enabled());

        assert!(
            OptionalWprsdConfig::args_parser()
                .to_options()
                .run_inner(&["--xwayland", "invalid"][..])
                .is_err()
        );
    }

    #[test]
    fn test_build_xwayland_command_all_backends() {
        // Disabled -> None (no child process, Capabilities.xwayland == false).
        let disabled = WprsdConfig {
            xwayland: XwaylandBackend::Disabled,
            ..WprsdConfig::default()
        };
        assert!(!disabled.xwayland.is_enabled());
        assert!(build_xwayland_command(&disabled).is_none());

        // Default XwaylandXdgShell.
        let xdg_default = WprsdConfig::default();
        assert!(xdg_default.xwayland.is_enabled());
        let cmd = build_xwayland_command(&xdg_default).unwrap();
        assert_eq!(cmd.get_program(), OsStr::new("xwayland-xdg-shell"));
        assert_eq!(
            command_envs(&cmd),
            vec![
                (OsStr::new("WAYLAND_DEBUG"), Some(OsStr::new("0"))),
                (OsStr::new("WAYLAND_DISPLAY"), Some(OsStr::new("wprs-0"))),
            ]
        );
        assert_eq!(
            command_args(&cmd),
            vec![OsStr::new("--display"), OsStr::new("100")]
        );

        // Customized XwaylandXdgShell.
        let xdg_custom = WprsdConfig {
            wayland_display: "wprs-custom".to_string(),
            xwayland: XwaylandBackend::XwaylandXdgShell,
            xwayland_display: 107,
            xwayland_path: Some("/custom/xwayland-xdg-shell".to_string()),
            xwayland_wayland_debug: true,
            xwayland_args: vec![
                "--decoration-behavior".to_string(),
                "AlwaysEnabled".to_string(),
            ],
            ..WprsdConfig::default()
        };
        let cmd = build_xwayland_command(&xdg_custom).unwrap();
        assert_eq!(cmd.get_program(), OsStr::new("/custom/xwayland-xdg-shell"));
        assert_eq!(
            command_envs(&cmd),
            vec![
                (OsStr::new("WAYLAND_DEBUG"), Some(OsStr::new("1"))),
                (
                    OsStr::new("WAYLAND_DISPLAY"),
                    Some(OsStr::new("wprs-custom"))
                ),
            ]
        );
        assert_eq!(
            command_args(&cmd),
            vec![
                OsStr::new("--display"),
                OsStr::new("107"),
                OsStr::new("--decoration-behavior"),
                OsStr::new("AlwaysEnabled"),
            ]
        );

        // Default XwaylandSatellite.
        let sat_default = WprsdConfig {
            xwayland: XwaylandBackend::XwaylandSatellite,
            ..WprsdConfig::default()
        };
        assert!(sat_default.xwayland.is_enabled());
        let cmd = build_xwayland_command(&sat_default).unwrap();
        assert_eq!(cmd.get_program(), OsStr::new("xwayland-satellite"));
        assert_eq!(
            command_envs(&cmd),
            vec![
                (OsStr::new("WAYLAND_DEBUG"), Some(OsStr::new("0"))),
                (OsStr::new("WAYLAND_DISPLAY"), Some(OsStr::new("wprs-0"))),
            ]
        );
        assert_eq!(command_args(&cmd), vec![OsStr::new(":100")]);

        // Customized XwaylandSatellite.
        let sat_custom = WprsdConfig {
            wayland_display: "wprs-1".to_string(),
            xwayland: XwaylandBackend::XwaylandSatellite,
            xwayland_display: 103,
            xwayland_path: Some("/opt/xwayland-satellite".to_string()),
            xwayland_wayland_debug: true,
            xwayland_args: vec!["--extra-flag".to_string()],
            ..WprsdConfig::default()
        };
        let cmd = build_xwayland_command(&sat_custom).unwrap();
        assert_eq!(cmd.get_program(), OsStr::new("/opt/xwayland-satellite"));
        assert_eq!(
            command_envs(&cmd),
            vec![
                (OsStr::new("WAYLAND_DEBUG"), Some(OsStr::new("1"))),
                (OsStr::new("WAYLAND_DISPLAY"), Some(OsStr::new("wprs-1"))),
            ]
        );
        assert_eq!(
            command_args(&cmd),
            vec![OsStr::new(":103"), OsStr::new("--extra-flag")]
        );
    }
}
