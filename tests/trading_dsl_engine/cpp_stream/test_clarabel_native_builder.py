from __future__ import annotations

from pathlib import Path
import subprocess

from trading_dsl_engine.cpp_stream.optimizer import clarabel_native


def test_build_current_clarabel_targets_host_cpu_and_preserves_rustflags(
    tmp_path: Path,
    monkeypatch,
) -> None:
    cpp_source = tmp_path / "Clarabel.cpp-source"
    (cpp_source / "include").mkdir(parents=True)
    (cpp_source / "include" / "clarabel.h").write_text("/* test header */\n")
    (cpp_source / "rust_wrapper").mkdir()
    (cpp_source / "rust_wrapper" / "Cargo.toml").write_text("[package]\n")

    rs_source = tmp_path / "Clarabel.rs-source"
    timer_path = rs_source / "src" / "timers" / "timers.rs"
    timer_path.parent.mkdir(parents=True)
    timer_path.write_text(
        """impl SubTimersMap {
    fn reset_subtimer(&mut self, key: &'static str) {
    }
}

impl Timers {
    fn reset(&mut self) {
        self.subtimers.clear();
    }
}
"""
    )

    monkeypatch.setenv("CLARABEL_CPP_SOURCE_DIR", str(cpp_source))
    monkeypatch.setenv("CLARABEL_RS_SOURCE_DIR", str(rs_source))
    monkeypatch.setenv("RUSTFLAGS", "-C debuginfo=1")

    calls: list[tuple[tuple[str, ...], dict[str, object]]] = []

    def fake_run(command, **kwargs):
        command = tuple(str(part) for part in command)
        calls.append((command, kwargs))
        if command[0] == "cargo":
            manifest = Path(command[command.index("--manifest-path") + 1])
            output = manifest.parent / "target" / "release" / "libclarabel_c.a"
            output.parent.mkdir(parents=True)
            output.write_bytes(b"test archive")
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(clarabel_native.subprocess, "run", fake_run)

    cache = tmp_path / "cache"
    paths = clarabel_native.build_current_clarabel(cache_dir=cache)

    cargo_command, cargo_kwargs = next(
        call for call in calls if call[0][0] == "cargo"
    )
    assert cargo_command[:3] == ("cargo", "build", "--release")
    assert cargo_kwargs["env"]["RUSTFLAGS"] == (
        "-C debuginfo=1 -C target-cpu=native"
    )
    assert "target-cpu=native" in (cache / "native" / "BUILD_ID").read_text()
    assert paths.static_library.read_bytes() == b"test archive"
