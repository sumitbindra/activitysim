from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
import uuid
import zipfile
from pathlib import Path

import modal

APP_NAME = "activitysim-runner"
VOLUME_NAME = "activitysim-runs"

image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("git", "curl", "ca-certificates", "build-essential")
    .run_commands(
        "curl -LsSf https://astral.sh/uv/install.sh | sh",
        "ln -sf /root/.local/bin/uv /usr/local/bin/uv",
    )
)

runs_volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)

app = modal.App(APP_NAME)


def _git_clone(repo_url: str, branch: str, dest: Path, github_token: str | None) -> str:
    url = repo_url
    if github_token and url.startswith("https://github.com/"):
        url = url.replace("https://", f"https://x-access-token:{github_token}@")

    subprocess.run(
        ["git", "clone", "--depth", "1", "--branch", branch, url, str(dest)],
        check=True,
    )
    sha = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=dest, capture_output=True, text=True, check=True
    ).stdout.strip()
    return sha


def _run_activitysim(
    repo_dir: Path,
    output_dir: Path,
    config_dirs: list[str],
    data_dir: str,
    sample_size: int | None,
    settings_file: str | None,
    chunk_size: str | None,
) -> None:
    subprocess.run(["uv", "sync", "--locked"], cwd=repo_dir, check=True)

    cmd = ["uv", "run", "activitysim", "run"]
    for c in config_dirs:
        cmd += ["-c", c]
    cmd += ["-d", data_dir, "-o", str(output_dir)]
    if sample_size is not None:
        cmd += ["--households_sample_size", str(sample_size)]
    if settings_file:
        cmd += ["-s", settings_file]
    if chunk_size:
        cmd += ["-g", chunk_size]

    print(f"\n>>> {' '.join(cmd)}\n", flush=True)
    subprocess.run(cmd, cwd=repo_dir, check=True)


def _zip_outputs(output_dir: Path, zip_path: Path) -> int:
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for f in output_dir.rglob("*"):
            if f.is_file():
                zf.write(f, f.relative_to(output_dir))
    return zip_path.stat().st_size


def _make_runner(cpu: float, memory_mb: int, timeout_s: int):
    secrets = []
    try:
        secrets.append(modal.Secret.from_name("github-token"))
    except Exception:
        pass

    @app.function(
        image=image,
        cpu=cpu,
        memory=memory_mb,
        timeout=timeout_s,
        volumes={"/runs": runs_volume},
        secrets=secrets,
    )
    def run_activitysim(
        repo_url: str,
        branch: str,
        sample_size: int | None,
        config_dirs: list[str],
        data_dir: str,
        settings_file: str | None,
        chunk_size: str | None,
    ) -> dict:
        run_id = f"{int(time.time())}-{uuid.uuid4().hex[:8]}"
        workdir = Path(f"/tmp/run-{run_id}")
        repo_dir = workdir / "repo"
        output_dir = workdir / "output"
        output_dir.mkdir(parents=True)

        github_token = os.environ.get("GITHUB_TOKEN")

        t0 = time.time()
        commit = _git_clone(repo_url, branch, repo_dir, github_token)
        t_clone = time.time() - t0

        t0 = time.time()
        _run_activitysim(
            repo_dir=repo_dir,
            output_dir=output_dir,
            config_dirs=config_dirs,
            data_dir=data_dir,
            sample_size=sample_size,
            settings_file=settings_file,
            chunk_size=chunk_size,
        )
        t_run = time.time() - t0

        manifest = {
            "run_id": run_id,
            "repo_url": repo_url,
            "branch": branch,
            "commit": commit,
            "sample_size": sample_size,
            "config_dirs": config_dirs,
            "data_dir": data_dir,
            "settings_file": settings_file,
            "chunk_size": chunk_size,
            "clone_seconds": round(t_clone, 2),
            "run_seconds": round(t_run, 2),
        }
        (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

        zip_path = Path(f"/runs/{run_id}/output.zip")
        size_bytes = _zip_outputs(output_dir, zip_path)
        (zip_path.parent / "manifest.json").write_text(json.dumps(manifest, indent=2))
        runs_volume.commit()

        shutil.rmtree(workdir, ignore_errors=True)

        manifest["zip_size_mb"] = round(size_bytes / (1024 * 1024), 2)
        manifest["download"] = (
            f"modal volume get {VOLUME_NAME} {run_id}/output.zip ./{run_id}.zip"
        )

        print("\n=== run complete ===")
        print(json.dumps(manifest, indent=2))
        return manifest

    return run_activitysim


@app.local_entrypoint()
def main(
    repo_url: str = "https://github.com/sumitbindra/activitysim.git",
    branch: str = "claude/model-run-management-tools-QvSyd",
    sample_size: int = 30,
    config_dirs: str = (
        "activitysim/examples/placeholder_sandag/configs_1_zone,"
        "activitysim/examples/prototype_mtc/configs"
    ),
    data_dir: str = "activitysim/examples/placeholder_sandag/data_1",
    settings_file: str = "",
    chunk_size: str = "",
    cpu: float = 4,
    memory_gb: int = 8,
    timeout_hours: float = 1.0,
):
    runner = _make_runner(
        cpu=cpu,
        memory_mb=memory_gb * 1024,
        timeout_s=int(timeout_hours * 3600),
    )
    result = runner.remote(
        repo_url=repo_url,
        branch=branch,
        sample_size=sample_size if sample_size > 0 else None,
        config_dirs=[c.strip() for c in config_dirs.split(",") if c.strip()],
        data_dir=data_dir,
        settings_file=settings_file or None,
        chunk_size=chunk_size or None,
    )
    print("\n--- launcher result ---")
    print(json.dumps(result, indent=2))
    print(f"\nDownload: {result['download']}")
