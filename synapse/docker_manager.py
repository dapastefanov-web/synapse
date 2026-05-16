"""
docker_manager.py — Docker DevContainer lifecycle and the DAC security jail.

Responsibilities
─────────────────
1. /init-docker bootstrap — creates .synapse/, generates Dockerfile and
   shell.nix, builds the image via Docker BuildKit.
2. Container lifecycle — start, stop, get running container.
3. Security jail — all AI file operations run as ai_user (no .synapse access).
   Test execution runs as test_user (owns .synapse, can source secrets).
4. run_tests() — executes the project test suite as test_user and returns
   raw stdout/stderr for the Debugger to analyse.
5. apply_patches() — writes unified diffs to disk after HITL approval.
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# The generated Dockerfile. Two users are created:
#   ai_user  — runs all LLM-driven file/command operations (no .synapse access)
#   test_user — owns .synapse, sources secrets, runs the test suite
_DOCKERFILE_TEMPLATE = """\
FROM debian:bookworm-slim

# Install Nix and runtime essentials
RUN apt-get update && apt-get install -y --no-install-recommends \\
        curl xz-utils ca-certificates git sudo bash procps \\
    && rm -rf /var/lib/apt/lists/*

# Install Nix package manager in single-user mode
RUN curl -L https://nixos.org/nix/install | bash -s -- --no-daemon

ENV PATH="/root/.nix-profile/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"

# Create the two security jail users
RUN useradd -m -s /bin/bash ai_user && \\
    useradd -m -s /bin/bash test_user

WORKDIR /workspace
"""

_SHELL_NIX_TEMPLATE = """\
{ pkgs ? import <nixpkgs> {} }:
pkgs.mkShell {
  buildInputs = with pkgs; [
    python3
    python3Packages.pip
    git
  ];
}
"""


class DockerManagerError(Exception):
    pass


class DockerManager:
    """
    Manages the Synapse DevContainer for a single project.

    One DockerManager instance is created per session by the orchestrator
    and injected into the ToolContext so the native run_command tool can
    delegate container operations here.
    """

    def __init__(self, project_root: str) -> None:
        self._root        = Path(project_root).resolve()
        self._synapse_dir = self._root / ".synapse"
        self._client: Any = None
        self._container: Any = None

        # Image name is derived from the project directory name so multiple
        # projects can coexist without overwriting each other's images.
        safe_name        = re.sub(r"[^a-z0-9_-]", "-", self._root.name.lower())
        self._image_name = f"synapse-{safe_name}"

    # ── Docker client ──────────────────────────────────────────────────────

    def _get_client(self) -> Any:
        """Lazy-load the Docker SDK client so import errors surface clearly."""
        if self._client is None:
            try:
                import docker
                self._client = docker.from_env()
            except Exception as exc:
                raise DockerManagerError(
                    f"Cannot connect to Docker daemon: {exc}\n"
                    "Ensure Docker is installed and the daemon is running."
                ) from exc
        return self._client

    # ── /init-docker ──────────────────────────────────────────────────────

    async def init_docker(self) -> str:
        """
        Bootstrap the DevContainer for the current project.

        Steps:
          1. Create .synapse/ directory.
          2. Write Dockerfile and shell.nix (skip if already present).
          3. Build the Docker image via BuildKit.
          4. Set up DAC permissions inside the container.

        Returns a status message for the terminal UI.
        """
        self._synapse_dir.mkdir(exist_ok=True)

        dockerfile_path = self._synapse_dir / "Dockerfile"
        shellnix_path   = self._synapse_dir / "shell.nix"

        if not dockerfile_path.exists():
            dockerfile_path.write_text(_DOCKERFILE_TEMPLATE, encoding="utf-8")
            logger.info("Generated %s", dockerfile_path)

        if not shellnix_path.exists():
            shellnix_path.write_text(_SHELL_NIX_TEMPLATE, encoding="utf-8")
            logger.info("Generated %s", shellnix_path)

        await self._build_image()
        return f"DevContainer initialised. Image: {self._image_name}"

    async def _build_image(self) -> None:
        """Build the Docker image from .synapse/Dockerfile using BuildKit."""
        def _build() -> None:
            client = self._get_client()
            logger.info("Building image '%s' from %s ...", self._image_name, self._synapse_dir)
            client.images.build(
                path=str(self._synapse_dir),
                tag=self._image_name,
                rm=True,
                buildargs={"BUILDKIT_INLINE_CACHE": "1"},
            )
            logger.info("Image '%s' built successfully.", self._image_name)

        await asyncio.to_thread(_build)

    # ── Container lifecycle ────────────────────────────────────────────────

    async def start_container(self) -> str:
        """
        Start the DevContainer with the project root bind-mounted to /workspace.

        Security jail setup runs immediately after start:
          - .synapse is owned by test_user (chmod 700) → ai_user cannot read it.
          - /workspace itself is owned by ai_user.
        """
        def _start() -> str:
            client    = self._get_client()
            container = client.containers.run(
                image=self._image_name,
                command="tail -f /dev/null",  # keep alive
                detach=True,
                remove=False,
                volumes={
                    str(self._root): {
                        "bind": "/workspace",
                        "mode": "rw",
                    }
                },
                working_dir="/workspace",
                name=f"{self._image_name}-session",
            )
            return container.id

        container_id = await asyncio.to_thread(_start)

        # Get the container object and apply the DAC jail
        self._container = self._get_client().containers.get(container_id)
        await self._setup_jail()

        logger.info("Container started: %s", container_id[:12])
        return container_id

    async def stop_container(self) -> None:
        """Stop and remove the running container."""
        if self._container is None:
            return

        def _stop() -> None:
            try:
                self._container.stop(timeout=10)
                self._container.remove()
            except Exception as exc:
                logger.warning("Error stopping container: %s", exc)

        await asyncio.to_thread(_stop)
        self._container = None

    async def _setup_jail(self) -> None:
        """
        Apply Linux DAC permissions to enforce the security jail.

          chown -R test_user:test_user /workspace/.synapse
          chmod 700 /workspace/.synapse
          chown -R ai_user:ai_user /workspace  (excludes .synapse)

        After this, ai_user cannot read .synapse even if the LLM
        generates a command like `cat .synapse/bashrc`.
        """
        if self._container is None:
            return

        def _apply() -> None:
            cmds = [
                "chown -R test_user:test_user /workspace/.synapse",
                "chmod 700 /workspace/.synapse",
                "find /workspace -maxdepth 1 ! -name .synapse "
                "-exec chown ai_user:ai_user {} +",
            ]
            for cmd in cmds:
                result = self._container.exec_run(
                    ["sh", "-c", cmd], user="root"
                )
                if result.exit_code != 0:
                    logger.warning("Jail setup command failed: %s", cmd)

        await asyncio.to_thread(_apply)

    # ── Test execution ─────────────────────────────────────────────────────

    async def run_tests(self, command: str | None = None) -> str:
        """
        Run the project's test suite as test_user inside the container.

        test_user owns .synapse so it can source secret keys before running
        tests. The raw stdout+stderr is returned to the caller (Debugger node)
        as a plain string — no sensitive key material is ever included because
        the shell only sources them for subprocess execution, not echo output.

        Args:
            command: Override the default test command. Auto-detected if None.

        Returns stdout+stderr as a single string.
        """
        if self._container is None:
            return "Error: container not running. Call start_container() first."

        test_cmd = command or self._detect_test_command()

        def _run() -> str:
            # Source .synapse secrets before running tests so the test suite
            # can authenticate against external APIs without the LLM ever
            # seeing the raw key values.
            full_cmd = (
                f"if [ -f /workspace/.synapse/.env ]; then "
                f"set -a && . /workspace/.synapse/.env && set +a; fi && {test_cmd}"
            )
            result = self._container.exec_run(
                ["bash", "-c", full_cmd],
                user="test_user",
                stdout=True,
                stderr=True,
                workdir="/workspace",
            )
            output = result.output.decode("utf-8", errors="replace") if result.output else ""
            exit_label = "PASSED" if result.exit_code == 0 else f"FAILED (exit {result.exit_code})"
            return f"[{exit_label}]\n{output}"

        return await asyncio.to_thread(_run)

    def _detect_test_command(self) -> str:
        """
        Infer the test command from the project structure.
        Checked in priority order: pytest → npm test → cargo test → make test.
        """
        if (self._root / "pytest.ini").exists() or (self._root / "pyproject.toml").exists():
            return "python -m pytest -x --tb=short 2>&1"
        if (self._root / "package.json").exists():
            return "npm test 2>&1"
        if (self._root / "Cargo.toml").exists():
            return "cargo test 2>&1"
        return "make test 2>&1"

    # ── Patch application ──────────────────────────────────────────────────

    async def apply_patches(self, patches: list[Any]) -> list[str]:
        """
        Apply a list of FilePatch objects to the project root on disk.

        Each patch's unified_diff is written to a temporary .patch file and
        applied with `patch -p1`. This keeps the application logic identical
        to standard Git patch workflow. Returns a list of applied file paths.

        This is called by the terminal dispatcher after the user accepts
        in no_trust mode, or automatically in trust mode.
        """
        applied: list[str] = []

        for patch in patches:
            try:
                await self._apply_single_patch(patch)
                applied.append(patch.file_path)
                logger.info("Applied patch to %s", patch.file_path)
            except Exception as exc:
                logger.error("Failed to apply patch to %s: %s", patch.file_path, exc)

        return applied

    async def _apply_single_patch(self, patch: Any) -> None:
        """Write the unified diff to a temp file and apply it with patch -p1."""
        import tempfile

        def _apply() -> None:
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".patch", delete=False, encoding="utf-8"
            ) as tmp:
                tmp.write(patch.unified_diff)
                tmp_path = tmp.name

            try:
                import subprocess
                result = subprocess.run(
                    ["patch", "-p1", "--input", tmp_path],
                    cwd=str(self._root),
                    capture_output=True,
                    text=True,
                )
                if result.returncode != 0:
                    raise DockerManagerError(
                        f"patch command failed for {patch.file_path}:\n{result.stderr}"
                    )
            finally:
                os.unlink(tmp_path)

        await asyncio.to_thread(_apply)

    # ── Dynamic Nix dependency injection ──────────────────────────────────

    async def add_nix_dependency(self, package_name: str) -> str:
        """
        Add a Nix package to shell.nix and rebuild the Docker image.

        This is the dynamic caching loop described in the master plan.
        The Coder calls this tool instead of running apt-get directly,
        keeping the environment fully declarative and reproducible.

        Args:
            package_name: Nix attribute name, e.g. 'rustc', 'nodejs', 'postgresql'.

        Returns a status string for the agent's tool result.
        """
        shell_nix = self._synapse_dir / "shell.nix"
        if not shell_nix.exists():
            return "Error: shell.nix not found. Run /init-docker first."

        content = shell_nix.read_text(encoding="utf-8")

        if package_name in content:
            return f"Package '{package_name}' is already in shell.nix."

        # Insert the new package before the closing bracket of buildInputs
        updated = content.replace(
            "  ];",
            f"    {package_name}\n  ];",
            1,
        )
        shell_nix.write_text(updated, encoding="utf-8")
        logger.info("Added '%s' to shell.nix — rebuilding image.", package_name)

        await self._build_image()
        return f"Package '{package_name}' added and image rebuilt successfully."