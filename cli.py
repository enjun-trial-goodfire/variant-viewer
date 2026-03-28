"""Variant Viewer CLI.

Usage:
    uv run vv build                          # fast build (no UMAP/neighbors)
    uv run vv build --umap --neighbors       # full build
    uv run vv build --sync webapp/build      # build + rsync to output dir
    uv run vv serve                          # serve with autointerp
    uv run vv serve --port 8501              # custom port
    uv run vv preflight                      # check all data exists
"""

from pathlib import Path
from typing import Annotated, Optional

import typer
from rich import print as rprint

app = typer.Typer(help="Variant Effect Viewer", no_args_is_help=True)


@app.command()
def build(
    umap: bool = typer.Option(False, help="Compute UMAP embedding (~40s)"),
    neighbors: bool = typer.Option(False, help="Compute nearest neighbors (GPU)"),
    sync: Optional[Path] = typer.Option(None, help="Rsync output to this directory"),
):
    """Build the static variant viewer site."""
    from build import main as build_main
    import sys

    # Translate typer args to the build.py argparse interface
    # TODO: refactor build.py to accept these directly instead of argparse
    argv = []
    if sync:
        argv.extend([str(sync), "--sync"])
    if umap:
        argv.append("--umap")
    if neighbors:
        argv.append("--neighbors")

    sys.argv = ["build"] + argv
    build_main()


@app.command()
def serve(
    build_dir: Path = typer.Option(Path("webapp/build"), help="Build directory to serve"),
    port: int = typer.Option(8080, help="Server port"),
    host: str = typer.Option("0.0.0.0", help="Server host"),
):
    """Serve the variant viewer with on-demand Claude interpretation."""
    import uvicorn
    from serve import create_app
    from loguru import logger
    import os

    if not build_dir.exists():
        rprint(f"[red]Build directory not found:[/] {build_dir}")
        raise typer.Exit(1)

    app = create_app(build_dir)
    logger.info(f"Serving {build_dir} on http://{host}:{port}")
    if not os.environ.get("ANTHROPIC_API_KEY"):
        logger.warning("ANTHROPIC_API_KEY not set — interpretation endpoint will return 503")

    uvicorn.run(app, host=host, port=port)


@app.command()
def preflight(
    probe: str = typer.Argument("probe_v9", help="Probe directory name"),
):
    """Validate all required data files exist before building."""
    import subprocess
    import sys

    result = subprocess.run(["bash", "preflight.sh", probe], cwd=Path(__file__).parent)
    raise typer.Exit(result.returncode)


@app.command()
def extract(
    probe: Path = typer.Option(..., help="Probe directory path"),
    activations: Path = typer.Option(..., help="Activations storage path"),
    shards: int = typer.Option(8, help="Number of SLURM array shards"),
):
    """Submit extract jobs to SLURM (8 shards by default)."""
    import subprocess

    rprint(f"Submitting {shards} extract shards for {probe.name}...")
    result = subprocess.run([
        "sbatch", "--parsable", f"--array=0-{shards - 1}",
        "pipeline/extract.sh",
        "--probe", str(probe),
        "--activations", str(activations),
    ], capture_output=True, text=True)

    if result.returncode == 0:
        job_id = result.stdout.strip()
        rprint(f"[green]Submitted:[/] job {job_id} ({shards} shards)")
        rprint(f"Monitor: squeue -j {job_id}")
    else:
        rprint(f"[red]Failed:[/] {result.stderr}")
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
