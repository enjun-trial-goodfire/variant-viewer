"""Variant Viewer CLI.

Usage:
    uv run vv check                          # validate data exists
    uv run vv build                          # fast build to /tmp
    uv run vv build --umap --neighbors       # full build
    uv run vv serve /tmp/variant_viewer_*    # serve a build dir
    uv run vv extract --probe $P --acts $A   # submit SLURM extract
    uv run vv pipeline --probe $P --acts $A  # extract → finalize → eval → build
"""

from pathlib import Path
from typing import Optional

import typer
from rich import print as rprint

app = typer.Typer(help="Variant Effect Viewer", no_args_is_help=True)


@app.command()
def check(probe: str = typer.Argument("probe_v9", help="Probe name")):
    """Validate all required data and artifacts exist."""
    import subprocess
    result = subprocess.run(["bash", "preflight.sh", probe], cwd=Path(__file__).parent)
    raise typer.Exit(result.returncode)


@app.command()
def build(
    umap: bool = typer.Option(False, help="Compute UMAP embedding (~40s)"),
    neighbors: bool = typer.Option(False, help="Compute nearest neighbors (GPU)"),
    sync: Optional[Path] = typer.Option(None, help="Rsync staging to this directory after build"),
):
    """Build the static variant viewer site to /tmp."""
    from build import main as _build
    import sys

    argv = []
    if sync:
        argv.extend([str(sync), "--sync"])
    if umap:
        argv.append("--umap")
    if neighbors:
        argv.append("--neighbors")
    sys.argv = ["build"] + argv
    _build()


@app.command()
def serve(
    build_dir: Path = typer.Argument(..., help="Build directory to serve"),
    port: int = typer.Option(8501, help="Server port"),
    host: str = typer.Option("0.0.0.0", help="Server host"),
):
    """Serve a build directory with on-demand Claude interpretation."""
    if not build_dir.exists():
        rprint(f"[red]Build directory not found:[/] {build_dir}")
        raise typer.Exit(1)

    import os
    import uvicorn
    from loguru import logger
    from serve import create_app

    server = create_app(build_dir)
    logger.info(f"Serving {build_dir} on http://{host}:{port}")
    if not os.environ.get("ANTHROPIC_API_KEY"):
        logger.warning("ANTHROPIC_API_KEY not set — interpretation endpoint will return 503")
    uvicorn.run(server, host=host, port=port)


@app.command()
def extract(
    probe: Path = typer.Option(..., help="Probe directory (e.g., $ACTS/probe_v9)"),
    activations: Path = typer.Option(..., help="Activations storage directory"),
    shards: int = typer.Option(8, help="Number of parallel SLURM shards"),
):
    """Submit extract jobs to SLURM (parallel shards)."""
    _sbatch(
        f"--array=0-{shards - 1}",
        "pipeline/extract.sh",
        "--probe", str(probe),
        "--activations", str(activations),
        desc=f"extract ({shards} shards)",
    )


@app.command()
def pipeline(
    probe: Path = typer.Argument(..., help="Probe directory (e.g., $ACTS/probe_v9)"),
    labeled_only: bool = typer.Option(False, help="Skip VUS extraction"),
):
    """Run the full pipeline: extract → finalize → eval → build."""
    import subprocess
    args = [str(probe)]
    if labeled_only:
        args.append("--labeled-only")
    result = subprocess.run(["bash", "pipeline/pipeline.sh"] + args, cwd=Path(__file__).parent)
    raise typer.Exit(result.returncode)


def _sbatch(*args: str, desc: str = "job"):
    """Submit a SLURM job and print the job ID."""
    import subprocess
    result = subprocess.run(
        ["sbatch", "--parsable", *args],
        capture_output=True, text=True, cwd=Path(__file__).parent,
    )
    if result.returncode == 0:
        job_id = result.stdout.strip()
        rprint(f"[green]Submitted {desc}:[/] job {job_id}")
        rprint(f"  Monitor: [dim]squeue -j {job_id}[/]")
    else:
        rprint(f"[red]Failed to submit {desc}:[/] {result.stderr.strip()}")
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
