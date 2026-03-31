"""Variant Viewer CLI.

Usage:
    uv run vv check                              # validate data exists
    uv run vv train pretrain-cmd --gpus 4        # submit probe training to SLURM
    uv run vv extract --probe $P --acts $A       # submit SLURM extract (8 shards)
    uv run vv eval $ACTS/probe_v11               # compute per-head metrics → eval.json
    uv run vv log-eval $ACTS/probe_v11           # upload eval.json to wandb
    uv run vv build                              # fast build to /tmp
    uv run vv serve /tmp/variant_viewer_*        # serve with on-demand interpretation
    uv run vv pipeline $ACTS/probe_v11           # full chain: extract → eval → build
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
    probe: str = typer.Option("probe_v9", help="Probe name (e.g. probe_v9, probe_v11)"),
    umap: bool = typer.Option(False, help="Compute UMAP embedding (~40s)"),
    neighbors: bool = typer.Option(False, help="Compute nearest neighbors (GPU)"),
    sync: Optional[Path] = typer.Option(None, help="Rsync staging to this directory after build"),
    dev: Optional[int] = typer.Option(None, help="Dev mode: limit to N variants, skip UMAP + neighbors"),
):
    """Build the static variant viewer site to /tmp."""
    from build import main as _build

    staging = _build(output=sync, sync=bool(sync), umap=umap, neighbors=neighbors, probe=probe, dev=dev)
    rprint(f"[green]Built to:[/] {staging}")


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
def train(
    phase: str = typer.Argument(..., help="pretrain-cmd or finetune-cmd"),
    gpus: int = typer.Option(4, help="Number of GPUs"),
    checkpoint: Optional[Path] = typer.Option(None, help="Pretrain checkpoint for finetune"),
    time: str = typer.Option("08:00:00", help="SLURM time limit"),
):
    """Submit probe training to SLURM (torchrun DDP)."""
    args = ["--gpus", str(gpus), f"--time={time}", "pipeline/train.sh", phase]
    if checkpoint:
        args.extend(["--checkpoint", str(checkpoint)])
    _sbatch(*args, desc=f"train ({phase}, {gpus} GPU)")


@app.command()
def eval(
    probe_dir: Path = typer.Argument(..., help="Probe directory with scores.feather"),
    preset: str = typer.Option("deconfounded-full", help="Annotation preset"),
):
    """Compute per-head eval metrics → eval.json (runs locally, no SLURM)."""
    import subprocess
    result = subprocess.run(
        ["uv", "run", "python", "pipeline/eval.py", "--probe-dir", str(probe_dir), "--preset", preset],
        cwd=Path(__file__).parent,
    )
    raise typer.Exit(result.returncode)


@app.command()
def log_eval(
    probe_dir: Path = typer.Argument(..., help="Probe directory containing eval.json"),
    project: str = typer.Option("gfm-probes", help="W&B project"),
    name: str = typer.Option(None, help="W&B run name (default: probe dir name)"),
):
    """Upload eval.json to wandb as a Table for cross-run comparison.

    Creates a wandb run with a per-head metrics table (AUC, correlation, accuracy)
    and summary scalars. Use wandb's run comparer to scatter v9 vs v10 heads.

    Works for both new and old probes — backfill historical eval.json files.
    """
    import json
    import wandb

    eval_path = probe_dir / "eval.json"
    if not eval_path.exists():
        rprint(f"[red]No eval.json in {probe_dir}[/]")
        raise typer.Exit(1)

    eval_data = json.loads(eval_path.read_text())
    run_name = name or probe_dir.name

    # Config from config.json if available
    config = {}
    config_path = probe_dir / "config.json"
    if config_path.exists():
        config = json.loads(config_path.read_text())
    config["probe_dir"] = str(probe_dir)

    run = wandb.init(project=project, name=run_name, config=config, job_type="eval")

    # Per-head table — the key artifact for cross-run scatter
    table = wandb.Table(columns=["head", "kind", "auc", "correlation", "accuracy", "mse", "n"])
    aucs, corrs = [], []
    for head, metrics in eval_data.items():
        table.add_data(
            head, metrics.get("kind"), metrics.get("auc"), metrics.get("correlation"),
            metrics.get("accuracy"), metrics.get("mse"), metrics.get("n", 0),
        )
        if "auc" in metrics and metrics["auc"] is not None:
            aucs.append(metrics["auc"])
        if "correlation" in metrics and metrics["correlation"] is not None:
            corrs.append(metrics["correlation"])

    wandb.log({"eval/heads": table})

    # Summary scalars — show in the wandb runs table for quick comparison
    if aucs:
        run.summary["mean_auc"] = sum(aucs) / len(aucs)
    if corrs:
        run.summary["mean_correlation"] = sum(corrs) / len(corrs)
    if "pathogenic" in eval_data and "auc" in eval_data["pathogenic"]:
        run.summary["pathogenic_auc"] = eval_data["pathogenic"]["auc"]
    run.summary["n_heads"] = len(eval_data)

    # Upload eval.json as artifact
    artifact = wandb.Artifact(f"eval-{run_name}", type="eval")
    artifact.add_file(str(eval_path))
    if config_path.exists():
        artifact.add_file(str(config_path))
    wandb.log_artifact(artifact)

    wandb.finish()
    rprint(f"[green]Logged {len(eval_data)} heads to wandb:{project}/{run_name}[/]")


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
