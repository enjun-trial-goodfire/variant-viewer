"""Variant Viewer CLI.

Usage:
    uv run vv check                              # validate data exists
    uv run vv train pretrain-cmd --gpus 4        # submit probe training to SLURM
    uv run vv extract --probe $P --acts $A       # submit SLURM extract (8 shards)
    uv run vv eval $ACTS/probe_v11               # compute per-head metrics → eval.json
    uv run vv log-eval $ACTS/probe_v11           # upload eval.json to wandb
    uv run vv build                              # build DuckDB → builds/variants.duckdb
    uv run vv serve                              # serve API + frontend on :8501
    uv run vv pipeline $ACTS/probe_v11           # full chain: extract → eval → build
"""

import subprocess
from pathlib import Path
from typing import Optional

import typer
from rich import print as rprint

app = typer.Typer(help="Variant Effect Viewer", no_args_is_help=True)
ROOT = Path(__file__).parent


# ── Helpers ──────────────────────────────────────────────────────────────


def _sbatch(*args: str) -> str:
    """Submit a SLURM job, return the job ID."""
    result = subprocess.run(
        ["sbatch", "--parsable", *args],
        capture_output=True, text=True, cwd=ROOT,
    )
    if result.returncode != 0:
        rprint(f"[red]sbatch failed:[/] {result.stderr.strip()}")
        raise typer.Exit(1)
    return result.stdout.strip()


# ── Commands ─────────────────────────────────────────────────────────────


@app.command()
def check(probe: str = typer.Argument("probe_v11", help="Probe name")):
    """Validate all required data and artifacts exist."""
    result = subprocess.run(["bash", "preflight.sh", probe], cwd=ROOT)
    raise typer.Exit(result.returncode)


@app.command()
def build(
    probe: str = typer.Option("probe_v11", help="Probe name (e.g. probe_v11)"),
    umap: bool = typer.Option(False, help="Compute UMAP embedding (~40s)"),
    neighbors: bool = typer.Option(False, help="Compute nearest neighbors (GPU)"),
    db: Path = typer.Option(Path("builds/variants.duckdb"), help="Output DuckDB path"),
    dev: Optional[int] = typer.Option(None, help="Dev mode: limit to N variants"),
):
    """Build the variant viewer DuckDB database."""
    from build import main as _build

    result = _build(db_path=db, umap=umap, neighbors=neighbors, probe=probe, dev=dev)
    rprint(f"[green]Built:[/] {result}")


@app.command()
def serve(
    db: Path = typer.Option(Path("builds/variants.duckdb"), help="DuckDB database path"),
    static: Optional[Path] = typer.Option(None, help="Frontend static files directory (default: frontend/dist)"),
    port: int = typer.Option(8501, help="Server port"),
    host: str = typer.Option("0.0.0.0", help="Server host"),
):
    """Serve the variant viewer API from DuckDB."""
    if not db.exists():
        rprint(f"[red]Database not found:[/] {db}\nRun: uv run vv build")
        raise typer.Exit(1)

    import os

    import uvicorn
    from loguru import logger

    from serve import create_app

    # Default to frontend/dist if it exists
    static_dir = static or (ROOT / "frontend" / "dist")
    if not static_dir.exists():
        static_dir = None
        logger.warning(f"No frontend build found at {ROOT / 'frontend' / 'dist'} — API-only mode")

    server = create_app(db, static_dir)
    logger.info(f"Serving {db} on http://{host}:{port}")
    if static_dir:
        logger.info(f"Frontend: {static_dir}")
    if not os.environ.get("ANTHROPIC_API_KEY"):
        logger.warning("ANTHROPIC_API_KEY not set — interpretation endpoint will return 503")
    uvicorn.run(server, host=host, port=port)


@app.command()
def extract(
    probe: Path = typer.Option(..., help="Probe directory (e.g., $ACTS/probe_v11)"),
    activations: Path = typer.Option(..., help="Activations storage directory"),
    shards: int = typer.Option(8, help="Number of parallel SLURM shards"),
):
    """Submit extract jobs to SLURM (parallel shards)."""
    job_id = _sbatch(f"--array=0-{shards - 1}", "pipeline/extract.sh",
                     "--probe", str(probe), "--activations", str(activations))
    rprint(f"[green]Submitted extract ({shards} shards):[/] job {job_id}")


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
    job_id = _sbatch(*args)
    rprint(f"[green]Submitted train ({phase}, {gpus} GPU):[/] job {job_id}")


@app.command()
def eval(
    probe_dir: Path = typer.Argument(..., help="Probe directory with scores.feather"),
    preset: str = typer.Option("deconfounded-full", help="Annotation preset"),
):
    """Compute per-head eval metrics → eval.json (runs locally, no SLURM)."""
    result = subprocess.run(
        ["uv", "run", "python", "pipeline/eval.py", "--probe-dir", str(probe_dir), "--preset", preset],
        cwd=ROOT,
    )
    raise typer.Exit(result.returncode)


@app.command()
def log_eval(
    probe_dir: Path = typer.Argument(..., help="Probe directory containing eval.json"),
    project: str = typer.Option("gfm-probes", help="W&B project"),
    name: str = typer.Option(None, help="W&B run name (default: probe dir name)"),
):
    """Upload eval.json to wandb as a Table for cross-run comparison."""
    import json

    import wandb

    eval_path = probe_dir / "eval.json"
    if not eval_path.exists():
        rprint(f"[red]No eval.json in {probe_dir}[/]")
        raise typer.Exit(1)

    eval_data = json.loads(eval_path.read_text())
    run_name = name or probe_dir.name

    config = {}
    config_path = probe_dir / "config.json"
    if config_path.exists():
        config = json.loads(config_path.read_text())
    config["probe_dir"] = str(probe_dir)

    run = wandb.init(project=project, name=run_name, config=config, job_type="eval")

    table = wandb.Table(columns=["head", "kind", "auc", "correlation", "accuracy", "mse", "n"])
    aucs, corrs = [], []
    for head, metrics in eval_data.items():
        table.add_data(
            head, metrics.get("kind"), metrics.get("auc"), metrics.get("correlation"),
            metrics.get("accuracy"), metrics.get("mse"), metrics.get("n", 0),
        )
        if metrics.get("auc") is not None:
            aucs.append(metrics["auc"])
        if metrics.get("correlation") is not None:
            corrs.append(metrics["correlation"])

    wandb.log({"eval/heads": table})

    if aucs:
        run.summary["mean_auc"] = sum(aucs) / len(aucs)
    if corrs:
        run.summary["mean_correlation"] = sum(corrs) / len(corrs)
    if "pathogenic" in eval_data and "auc" in eval_data["pathogenic"]:
        run.summary["pathogenic_auc"] = eval_data["pathogenic"]["auc"]
    run.summary["n_heads"] = len(eval_data)

    artifact = wandb.Artifact(f"eval-{run_name}", type="eval")
    artifact.add_file(str(eval_path))
    if config_path.exists():
        artifact.add_file(str(config_path))
    wandb.log_artifact(artifact)

    wandb.finish()
    rprint(f"[green]Logged {len(eval_data)} heads to wandb:{project}/{run_name}[/]")


@app.command()
def pipeline(
    probe: Path = typer.Argument(..., help="Probe directory (e.g., $ACTS/probe_v11)"),
    labeled_only: bool = typer.Option(False, help="Skip VUS extraction"),
):
    """Full pipeline: extract → finalize → eval → build.

    Submits SLURM jobs with dependency chains and exits immediately.
    """
    probe_name = probe.name
    acts = probe.parent
    vus = acts.parent / "clinvar_evo2_vus"

    for f in ("weights.pt", "config.json"):
        if not (probe / f).exists():
            rprint(f"[red]Missing {probe / f} — run training first[/]")
            raise typer.Exit(1)

    # Clean stale SQLite WAL locks (NFS + WAL = deadlock)
    for base in (acts, vus):
        for suffix in ("-shm", "-wal"):
            (base / "activations" / f"index.sqlite{suffix}").unlink(missing_ok=True)

    # Datasets to process: always labeled, optionally VUS
    datasets = [("labeled", acts)]
    if not labeled_only:
        datasets.append(("vus", vus))

    rprint(f"[bold]=== Pipeline: {probe_name} ===[/]")

    # Extract → Finalize per dataset
    finalizes = []
    for name, path in datasets:
        ext = _sbatch("--array=0-7", "pipeline/extract.sh",
                       "--probe", str(probe), "--activations", str(path))
        fin = _sbatch(f"--dependency=afterok:{ext}",
                       "pipeline/finalize.sh", str(path / probe_name))
        rprint(f"  {name}: extract={ext} → finalize={fin}")
        finalizes.append(fin)

    # Eval + log-eval + build after all finalizes
    wait_for = ":".join(finalizes)
    eval_cmd = (
        f"cd ${{SLURM_SUBMIT_DIR}} && "
        f"uv run --frozen python pipeline/eval.py --probe-dir '{acts / probe_name}' && "
        f"uv run --frozen --extra train vv log-eval '{acts / probe_name}' && "
        f"uv run --frozen vv build --probe {probe_name}"
    )
    eval_id = _sbatch(f"--dependency=afterok:{wait_for}",
                      "--job-name=eval-build", "--gpus=1", "--time=02:00:00",
                      "--output=outputs/eval_build_%j.out", f"--wrap={eval_cmd}")
    rprint(f"  eval+build: {eval_id} (after {wait_for})")
    rprint("\n[green]Pipeline submitted.[/] Monitor: squeue -u $(whoami)")


if __name__ == "__main__":
    app()
