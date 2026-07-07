"""Hugging Face Hub search / install CLI commands for OllamaDiffuser."""

import sys
from typing import Optional

import click
from rich.console import Console
from rich.table import Table
from rich import print as rprint

from .model_commands import OllamaStyleProgress
from ..core.utils import hf_client
from ..core.utils.hf_client import hf_manager, HuggingFaceError

console = Console()


@click.group(name="huggingface")
def huggingface():
    """Search and install models / LoRAs from Hugging Face Hub."""
    pass


def _progress_cb():
    tracker = OllamaStyleProgress(console)
    return tracker.update


@huggingface.command()
@click.argument("query")
@click.option("--type", "-t", "model_type",
              type=click.Choice(["lora", "checkpoint"]),
              help="Filter by kind: LoRA adapters or full diffusers checkpoints.")
@click.option("--base-model", "-b",
              help="Filter by base model tag (e.g. 'black-forest-labs/FLUX.2-klein-9B').")
@click.option("--limit", "-l", default=20, show_default=True, help="Max results.")
@click.option("--files", is_flag=True,
              help="Also list each result's .safetensors weight file(s) (slower).")
def search(query: str, model_type: Optional[str], base_model: Optional[str],
           limit: int, files: bool):
    """Search Hugging Face by keyword, sorted by downloads."""
    try:
        rows = hf_client.search(query, model_type=model_type, base_model=base_model,
                                limit=limit, include_files=files)
    except HuggingFaceError as e:
        rprint(f"[red]Search failed: {e}[/red]")
        sys.exit(1)

    if not rows:
        rprint(f"[yellow]No results for '{query}'.[/yellow]")
        return

    table = Table(title=f"Hugging Face results for '{query}'")
    table.add_column("Repo ID", style="cyan", no_wrap=True)
    table.add_column("Kind")
    table.add_column("Base model")
    table.add_column("Downloads", justify="right")
    table.add_column("Likes", justify="right")
    if files:
        table.add_column("Weight file(s)")
    for r in rows:
        kind = "LoRA" if r.get("is_lora") else (r.get("pipeline_tag") or "model")
        cells = [
            r.get("repo_id") or "-",
            kind,
            r.get("base_model") or "-",
            f"{r.get('downloads', 0):,}",
            f"{r.get('likes', 0):,}",
        ]
        if files:
            weights = r.get("lora_weights") or []
            cells.append("\n".join(weights) if weights else "-")
        table.add_row(*cells)
    console.print(table)
    rprint("[dim]LoRA:[/dim] ollamadiffuser hf pull <Repo ID> [--weight-name <f>]")
    rprint("[dim]Model:[/dim] ollamadiffuser hf pull <Repo ID> --as-model --type <flux|sdxl|sd3|qwen|...>")


@huggingface.command()
@click.argument("repo_id")
def info(repo_id: str):
    """Show metadata and weight files for a Hugging Face repo without downloading."""
    try:
        row = hf_client.get_model_info(repo_id)
    except HuggingFaceError as e:
        rprint(f"[red]{e}[/red]")
        sys.exit(1)

    kind = "LoRA" if row.get("is_lora") else (row.get("pipeline_tag") or "model")
    rprint(f"[bold cyan]{row['repo_id']}[/bold cyan]")
    rprint(f"Kind: {kind}")
    if row.get("base_model"):
        rprint(f"Base model: {row['base_model']}")
    rprint(f"Downloads: {row.get('downloads', 0):,}   Likes: {row.get('likes', 0):,}")
    weights = row.get("lora_weights") or []
    if weights:
        rprint("[yellow]Weight files (.safetensors):[/yellow]")
        for w in weights:
            rprint(f"  - {w}")
    if row.get("tags"):
        rprint(f"[dim]Tags: {', '.join(row['tags'][:12])}[/dim]")
    if row.get("is_lora"):
        pick = f" --weight-name '{weights[0]}'" if len(weights) == 1 else ""
        rprint(f"\n[dim]Install with:[/dim] ollamadiffuser hf pull {repo_id}{pick}")
    else:
        rprint(f"\n[dim]Install with:[/dim] ollamadiffuser hf pull {repo_id} --as-model --type <type>")


@huggingface.command()
@click.argument("repo_id")
@click.option("--weight-name", "-w", help="Specific .safetensors weight file (LoRA repos).")
@click.option("--alias", "-a", help="Local name to register the LoRA/model under.")
@click.option("--as-model", "as_model", is_flag=True,
              help="Install a full diffusers model (into the registry) instead of a LoRA.")
@click.option("--type", "-t", "model_type",
              type=click.Choice(["flux", "sdxl", "sd15", "sd3", "mlx", "qwen"]),
              help="Model type (required with --as-model).")
@click.option("--variant", help="Model variant (with --as-model), e.g. 'fp16'.")
@click.option("--force", "-f", is_flag=True, help="Re-download even if already installed.")
def pull(repo_id: str, weight_name: Optional[str], alias: Optional[str],
         as_model: bool, model_type: Optional[str], variant: Optional[str], force: bool):
    """Install a LoRA (default) or a full model (--as-model) from Hugging Face.

    REPO_ID is a Hugging Face repo like 'diroverflo/FLux_Klein_9B_NSFW'.
    """
    if as_model:
        if not model_type:
            rprint("[red]--as-model requires --type (flux, sdxl, sd15, sd3, mlx, qwen).[/red]")
            sys.exit(1)
        rprint(f"[blue]Installing Hugging Face model: {repo_id}[/blue]")
        try:
            result = hf_manager.pull_model(
                repo_id, model_type=model_type, alias=alias, variant=variant,
                force=force, progress_callback=_progress_cb())
        except HuggingFaceError as e:
            rprint(f"[red]Install failed: {e}[/red]")
            sys.exit(1)
        except KeyboardInterrupt:
            rprint("\n[yellow]Download cancelled by user[/yellow]")
            sys.exit(1)
        name = result["name"]
        rprint(f"[green]✅ Installed model '{name}' ({result['model_type']})[/green]")
        rprint(f"[dim]Run it with:[/dim] ollamadiffuser run {name}")
        return

    rprint(f"[blue]Installing Hugging Face LoRA: {repo_id}[/blue]")
    try:
        result = hf_manager.pull_lora(
            repo_id, weight_name=weight_name, alias=alias,
            progress_callback=_progress_cb())
    except HuggingFaceError as e:
        rprint(f"[red]Install failed: {e}[/red]")
        sys.exit(1)
    except KeyboardInterrupt:
        rprint("\n[yellow]Download cancelled by user[/yellow]")
        sys.exit(1)

    name = result["name"]
    rprint(f"[green]✅ Installed LoRA '{name}'[/green]")
    if result.get("weight_name"):
        rprint(f"[dim]Weight file:[/dim] {result['weight_name']}")
    if result.get("base_model"):
        rprint(f"[dim]Base model:[/dim] {result['base_model']}")
    rprint(f"[dim]Load it onto a running model with:[/dim] ollamadiffuser lora load {name}")


# Alias: `ollamadiffuser hf ...`
@click.group(name="hf")
def hf():
    """Alias for the 'huggingface' command group."""
    pass


hf.add_command(search)
hf.add_command(info)
hf.add_command(pull)
