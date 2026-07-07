"""CivitAI / CivitAI Red CLI commands for OllamaDiffuser."""

import sys
from typing import Optional

import click
from rich.console import Console
from rich.table import Table
from rich import print as rprint

from .model_commands import OllamaStyleProgress
from ..core.utils import civitai_client
from ..core.utils.civitai_client import civitai_manager, CivitaiError

console = Console()

# CivitAI model.type values accepted for --type / search --type.
_TYPE_CHOICES = ["Checkpoint", "LORA", "LoCon", "TextualInversion", "VAE"]


@click.group()
def civitai():
    """Download models from CivitAI (civitai.com) and CivitAI Red (civitai.red)."""
    pass


def _progress_cb():
    tracker = OllamaStyleProgress(console)
    return tracker.update


@civitai.command()
@click.argument("ref")
@click.option("--type", "-t", "model_type",
              type=click.Choice(["sd15", "sdxl", "sd3", "flux"]),
              help="Override the inferred model type (checkpoints).")
@click.option("--alias", "-a", help="Local name to register the model under.")
@click.option("--force", "-f", is_flag=True, help="Re-download / overwrite if it exists.")
@click.option("--red", is_flag=True, help="Treat a bare id/host-less ref as civitai.red.")
@click.option("--experimental", is_flag=True,
              help="Attempt FLUX/SD3 single-file checkpoints (may be incomplete).")
def pull(ref: str, model_type: Optional[str], alias: Optional[str], force: bool,
         red: bool, experimental: bool):
    """Download a model by URL or version id.

    REF is a civitai.com/civitai.red URL, a ?modelVersionId= URL, an
    /api/download/models/<id> URL, or a bare model-version id.
    """
    rprint(f"[blue]Resolving CivitAI reference: {ref}[/blue]")
    try:
        result = civitai_manager.pull(
            ref, model_type=model_type, alias=alias, force=force, red=red,
            allow_experimental=experimental, progress_callback=_progress_cb())
    except CivitaiError as e:
        rprint(f"[red]CivitAI pull failed: {e}[/red]")
        sys.exit(1)
    except KeyboardInterrupt:
        rprint("\n[yellow]Download cancelled by user[/yellow]")
        sys.exit(1)

    name = result["name"]
    rprint(f"[green]✅ Installed '{name}' ({result.get('model_type') or result['content_category']})[/green]")
    if result.get("trained_words"):
        rprint(f"[yellow]Trigger words:[/yellow] {', '.join(result['trained_words'])}")
    if result["content_category"] == "checkpoint":
        rprint(f"[dim]Run it with:[/dim] ollamadiffuser run {name}")
    else:
        rprint(f"[dim]Load it onto a running model with:[/dim] ollamadiffuser lora load {name}")


@civitai.command()
@click.argument("query")
@click.option("--type", "-t", "types", type=click.Choice(_TYPE_CHOICES),
              help="Filter by CivitAI model type.")
@click.option("--base-model", "-b", help="Filter by base model (e.g. 'SDXL 1.0', 'Pony').")
@click.option("--limit", "-l", default=20, show_default=True, help="Max results.")
@click.option("--nsfw", is_flag=True, help="Include mature content.")
@click.option("--red", is_flag=True, help="Search via civitai.red.")
def search(query: str, types: Optional[str], base_model: Optional[str], limit: int,
           nsfw: bool, red: bool):
    """Search CivitAI by keyword."""
    base = civitai_client.RED_BASE if red else civitai_client.DEFAULT_BASE
    try:
        rows = civitai_client.search(query, types=types, base_model=base_model,
                                     limit=limit, nsfw=nsfw, base_url=base)
    except CivitaiError as e:
        rprint(f"[red]Search failed: {e}[/red]")
        sys.exit(1)

    if not rows:
        rprint(f"[yellow]No results for '{query}'.[/yellow]")
        return

    table = Table(title=f"CivitAI results for '{query}'")
    table.add_column("Version ID", style="cyan", no_wrap=True)
    table.add_column("Name")
    table.add_column("Type")
    table.add_column("Base model")
    table.add_column("Downloads", justify="right")
    table.add_column("NSFW", justify="center")
    for r in rows:
        table.add_row(
            str(r.get("version_id") or "-"),
            (r.get("name") or "")[:48],
            r.get("type") or "-",
            r.get("base_model") or "-",
            f"{r.get('download_count', 0):,}",
            "🔞" if r.get("nsfw") else "",
        )
    console.print(table)
    rprint("[dim]Pull one with:[/dim] ollamadiffuser civitai pull <Version ID>")


@civitai.command()
@click.argument("ref")
@click.option("--red", is_flag=True, help="Treat a bare id/host-less ref as civitai.red.")
def info(ref: str, red: bool):
    """Show metadata for a CivitAI model/version without downloading."""
    try:
        parsed = civitai_client.parse_civitai_ref(ref, red=red)
        version = civitai_client.resolve(parsed)
    except CivitaiError as e:
        rprint(f"[red]{e}[/red]")
        sys.exit(1)

    rprint(f"[bold cyan]{version.name}[/bold cyan]")
    rprint(f"Source: {version.base_url}")
    rprint(f"Model id / Version id: {version.model_id} / {version.version_id}")
    rprint(f"CivitAI type: {version.civitai_type}  ->  category: {version.content_category}")
    rprint(f"Base model: {version.base_model}  ->  model_type: {version.model_type or '[unmapped]'}")
    if version.nsfw:
        rprint("[red]Mature content[/red]")
    if version.trained_words:
        rprint(f"[yellow]Trigger words:[/yellow] {', '.join(version.trained_words)}")
    try:
        primary = civitai_client.select_primary_file(version)
        size_mb = (primary.get('sizeKB') or 0) / 1024
        rprint(f"Primary file: {primary.get('name')} ({size_mb:.0f} MB)")
    except CivitaiError:
        pass
    if version.description:
        rprint(f"[dim]{version.description[:300]}[/dim]")
    rprint(f"\n[dim]Pull it with:[/dim] ollamadiffuser civitai pull {version.version_id}")


@civitai.command(name="import")
@click.argument("path", type=click.Path(exists=True))
@click.option("--type", "-t", "content_type",
              type=click.Choice(["checkpoint", "lora", "embedding", "vae"]),
              help="Force the content type (else inferred from sidecar/hash/folder).")
@click.option("--model-type", "-m", "model_type",
              type=click.Choice(["sd15", "sdxl", "sd3", "flux"]),
              help="Force the model type for checkpoints.")
@click.option("--alias", "-a", help="Register a single file under this name.")
@click.option("--recursive", "-r", is_flag=True, help="Recurse into subdirectories.")
@click.option("--no-lookup", is_flag=True,
              help="Do not query CivitAI by hash; use sidecars/flags only (offline).")
def import_cmd(path: str, content_type: Optional[str], model_type: Optional[str],
               alias: Optional[str], recursive: bool, no_lookup: bool):
    """Register already-downloaded local models/LoRAs in place (no copy)."""
    rprint(f"[blue]Importing from {path}[/blue]")
    try:
        results = civitai_manager.import_local(
            path, content_type=content_type, model_type=model_type, alias=alias,
            recursive=recursive, do_lookup=not no_lookup,
            progress_callback=_progress_cb())
    except CivitaiError as e:
        rprint(f"[red]Import failed: {e}[/red]")
        sys.exit(1)

    table = Table(title="Import results")
    table.add_column("File")
    table.add_column("Registered as")
    table.add_column("Category")
    table.add_column("Model type")
    table.add_column("Status")
    registered = 0
    for r in results:
        fname = r.get("file", "").split("/")[-1]
        if r.get("skipped"):
            table.add_row(fname, "-", "-", "-", f"[yellow]skipped: {r['skipped']}[/yellow]")
        else:
            registered += 1
            table.add_row(fname, r.get("name", "-"), r.get("content_category", "-"),
                          r.get("model_type") or "-", "[green]ok[/green]")
    console.print(table)
    rprint(f"[green]Registered {registered}/{len(results)} file(s).[/green]")
