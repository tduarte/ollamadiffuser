"""Standalone VAE management CLI commands."""

import sys

import click
from rich.console import Console
from rich.table import Table
from rich import print as rprint

console = Console()


@click.group()
def vae():
    """VAE management commands.

    Download VAEs with `ollamadiffuser civitai pull <url>`, then attach one to a
    running model here. The VAE stays attached until the model is reloaded.
    """
    pass


@vae.command()
@click.argument("name")
def load(name: str):
    """Attach a VAE to the current model."""
    from ..core.utils.vae_manager import vae_manager

    rprint(f"[blue]Attaching VAE: {name}[/blue]")
    if vae_manager.attach_vae(name):
        rprint(f"[green]VAE {name} attached![/green]")
    else:
        rprint(f"[red]Failed to attach VAE {name}![/red]")
        sys.exit(1)


@vae.command()
def restore():
    """Restore the current model's original VAE."""
    from ..core.utils.vae_manager import vae_manager

    if vae_manager.restore_default_vae():
        rprint("[green]Original VAE restored![/green]")
    else:
        rprint("[yellow]No replaced VAE to restore (or no model loaded).[/yellow]")


@vae.command(name="list")
def list_cmd():
    """List installed VAEs."""
    from ..core.utils.vae_manager import vae_manager

    vaes = vae_manager.list_vaes()
    if not vaes:
        rprint("[yellow]No VAEs installed.[/yellow]")
        return
    table = Table(title="VAEs")
    table.add_column("Name", style="cyan")
    table.add_column("Base model")
    table.add_column("Source")
    table.add_column("Size")
    for name, info in vaes.items():
        table.add_row(name, info.get("base_model") or "-",
                      info.get("source", "-"), info.get("size", "-"))
    console.print(table)


@vae.command()
@click.argument("name")
def show(name: str):
    """Show VAE details."""
    from ..core.utils.vae_manager import vae_manager

    info = vae_manager.get_vae_info(name)
    if not info:
        rprint(f"[red]VAE {name} not found.[/red]")
        sys.exit(1)
    rprint(f"[bold cyan]VAE: {name}[/bold cyan]")
    rprint(f"Weight file: {info.get('weight_name', 'Unknown')}")
    rprint(f"Path: {info.get('path', 'Unknown')}")
    rprint(f"Base model: {info.get('base_model') or 'Unknown'}")
    rprint(f"Source: {info.get('source', 'Unknown')}")


@vae.command()
@click.argument("name")
@click.confirmation_option(prompt="Are you sure you want to delete this VAE?")
def rm(name: str):
    """Remove a VAE (in-place imports are unregistered, files kept)."""
    from ..core.utils.vae_manager import vae_manager

    if vae_manager.remove_vae(name):
        rprint(f"[green]VAE {name} removed![/green]")
    else:
        rprint(f"[red]Failed to remove VAE {name}![/red]")
        sys.exit(1)
