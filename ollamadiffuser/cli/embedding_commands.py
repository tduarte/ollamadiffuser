"""Textual-inversion embedding management CLI commands."""

import sys

import click
from rich.console import Console
from rich.table import Table
from rich import print as rprint

console = Console()


@click.group()
def embedding():
    """Textual-inversion embedding management commands.

    Download embeddings with `ollamadiffuser civitai pull <url>`, then load one
    onto a running model here.
    """
    pass


@embedding.command()
@click.argument("name")
def load(name: str):
    """Load an embedding into the current model."""
    from ..core.utils.embedding_manager import embedding_manager

    rprint(f"[blue]Loading embedding: {name}[/blue]")
    if embedding_manager.load_embedding(name):
        info = embedding_manager.get_embedding_info(name) or {}
        rprint(f"[green]Embedding {name} loaded![/green]")
        if info.get("token"):
            rprint(f"[yellow]Use it in prompts with:[/yellow] {info['token']}")
    else:
        rprint(f"[red]Failed to load embedding {name}![/red]")
        sys.exit(1)


@embedding.command(name="list")
def list_cmd():
    """List installed embeddings."""
    from ..core.utils.embedding_manager import embedding_manager

    embeddings = embedding_manager.list_embeddings()
    if not embeddings:
        rprint("[yellow]No embeddings installed.[/yellow]")
        return
    table = Table(title="Embeddings")
    table.add_column("Name", style="cyan")
    table.add_column("Token")
    table.add_column("Base model")
    table.add_column("Source")
    table.add_column("Size")
    for name, info in embeddings.items():
        table.add_row(name, info.get("token", "-"), info.get("base_model") or "-",
                      info.get("source", "-"), info.get("size", "-"))
    console.print(table)


@embedding.command()
@click.argument("name")
def show(name: str):
    """Show embedding details."""
    from ..core.utils.embedding_manager import embedding_manager

    info = embedding_manager.get_embedding_info(name)
    if not info:
        rprint(f"[red]Embedding {name} not found.[/red]")
        sys.exit(1)
    rprint(f"[bold cyan]Embedding: {name}[/bold cyan]")
    rprint(f"Token: {info.get('token', 'Unknown')}")
    rprint(f"Weight file: {info.get('weight_name', 'Unknown')}")
    rprint(f"Path: {info.get('path', 'Unknown')}")
    rprint(f"Base model: {info.get('base_model') or 'Unknown'}")
    rprint(f"Source: {info.get('source', 'Unknown')}")
    if info.get("trained_words"):
        rprint(f"[yellow]Trigger words:[/yellow] {', '.join(info['trained_words'])}")


@embedding.command()
@click.argument("name")
@click.confirmation_option(prompt="Are you sure you want to delete this embedding?")
def rm(name: str):
    """Remove an embedding (in-place imports are unregistered, files kept)."""
    from ..core.utils.embedding_manager import embedding_manager

    if embedding_manager.remove_embedding(name):
        rprint(f"[green]Embedding {name} removed![/green]")
    else:
        rprint(f"[red]Failed to remove embedding {name}![/red]")
        sys.exit(1)
