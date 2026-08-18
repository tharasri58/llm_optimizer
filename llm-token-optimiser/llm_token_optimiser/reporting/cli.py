"""cli.py — Click-based CLI: `token-optimiser analyse <file>` (Section 8.3)."""
from __future__ import annotations

import click

from ..compression.compressor import compress_prompt
from .token_counter import estimate_cost


@click.group()
def main():
    """LLM Token Optimiser command-line tool."""


@main.command()
@click.argument("file", type=click.Path(exists=True))
@click.option("--target-ratio", default=0.5, show_default=True, help="Target compression ratio.")
@click.option("--model", default="gpt-4o-mini", show_default=True, help="Model used for token counting / pricing.")
def analyse(file: str, target_ratio: float, model: str):
    """Analyse token savings for a prompt saved in FILE."""
    with open(file, "r", encoding="utf-8") as f:
        prompt = f.read()

    result = compress_prompt(prompt, target_ratio=target_ratio, model=model)
    cost = estimate_cost(result.optimised_tokens, model=model)

    click.echo(f"Baseline tokens : {result.baseline_tokens}")
    click.echo(f"Optimised tokens: {result.optimised_tokens}")
    click.echo(f"Reduction       : {result.reduction_pct}%")
    click.echo(f"Estimated saving: ${cost} (at published {model} pricing)")


if __name__ == "__main__":
    main()
