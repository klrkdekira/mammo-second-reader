"""Create a writable report draft while preserving the template."""

from __future__ import annotations

from pathlib import Path

import click


def create_report_draft(template: Path, output: Path, *, force: bool = False) -> bool:
    """Copy the report template without changing its relative links."""
    template = template.resolve()
    output = output.resolve()
    if template == output:
        raise ValueError("The report template and writable draft must differ.")
    if output.exists() and not force:
        return False
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(template.read_text())
    return True


@click.command()
@click.option(
    "--template",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    required=True,
)
@click.option(
    "--output",
    type=click.Path(dir_okay=False, path_type=Path),
    required=True,
)
@click.option("--force", is_flag=True, default=False)
def cli(template: Path, output: Path, force: bool) -> None:
    """Create the editable report draft if it does not already exist."""
    try:
        created = create_report_draft(template, output, force=force)
    except (OSError, ValueError) as exc:
        raise click.ClickException(str(exc)) from exc
    if created:
        click.echo(f"Created writable report draft at {output}")
    else:
        click.echo(f"Kept existing writable report draft at {output}")


if __name__ == "__main__":
    cli()
