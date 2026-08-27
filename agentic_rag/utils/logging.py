"""Shared Rich logging helpers for the agentic_rag package."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import Any, Final

from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule
from rich.table import Table
from rich.text import Text

console: Final[Console] = Console()

DEFAULT_RULE_STYLE: Final[str] = "bold cyan"
_EMPTY = "(empty)"


def log_rule(title: str, style: str = DEFAULT_RULE_STYLE) -> None:
    """Print a section divider."""
    console.print()
    console.print(Rule(title, style=style))


def log_panel(title: str, content: str, border_style: str = "cyan") -> None:
    """Print a titled panel. Empty content is shown as a dim placeholder."""
    body = (content or "").strip()
    console.print(
        Panel(
            Text(body) if body else Text(_EMPTY, style="dim"),
            title=title,
            title_align="left",
            border_style=border_style,
            padding=(1, 2),
        )
    )


def log_table(
    title: str,
    columns: Sequence[str],
    rows: Iterable[Sequence[Any]],
    *,
    justify: Mapping[str, str] | None = None,
    show_lines: bool = True,
) -> None:
    """Build and print a table. `justify` maps column header → Rich justify value."""
    table = Table(title=title, show_lines=show_lines)
    for name in columns:
        kwargs: dict[str, str] = {"overflow": "fold"}
        if justify and name in justify:
            kwargs["justify"] = justify[name]
        table.add_column(name, **kwargs)
    for row in rows:
        table.add_row(*[str(cell) for cell in row])
    console.print(table)


def preview_text(text: str | None, *, max_chars: int = 120) -> str:
    """Collapse whitespace and truncate for table/log previews."""
    collapsed = " ".join((text or "").split())
    if len(collapsed) <= max_chars:
        return collapsed
    return f"{collapsed[: max_chars - 1]}…"
