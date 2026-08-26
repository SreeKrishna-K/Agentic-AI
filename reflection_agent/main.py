from dotenv import load_dotenv
load_dotenv()

from langchain_core.messages import HumanMessage
from langgraph.graph import END, StateGraph, MessagesState
from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule
from rich.table import Table
from rich.text import Text

from chains import generate_chain, reflect_chain


REFLECT = "reflect"
GENERATE = "generate"
MAX_ITERATIONS = 4

console = Console()


def _log_rule(title: str, style: str = "bold cyan") -> None:
    console.print()
    console.print(Rule(title, style=style))


def _log_panel(title: str, content: str, border_style: str) -> None:
    body = (content or "").strip()
    console.print(
        Panel(
            Text(body) if body else Text("(empty)", style="dim"),
            title=title,
            title_align="left",
            border_style=border_style,
            padding=(1, 2),
        )
    )


def _generation_count(state: MessagesState) -> int:
    return sum(1 for message in state["messages"] if message.type == "ai")


def generation_node(state: MessagesState):
    draft_number = _generation_count(state) + 1
    _log_rule(f"GENERATE  ·  draft {draft_number}/{MAX_ITERATIONS}", "bold green")

    response = generate_chain.invoke({
        "messages": state["messages"]
    })

    _log_panel("Generated tweet", response.content, "green")
    return {"messages": [response]}


def reflection_node(state: MessagesState):
    _log_rule("REFLECT  ·  grading tweet", "bold yellow")

    result = reflect_chain.invoke({
        "messages": state["messages"]
    })

    status = (
        "[bold green]APPROVED[/bold green]"
        if result.approved
        else "[bold red]NEEDS REVISION[/bold red]"
    )
    _log_panel(f"Critique  ·  {status}", result.critique, "yellow")

    return {
        "messages": [
            HumanMessage(
                content=result.critique,
                additional_kwargs={"approved": result.approved},
            )
        ]
    }


builder = StateGraph(MessagesState)

builder.add_node(GENERATE, generation_node)
builder.add_node(REFLECT, reflection_node)

builder.set_entry_point(GENERATE)


def should_continue(state: MessagesState):
    generation_count = _generation_count(state)
    hit_limit = generation_count >= MAX_ITERATIONS
    nxt = END if hit_limit else REFLECT

    reason = (
        f"hit MAX_ITERATIONS ({generation_count}/{MAX_ITERATIONS})"
        if hit_limit
        else f"still under limit ({generation_count}/{MAX_ITERATIONS}) — send to critic"
    )
    console.print(
        f"  [dim]route[/dim]  generate → [bold]{nxt if nxt != END else 'END'}[/bold]"
        f"  [dim]· {reason}[/dim]"
    )
    return nxt


def should_revise(state: MessagesState):
    last_message = state["messages"][-1]
    approved = last_message.additional_kwargs.get("approved", False)
    nxt = END if approved else GENERATE

    reason = (
        "critic approved the tweet — stop"
        if approved
        else "critic requested changes — rewrite"
    )
    console.print(
        f"  [dim]route[/dim]  reflect → [bold]{nxt if nxt != END else 'END'}[/bold]"
        f"  [dim]· {reason}[/dim]"
    )
    return nxt


def log_transcript(messages) -> None:
    _log_rule("FINAL TRANSCRIPT", "bold magenta")

    summary = Table(show_header=True, header_style="bold", box=None, pad_edge=False)
    summary.add_column("#", style="dim", width=3)
    summary.add_column("Role", width=18)
    summary.add_column("Preview")

    critique_index = 0
    for i, message in enumerate(messages, start=1):
        is_critique = (
            message.type == "human"
            and "approved" in getattr(message, "additional_kwargs", {})
        )
        if message.type == "ai":
            role = "assistant (tweet)"
        elif is_critique:
            critique_index += 1
            approved = message.additional_kwargs.get("approved")
            role = f"critic #{critique_index}" + (" ✓" if approved else " ✗")
        else:
            role = "user (request)"

        preview = " ".join((message.content or "").split())
        if len(preview) > 80:
            preview = preview[:77] + "..."
        summary.add_row(str(i), role, preview)

    console.print(summary)
    console.print()

    draft_number = 0
    critique_index = 0
    for message in messages:
        is_critique = (
            message.type == "human"
            and "approved" in getattr(message, "additional_kwargs", {})
        )
        if message.type == "ai":
            draft_number += 1
            title, border = f"Tweet draft {draft_number}", "green"
        elif is_critique:
            critique_index += 1
            approved = message.additional_kwargs.get("approved")
            verdict = "APPROVED" if approved else "NEEDS REVISION"
            title, border = f"Critique {critique_index}  ·  {verdict}", "yellow"
        else:
            title, border = "Original request", "cyan"

        _log_panel(title, message.content, border)


builder.add_conditional_edges(GENERATE, should_continue)
builder.add_conditional_edges(REFLECT, should_revise)

graph = builder.compile()


if __name__ == "__main__":

    inputs = {
        "messages": [
            HumanMessage(
                content="""Make this tweet better:

@LangChainAI
— newly Tool Calling feature is seriously underrated.

After a long wait, it's here - making the implementation of agents across different models with function calling - super easy.

Made a video covering their newest blog post"""
            )
        ]
    }

    console.print()
    console.print(
        Text("Reflection agent", style="bold white"),
        Text("  tweet generate → critique → revise", style="dim"),
    )
    _log_panel("Original request", inputs["messages"][0].content, "cyan")

    response = graph.invoke(inputs)
    log_transcript(response["messages"])