from typing import Optional
import datetime
import typer
import questionary
from pathlib import Path
from functools import wraps
from rich.console import Console
from rich.panel import Panel
from rich.spinner import Spinner
from rich.live import Live
from rich.columns import Columns
from rich.markdown import Markdown
from rich.layout import Layout
from rich.text import Text
from rich.table import Table
from collections import deque
import time
from rich.tree import Tree
from rich import box
from rich.align import Align
from rich.rule import Rule

from tradingagents.graph.trading_graph import TradingAgentsGraph
from tradingagents.default_config import DEFAULT_CONFIG
from cli.models import AnalystType
from cli.utils import *
from cli.announcements import fetch_announcements, display_announcements
from cli.stats_handler import StatsCallbackHandler

console = Console()

app = typer.Typer(
    name="TradingAgents",
    help="TradingAgents CLI: Multi-Agents LLM Financial Trading Framework",
    add_completion=True,  # Enable shell completion
)

# Defaults applied when CLI flags are omitted. Bare `tradingagents` runs with
# these silently; pass `--interactive` to opt back into the step-by-step UI.
DEFAULTS = {
    "ticker": "SOUTHBANK.NS",
    "analysts": "market",
    "depth": "deep",
    "language": "English",
    "provider": "glm-anthropic",
    "quick_model": "glm-5.1",
    "deep_model": "glm-5.1",
    "horizon": "swing",
}
DEPTH_MAP = {"shallow": 1, "medium": 2, "deep": 3}
HORIZON_CHOICES = ("swing", "position", "long-term")

# Named bundles of flag values. `--profile <name>` applies these in one shot;
# explicit flags passed alongside still win (profile fills only the slots the
# user left at their default).
PROFILES = {
    "full": {
        "analysts": "market,social,news,fundamentals",
        "horizon": "long-term",
        "depth": "deep",
    },
    "short-term": {
        "analysts": "market",
        "horizon": "swing",
        "depth": "deep",
    },
}


# Create a deque to store recent messages with a maximum length
class MessageBuffer:
    # Fixed teams that always run (not user-selectable)
    FIXED_AGENTS = {
        "Research Team": ["Bull Researcher", "Bear Researcher", "Research Manager"],
        "Trading Team": ["Trader"],
        "Risk Management": ["Aggressive Analyst", "Neutral Analyst", "Conservative Analyst"],
        "Portfolio Management": ["Portfolio Manager"],
    }

    # Analyst name mapping
    ANALYST_MAPPING = {
        "market": "Market Analyst",
        "social": "Sentiment Analyst",
        "news": "News Analyst",
        "fundamentals": "Fundamentals Analyst",
    }

    # Report section mapping: section -> (analyst_key for filtering, finalizing_agent)
    # analyst_key: which analyst selection controls this section (None = always included)
    # finalizing_agent: which agent must be "completed" for this report to count as done
    REPORT_SECTIONS = {
        "market_report": ("market", "Market Analyst"),
        "sentiment_report": ("social", "Sentiment Analyst"),
        "news_report": ("news", "News Analyst"),
        "fundamentals_report": ("fundamentals", "Fundamentals Analyst"),
        "investment_plan": (None, "Research Manager"),
        "trader_investment_plan": (None, "Trader"),
        "final_trade_decision": (None, "Portfolio Manager"),
    }

    def __init__(self, max_length=100):
        self.messages = deque(maxlen=max_length)
        self.tool_calls = deque(maxlen=max_length)
        self.current_report = None
        self.final_report = None  # Store the complete final report
        self.agent_status = {}
        self.current_agent = None
        self.report_sections = {}
        self.selected_analysts = []
        self._processed_message_ids = set()

    def init_for_analysis(self, selected_analysts):
        """Initialize agent status and report sections based on selected analysts.

        Args:
            selected_analysts: List of analyst type strings (e.g., ["market", "news"])
        """
        self.selected_analysts = [a.lower() for a in selected_analysts]

        # Build agent_status dynamically
        self.agent_status = {}

        # Add selected analysts
        for analyst_key in self.selected_analysts:
            if analyst_key in self.ANALYST_MAPPING:
                self.agent_status[self.ANALYST_MAPPING[analyst_key]] = "pending"

        # Add fixed teams
        for team_agents in self.FIXED_AGENTS.values():
            for agent in team_agents:
                self.agent_status[agent] = "pending"

        # Build report_sections dynamically
        self.report_sections = {}
        for section, (analyst_key, _) in self.REPORT_SECTIONS.items():
            if analyst_key is None or analyst_key in self.selected_analysts:
                self.report_sections[section] = None

        # Reset other state
        self.current_report = None
        self.final_report = None
        self.current_agent = None
        self.messages.clear()
        self.tool_calls.clear()
        self._processed_message_ids.clear()

    def get_completed_reports_count(self):
        """Count reports that are finalized (their finalizing agent is completed).

        A report is considered complete when:
        1. The report section has content (not None), AND
        2. The agent responsible for finalizing that report has status "completed"

        This prevents interim updates (like debate rounds) from counting as completed.
        """
        count = 0
        for section in self.report_sections:
            if section not in self.REPORT_SECTIONS:
                continue
            _, finalizing_agent = self.REPORT_SECTIONS[section]
            # Report is complete if it has content AND its finalizing agent is done
            has_content = self.report_sections.get(section) is not None
            agent_done = self.agent_status.get(finalizing_agent) == "completed"
            if has_content and agent_done:
                count += 1
        return count

    def add_message(self, message_type, content):
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        self.messages.append((timestamp, message_type, content))

    def add_tool_call(self, tool_name, args):
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        self.tool_calls.append((timestamp, tool_name, args))

    def update_agent_status(self, agent, status):
        if agent in self.agent_status:
            self.agent_status[agent] = status
            self.current_agent = agent

    def update_report_section(self, section_name, content):
        if section_name in self.report_sections:
            self.report_sections[section_name] = content
            self._update_current_report()

    def _update_current_report(self):
        # For the panel display, only show the most recently updated section
        latest_section = None
        latest_content = None

        # Find the most recently updated section
        for section, content in self.report_sections.items():
            if content is not None:
                latest_section = section
                latest_content = content
               
        if latest_section and latest_content:
            # Format the current section for display
            section_titles = {
                "market_report": "Market Analysis",
                "sentiment_report": "Social Sentiment",
                "news_report": "News Analysis",
                "fundamentals_report": "Fundamentals Analysis",
                "investment_plan": "Research Team Decision",
                "trader_investment_plan": "Trading Team Plan",
                "final_trade_decision": "Portfolio Management Decision",
            }
            self.current_report = (
                f"### {section_titles[latest_section]}\n{latest_content}"
            )

        # Update the final complete report
        self._update_final_report()

    def _update_final_report(self):
        report_parts = []

        # Analyst Team Reports - use .get() to handle missing sections
        analyst_sections = ["market_report", "sentiment_report", "news_report", "fundamentals_report"]
        if any(self.report_sections.get(section) for section in analyst_sections):
            report_parts.append("## Analyst Team Reports")
            if self.report_sections.get("market_report"):
                report_parts.append(
                    f"### Market Analysis\n{self.report_sections['market_report']}"
                )
            if self.report_sections.get("sentiment_report"):
                report_parts.append(
                    f"### Social Sentiment\n{self.report_sections['sentiment_report']}"
                )
            if self.report_sections.get("news_report"):
                report_parts.append(
                    f"### News Analysis\n{self.report_sections['news_report']}"
                )
            if self.report_sections.get("fundamentals_report"):
                report_parts.append(
                    f"### Fundamentals Analysis\n{self.report_sections['fundamentals_report']}"
                )

        # Research Team Reports
        if self.report_sections.get("investment_plan"):
            report_parts.append("## Research Team Decision")
            report_parts.append(f"{self.report_sections['investment_plan']}")

        # Trading Team Reports
        if self.report_sections.get("trader_investment_plan"):
            report_parts.append("## Trading Team Plan")
            report_parts.append(f"{self.report_sections['trader_investment_plan']}")

        # Portfolio Management Decision
        if self.report_sections.get("final_trade_decision"):
            report_parts.append("## Portfolio Management Decision")
            report_parts.append(f"{self.report_sections['final_trade_decision']}")

        self.final_report = "\n\n".join(report_parts) if report_parts else None


message_buffer = MessageBuffer()


def create_layout():
    layout = Layout()
    layout.split_column(
        Layout(name="header", size=3),
        Layout(name="main"),
        Layout(name="footer", size=3),
    )
    layout["main"].split_column(
        Layout(name="upper", ratio=3), Layout(name="analysis", ratio=5)
    )
    layout["upper"].split_row(
        Layout(name="progress", ratio=2), Layout(name="messages", ratio=3)
    )
    return layout


def format_tokens(n):
    """Format token count for display."""
    if n >= 1000:
        return f"{n/1000:.1f}k"
    return str(n)


def update_display(layout, spinner_text=None, stats_handler=None, start_time=None):
    # Header with welcome message
    layout["header"].update(
        Panel(
            "[bold green]Welcome to TradingAgents CLI[/bold green]\n"
            "[dim]© [Tauric Research](https://github.com/TauricResearch)[/dim]",
            title="Welcome to TradingAgents",
            border_style="green",
            padding=(1, 2),
            expand=True,
        )
    )

    # Progress panel showing agent status
    progress_table = Table(
        show_header=True,
        header_style="bold magenta",
        show_footer=False,
        box=box.SIMPLE_HEAD,  # Use simple header with horizontal lines
        title=None,  # Remove the redundant Progress title
        padding=(0, 2),  # Add horizontal padding
        expand=True,  # Make table expand to fill available space
    )
    progress_table.add_column("Team", style="cyan", justify="center", width=20)
    progress_table.add_column("Agent", style="green", justify="center", width=20)
    progress_table.add_column("Status", style="yellow", justify="center", width=20)

    # Group agents by team - filter to only include agents in agent_status
    all_teams = {
        "Analyst Team": [
            "Market Analyst",
            "Sentiment Analyst",
            "News Analyst",
            "Fundamentals Analyst",
        ],
        "Research Team": ["Bull Researcher", "Bear Researcher", "Research Manager"],
        "Trading Team": ["Trader"],
        "Risk Management": ["Aggressive Analyst", "Neutral Analyst", "Conservative Analyst"],
        "Portfolio Management": ["Portfolio Manager"],
    }

    # Filter teams to only include agents that are in agent_status
    teams = {}
    for team, agents in all_teams.items():
        active_agents = [a for a in agents if a in message_buffer.agent_status]
        if active_agents:
            teams[team] = active_agents

    for team, agents in teams.items():
        # Add first agent with team name
        first_agent = agents[0]
        status = message_buffer.agent_status.get(first_agent, "pending")
        if status == "in_progress":
            spinner = Spinner(
                "dots", text="[blue]in_progress[/blue]", style="bold cyan"
            )
            status_cell = spinner
        else:
            status_color = {
                "pending": "yellow",
                "completed": "green",
                "error": "red",
            }.get(status, "white")
            status_cell = f"[{status_color}]{status}[/{status_color}]"
        progress_table.add_row(team, first_agent, status_cell)

        # Add remaining agents in team
        for agent in agents[1:]:
            status = message_buffer.agent_status.get(agent, "pending")
            if status == "in_progress":
                spinner = Spinner(
                    "dots", text="[blue]in_progress[/blue]", style="bold cyan"
                )
                status_cell = spinner
            else:
                status_color = {
                    "pending": "yellow",
                    "completed": "green",
                    "error": "red",
                }.get(status, "white")
                status_cell = f"[{status_color}]{status}[/{status_color}]"
            progress_table.add_row("", agent, status_cell)

        # Add horizontal line after each team
        progress_table.add_row("─" * 20, "─" * 20, "─" * 20, style="dim")

    layout["progress"].update(
        Panel(progress_table, title="Progress", border_style="cyan", padding=(1, 2))
    )

    # Messages panel showing recent messages and tool calls
    messages_table = Table(
        show_header=True,
        header_style="bold magenta",
        show_footer=False,
        expand=True,  # Make table expand to fill available space
        box=box.MINIMAL,  # Use minimal box style for a lighter look
        show_lines=True,  # Keep horizontal lines
        padding=(0, 1),  # Add some padding between columns
    )
    messages_table.add_column("Time", style="cyan", width=8, justify="center")
    messages_table.add_column("Type", style="green", width=10, justify="center")
    messages_table.add_column(
        "Content", style="white", no_wrap=False, ratio=1
    )  # Make content column expand

    # Combine tool calls and messages
    all_messages = []

    # Add tool calls
    for timestamp, tool_name, args in message_buffer.tool_calls:
        formatted_args = format_tool_args(args)
        all_messages.append((timestamp, "Tool", f"{tool_name}: {formatted_args}"))

    # Add regular messages
    for timestamp, msg_type, content in message_buffer.messages:
        content_str = str(content) if content else ""
        if len(content_str) > 200:
            content_str = content_str[:197] + "..."
        all_messages.append((timestamp, msg_type, content_str))

    # Sort by timestamp descending (newest first)
    all_messages.sort(key=lambda x: x[0], reverse=True)

    # Calculate how many messages we can show based on available space
    max_messages = 12

    # Get the first N messages (newest ones)
    recent_messages = all_messages[:max_messages]

    # Add messages to table (already in newest-first order)
    for timestamp, msg_type, content in recent_messages:
        # Format content with word wrapping
        wrapped_content = Text(content, overflow="fold")
        messages_table.add_row(timestamp, msg_type, wrapped_content)

    layout["messages"].update(
        Panel(
            messages_table,
            title="Messages & Tools",
            border_style="blue",
            padding=(1, 2),
        )
    )

    # Analysis panel showing current report
    if message_buffer.current_report:
        layout["analysis"].update(
            Panel(
                Markdown(message_buffer.current_report),
                title="Current Report",
                border_style="green",
                padding=(1, 2),
            )
        )
    else:
        layout["analysis"].update(
            Panel(
                "[italic]Waiting for analysis report...[/italic]",
                title="Current Report",
                border_style="green",
                padding=(1, 2),
            )
        )

    # Footer with statistics
    # Agent progress - derived from agent_status dict
    agents_completed = sum(
        1 for status in message_buffer.agent_status.values() if status == "completed"
    )
    agents_total = len(message_buffer.agent_status)

    # Report progress - based on agent completion (not just content existence)
    reports_completed = message_buffer.get_completed_reports_count()
    reports_total = len(message_buffer.report_sections)

    # Build stats parts
    stats_parts = [f"Agents: {agents_completed}/{agents_total}"]

    # LLM and tool stats from callback handler
    if stats_handler:
        stats = stats_handler.get_stats()
        stats_parts.append(f"LLM: {stats['llm_calls']}")
        stats_parts.append(f"Tools: {stats['tool_calls']}")

        # Token display with graceful fallback
        if stats["tokens_in"] > 0 or stats["tokens_out"] > 0:
            tokens_str = f"Tokens: {format_tokens(stats['tokens_in'])}\u2191 {format_tokens(stats['tokens_out'])}\u2193"
        else:
            tokens_str = "Tokens: --"
        stats_parts.append(tokens_str)

    stats_parts.append(f"Reports: {reports_completed}/{reports_total}")

    # Elapsed time
    if start_time:
        elapsed = time.time() - start_time
        elapsed_str = f"\u23f1 {int(elapsed // 60):02d}:{int(elapsed % 60):02d}"
        stats_parts.append(elapsed_str)

    stats_table = Table(show_header=False, box=None, padding=(0, 2), expand=True)
    stats_table.add_column("Stats", justify="center")
    stats_table.add_row(" | ".join(stats_parts))

    layout["footer"].update(Panel(stats_table, border_style="grey50"))


def get_user_selections(overrides: dict | None = None, interactive: bool = True):
    """Get all user selections before starting the analysis display.

    When ``interactive`` is False, every value in ``overrides`` is used verbatim
    (no questionary prompt). When True, overrides are ignored and the
    step-by-step prompts run as before.
    """
    overrides = overrides or {}
    # Display ASCII art welcome message
    with open(Path(__file__).parent / "static" / "welcome.txt", "r", encoding="utf-8") as f:
        welcome_ascii = f.read()

    # Create welcome box content
    welcome_content = f"{welcome_ascii}\n"
    welcome_content += "[bold green]TradingAgents: Multi-Agents LLM Financial Trading Framework - CLI[/bold green]\n\n"
    welcome_content += "[bold]Workflow Steps:[/bold]\n"
    welcome_content += "I. Analyst Team → II. Research Team → III. Trader → IV. Risk Management → V. Portfolio Management\n\n"
    welcome_content += (
        "[dim]Built by [Tauric Research](https://github.com/TauricResearch)[/dim]"
    )

    # Run config summary (only when caller passed overrides — i.e. non-interactive)
    if not interactive and overrides:
        depth_label = {1: "shallow", 2: "medium", 3: "deep"}.get(
            overrides.get("research_depth"), str(overrides.get("research_depth", "?"))
        )
        analysts_val = overrides.get("analysts", [])
        analysts_label = ", ".join(
            getattr(a, "value", str(a)) for a in analysts_val
        ) if analysts_val else "?"
        summary_rows = [
            ("Ticker",   overrides.get("ticker", "?")),
            ("Analysts", analysts_label),
            ("Depth",    depth_label),
            ("Horizon",  overrides.get("trading_horizon", "?")),
        ]
        welcome_content += "\n\n[bold]Run config:[/bold]\n"
        welcome_content += "\n".join(
            f"  [dim]{k}:[/dim] [bold]{v}[/bold]" for k, v in summary_rows
        )

    # Create and center the welcome box
    welcome_box = Panel(
        welcome_content,
        border_style="green",
        padding=(1, 2),
        title="Welcome to TradingAgents",
        subtitle="Multi-Agents LLM Financial Trading Framework",
    )
    console.print(Align.center(welcome_box))
    console.print()
    console.print()  # Add vertical space before announcements

    # Fetch and display announcements (silent on failure)
    announcements = fetch_announcements()
    display_announcements(console, announcements)

    # Create a boxed questionnaire for each step
    def create_question_box(title, prompt, default=None):
        box_content = f"[bold]{title}[/bold]\n"
        box_content += f"[dim]{prompt}[/dim]"
        if default:
            box_content += f"\n[dim]Default: {default}[/dim]"
        return Panel(box_content, border_style="blue", padding=(1, 2))

    # Step 1: Ticker symbol
    if not interactive and "ticker" in overrides:
        selected_ticker = overrides["ticker"]
        console.print(f"[dim]Ticker:[/dim] [bold]{selected_ticker}[/bold]")
    else:
        console.print(
            create_question_box(
                "Step 1: Ticker Symbol",
                "Enter the exact ticker symbol to analyze, including exchange suffix when needed (examples: SPY, CNC.TO, 7203.T, 0700.HK)",
                "SPY",
            )
        )
        selected_ticker = get_ticker()

    # Step 2: Analysis date
    default_date = datetime.datetime.now().strftime("%Y-%m-%d")
    if not interactive and "analysis_date" in overrides:
        analysis_date = overrides["analysis_date"]
        console.print(f"[dim]Analysis date:[/dim] [bold]{analysis_date}[/bold]")
    else:
        console.print(
            create_question_box(
                "Step 2: Analysis Date",
                "Enter the analysis date (YYYY-MM-DD)",
                default_date,
            )
        )
        analysis_date = get_analysis_date()

    # Step 3: Output language
    if not interactive and "output_language" in overrides:
        output_language = overrides["output_language"]
        console.print(f"[dim]Output language:[/dim] [bold]{output_language}[/bold]")
    else:
        console.print(
            create_question_box(
                "Step 3: Output Language",
                "Select the language for analyst reports and final decision"
            )
        )
        output_language = ask_output_language()

    # Step 4: Select analysts
    if not interactive and "analysts" in overrides:
        selected_analysts = overrides["analysts"]
        console.print(
            f"[dim]Analysts:[/dim] {', '.join(analyst.value for analyst in selected_analysts)}"
        )
    else:
        console.print(
            create_question_box(
                "Step 4: Analysts Team", "Select your LLM analyst agents for the analysis"
            )
        )
        selected_analysts = select_analysts()
        console.print(
            f"[green]Selected analysts:[/green] {', '.join(analyst.value for analyst in selected_analysts)}"
        )

    # Step 5: Research depth
    if not interactive and "research_depth" in overrides:
        selected_research_depth = overrides["research_depth"]
        console.print(f"[dim]Research depth:[/dim] [bold]{selected_research_depth}[/bold] rounds")
    else:
        console.print(
            create_question_box(
                "Step 5: Research Depth", "Select your research depth level"
            )
        )
        selected_research_depth = select_research_depth()

    # Trading horizon (driven by --horizon flag; no interactive picker for now)
    selected_trading_horizon = overrides.get("trading_horizon", DEFAULTS["horizon"])
    console.print(f"[dim]Trading horizon:[/dim] [bold]{selected_trading_horizon}[/bold]")

    # Step 6: LLM Provider
    if not interactive and "llm_provider" in overrides:
        selected_llm_provider = overrides["llm_provider"]
        backend_url = overrides.get(
            "backend_url", get_provider_backend_url(selected_llm_provider)
        )
        console.print(
            f"[dim]Provider:[/dim] [bold]{selected_llm_provider}[/bold] "
            f"[dim]({backend_url or 'no backend_url'})[/dim]"
        )
    else:
        console.print(
            create_question_box(
                "Step 6: LLM Provider", "Select your LLM provider"
            )
        )
        selected_llm_provider, backend_url = select_llm_provider()

    # Providers with regional endpoints prompt for the region as a secondary
    # step so the main dropdown stays clean (mainland China and international
    # accounts cannot share API keys).
    if selected_llm_provider == "qwen":
        selected_llm_provider, backend_url = ask_qwen_region()
    elif selected_llm_provider == "minimax":
        selected_llm_provider, backend_url = ask_minimax_region()
    elif selected_llm_provider == "glm":
        selected_llm_provider, backend_url = ask_glm_region()

    # For Ollama, surface the resolved endpoint (OLLAMA_BASE_URL vs default)
    # before model selection so it's obvious where we're connecting.
    if selected_llm_provider == "ollama":
        confirm_ollama_endpoint(backend_url)

    # Confirm the provider's API key is present; prompt the user to paste
    # one and persist it to .env if it's missing, so the analysis run
    # doesn't fail later at the first API call.
    ensure_api_key(selected_llm_provider)

    # Step 7: Thinking agents
    if (
        not interactive
        and "shallow_thinker" in overrides
        and "deep_thinker" in overrides
    ):
        selected_shallow_thinker = overrides["shallow_thinker"]
        selected_deep_thinker = overrides["deep_thinker"]
        console.print(
            f"[dim]Quick model:[/dim] [bold]{selected_shallow_thinker}[/bold]   "
            f"[dim]Deep model:[/dim] [bold]{selected_deep_thinker}[/bold]"
        )
    else:
        console.print(
            create_question_box(
                "Step 7: Thinking Agents", "Select your thinking agents for analysis"
            )
        )
        selected_shallow_thinker = select_shallow_thinking_agent(selected_llm_provider)
        selected_deep_thinker = select_deep_thinking_agent(selected_llm_provider)

    # Step 8: Provider-specific thinking configuration. Skipped entirely in
    # non-interactive mode unless the corresponding override is supplied; pass
    # --thinking-level / --reasoning-effort / --effort to set explicitly.
    thinking_level = overrides.get("google_thinking_level")
    reasoning_effort = overrides.get("openai_reasoning_effort")
    anthropic_effort = overrides.get("anthropic_effort")

    provider_lower = selected_llm_provider.lower()
    if interactive:
        if provider_lower == "google":
            console.print(
                create_question_box(
                    "Step 8: Thinking Mode",
                    "Configure Gemini thinking mode"
                )
            )
            thinking_level = ask_gemini_thinking_config()
        elif provider_lower == "openai":
            console.print(
                create_question_box(
                    "Step 8: Reasoning Effort",
                    "Configure OpenAI reasoning effort level"
                )
            )
            reasoning_effort = ask_openai_reasoning_effort()
        elif provider_lower == "anthropic":
            console.print(
                create_question_box(
                    "Step 8: Effort Level",
                    "Configure Claude effort level"
                )
            )
            anthropic_effort = ask_anthropic_effort()

    return {
        "ticker": selected_ticker,
        "analysis_date": analysis_date,
        "analysts": selected_analysts,
        "research_depth": selected_research_depth,
        "llm_provider": selected_llm_provider.lower(),
        "backend_url": backend_url,
        "shallow_thinker": selected_shallow_thinker,
        "deep_thinker": selected_deep_thinker,
        "google_thinking_level": thinking_level,
        "openai_reasoning_effort": reasoning_effort,
        "anthropic_effort": anthropic_effort,
        "output_language": output_language,
        "trading_horizon": selected_trading_horizon,
    }


def get_ticker():
    """Get ticker symbol from user input, preserving exchange suffixes."""
    # typer.prompt strips trailing dot-suffixes on some shells (e.g. 000404.SH
    # collapses to 000404). questionary.text reads the raw line.
    ticker = questionary.text(
        "",
        validate=lambda value: (
            not value.strip()
            or (
                all(ch.isalnum() or ch in "._-^" for ch in value.strip())
                and len(value.strip()) <= 32
            )
        )
        or "Please enter a valid ticker symbol, e.g. AAPL, 000404.SZ, 0700.HK.",
    ).ask()

    if ticker is None:
        console.print("\n[red]No ticker symbol provided. Exiting...[/red]")
        raise typer.Exit(1)

    return (ticker.strip() or "SPY").upper()


def get_analysis_date():
    """Get the analysis date from user input."""
    while True:
        date_str = typer.prompt(
            "", default=datetime.datetime.now().strftime("%Y-%m-%d")
        )
        try:
            # Validate date format and ensure it's not in the future
            analysis_date = datetime.datetime.strptime(date_str, "%Y-%m-%d")
            if analysis_date.date() > datetime.datetime.now().date():
                console.print("[red]Error: Analysis date cannot be in the future[/red]")
                continue
            return date_str
        except ValueError:
            console.print(
                "[red]Error: Invalid date format. Please use YYYY-MM-DD[/red]"
            )


# Probe-ordered candidates: macOS SFNS (system font) and Geneva carry the
# Indian Rupee Sign glyph; macOS Arial Unicode and core PDF Helvetica do not.
# DejaVu Sans is the canonical Linux option.
_UNICODE_FONT_CANDIDATES = (
    "/System/Library/Fonts/SFNS.ttf",
    "/System/Library/Fonts/Geneva.ttf",
    "/System/Library/Fonts/NewYork.ttf",
    "/System/Library/Fonts/Supplemental/Georgia.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/dejavu/DejaVuSans.ttf",
    "/Library/Fonts/Arial Unicode.ttf",
    "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
    "C:\\Windows\\Fonts\\arial.ttf",
)

# Codepoints we want the chosen font to actually have. INDIAN RUPEE SIGN
# (U+20B9) is the canary — older fonts predate Unicode 6.0 and lack it.
_REQUIRED_GLYPHS = (0x20B9,)


def _find_unicode_font_path() -> Optional[str]:
    """Pick the first candidate TTF whose cmap covers the required glyphs."""
    try:
        from reportlab.pdfbase.ttfonts import TTFont
    except ImportError:
        return None
    for path in _UNICODE_FONT_CANDIDATES:
        if not Path(path).exists():
            continue
        try:
            face = TTFont("_probe", path).face
        except Exception:
            continue
        if all(face.charToGlyph.get(cp) for cp in _REQUIRED_GLYPHS):
            return path
    return None


# Color emoji can't be rendered by reportlab/xhtml2pdf from a regular TTF (PDF
# has no native color-emoji story for embedded fonts that ship the COLR/sbix
# tables). The pragmatic workaround is to swap each emoji codepoint for an
# inline ``<img>`` tag pointing at a small Twemoji PNG, cached on first use.
_EMOJI_RE = __import__("re").compile(
    "["
    "\U0001F300-\U0001FAFF"
    "\U00002600-\U000027BF"
    "\U0001F600-\U0001F64F"
    "]"
)
_TWEMOJI_URL = "https://cdn.jsdelivr.net/gh/jdecked/twemoji@latest/assets/72x72/{cp}.png"


def _fetch_emoji_png(cp: str, cache_dir: Path) -> Optional[Path]:
    """Download a Twemoji PNG and normalize it to RGBA so reportlab renders the
    transparent background as transparent rather than opaque black.
    """
    target = cache_dir / f"{cp}.png"
    if target.exists() and target.stat().st_size > 0:
        return target
    import urllib.request, io
    try:
        with urllib.request.urlopen(_TWEMOJI_URL.format(cp=cp), timeout=5) as r:
            raw = r.read()
        from PIL import Image
        with Image.open(io.BytesIO(raw)) as im:
            im = im.convert("RGBA")
            im.save(target, "PNG")
        return target
    except Exception:
        if target.exists():
            target.unlink()
        return None


# Variation selectors (U+FE00-U+FE0F) are non-printing, but some fonts render
# them as visible glyphs once the preceding emoji is replaced by an <img>.
# Strip them after substitution so they don't leave artefacts.
_VARIATION_SELECTOR_RE = __import__("re").compile("[\U0000FE00-\U0000FE0F]")


def _replace_emoji_with_img(html: str) -> str:
    """Replace emoji characters with inline <img> tags backed by Twemoji PNGs."""
    import tempfile
    cache_dir = Path(tempfile.gettempdir()) / "tradingagents_emoji"
    cache_dir.mkdir(exist_ok=True)

    def sub(m):
        ch = m.group(0)
        cp = f"{ord(ch):x}"
        png = _fetch_emoji_png(cp, cache_dir)
        if not png:
            return ch
        # xhtml2pdf doesn't honour inline margin on <img>, so we append a
        # non-breaking space to create visible separation between glyph and text.
        return (
            f'<img src="{png.resolve()}" '
            'style="height: 11pt; width: 11pt; vertical-align: middle;"/>&nbsp;'
        )

    return _VARIATION_SELECTOR_RE.sub("", _EMOJI_RE.sub(sub, html))


def _xhtml2pdf_styling(body_html: str) -> tuple[str, str]:
    """Build the (html_doc, body_html) pair tuned for xhtml2pdf's CSS 2.1 subset.

    Includes the ₹-glyph @font-face workaround and emoji-as-img substitution
    that xhtml2pdf needs but WeasyPrint does not.
    """
    body_html = _replace_emoji_with_img(body_html)
    font_path = _find_unicode_font_path()
    if font_path:
        # xhtml2pdf reads @font-face url() as a filesystem path. It chokes on
        # spaces and on file:// URIs, so we copy the TTF to a temp file with
        # a safe name and reference it as an absolute path.
        import shutil, tempfile
        safe_dir = Path(tempfile.gettempdir()) / "tradingagents_fonts"
        safe_dir.mkdir(exist_ok=True)
        safe_font = safe_dir / "ReportUnicode.ttf"
        if not safe_font.exists():
            shutil.copy(font_path, safe_font)
        ref = str(safe_font.resolve())
        font_face_css = (
            f"@font-face {{ font-family: ReportUnicode; src: url({ref}); }}"
            f"@font-face {{ font-family: ReportUnicode; font-weight: bold; src: url({ref}); }}"
            f"@font-face {{ font-family: ReportUnicode; font-style: italic; src: url({ref}); }}"
            f"@font-face {{ font-family: ReportUnicode; font-weight: bold; font-style: italic; src: url({ref}); }}"
        )
        body_font = "ReportUnicode, Helvetica, Arial, sans-serif"
        mono_font = "ReportUnicode, 'Menlo', 'Courier New', monospace"
    else:
        font_face_css = ""
        body_font = "Helvetica, Arial, sans-serif"
        mono_font = "'Menlo', 'Courier New', monospace"
    style = (
        font_face_css +
        f"body {{ font-family: {body_font}; font-size: 10pt; line-height: 1.4; }}"
        "h1 { font-size: 18pt; } h2 { font-size: 14pt; margin-top: 18pt; }"
        "h3 { font-size: 12pt; } h4 { font-size: 11pt; }"
        "table { border-collapse: collapse; margin: 6pt 0; }"
        "th, td { border: 1px solid #999; padding: 4pt 6pt; }"
        f"code, pre {{ font-family: {mono_font}; font-size: 9pt; }}"
        "pre { background: #f4f4f4; padding: 6pt; }"
    )
    html_doc = (
        f"<html><head><meta charset='utf-8'><style>{style}</style></head>"
        f"<body>{body_html}</body></html>"
    )
    return html_doc, body_html


_PROPOSAL_RE = __import__("re").compile(
    r"FINAL TRANSACTION PROPOSAL:\s*(?:<strong>)?\s*\*{0,2}\s*(BUY|SELL|HOLD)\s*\*{0,2}\s*(?:</strong>)?",
    __import__("re").IGNORECASE,
)


def _wrap_proposal_callouts(html: str) -> str:
    """Wrap each FINAL TRANSACTION PROPOSAL: BUY/SELL/HOLD line in a styled callout."""
    def sub(m):
        verdict = m.group(1).upper()
        return (
            f'<span class="ta-proposal ta-proposal-{verdict.lower()}">'
            f'FINAL TRANSACTION PROPOSAL: <span class="ta-proposal-verdict">{verdict}</span>'
            f'</span>'
        )
    return _PROPOSAL_RE.sub(sub, html)


_ANALYST_H3_RE = __import__("re").compile(
    r"<h3>((?:Market|News|Social|Fundamentals)\s+Analyst)</h3>",
    __import__("re").IGNORECASE,
)


def _wrap_analyst_headings(html: str) -> str:
    """Tag analyst H3s with .ta-analyst so CSS can render them as a chip."""
    return _ANALYST_H3_RE.sub(r'<h3 class="ta-analyst">\1</h3>', html)


# WeasyPrint stylesheet. Design intent: financial-broadsheet feel — restrained
# navy/gold palette, bold banded section headings, monospace tabular numerals
# for data tables, page footer with the report title plus page numbering, and
# clickable PDF bookmarks per heading. Tuned for letter-size A-style reports.
_WEASYPRINT_STYLE = """
@page {
  size: letter;
  margin: 22mm 18mm 22mm 18mm;
  @bottom-left   { content: string(doctitle); font-family: 'Inter', 'Helvetica Neue', sans-serif; font-size: 8pt; color: #6b7280; }
  @bottom-right  { content: "page " counter(page) " of " counter(pages); font-family: 'Inter', 'Helvetica Neue', sans-serif; font-size: 8pt; color: #6b7280; }
  @top-right     { content: string(section); font-family: 'Inter', 'Helvetica Neue', sans-serif; font-size: 8pt; color: #9ca3af; letter-spacing: 0.06em; text-transform: uppercase; }
}
@page :first {
  @top-right     { content: ""; }
  @bottom-left   { content: ""; }
  @bottom-right  { content: ""; }
}

body {
  font-family: 'Inter', 'Helvetica Neue', Helvetica, Arial, sans-serif;
  font-size: 10pt;
  line-height: 1.45;
  color: #111827;
  font-variant-numeric: tabular-nums;
  font-feature-settings: "ss01", "cv11";
}

h1, h2, h3, h4 {
  page-break-after: avoid;
  break-after: avoid;
  font-family: 'Inter Display', 'Inter', 'Helvetica Neue', sans-serif;
  color: #111827;
  font-feature-settings: "ss01", "cv11";
}

h1 {
  string-set: doctitle content();
  bookmark-level: 1;
  bookmark-label: content();
  font-size: 22pt;
  font-weight: 700;
  margin: 0 0 6pt 0;
  padding-bottom: 8pt;
  border-bottom: 2pt solid #1e3a8a;
  letter-spacing: -0.02em;
}

h2 {
  string-set: section content();
  bookmark-level: 2;
  bookmark-label: content();
  page-break-before: auto;
  break-before: auto;
  font-size: 14pt;
  font-weight: 600;
  color: #ffffff;
  background: #1e3a8a;
  padding: 8pt 14pt;
  margin: 22pt -10mm 12pt -10mm;
  border-left: 4pt solid #f59e0b;
  letter-spacing: -0.005em;
}

h3 {
  bookmark-level: 3;
  font-size: 12pt;
  font-weight: 600;
  margin-top: 16pt;
  margin-bottom: 4pt;
  color: #1e3a8a;
  border-bottom: 0.5pt solid #d1d5db;
  padding-bottom: 2pt;
  letter-spacing: -0.01em;
}

h4 {
  font-size: 9.5pt;
  font-weight: 600;
  margin-top: 12pt;
  margin-bottom: 2pt;
  color: #374151;
  text-transform: uppercase;
  letter-spacing: 0.08em;
}

p { margin: 6pt 0; }
strong { color: #111827; }
em { color: #4b5563; }

ul, ol { margin: 6pt 0 6pt 18pt; padding: 0; }
li { margin: 2pt 0; }

table {
  border-collapse: collapse;
  margin: 8pt 0 12pt 0;
  width: 100%;
  page-break-inside: avoid;
  break-inside: avoid;
  font-size: 9pt;
  font-variant-numeric: tabular-nums;
}
thead { display: table-header-group; }
th {
  background: #1f2937;
  color: #ffffff;
  font-weight: 600;
  text-align: left;
  padding: 5pt 8pt;
  border: 0;
  letter-spacing: 0.02em;
}
td {
  padding: 4pt 8pt;
  border-bottom: 0.5pt solid #e5e7eb;
}
tbody tr:nth-child(even) td { background: #f9fafb; }

code {
  font-family: 'Menlo', 'Monaco', 'Courier New', monospace;
  font-size: 8.5pt;
  background: #f3f4f6;
  padding: 1pt 3pt;
  border-radius: 2pt;
}
pre {
  font-family: 'Menlo', 'Monaco', 'Courier New', monospace;
  font-size: 8.5pt;
  background: #f3f4f6;
  padding: 8pt 10pt;
  border-left: 3pt solid #1e3a8a;
  page-break-inside: avoid;
  break-inside: avoid;
  white-space: pre-wrap;
  word-wrap: break-word;
}

blockquote {
  margin: 8pt 0;
  padding: 4pt 12pt;
  border-left: 3pt solid #f59e0b;
  background: #fffbeb;
  color: #374151;
  font-style: italic;
}

hr {
  border: 0;
  height: 0.5pt;
  background: #d1d5db;
  margin: 14pt 0;
}

a { color: #1e3a8a; text-decoration: none; }

/* FINAL TRANSACTION PROPOSAL callouts. The colour signals trade direction at a glance. */
.ta-proposal {
  display: inline-block;
  padding: 4pt 12pt;
  margin: 4pt 0;
  font-weight: 600;
  font-family: 'Inter', 'Helvetica Neue', sans-serif;
  font-size: 10pt;
  letter-spacing: 0.04em;
  border-radius: 3pt;
  color: #ffffff;
}
.ta-proposal-buy  { background: #047857; }
.ta-proposal-sell { background: #b91c1c; }
.ta-proposal-hold { background: #4b5563; }
.ta-proposal-verdict { font-weight: 700; letter-spacing: 0.08em; }

/* Analyst-name headings inside "I. Analyst Team Reports". Slanted amber chip
   that's distinct from regular h3s but lighter than the navy h2 bands, so the
   reader still feels the section hierarchy. */
h3.ta-analyst {
  display: inline-block;
  font-family: 'Inter Display', 'Inter', sans-serif;
  font-size: 13pt;
  font-weight: 600;
  font-style: italic;
  color: #92400e;
  background: #fef3c7;
  border-left: 4pt solid #f59e0b;
  border-bottom: 0;
  padding: 5pt 14pt 5pt 12pt;
  margin: 16pt 0 8pt -10mm;
  letter-spacing: 0;
  page-break-after: avoid;
  break-after: avoid;
}
"""


def _weasyprint_styling(body_html: str) -> str:
    """Build the html_doc tuned for WeasyPrint.

    WeasyPrint reads system fonts via fontconfig+pango, so the manual
    @font-face and Twemoji-as-img workarounds xhtml2pdf needs are unnecessary.
    The full stylesheet (page footers, banded section headings, BUY/SELL/HOLD
    callouts, zebra tables, PDF bookmarks) lives in ``_WEASYPRINT_STYLE``.
    """
    body_html = _wrap_proposal_callouts(body_html)
    body_html = _wrap_analyst_headings(body_html)
    return (
        f"<html><head><meta charset='utf-8'><style>{_WEASYPRINT_STYLE}</style></head>"
        f"<body>{body_html}</body></html>"
    )


def _markdown_to_pdf(md_path: Path) -> Optional[Path]:
    """Render a markdown report to a sibling PDF. Returns the PDF path or None on failure.

    Engine is chosen by ``DEFAULT_CONFIG['pdf_engine']`` (overridable via
    $TRADINGAGENTS_PDF_ENGINE). ``weasyprint`` is the modern default; it
    supports CSS3, @page rules, system fonts, and OpenType features.
    ``xhtml2pdf`` is the legacy pure-Python fallback for environments
    without Pango/Cairo. If the configured engine's deps are missing the
    other engine is tried before giving up.
    """
    try:
        import markdown as md_lib
    except ImportError:
        console.print("[yellow]PDF generation skipped — install with: pip install markdown[/yellow]")
        return None

    pdf_path = md_path.with_suffix(".pdf")
    try:
        body_html = md_lib.markdown(
            md_path.read_text(encoding="utf-8"),
            extensions=["tables", "fenced_code", "sane_lists"],
        )
    except Exception as e:
        console.print(f"[yellow]Markdown→HTML conversion failed: {e}[/yellow]")
        return None

    engine = DEFAULT_CONFIG.get("pdf_engine", "weasyprint")
    order = ("weasyprint", "xhtml2pdf") if engine == "weasyprint" else ("xhtml2pdf", "weasyprint")
    for candidate in order:
        try:
            if candidate == "weasyprint":
                html_doc = _weasyprint_styling(body_html)
                from weasyprint import HTML  # type: ignore[import-not-found]
                HTML(string=html_doc, base_url=str(md_path.parent)).write_pdf(pdf_path)
                return pdf_path
            else:
                html_doc, _ = _xhtml2pdf_styling(body_html)
                from xhtml2pdf import pisa  # type: ignore[import-not-found]
                with open(pdf_path, "wb") as f:
                    result = pisa.CreatePDF(html_doc, dest=f)
                if result.err:
                    console.print("[yellow]xhtml2pdf reported errors; trying next engine[/yellow]")
                    continue
                return pdf_path
        except ImportError:
            continue
        except Exception as e:
            console.print(f"[yellow]{candidate} render failed ({e}); trying next engine[/yellow]")
            continue

    console.print(
        "[yellow]No PDF engine available — install with: "
        "pip install weasyprint  (preferred; needs `brew install pango`)  OR  "
        "pip install xhtml2pdf  (no system deps)[/yellow]"
    )
    return None


def _open_in_default_viewer(path: Path) -> None:
    """Open a file with the OS default application. No-op on failure."""
    import platform
    import subprocess
    import os

    system = platform.system()
    try:
        if system == "Darwin":
            subprocess.run(["open", str(path)], check=False)
        elif system == "Windows":
            os.startfile(str(path))  # type: ignore[attr-defined]
        else:
            subprocess.run(["xdg-open", str(path)], check=False)
    except Exception as e:
        console.print(f"[yellow]Could not open report in default viewer: {e}[/yellow]")


_PREAMBLE_OPENERS = (
    "excellent", "great", "perfect", "sure", "of course", "got it",
    "here is", "here's", "here is the", "all data",
    "i'll", "i will", "let me", "alright", "okay",
)


def _strip_preamble(text: str) -> str:
    """Drop a conversational opening paragraph if present.

    The model occasionally prefaces analyst reports with chatty lines like
    'Excellent. All data is now in hand. Here is the analysis...'. Detect such
    a paragraph (before the first markdown heading) and remove it.
    """
    if not text:
        return text
    parts = text.split("\n\n", 1)
    if len(parts) < 2:
        return text
    head, rest = parts[0].strip(), parts[1]
    if not head or head.startswith("#") or head.startswith("|"):
        return text
    if any(head.lower().startswith(opener) for opener in _PREAMBLE_OPENERS):
        return rest.lstrip()
    return text


def _strip_outer_hrules(text: str) -> str:
    """Drop leading/trailing markdown horizontal-rule lines from analyst output.

    Analysts often bracket their report with `---` separators. When such a
    block is wrapped under a `### Market Analyst` heading the leading `---`
    becomes a redundant second line directly under the heading; the trailing
    one duplicates the section break. Strip both — preserves any `---` used
    mid-content.
    """
    if not text:
        return text
    lines = text.splitlines()
    while lines and lines[0].strip() in ("", "---", "***", "___"):
        if lines[0].strip() in ("---", "***", "___"):
            lines.pop(0)
        else:
            lines.pop(0)
            break
    while lines and lines[-1].strip() in ("", "---", "***", "___"):
        if lines[-1].strip() in ("---", "***", "___"):
            lines.pop()
        else:
            lines.pop()
            break
    return "\n".join(lines)


def _build_trade_setup_block(
    trader_plan: str | None,
    pm_decision: str | None,
    key_levels: str | None = None,
) -> str | None:
    """Extract the headline trade-setup numbers and render them as a top-of-report block.

    Parses the rendered markdown produced by `render_trader_proposal` (Action,
    Entry Price, Stop Loss, Position Sizing) and `render_pm_decision` (Rating,
    Price Target, Time Horizon). The reader sees the assumed entry and stop
    immediately, before scrolling.

    When the Trader leaves Entry Price blank — typical on Hold/Sell, where
    "where to buy" doesn't apply — we surface the latest close from
    `key_levels` as a "Reference Price (current close)" so the reader still
    sees the price at which the Hold/Sell verdict is being delivered. Without
    this, a Hold report has no anchor for the recommendation.

    Returns None if neither side yields any fields (e.g. a free-text fallback
    that didn't preserve the schema shape).
    """
    import re

    def _grab(text: str | None, label: str) -> str | None:
        if not text:
            return None
        m = re.search(rf"\*\*{re.escape(label)}\*\*:\s*([^\n]+)", text)
        return m.group(1).strip() if m else None

    entry_price = _grab(trader_plan, "Entry Price")
    if not entry_price and key_levels:
        # Trader did not specify an anchor level. The latest close is the
        # *current* price (where you'd buy at market), not an anchor for
        # the recommendation — Hold/Sell anchors should be a prior level.
        # Fall back to the 50-DMA from key_levels: it's the canonical
        # "swing entry" anchor and almost always a meaningful prior level.
        m = re.search(r"50-DMA:\s*([0-9][0-9.,]*)", key_levels)
        if m:
            entry_price = f"{m.group(1).strip()} — _50-DMA fallback (Trader did not specify an anchor level)_"

    fields = [
        ("Action", _grab(trader_plan, "Action")),
        ("Rating", _grab(pm_decision, "Rating")),
        ("Entry Price", entry_price),
        ("Initial Stop", _grab(trader_plan, "Initial Stop")),
        ("Trailing Stop", _grab(trader_plan, "Trailing Stop")),
        ("Position Sizing", _grab(trader_plan, "Position Sizing")),
        ("Price Target", _grab(pm_decision, "Price Target")),
        ("Time Horizon", _grab(pm_decision, "Time Horizon")),
    ]
    rows = [(label, value) for label, value in fields if value]
    if not rows:
        return None

    body = "\n".join(f"- **{label}**: {value}" for label, value in rows)
    return (
        "## Trade Setup at a Glance\n\n"
        f"{body}\n\n"
        "_Numbers below are the assumed levels at the time of writing. "
        "Scroll for the full analyst, research, trader, risk, and portfolio sections._"
    )


def _build_market_regime_block(market_regime: str | None) -> str | None:
    """Render the deterministic market-regime block for the report header.

    `market_regime` is the markdown produced by `compute_market_regime` —
    already a bulleted list with broad (Nifty 50/500), cap-tier, and sector
    lines. We wrap it under a top-level heading and add a one-line preface
    so a reader landing on the report knows the broad market regime AND the
    ticker-specific cap/sector regime before they hit the trade setup.

    Empty / non-Indian → returns None so the section is skipped.
    """
    if not market_regime or not market_regime.strip():
        return None
    body = market_regime.strip()
    if body.startswith("**") and "\n" in body:
        body = body.split("\n", 1)[1].strip()
    return (
        "## Market Regime Context\n\n"
        "_Broad market and ticker-specific regime at the time of analysis "
        "(deterministic snapshot, not LLM-generated)._\n\n"
        f"{body}"
    )


def save_report_to_disk(final_state, ticker: str, save_path: Path):
    """Save complete analysis report to disk with organized subfolders."""
    save_path.mkdir(parents=True, exist_ok=True)
    sections = []

    setup_block = _build_trade_setup_block(
        final_state.get("trader_investment_plan"),
        (final_state.get("risk_debate_state") or {}).get("judge_decision"),
        final_state.get("key_levels"),
    )
    if setup_block:
        sections.append(setup_block)

    regime_block = _build_market_regime_block(final_state.get("market_regime"))
    if regime_block:
        sections.append(regime_block)

    # 1. Analysts (preamble stripped to drop any chatty model preamble)
    analysts_dir = save_path / "1_analysts"
    analyst_parts = []
    if final_state.get("market_report"):
        analysts_dir.mkdir(exist_ok=True)
        market_text = _strip_outer_hrules(_strip_preamble(final_state["market_report"]))
        (analysts_dir / "market.md").write_text(market_text, encoding="utf-8")
        analyst_parts.append(("Market Analyst", market_text))
    if final_state.get("sentiment_report"):
        analysts_dir.mkdir(exist_ok=True)
        sentiment_text = _strip_outer_hrules(_strip_preamble(final_state["sentiment_report"]))
        (analysts_dir / "sentiment.md").write_text(sentiment_text, encoding="utf-8")
        analyst_parts.append(("Sentiment Analyst", sentiment_text))
    if final_state.get("news_report"):
        analysts_dir.mkdir(exist_ok=True)
        news_text = _strip_outer_hrules(_strip_preamble(final_state["news_report"]))
        (analysts_dir / "news.md").write_text(news_text, encoding="utf-8")
        analyst_parts.append(("News Analyst", news_text))
    if final_state.get("fundamentals_report"):
        analysts_dir.mkdir(exist_ok=True)
        fundamentals_text = _strip_outer_hrules(_strip_preamble(final_state["fundamentals_report"]))
        (analysts_dir / "fundamentals.md").write_text(fundamentals_text, encoding="utf-8")
        analyst_parts.append(("Fundamentals Analyst", fundamentals_text))
    if analyst_parts:
        content = "\n\n".join(f"### {name}\n{text}" for name, text in analyst_parts)
        sections.append(f"## I. Analyst Team Reports\n\n{content}")

    # 2. Research — full transcripts go to subdir; consolidated report shows
    # only the Research Manager's synthesis + a pointer to the transcripts.
    # The raw debate is faithfully preserved in 2_research/transcripts/ but
    # not duplicated into the headline report (the manager already absorbed it).
    if final_state.get("investment_debate_state"):
        research_dir = save_path / "2_research"
        transcripts_dir = research_dir / "transcripts"
        debate = final_state["investment_debate_state"]
        if debate.get("bull_history"):
            transcripts_dir.mkdir(parents=True, exist_ok=True)
            (transcripts_dir / "bull.md").write_text(debate["bull_history"], encoding="utf-8")
        if debate.get("bear_history"):
            transcripts_dir.mkdir(parents=True, exist_ok=True)
            (transcripts_dir / "bear.md").write_text(debate["bear_history"], encoding="utf-8")
        if debate.get("judge_decision"):
            research_dir.mkdir(exist_ok=True)
            (research_dir / "manager.md").write_text(debate["judge_decision"], encoding="utf-8")
            sections.append(
                f"## II. Research Team Decision\n\n"
                f"### Research Manager\n{debate['judge_decision']}\n\n"
                f"_Full bull/bear debate transcripts: `2_research/transcripts/`._"
            )

    # 3. Trading
    if final_state.get("trader_investment_plan"):
        trading_dir = save_path / "3_trading"
        trading_dir.mkdir(exist_ok=True)
        (trading_dir / "trader.md").write_text(final_state["trader_investment_plan"], encoding="utf-8")
        sections.append(f"## III. Trading Team Plan\n\n### Trader\n{final_state['trader_investment_plan']}")

    # 4. Risk Management — full transcripts go to subdir; consolidated report
    # shows only each debater's *final* turn (the most-developed argument
    # under the novelty constraint in their prompts). Full multi-round
    # transcripts remain in 4_risk/transcripts/ for forensic use.
    if final_state.get("risk_debate_state"):
        risk_dir = save_path / "4_risk"
        risk_transcripts_dir = risk_dir / "transcripts"
        risk = final_state["risk_debate_state"]
        risk_parts = []
        if risk.get("aggressive_history"):
            risk_transcripts_dir.mkdir(parents=True, exist_ok=True)
            (risk_transcripts_dir / "aggressive.md").write_text(risk["aggressive_history"], encoding="utf-8")
            final_turn = risk.get("current_aggressive_response") or risk["aggressive_history"]
            risk_parts.append(("Aggressive Analyst (final position)", final_turn))
        if risk.get("conservative_history"):
            risk_transcripts_dir.mkdir(parents=True, exist_ok=True)
            (risk_transcripts_dir / "conservative.md").write_text(risk["conservative_history"], encoding="utf-8")
            final_turn = risk.get("current_conservative_response") or risk["conservative_history"]
            risk_parts.append(("Conservative Analyst (final position)", final_turn))
        if risk.get("neutral_history"):
            risk_transcripts_dir.mkdir(parents=True, exist_ok=True)
            (risk_transcripts_dir / "neutral.md").write_text(risk["neutral_history"], encoding="utf-8")
            final_turn = risk.get("current_neutral_response") or risk["neutral_history"]
            risk_parts.append(("Neutral Analyst (final position)", final_turn))
        if risk_parts:
            content = "\n\n".join(f"### {name}\n{text}" for name, text in risk_parts)
            sections.append(
                f"## IV. Risk Management Team Decision\n\n{content}\n\n"
                f"_Full multi-round risk debate transcripts: `4_risk/transcripts/`._"
            )

        # 5. Portfolio Manager
        if risk.get("judge_decision"):
            portfolio_dir = save_path / "5_portfolio"
            portfolio_dir.mkdir(exist_ok=True)
            (portfolio_dir / "decision.md").write_text(risk["judge_decision"], encoding="utf-8")
            sections.append(f"## V. Portfolio Manager Decision\n\n### Portfolio Manager\n{risk['judge_decision']}")

    # Write consolidated report. The file is prefixed with the ticker base
    # (e.g. NDRAUTO.NS -> ndrauto_complete_report.md) so a folder containing
    # multiple runs is easy to scan.
    prefix = ticker.split(".")[0].lower() if ticker else "report"
    header = f"# Trading Analysis Report: {ticker}\n\nGenerated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
    md_path = save_path / f"{prefix}_complete_report.md"
    md_path.write_text(header + "\n\n".join(sections), encoding="utf-8")
    return md_path


def display_complete_report(final_state):
    """Display the complete analysis report sequentially (avoids truncation)."""
    console.print()
    console.print(Rule("Complete Analysis Report", style="bold green"))

    # I. Analyst Team Reports
    analysts = []
    if final_state.get("market_report"):
        analysts.append(("Market Analyst", final_state["market_report"]))
    if final_state.get("sentiment_report"):
        analysts.append(("Sentiment Analyst", final_state["sentiment_report"]))
    if final_state.get("news_report"):
        analysts.append(("News Analyst", final_state["news_report"]))
    if final_state.get("fundamentals_report"):
        analysts.append(("Fundamentals Analyst", final_state["fundamentals_report"]))
    if analysts:
        console.print(Panel("[bold]I. Analyst Team Reports[/bold]", border_style="cyan"))
        for title, content in analysts:
            console.print(Panel(Markdown(content), title=title, border_style="blue", padding=(1, 2)))

    # II. Research Team Reports
    if final_state.get("investment_debate_state"):
        debate = final_state["investment_debate_state"]
        research = []
        if debate.get("bull_history"):
            research.append(("Bull Researcher", debate["bull_history"]))
        if debate.get("bear_history"):
            research.append(("Bear Researcher", debate["bear_history"]))
        if debate.get("judge_decision"):
            research.append(("Research Manager", debate["judge_decision"]))
        if research:
            console.print(Panel("[bold]II. Research Team Decision[/bold]", border_style="magenta"))
            for title, content in research:
                console.print(Panel(Markdown(content), title=title, border_style="blue", padding=(1, 2)))

    # III. Trading Team
    if final_state.get("trader_investment_plan"):
        console.print(Panel("[bold]III. Trading Team Plan[/bold]", border_style="yellow"))
        console.print(Panel(Markdown(final_state["trader_investment_plan"]), title="Trader", border_style="blue", padding=(1, 2)))

    # IV. Risk Management Team
    if final_state.get("risk_debate_state"):
        risk = final_state["risk_debate_state"]
        risk_reports = []
        if risk.get("aggressive_history"):
            risk_reports.append(("Aggressive Analyst", risk["aggressive_history"]))
        if risk.get("conservative_history"):
            risk_reports.append(("Conservative Analyst", risk["conservative_history"]))
        if risk.get("neutral_history"):
            risk_reports.append(("Neutral Analyst", risk["neutral_history"]))
        if risk_reports:
            console.print(Panel("[bold]IV. Risk Management Team Decision[/bold]", border_style="red"))
            for title, content in risk_reports:
                console.print(Panel(Markdown(content), title=title, border_style="blue", padding=(1, 2)))

        # V. Portfolio Manager Decision
        if risk.get("judge_decision"):
            console.print(Panel("[bold]V. Portfolio Manager Decision[/bold]", border_style="green"))
            console.print(Panel(Markdown(risk["judge_decision"]), title="Portfolio Manager", border_style="blue", padding=(1, 2)))


def update_research_team_status(status):
    """Update status for research team members (not Trader)."""
    research_team = ["Bull Researcher", "Bear Researcher", "Research Manager"]
    for agent in research_team:
        message_buffer.update_agent_status(agent, status)


# Ordered list of analysts for status transitions
ANALYST_ORDER = ["market", "social", "news", "fundamentals"]
ANALYST_AGENT_NAMES = {
    "market": "Market Analyst",
    "social": "Sentiment Analyst",
    "news": "News Analyst",
    "fundamentals": "Fundamentals Analyst",
}
ANALYST_REPORT_MAP = {
    "market": "market_report",
    "social": "sentiment_report",
    "news": "news_report",
    "fundamentals": "fundamentals_report",
}


def update_analyst_statuses(message_buffer, chunk):
    """Update analyst statuses based on accumulated report state.

    Logic:
    - Store new report content from the current chunk if present
    - Check accumulated report_sections (not just current chunk) for status
    - Analysts with reports = completed
    - First analyst without report = in_progress
    - Remaining analysts without reports = pending
    - When all analysts done, set Bull Researcher to in_progress
    """
    selected = message_buffer.selected_analysts
    found_active = False

    for analyst_key in ANALYST_ORDER:
        if analyst_key not in selected:
            continue

        agent_name = ANALYST_AGENT_NAMES[analyst_key]
        report_key = ANALYST_REPORT_MAP[analyst_key]

        # Capture new report content from current chunk
        if chunk.get(report_key):
            message_buffer.update_report_section(report_key, chunk[report_key])

        # Determine status from accumulated sections, not just current chunk
        has_report = bool(message_buffer.report_sections.get(report_key))

        if has_report:
            message_buffer.update_agent_status(agent_name, "completed")
        elif not found_active:
            message_buffer.update_agent_status(agent_name, "in_progress")
            found_active = True
        else:
            message_buffer.update_agent_status(agent_name, "pending")

    # When all analysts complete, transition research team to in_progress
    if not found_active and selected:
        if message_buffer.agent_status.get("Bull Researcher") == "pending":
            message_buffer.update_agent_status("Bull Researcher", "in_progress")

def extract_content_string(content):
    """Extract string content from various message formats.
    Returns None if no meaningful text content is found.
    """
    import ast

    def is_empty(val):
        """Check if value is empty using Python's truthiness."""
        if val is None or val == '':
            return True
        if isinstance(val, str):
            s = val.strip()
            if not s:
                return True
            try:
                return not bool(ast.literal_eval(s))
            except (ValueError, SyntaxError):
                return False  # Can't parse = real text
        return not bool(val)

    if is_empty(content):
        return None

    if isinstance(content, str):
        return content.strip()

    if isinstance(content, dict):
        text = content.get('text', '')
        return text.strip() if not is_empty(text) else None

    if isinstance(content, list):
        text_parts = [
            item.get('text', '').strip() if isinstance(item, dict) and item.get('type') == 'text'
            else (item.strip() if isinstance(item, str) else '')
            for item in content
        ]
        result = ' '.join(t for t in text_parts if t and not is_empty(t))
        return result if result else None

    return str(content).strip() if not is_empty(content) else None


def classify_message_type(message) -> tuple[str, str | None]:
    """Classify LangChain message into display type and extract content.

    Returns:
        (type, content) - type is one of: User, Agent, Data, Control
                        - content is extracted string or None
    """
    from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

    content = extract_content_string(getattr(message, 'content', None))

    if isinstance(message, HumanMessage):
        if content and content.strip() == "Continue":
            return ("Control", content)
        return ("User", content)

    if isinstance(message, ToolMessage):
        return ("Data", content)

    if isinstance(message, AIMessage):
        return ("Agent", content)

    # Fallback for unknown types
    return ("System", content)


def format_tool_args(args, max_length=80) -> str:
    """Format tool arguments for terminal display."""
    result = str(args)
    if len(result) > max_length:
        return result[:max_length - 3] + "..."
    return result

def run_analysis(
    checkpoint: bool = False,
    overrides: dict | None = None,
    interactive: bool = True,
    save_report: bool = True,
    report_name: Optional[str] = None,
    display_report: bool = False,
    open_report: bool = True,
):
    # First get all user selections
    selections = get_user_selections(overrides=overrides, interactive=interactive)

    # Create config with selected research depth
    config = DEFAULT_CONFIG.copy()
    config["max_debate_rounds"] = selections["research_depth"]
    config["max_risk_discuss_rounds"] = selections["research_depth"]
    config["quick_think_llm"] = selections["shallow_thinker"]
    config["deep_think_llm"] = selections["deep_thinker"]
    config["backend_url"] = selections["backend_url"]
    config["llm_provider"] = selections["llm_provider"].lower()
    # Provider-specific thinking configuration
    config["google_thinking_level"] = selections.get("google_thinking_level")
    config["openai_reasoning_effort"] = selections.get("openai_reasoning_effort")
    config["anthropic_effort"] = selections.get("anthropic_effort")
    config["output_language"] = selections.get("output_language", "English")
    config["trading_horizon"] = selections.get("trading_horizon", "position")
    config["checkpoint_enabled"] = checkpoint

    # Create stats callback handler for tracking LLM/tool calls
    stats_handler = StatsCallbackHandler()

    # Normalize analyst selection to predefined order (selection is a 'set', order is fixed)
    selected_set = {analyst.value for analyst in selections["analysts"]}
    selected_analyst_keys = [a for a in ANALYST_ORDER if a in selected_set]

    # Initialize the graph with callbacks bound to LLMs
    graph = TradingAgentsGraph(
        selected_analyst_keys,
        config=config,
        debug=True,
        callbacks=[stats_handler],
    )

    # Initialize message buffer with selected analysts
    message_buffer.init_for_analysis(selected_analyst_keys)

    # Track start time for elapsed display
    start_time = time.time()

    # Create result directory
    results_dir = Path(config["results_dir"]) / selections["ticker"] / selections["analysis_date"]
    results_dir.mkdir(parents=True, exist_ok=True)
    report_dir = results_dir / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    log_file = results_dir / "message_tool.log"
    log_file.touch(exist_ok=True)

    def save_message_decorator(obj, func_name):
        func = getattr(obj, func_name)
        @wraps(func)
        def wrapper(*args, **kwargs):
            func(*args, **kwargs)
            timestamp, message_type, content = obj.messages[-1]
            content = content.replace("\n", " ")  # Replace newlines with spaces
            with open(log_file, "a", encoding="utf-8") as f:
                f.write(f"{timestamp} [{message_type}] {content}\n")
        return wrapper
    
    def save_tool_call_decorator(obj, func_name):
        func = getattr(obj, func_name)
        @wraps(func)
        def wrapper(*args, **kwargs):
            func(*args, **kwargs)
            timestamp, tool_name, args = obj.tool_calls[-1]
            args_str = ", ".join(f"{k}={v}" for k, v in args.items())
            with open(log_file, "a", encoding="utf-8") as f:
                f.write(f"{timestamp} [Tool Call] {tool_name}({args_str})\n")
        return wrapper

    def save_report_section_decorator(obj, func_name):
        func = getattr(obj, func_name)
        @wraps(func)
        def wrapper(section_name, content):
            func(section_name, content)
            if section_name in obj.report_sections and obj.report_sections[section_name] is not None:
                content = obj.report_sections[section_name]
                if content:
                    file_name = f"{section_name}.md"
                    text = "\n".join(str(item) for item in content) if isinstance(content, list) else content
                    with open(report_dir / file_name, "w", encoding="utf-8") as f:
                        f.write(text)
        return wrapper

    message_buffer.add_message = save_message_decorator(message_buffer, "add_message")
    message_buffer.add_tool_call = save_tool_call_decorator(message_buffer, "add_tool_call")
    message_buffer.update_report_section = save_report_section_decorator(message_buffer, "update_report_section")

    # Now start the display layout
    layout = create_layout()

    with Live(layout, refresh_per_second=4) as live:
        # Initial display
        update_display(layout, stats_handler=stats_handler, start_time=start_time)

        # Add initial messages
        message_buffer.add_message("System", f"Selected ticker: {selections['ticker']}")
        message_buffer.add_message(
            "System", f"Analysis date: {selections['analysis_date']}"
        )
        message_buffer.add_message(
            "System",
            f"Selected analysts: {', '.join(analyst.value for analyst in selections['analysts'])}",
        )
        update_display(layout, stats_handler=stats_handler, start_time=start_time)

        # Update agent status to in_progress for the first analyst
        first_analyst = f"{selections['analysts'][0].value.capitalize()} Analyst"
        message_buffer.update_agent_status(first_analyst, "in_progress")
        update_display(layout, stats_handler=stats_handler, start_time=start_time)

        # Create spinner text
        spinner_text = (
            f"Analyzing {selections['ticker']} on {selections['analysis_date']}..."
        )
        update_display(layout, spinner_text, stats_handler=stats_handler, start_time=start_time)

        # Initialize state and get graph args with callbacks
        init_agent_state = graph.propagator.create_initial_state(
            selections["ticker"], selections["analysis_date"]
        )
        # Pass callbacks to graph config for tool execution tracking
        # (LLM tracking is handled separately via LLM constructor)
        args = graph.propagator.get_graph_args(callbacks=[stats_handler])

        # Stream the analysis
        trace = []
        for chunk in graph.graph.stream(init_agent_state, **args):
            # Process all messages in chunk, deduplicating by message ID
            for message in chunk.get("messages", []):
                msg_id = getattr(message, "id", None)
                if msg_id is not None:
                    if msg_id in message_buffer._processed_message_ids:
                        continue
                    message_buffer._processed_message_ids.add(msg_id)

                msg_type, content = classify_message_type(message)
                if content and content.strip():
                    message_buffer.add_message(msg_type, content)

                if hasattr(message, "tool_calls") and message.tool_calls:
                    for tool_call in message.tool_calls:
                        if isinstance(tool_call, dict):
                            message_buffer.add_tool_call(tool_call["name"], tool_call["args"])
                        else:
                            message_buffer.add_tool_call(tool_call.name, tool_call.args)

            # Update analyst statuses based on report state (runs on every chunk)
            update_analyst_statuses(message_buffer, chunk)

            # Research Team - Handle Investment Debate State
            if chunk.get("investment_debate_state"):
                debate_state = chunk["investment_debate_state"]
                bull_hist = debate_state.get("bull_history", "").strip()
                bear_hist = debate_state.get("bear_history", "").strip()
                judge = debate_state.get("judge_decision", "").strip()

                # Only update status when there's actual content
                if bull_hist or bear_hist:
                    update_research_team_status("in_progress")
                if bull_hist:
                    message_buffer.update_report_section(
                        "investment_plan", f"### Bull Researcher Analysis\n{bull_hist}"
                    )
                if bear_hist:
                    message_buffer.update_report_section(
                        "investment_plan", f"### Bear Researcher Analysis\n{bear_hist}"
                    )
                if judge:
                    message_buffer.update_report_section(
                        "investment_plan", f"### Research Manager Decision\n{judge}"
                    )
                    update_research_team_status("completed")
                    message_buffer.update_agent_status("Trader", "in_progress")

            # Trading Team
            if chunk.get("trader_investment_plan"):
                message_buffer.update_report_section(
                    "trader_investment_plan", chunk["trader_investment_plan"]
                )
                if message_buffer.agent_status.get("Trader") != "completed":
                    message_buffer.update_agent_status("Trader", "completed")
                    message_buffer.update_agent_status("Aggressive Analyst", "in_progress")

            # Risk Management Team - Handle Risk Debate State
            if chunk.get("risk_debate_state"):
                risk_state = chunk["risk_debate_state"]
                agg_hist = risk_state.get("aggressive_history", "").strip()
                con_hist = risk_state.get("conservative_history", "").strip()
                neu_hist = risk_state.get("neutral_history", "").strip()
                judge = risk_state.get("judge_decision", "").strip()

                if agg_hist:
                    if message_buffer.agent_status.get("Aggressive Analyst") != "completed":
                        message_buffer.update_agent_status("Aggressive Analyst", "in_progress")
                    message_buffer.update_report_section(
                        "final_trade_decision", f"### Aggressive Analyst Analysis\n{agg_hist}"
                    )
                if con_hist:
                    if message_buffer.agent_status.get("Conservative Analyst") != "completed":
                        message_buffer.update_agent_status("Conservative Analyst", "in_progress")
                    message_buffer.update_report_section(
                        "final_trade_decision", f"### Conservative Analyst Analysis\n{con_hist}"
                    )
                if neu_hist:
                    if message_buffer.agent_status.get("Neutral Analyst") != "completed":
                        message_buffer.update_agent_status("Neutral Analyst", "in_progress")
                    message_buffer.update_report_section(
                        "final_trade_decision", f"### Neutral Analyst Analysis\n{neu_hist}"
                    )
                if judge:
                    if message_buffer.agent_status.get("Portfolio Manager") != "completed":
                        message_buffer.update_agent_status("Portfolio Manager", "in_progress")
                        message_buffer.update_report_section(
                            "final_trade_decision", f"### Portfolio Manager Decision\n{judge}"
                        )
                        message_buffer.update_agent_status("Aggressive Analyst", "completed")
                        message_buffer.update_agent_status("Conservative Analyst", "completed")
                        message_buffer.update_agent_status("Neutral Analyst", "completed")
                        message_buffer.update_agent_status("Portfolio Manager", "completed")

            # Update the display
            update_display(layout, stats_handler=stats_handler, start_time=start_time)

            trace.append(chunk)

        # Streamed chunks are per-node deltas, not full state. Merge them
        # so every report field populated across the run is present.
        final_state = {}
        for chunk in trace:
            final_state.update(chunk)
        decision = graph.process_signal(final_state["final_trade_decision"])

        # Update all agent statuses to completed
        for agent in message_buffer.agent_status:
            message_buffer.update_agent_status(agent, "completed")

        message_buffer.add_message(
            "System", f"Completed analysis for {selections['analysis_date']}"
        )

        # Update final report sections
        for section in message_buffer.report_sections.keys():
            if section in final_state:
                message_buffer.update_report_section(section, final_state[section])

        update_display(layout, stats_handler=stats_handler, start_time=start_time)

    # Post-analysis actions (outside Live context for clean output)
    console.print("\n[bold cyan]Analysis Complete![/bold cyan]\n")

    if save_report:
        if report_name:
            save_path = Path(report_name)
            if not save_path.is_absolute():
                save_path = Path.cwd() / "reports" / save_path
        else:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = Path.cwd() / "reports" / f"{selections['ticker']}_{timestamp}"
        try:
            report_file = save_report_to_disk(final_state, selections["ticker"], save_path)
            console.print(f"[green]✓ Report saved to:[/green] {report_file.resolve()}")
            pdf_file = _markdown_to_pdf(report_file)
            if pdf_file:
                console.print(f"[green]✓ PDF saved to:[/green]    {pdf_file.resolve()}")
            if open_report:
                target = pdf_file if pdf_file else report_file
                _open_in_default_viewer(target.resolve())
        except Exception as e:
            console.print(f"[red]Error saving report: {e}[/red]")
    else:
        console.print("[dim]Skipped report save (--skip-save-report).[/dim]")

    if display_report:
        display_complete_report(final_state)


@app.command()
def analyze(
    ticker: str = typer.Option(
        DEFAULTS["ticker"], "--ticker", "-t",
        help="Ticker symbol with exchange suffix when needed (e.g. SPY, CNC.TO, 7203.T).",
    ),
    date: Optional[str] = typer.Option(
        None, "--date", "-d",
        help="Analysis date (YYYY-MM-DD). Defaults to today.",
    ),
    language: str = typer.Option(
        DEFAULTS["language"], "--language", "-l",
        help="Output language for analyst reports and final decision.",
    ),
    analysts: str = typer.Option(
        DEFAULTS["analysts"], "--analysts",
        help="Comma-separated analysts: market,social,news,fundamentals.",
    ),
    depth: str = typer.Option(
        DEFAULTS["depth"], "--depth",
        help="Research depth: shallow | medium | deep.",
    ),
    horizon: str = typer.Option(
        DEFAULTS["horizon"], "--horizon",
        help="Trading horizon: swing (2-6 weeks) | position (3-6 months) | long-term (12+ months). "
             "Drives analyst lookback windows and the holding period the trader / PM target.",
    ),
    provider: str = typer.Option(
        DEFAULTS["provider"], "--provider", "-p",
        help="LLM provider key (e.g. glm-anthropic, openai, anthropic, xai).",
    ),
    quick_model: str = typer.Option(
        DEFAULTS["quick_model"], "--quick-model",
        help="Model id for quick-thinking agents.",
    ),
    deep_model: str = typer.Option(
        DEFAULTS["deep_model"], "--deep-model",
        help="Model id for deep-thinking agents.",
    ),
    effort: Optional[str] = typer.Option(
        None, "--effort",
        help="Anthropic effort level (low|medium|high). Only applies to anthropic provider.",
    ),
    reasoning_effort: Optional[str] = typer.Option(
        None, "--reasoning-effort",
        help="OpenAI reasoning effort (low|medium|high). Only applies to openai provider.",
    ),
    thinking_level: Optional[str] = typer.Option(
        None, "--thinking-level",
        help="Gemini thinking level. Only applies to google provider.",
    ),
    interactive: bool = typer.Option(
        False, "--interactive", "-i",
        help="Show step-by-step prompts instead of using flag defaults.",
    ),
    checkpoint: bool = typer.Option(
        False,
        "--checkpoint",
        help="Enable checkpoint/resume: save state after each node so a crashed run can resume.",
    ),
    clear_checkpoints: bool = typer.Option(
        False,
        "--clear-checkpoints",
        help="Delete all saved checkpoints before running (force fresh start).",
    ),
    report_name: Optional[str] = typer.Option(
        None, "--report-name",
        help="Override generated report folder name. Relative paths land under ./reports/.",
    ),
    skip_save_report: bool = typer.Option(
        False, "--skip-save-report",
        help="Do not save the consolidated report to ./reports/ at the end of the run.",
    ),
    display_report: bool = typer.Option(
        False, "--display-report",
        help="Print the full consolidated report to the terminal after the run.",
    ),
    skip_open_report: bool = typer.Option(
        False, "--skip-open-report",
        help="Do not auto-launch the saved report in the default OS viewer.",
    ),
    profile: Optional[str] = typer.Option(
        None, "--profile",
        help=f"Named bundle of flag values: {', '.join(PROFILES)}. "
             "Explicit flags override profile values.",
    ),
):
    if clear_checkpoints:
        from tradingagents.graph.checkpointer import clear_all_checkpoints
        n = clear_all_checkpoints(DEFAULT_CONFIG["data_cache_dir"])
        console.print(f"[yellow]Cleared {n} checkpoint(s).[/yellow]")

    if profile is not None:
        profile_key = profile.lower()
        if profile_key not in PROFILES:
            raise typer.BadParameter(
                f"--profile must be one of {list(PROFILES)}, got {profile!r}"
            )
        bundle = PROFILES[profile_key]
        if analysts == DEFAULTS["analysts"] and "analysts" in bundle:
            analysts = bundle["analysts"]
        if horizon == DEFAULTS["horizon"] and "horizon" in bundle:
            horizon = bundle["horizon"]
        if depth == DEFAULTS["depth"] and "depth" in bundle:
            depth = bundle["depth"]
        console.print(f"[dim]Profile:[/dim] [bold]{profile_key}[/bold]")

    depth_key = depth.lower()
    if depth_key not in DEPTH_MAP:
        raise typer.BadParameter(
            f"--depth must be one of {list(DEPTH_MAP)}, got {depth!r}"
        )

    horizon_key = horizon.lower()
    if horizon_key not in HORIZON_CHOICES:
        raise typer.BadParameter(
            f"--horizon must be one of {list(HORIZON_CHOICES)}, got {horizon!r}"
        )

    try:
        analyst_enums = [
            AnalystType(a.strip().lower())
            for a in analysts.split(",") if a.strip()
        ]
    except ValueError as e:
        raise typer.BadParameter(
            f"--analysts contains an unknown analyst: {e}. "
            f"Valid: {[a.value for a in AnalystType]}"
        )
    if not analyst_enums:
        raise typer.BadParameter("--analysts must list at least one analyst.")

    overrides = {
        "ticker": ticker,
        "analysis_date": date or datetime.datetime.now().strftime("%Y-%m-%d"),
        "output_language": language,
        "analysts": analyst_enums,
        "research_depth": DEPTH_MAP[depth_key],
        "llm_provider": provider.lower(),
        "backend_url": get_provider_backend_url(provider),
        "shallow_thinker": quick_model,
        "deep_thinker": deep_model,
        "anthropic_effort": effort,
        "openai_reasoning_effort": reasoning_effort,
        "google_thinking_level": thinking_level,
        "trading_horizon": horizon_key,
    }

    run_analysis(
        checkpoint=checkpoint,
        overrides=overrides,
        interactive=interactive,
        save_report=not skip_save_report,
        report_name=report_name,
        display_report=display_report,
        open_report=not skip_open_report,
    )


if __name__ == "__main__":
    app()
