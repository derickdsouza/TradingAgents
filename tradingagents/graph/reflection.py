# TradingAgents/graph/reflection.py

from typing import Any


class Reflector:
    """Handles reflection on trading decisions."""

    def __init__(self, quick_thinking_llm: Any):
        """Initialize the reflector with an LLM."""
        self.quick_thinking_llm = quick_thinking_llm
        self.log_reflection_prompt = self._get_log_reflection_prompt()

    def _get_log_reflection_prompt(self) -> str:
        """Concise prompt for reflect_on_final_decision (Phase B log entries).

        Produces 2-4 sentences of plain prose — compact enough to be re-injected
        into future agent prompts without bloating the context window.
        """
        return (
            "You are a trading analyst reviewing your own past decision now that the outcome is known.\n"
            "Write exactly 2-4 sentences of plain prose (no bullets, no headers, no markdown).\n\n"
            "Cover in order:\n"
            "1. Was the directional call correct? (cite the alpha figure)\n"
            "2. Which part of the investment thesis held or failed?\n"
            "3. One concrete lesson to apply to the next similar analysis.\n\n"
            "Be specific and terse. Your output will be stored verbatim in a decision log "
            "and re-read by future analysts, so every word must earn its place."
        )

    def reflect_on_final_decision(
        self,
        final_decision: str,
        raw_return: float,
        alpha_return: float,
        benchmark: str = "SPY",
        holding_days: int | None = None,
        horizon: str | None = None,
    ) -> str:
        """Single reflection call on the final trade decision with outcome context.

        Used by Phase B deferred reflection. The final_trade_decision already
        synthesises all analyst insights, so no separate market context is
        needed. ``benchmark``, ``holding_days``, and ``horizon`` describe the
        measurement window used and are surfaced in the prompt so generated
        lessons are interpretable when re-read by future agents (e.g. an
        Indian-ticker decision measured over 63 trading days vs Nifty 500 is
        not comparable to a US-ticker decision over 5 days vs SPY).
        """
        window_lines = []
        if horizon:
            window_lines.append(f"Horizon: {horizon}")
        if holding_days is not None:
            window_lines.append(f"Holding window: {holding_days} trading days")
        window_block = ("\n".join(window_lines) + "\n") if window_lines else ""

        messages = [
            ("system", self.log_reflection_prompt),
            (
                "human",
                (
                    f"{window_block}"
                    f"Raw return: {raw_return:+.1%}\n"
                    f"Alpha vs {benchmark}: {alpha_return:+.1%}\n\n"
                    f"Final Decision:\n{final_decision}"
                ),
            ),
        ]
        return self.quick_thinking_llm.invoke(messages).content
