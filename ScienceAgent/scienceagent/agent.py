"""
DiscoveryAgent: runs the experiment-design loop until <final_law> is submitted.

The agent follows the protocol defined in PhysicsSchool/prompts/run_experiments.md:
  - Submits experiments via <run_experiment>...</run_experiment> XML tags
  - Receives results in <experiment_output>...</experiment_output> tags
  - Submits its discovered law in <final_law>...</final_law> tags
  - Maximum MAX_ROUNDS rounds before the loop is terminated
"""

import json
import re
from typing import Optional

from scienceagent import llm_client
from scienceagent.executor import SimulationExecutor

MAX_ROUNDS = 10
MIN_ROUNDS = 2

_SYSTEM_PROMPT_PATH = "PhysicsSchool/prompts/run_experiments.md"

_DEFAULT_LAW_STUB = (
    "def discovered_law(pos1, pos2, p1, p2, velocity2, duration):\n"
    "    # your best implementation\n"
    "    return final_pos2, final_vel2\n"
)

_DEFAULT_EXPERIMENT_FORMAT = (
    "<run_experiment>[{\"p1\": 1.0, \"p2\": 1.0, \"pos2\": [3.0, 0.0], "
    "\"velocity2\": [0.0, 0.0], \"measurement_times\": [0.5, 1.0, 2.0]}]</run_experiment>"
)


def _load_system_prompt(prompt_path: str = None) -> str:
    import os
    prompt_path = prompt_path or _SYSTEM_PROMPT_PATH
    here = os.path.dirname(__file__)
    root = os.path.abspath(os.path.join(here, "..", "..", ".."))
    path = os.path.join(root, prompt_path)
    if not os.path.exists(path):
        # Fallback: assume CWD is repo root
        path = prompt_path
    with open(path) as f:
        return f.read()


class DiscoveryAgent:
    """
    Drives an LLM through the physics-discovery experiment loop.

    Args:
        model: Model string passed to llm_client.complete().
        executor: SimulationExecutor instance wrapping the target world.
        mission: Optional mission description appended to the system prompt.
            Describes what the agent should discover without revealing the answer.
        max_tokens: Max tokens per LLM call.
        verbose: If True, print each round's exchange to stdout.
        show_experiment_output: If True, also print the simulator's experiment output each round.
    """

    def __init__(
        self,
        model: str,
        executor,
        mission: Optional[str] = None,
        max_tokens: int = 4096,
        verbose: bool = True,
        show_experiment_output: bool = False,
        system_prompt_path: str = None,
        law_stub: str = None,
        experiment_format: str = None,
        critic=None,
    ):
        self.model = model
        self.executor = executor
        self.mission = mission
        self.max_tokens = max_tokens
        self.verbose = verbose
        self.show_experiment_output = show_experiment_output
        self.critic = critic
        self._system_prompt_path = system_prompt_path or _SYSTEM_PROMPT_PATH
        self._law_stub = law_stub or _DEFAULT_LAW_STUB
        self._experiment_format = experiment_format or _DEFAULT_EXPERIMENT_FORMAT
        self._system = self._build_system_prompt()
        # Populated during run(); each entry is a dict describing one round.
        self.conversation_log: list[dict] = []

    def run(self) -> Optional[str]:
        """
        Run the discovery loop.

        Returns:
            The discovered law as a Python source string, or None if the agent
            did not submit a final law within MAX_ROUNDS.

        Side-effect:
            Populates self.conversation_log with one entry per round.
        """
        self.conversation_log = []
        messages = []
        if self.mission:
            messages.append({"role": "user", "content": self.mission})

        for round_num in range(1, MAX_ROUNDS + 1):
            if self.verbose:
                print(f"\n{'='*52}")
                print(f"󰙨 󰧑  science agent experimenting at round {round_num}/{MAX_ROUNDS} 󰧑  󰙨")
                print(f"{'='*52}")

            round_entry = {
                "round": round_num,
                "system_message": None,
                "llm_reply": None,
                "action": None,            # "experiment" | "final_law" | "warning" | "no_tag"
                "experiment_input": None,  # parsed JSON list, or None
                "experiment_output": None, # parsed JSON list, or None
                "experiment_error": None,
                "final_law": None,
                "critic_feedback": None,
            }

            # On the second-to-last round, warn the agent it must submit next round
            if round_num == MAX_ROUNDS - 1:
                warn_msg = (
                    f"Warning: this is round {round_num} of {MAX_ROUNDS}. "
                    "You have one round remaining. Your next response MUST be a <final_law> submission — "
                    "you will not be able to run further experiments after this."
                )
                messages.append({"role": "user", "content": warn_msg})
                round_entry["system_message"] = warn_msg

            # On the final round, force a final_law submission
            if round_num == MAX_ROUNDS:
                force_msg = (
                    "This is your final round. You MUST now submit your best guess "
                    "as a <final_law> regardless of confidence. Do not run more experiments.\n\n"
                    "<final_law>\n"
                    + self._law_stub
                    + "</final_law>"
                )
                messages.append({"role": "user", "content": force_msg})
                round_entry["system_message"] = force_msg

            reply = llm_client.complete(
                model=self.model,
                messages=messages,
                system=self._system,
                max_tokens=self.max_tokens,
            )
            round_entry["llm_reply"] = reply

            if self.verbose:
                print(f"\n[Science Agent]\n{reply}")

            messages.append({"role": "assistant", "content": reply})

            # Check for final law submission
            final_law = _extract_tag(reply, "final_law")
            if final_law is not None:
                if round_num >= MIN_ROUNDS:
                    if self.verbose:
                        print("\n[Agent submitted final law]")
                    round_entry["action"] = "final_law"
                    round_entry["final_law"] = final_law.strip()
                    self.conversation_log.append(round_entry)
                    return final_law.strip()
                else:
                    warn = (
                        f"You have only run {round_num} round(s) of experiments. "
                        f"You must run at least {MIN_ROUNDS} rounds before submitting a final law. "
                        "Please design and run at least one more experiment."
                    )
                    if self.verbose:
                        print(f"[Warning] Agent submitted final law too early (round {round_num}/{MIN_ROUNDS} minimum). Requiring more experiments.")
                    round_entry["action"] = "warning"
                    round_entry["system_message"] = warn
                    self.conversation_log.append(round_entry)
                    messages.append({"role": "user", "content": warn})
                    continue

            # Check for experiment request
            experiment_block = _extract_tag(reply, "run_experiment")
            if experiment_block is None:
                if self.verbose:
                    print("[Warning] No recognized tag in response. Prompting agent to continue.")
                no_tag_msg = (
                    "Your response did not contain a recognized XML tag. "
                    "You MUST wrap your output in one of these two formats exactly:\n\n"
                    "To run an experiment:\n"
                    + self._experiment_format + "\n\n"
                    + "To submit your final law:\n"
                    "<final_law>\n"
                    + self._law_stub
                    + "</final_law>\n\n"
                    "Do not describe what you will do — output the tag directly now."
                )
                round_entry["action"] = "no_tag"
                round_entry["system_message"] = no_tag_msg
                self.conversation_log.append(round_entry)
                messages.append({"role": "user", "content": no_tag_msg})
                continue

            # Run the experiments
            try:
                exp_input = json.loads(experiment_block)
                results = self.executor.run(exp_input)
                output_content = (
                    "<experiment_output>\n"
                    + json.dumps(results, indent=2)
                    + "\n</experiment_output>"
                )
                round_entry["action"] = "experiment"
                round_entry["experiment_input"] = exp_input
                round_entry["experiment_output"] = results
            except Exception as e:
                output_content = (
                    f"<experiment_output>\nError running experiment: {e}\n</experiment_output>"
                )
                round_entry["action"] = "experiment"
                round_entry["experiment_error"] = str(e)

            if self.verbose and self.show_experiment_output:
                print(f"\n[Simulator]\n{output_content}")

            messages.append({"role": "user", "content": output_content})

            # Critic feedback injection (skip round 1)
            if self.critic and round_num >= 2 and round_entry["action"] == "experiment":
                critic_feedback = self.critic.review(
                    agent_system_prompt=self._system,
                    messages=messages,
                    round_num=round_num,
                )
                if self.verbose:
                    print(f"\n[Supervisor Agent]\n{critic_feedback}")
                critic_msg = (
                    f"Supervisor feedback:\n{critic_feedback}\n\n"
                    "Before proceeding with your next experiment or final law submission, "
                    "briefly acknowledge the supervisor's feedback above: state what you will "
                    "change or why you disagree. Then continue with your action."
                )
                messages.append({"role": "user", "content": critic_msg})
                round_entry["critic_feedback"] = critic_feedback

            self.conversation_log.append(round_entry)

        if self.verbose:
            print(f"\n[Agent did not submit a final law within {MAX_ROUNDS} rounds]")
        return None

    # ── Internal ──────────────────────────────────────────────────────────────

    def _build_system_prompt(self) -> str:
        try:
            return _load_system_prompt(self._system_prompt_path)
        except FileNotFoundError:
            return (
                "You are a scientific discovery agent. Design experiments, "
                "analyze results, and discover the underlying law of physics."
            )


# ── Tag parsing ───────────────────────────────────────────────────────────────

def _extract_tag(text: str, tag: str) -> Optional[str]:
    """Return the content between <tag>...</tag>, or None if not present."""
    if not text:
        return None
    pattern = rf"<{tag}>(.*?)</{tag}>"
    match = re.search(pattern, text, re.DOTALL)
    return match.group(1) if match else None
