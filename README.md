# Discovery Agents

Training scientific discovery agents.

LLM agents are placed in simulated physical worlds with unknown governing laws. Through iterative experimentation, observing particle trajectories, designing new experiments, and proposing equations, they must discover the hidden physics from scratch. Can they do this? Can we help them? All to be determined.

The simulator generates diverse worlds by randomizing field equations, particle-field couplings, and symmetry structures, forcing agents to perform genuine scientific reasoning rather than pattern matching against known physics.

## How It Works

Each world is governed by a generalized field equation:

$$\frac{\partial^n \varphi}{\partial t^n} = \mathcal{L}[\varphi] + \mathcal{N}[\varphi] + S(\text{particles})$$

where $n \in \{0, 1, 2\}$ sets the temporal order (constraint, diffusion, or wave), $\mathcal{L}$ is a linear spatial operator, $\mathcal{N}$ contains nonlinear terms, and $S$ couples particles to the field. Particles feel forces from the field and move according to Newton's second law.

The agent doesn't see any of this. It only sees noisy particle positions over time — and must figure out the rest.

**Discovery loop:**

1. The agent receives a mission describing what it can observe and control
2. It designs an experiment (particle positions, velocities, properties)
3. The simulator runs the experiment and returns trajectory data
4. The agent analyzes results, forms hypotheses, and designs follow-up experiments
5. After sufficient evidence, it submits a proposed law as executable Python
6. The law is evaluated against held-out test trajectories

## Repository Structure

```
discovery-agents/
├── PhysicsSchool/                        # Physics simulation engine
│   ├── setup.py
│   ├── physchool/
│   │   ├── __init__.py
│   │   └── worlds/
│   │       ├── field_sampler.py          # Core FieldSampler class
│   │       ├── utils.py                  # Cloud-in-Cell interpolation
│   │       └── main.py
│   ├── tests/
│   │   ├── test_cic.py                   # CIC paint/read tests
│   │   ├── test_forces.py                # Force computation tests
│   │   ├── test_field_evolution.py        # Field time-stepping tests
│   │   ├── test_trajectories.py          # Particle trajectory tests
│   │   ├── test_circle_world.py          # Circle geometry world tests
│   │   └── test_species_world.py         # Hidden species world tests
│   ├── nbs/
│   │   └── gravity_example.ipynb         # Example notebook
│   └── prompts/                          # Agent system prompts
│       ├── run_experiments.md
│       ├── run_experiments_circle.md
│       └── run_experiments_species.md
│
├── ScienceAgent/                         # LLM discovery agent
│   ├── setup.py
│   ├── run_discovery.py                  # CLI entry point
│   ├── scienceagent/
│   │   ├── __init__.py
│   │   ├── agent.py                      # DiscoveryAgent main loop
│   │   ├── critic.py                     # Supervisor critic agent
│   │   ├── executor.py                   # Simulation executors
│   │   ├── evaluator.py                  # Law evaluation and scoring
│   │   ├── llm_client.py                 # Multi-provider LLM client
│   │   └── worlds.py                     # Predefined world configs
│   └── tests/
│       └── test_executor.py
│
├── .gitignore
├── LICENSE
└── README.md
```

## Predefined Worlds

| World | Temporal Order | Operator | What the Agent Must Discover |
|-------|---------------|----------|------------------------------|
| **Gravity** | $n=0$ (constraint) | Laplacian | Classical inverse-distance force law |
| **Fractional** | $n=0$ (constraint) | Fractional Laplacian | Anomalous power-law force |
| **Circle** | $n=0$ (constraint) | Fractional Laplacian | Force law from ring geometry (11 particles) |

## Getting Started

### Prerequisites

- Python 3.9+
- [JAX](https://github.com/jax-ml/jax)

### Installation

```bash
# Clone the repository
git clone https://github.com/SampsonML/discovery-agents.git
cd discovery-agents

# Install the physics simulator
pip install -e PhysicsSchool/

# Install the discovery agent
pip install -e ScienceAgent/
```

### Running the Tests

```bash
pytest PhysicsSchool/tests/
```

### Running a Discovery Agent

Set your API key for the LLM provider:

```bash
export ANTHROPIC_API_KEY="your-key-here"
```

Run the agent on a world:

```bash
python ScienceAgent/run_discovery.py --world gravity --model claude-sonnet-4-5
```

The agent will iteratively design experiments, observe results, and propose a governing law. Results are saved as JSON logs and trajectory plots.

### Supervisor Critic

Enable an optional supervisor agent that reviews each experiment round (from round 2 onward) for rule compliance and information gain:

```bash
python ScienceAgent/run_discovery.py --world gravity --model claude-sonnet-4-5 --use-critic
```

The critic defaults to `claude-haiku-4-5-20251001` for fast, low-cost feedback. Override with `--critic-model`:

```bash
python ScienceAgent/run_discovery.py --world gravity --model claude-sonnet-4-20250514 --use-critic --critic-model claude-sonnet-4-20250514
```

The critic checks that the science agent follows its experimental protocol and that each experiment provides new information not seen in previous rounds. Feedback is injected into the conversation so the science agent can course-correct.

### Example: Fractional Gravity on a Ring

Eleven particles are placed on a ring and interact via a fractional-Laplacian gravity field. The agent must discover the anomalous power-law force from noisy trajectories alone. Run with Opus 4.6 as the discovery agent and Sonnet 4.5 as the critic:

```bash
python ScienceAgent/run_discovery.py \
  --world circle \
  --model claude-opus-4-6 \
  --use-critic \
  --critic-model claude-sonnet-4-5 \
  --plot circle_plot.png
```

The true fractional exponent is $\alpha = 1.5$. The agent discovers a force law with fractional exponent $\alpha = (1+\sqrt{5})/2$ (the golden ratio) and achieves a mean position error of ~0.064:

![Trajectory comparison](imgs/circle_plot.png)

The discovered law submitted by the agent:

![Discovered law](imgs/circle_law.png)

## Supported LLM Providers

The agent supports multiple LLM backends though Groq seems to be the most frictionless free option:

- **Anthropic** (Claude)
- **OpenAI** (GPT, o1)
- **OpenRouter**, **Groq**, **HuggingFace**, **Ollama**

Set the corresponding environment variable (`ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, etc.) and pass the model name to `run_discovery.py`.

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

## Author

scientific discovery team
