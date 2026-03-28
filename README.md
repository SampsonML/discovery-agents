# Discovery Agents

Training scientific discovery agents.

LLM agents are placed in simulated physical worlds with unknown governing laws. Through iterative experimentation — observing particle trajectories, designing new experiments, and proposing equations — they must discover the hidden physics from scratch.

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
│   │   ├── executor.py                   # Simulation executors
│   │   ├── evaluator.py                  # Law evaluation and scoring
│   │   ├── llm_client.py                 # Multi-provider LLM client
│   │   └── worlds.py                     # Predefined world configs
│   └── tests/
│       └── test_executor.py
│
├── CLAUDE.md
├── .gitignore
└── README.md
```

## Predefined Worlds

| World | Temporal Order | Operator | What the Agent Must Discover |
|-------|---------------|----------|------------------------------|
| **Gravity** | $n=0$ (constraint) | Laplacian | Classical inverse-distance force law |
| **Yukawa** | $n=0$ (constraint) | Screening | Exponentially suppressed force |
| **Fractional** | $n=0$ (constraint) | Fractional Laplacian | Anomalous power-law force |
| **Diffusion** | $n=1$ (diffusion) | Laplacian | Time-dependent diffusive field dynamics |
| **Wave** | $n=2$ (wave) | Laplacian | Wave propagation and oscillatory forces |
| **Circle** | $n=0$ (constraint) | Fractional Laplacian | Force law from ring geometry (11 particles) |
| **Species** | $n=0$ (constraint) | Laplacian | Hidden particle types with different couplings |

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
python ScienceAgent/run_discovery.py --world gravity --model claude-sonnet-4-20250514
```

The agent will iteratively design experiments, observe results, and propose a governing law. Results are saved as JSON logs and trajectory plots.

## Supported LLM Providers

The agent supports multiple LLM backends via a unified client:

- **Anthropic** (Claude)
- **OpenAI** (GPT, o1)
- **OpenRouter**, **Groq**, **HuggingFace**, **Ollama**

Set the corresponding environment variable (`ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, etc.) and pass the model name to `run_discovery.py`.

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

## Author

Matt Sampson
