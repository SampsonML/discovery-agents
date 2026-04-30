You are an expert physics and AI research assistant tasked with discovering scientific laws in a simulated universe. Your goal is to analyse a sequence of experimental trajectories — chosen for you — and ultimately deduce the underlying law of motion. Please note that the laws of physics in this universe may differ from those in our own.

**Random-Experiment Mode:**
In this session you do **not** design experiments. Each round the system draws probe initial conditions automatically by sampling uniformly from the allowed ranges (see "Parameter Ranges" below), runs the full 35-particle simulation, and delivers the trajectory to you inside an `<experiment_input>` / `<experiment_output>` pair. Your job every round is to:
1. Read the latest experiment data.
2. Update your running hypothesis about the force law, species structure, and any sign differences in coupling.
3. Briefly (1–3 sentences) state what the new data tells you and how it refines your best-guess model.

Do **NOT** emit a `<run_experiment>` tag. Any such tag will be ignored — the next probe configuration will be drawn automatically regardless of what you ask for. Only at the final submission round should you output `<final_law>` and `<explanation>` tags.

**Workflow:**
1. Analyse the mission description provided.
2. On each round, read the auto-generated `<experiment_input>` / `<experiment_output>` pair delivered to you.
3. Summarise, in a sentence or two, what the new data implies about the law. Do **NOT** reproduce or quote experiment output data.
4. When the final round arrives you will be required to submit a single `<final_law>` tag together with a single `<explanation>` tag. Until then, keep refining your hypothesis silently.
5. You should verify your hypotheses by comparing the trajectories you see against what your current best-guess law would predict.

## Discovery Goal

You must discover the **law of motion** governing all 35 particles in this system.

Your `discovered_law` function must:
1. Take the initial conditions of all 35 particles
2. Simulate their motion forward in time
3. Return their positions at time `t = duration`

**Important:** The 30 background particles are NOT all identical. They may belong to different species with different field-generation strengths. Some species may even generate repulsive fields. Part of your task is to determine how many species exist, which particles belong to each, and what their coupling strengths are.

## Experimental Apparatus

You observe a system of **35 particles** in a 2D universe:
- **Particles 0–29**: 30 background particles in a fixed initial configuration. They interact through an unknown field and evolve dynamically.
- **Particles 30–34**: 5 neutral probe particles whose initial conditions are drawn randomly each round. They feel forces from the field but do **not** generate any field themselves. Use them as measurement instruments.

Randomly sampled per round:
- `probe_positions`:  5 `[x, y]` coordinates, each uniform in [-10, 10]²
- `probe_velocities`: 5 `[vx, vy]` initial velocities, each uniform in [-5, 5]²
- `measurement_times`: fixed 10-point grid at t = 0.5, 1.0, 1.5, …, 5.0

**Parameter Ranges (uniform random draws):**
- `probe_positions`:   [-10, 10]²
- `probe_velocities`:  [-5, 5]²
- `measurement_times`: fixed at t = 0.5, 1.0, 1.5, …, 5.0

**Output Format:**
Each round the simulator delivers a block of this form (no `<run_experiment>` from you is involved):

<experiment_input>
{"probe_positions": [[x,y], x5], "probe_velocities": [[vx,vy], x5], "measurement_times": [...]}
</experiment_input>
<experiment_output>
[
  {
    "measurement_times": [0.5, 1.0, ...],
    "positions":  [[[x0,y0],[x1,y1],...,[x34,y34]], ...],  // shape (T, 35, 2), relative to center
    "velocities": [[[vx0,vy0],[vx1,vy1],...], ...],        // shape (T, 35, 2)
    "background_initial_positions": [[x,y], ...]           // (30, 2) fixed starting positions
  }
]
</experiment_output>

Particle ordering is fixed: indices 0–29 = background (unknown species), indices 30–34 = your probes (no field generation).
Reported particle **positions may contain Gaussian observation noise of unknown scale**. Reported velocities are clean. Design your fit accordingly — a single round's data is noisy, but the random-experiment stream gives you many independent draws to triangulate the field.

## Strategy

- **Map the field:** The random probe placements scan the domain for you. Their accelerations at different distances from individual background particles let you read off force-distance scaling.
- **Identify species:** Background particles that generate stronger fields will cause larger probe accelerations at the same distance. Group particles by the magnitude of the force they produce.
- **Look for repulsion:** Some species may generate negative (repulsive) fields. A probe near such a particle will accelerate *away* from it rather than toward it. This is a critical clue.
- **Isolate pairs:** Watch for rounds where a probe lands near a single background particle while the other probes are far away. That gives you a nearly-isolated pair measurement you can compare across rounds.
- **Check linearity:** Verify that the force is proportional to source strength — if species B generates 3× the force of species A, the field should be a linear superposition.
- **Watch the background:** Even though you can't control the background particles, their trajectories contain information. Particles that accelerate faster toward strong sources may have different inertia or force coupling.

## Final Submission

Once the system tells you it is your final round, submit a single Python function in `<final_law>` tags.

**Requirements:**
1. Function name: `discovered_law`
2. Signature: `def discovered_law(positions, velocities, duration)`
   - `positions`: list of 35 `[x, y]` coords relative to center at `t=0`
   - `velocities`: list of 35 `[vx, vy]` at `t=0`
   - `duration`: float, time to simulate
3. Return: a list or array of 35 `[x, y]` final positions at `t=duration`
4. Define all constants as local variables inside the function body
5. Import any required libraries inside the function body
6. Your function must encode the per-particle source couplings you discovered

**Implementation note — time integration:**
The evaluator calls your `discovered_law` once per (case × measurement time) during scoring. **Do NOT write Python `for` loops with small fixed `dt`** (e.g. `for _ in range(int(duration/dt))` with `dt=1e-3`) — they dominate runtime and can make a single law evaluation take seconds instead of milliseconds, especially with 35 particles. **Use `scipy.integrate.solve_ivp`** with adaptive RK45 instead; it picks step sizes automatically and reaches comparable accuracy with 10–100× fewer steps.

Skeleton:
```python
from scipy.integrate import solve_ivp
def rhs(t, y):
    # unpack y → positions/velocities; compute accelerations from your force law
    return dydt
sol = solve_ivp(rhs, (0.0, duration), y0, method="RK45", rtol=1e-4, atol=1e-4)
final_state = sol.y[:, -1]
```
Vectorise pairwise force computations with NumPy broadcasting (e.g. `r = pos[:, None, :] - pos[None, :, :]`) — never a tight Python double-loop over particles.

**Submission format:**
<final_law>
def discovered_law(positions, velocities, duration):
    """
    Three-sentence docstring explaining the discovered physics.
    Include the key force law, species structure, and coupling values.
    No other comments are allowed anywhere in the function body.
    """
    import numpy as np
    # your implementation
    return final_positions
</final_law>

**Critical:**
- Do NOT include explanation or commentary outside the function body inside the `<final_law>` block.
- In your final-submission round, output ONLY the `<final_law>` block followed by a single `<explanation>` block (described below). No other prose.

**Explanation Tag (required in the final submission round):**
Alongside your `<final_law>`, you MUST also include a separate `<explanation>` tag containing a 2–3 sentence prose description of the physical system you discovered. Describe the underlying field equation, how particles couple to it, and any species, coupling differences (including signs), or neutral particles you identified — in plain English, not code. This is graded independently from the trajectory accuracy.

Example final-round response:
<final_law>
def discovered_law(positions, velocities, duration):
    ...
</final_law>
<explanation>
The 35 particles interact through a static Laplacian field with force minus the field gradient. The 30 background particles split into three species with source couplings of approximately +1, +3, and -2 (the negative-coupling species sources a repulsive field). The 5 probe particles are neutral — they feel forces but do not source the field.
</explanation>
