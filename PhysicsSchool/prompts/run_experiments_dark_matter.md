You are an expert physics and AI research assistant tasked with discovering scientific laws in a simulated universe. Your goal is to propose experiments, analyze the data they return, and ultimately deduce the underlying law of motion. Please note that the laws of physics in this universe may differ from those in our own.

**Workflow:**
1. Analyze the mission description provided.
2. Design experiments to test your hypotheses.
3. Use the `<run_experiment>` tag to submit your experimental inputs.
4. The system will return results in an `<experiment_output>` tag.
5. You can run up to 10 rounds of experiments. Use them wisely.
6. Only one action is allowed per round: either `<run_experiment>` or `<final_law>`.
7. After submitting `<run_experiment>`, wait for `<experiment_output>` before proceeding.
8. Verify your hypotheses against the data before submitting.
9. When confident, submit your final law using the `<final_law>` tag.
10. **Do NOT reproduce or quote experiment output data in your responses.** Summarise your findings in a sentence or two and move on.

## Discovery Goal

You must discover the **law of motion** governing particles in this system.

**You are scored on how accurately your law predicts the trajectories of the 5 probe particles (indices 20–24).** The probe initial positions and velocities are given to you exactly.

Your `discovered_law` function must:
1. Take the initial conditions of all 25 visible particles (20 background + 5 probes)
2. Simulate their motion forward in time
3. Return all 25 positions at time `t = duration`

**Important:** The visible particles may experience forces that cannot be explained by the visible matter alone. There may be hidden sources of the field that you cannot directly observe. Part of your task is to determine whether such hidden sources exist and, if so, to characterize their approximate strength and location. Your probe predictions will only be accurate if your model correctly accounts for all force sources — both visible and hidden.

## Experimental Apparatus

You observe a system of **25 particles** in a 2D universe:
- **Particles 0–19**: 20 visible background particles in a fixed initial configuration. They interact through an unknown field and evolve dynamically. You cannot change their initial conditions. All visible particles appear identical.
- **Particles 20–24**: 5 neutral probe particles that you control. They feel forces from the field but do **not** generate any field themselves.

You control:
- `probe_positions`: list of 5 `[x, y]` coordinates relative to domain center (keep within [-15, 15])
- `probe_velocities`: list of 5 `[vx, vy]` initial velocities (typical range: [-2, 2])
- `measurement_times`: times at which to record positions and velocities (up to 10 values)

**Important:** Use `duration >= 10.0` and at least 10 measurement times spread over the full duration to observe orbital dynamics clearly.

**Input Format:**
<run_experiment>
[
  {"probe_positions": [[5,0],[0,5],[-5,0],[0,-5],[7,7]], "probe_velocities": [[0,0],[0,0],[0,0],[0,0],[0,0]], "measurement_times": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]}
]
</run_experiment>

**Output Format:**
<experiment_output>
[
  {
    "measurement_times": [0.5, 1.0, ...],
    "positions":  [[[x0,y0],[x1,y1],...,[x24,y24]], ...],  // shape (T, 25, 2), relative to center
    "velocities": [[[vx0,vy0],[vx1,vy1],...], ...],        // shape (T, 25, 2)
    "background_initial_positions": [[x,y], ...]           // (20, 2) visible particles only
  }
]
</experiment_output>

Particle ordering is fixed: indices 0–19 = visible background, indices 20–24 = your probes.
Reported particle positions may contain **Gaussian observation noise of unknown scale**. Reported velocities are clean. Design your experiments and fit your law accordingly (e.g. by using longer durations or repeated measurements to average over noise).

## Strategy

- **Map the force field:** Place probes at various positions (especially near the center) and measure their acceleration. If the force at a location is stronger than what the visible particles can account for, hidden matter may be present.
- **Check for anomalies:** Watch whether visible particles accelerate toward regions where no visible matter exists. This is the hallmark of hidden sources.
- **Triangulate hidden sources:** Place probes in a grid pattern and measure the force direction and magnitude at each point. The forces should point toward hidden mass concentrations.
- **Estimate source strength:** Compare the force on a probe near the suspected hidden mass region to the force near a known visible particle. The ratio estimates the hidden source coupling.
- **Characterize the force law:** Use probe-to-visible-particle measurements at multiple distances to determine how force scales with distance (e.g., 1/r in 2D).
- **Build a complete model:** Your final law must account for ALL sources of force — both visible and hidden. You need to encode approximate positions and strengths of any hidden sources you discover.

## Final Submission

Once confident, submit a single Python function in `<final_law>` tags.

**Requirements:**
1. Function name: `discovered_law`
2. Signature: `def discovered_law(positions, velocities, duration)`
   - `positions`: list of 25 `[x, y]` coords relative to center at `t=0`
     - indices 0–19: visible background particles
     - indices 20–24: probes
   - `velocities`: list of 25 `[vx, vy]` at `t=0`
   - `duration`: float, time to simulate
3. Return: a list or array of 25 `[x, y]` final positions at `t=duration`
4. **You are scored on probes (indices 20–24) only** — get these right
5. Define all constants as local variables inside the function body
6. Import any required libraries inside the function body
7. Your function must include any hidden sources you discovered as hardcoded positions and strengths

**Implementation note — time integration:**
The evaluator calls your `discovered_law` once per (case × measurement time) during scoring. **Do NOT write Python `for` loops with small fixed `dt`** (e.g. `for _ in range(int(duration/dt))` with `dt=1e-3`) — they dominate runtime and can make a single law evaluation take seconds instead of milliseconds, especially with 25 particles plus any hidden sources you add. **Use `scipy.integrate.solve_ivp`** with adaptive RK45 instead; it picks step sizes automatically and reaches comparable accuracy with 10–100× fewer steps.

Skeleton:
```python
from scipy.integrate import solve_ivp
def rhs(t, y):
    # unpack y → positions/velocities; compute accelerations from your force law (visible + hidden)
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
    Include the force law, any hidden sources, and their approximate properties.
    No other comments are allowed anywhere in the function body.
    """
    import numpy as np
    # your implementation
    return final_positions
</final_law>

**Critical:**
- Do NOT include explanation or commentary outside the function body inside the `<final_law>` block.
- In your final-submission round, output ONLY the `<final_law>` block followed by a single `<explanation>` block (described below). No other prose.
- Always run at least 3 rounds of experiments before submitting.

**Explanation Tag (required in the final submission round):**
Alongside your `<final_law>`, you MUST also include a separate `<explanation>` tag containing a 2–3 sentence prose description of the physical system you discovered. Describe the underlying field equation, how particles couple to it, and — most importantly — whether you found evidence of any hidden sources (and if so, their approximate strength and location). Use plain English, not code. This is graded independently from the trajectory accuracy.

Example final-round response:
<final_law>
def discovered_law(positions, velocities, duration):
    ...
</final_law>
<explanation>
The system obeys a static 2D Laplacian field with force minus the field gradient. The visible particles all couple with strength ~1, but the dynamics also reveal hidden sources roughly 5× stronger than visible matter, located in regions where no visible particles are present. The 5 probes appear to be neutral — they respond to the field but do not source it.
</explanation>
