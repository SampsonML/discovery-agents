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

You must discover the **law of motion** governing all 6 particles in this system.

Your `discovered_law` function must:
1. Take the initial conditions of all 6 particles
2. Simulate their motion forward in time
3. Return their positions at time `t = duration`

**Important:** The particles may not all be identical. Some may generate stronger or weaker fields, or respond to forces differently. Part of your task is to determine whether the particles differ and, if so, how.

## Experimental Apparatus

You observe a system of **6 particles** (labelled 0-5) in a 2D universe. All particles interact through an unknown field.

You have full control over initial conditions:
- `positions`: list of 6 `[x, y]` coordinates relative to domain center (keep within [-8, 8])
- `velocities`: list of 6 `[vx, vy]` initial velocities (typical range: [-2, 2])
- `measurement_times`: times at which to record positions and velocities (up to 10 values)

**Important:** Use `duration >= 5.0` and at least 10 measurement times to observe the force law clearly.

**Input Format:**
<run_experiment>
[
  {"positions": [[0,0],[3,0],[-3,0],[0,3],[0,-3],[4,4]], "velocities": [[0,0],[0,0],[0,0],[0,0],[0,0],[0,0]], "measurement_times": [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]}
]
</run_experiment>

**Output Format:**
<experiment_output>
[
  {
    "measurement_times": [0.5, 1.0, ...],
    "positions":  [[[x0,y0],[x1,y1],...,[x5,y5]], ...],  // shape (T, 6, 2), relative to center
    "velocities": [[[vx0,vy0],[vx1,vy1],...], ...]       // shape (T, 6, 2)
  }
]
</experiment_output>

Particle ordering is fixed across all experiments: particle 0 is always particle 0.
Reported particle positions may contain **Gaussian observation noise of unknown scale**. Reported velocities are clean. Design your experiments and fit your law accordingly (e.g. by using longer durations or repeated measurements to average over noise).

## Strategy

- **Isolate pairs:** Place two particles close together and the rest far away (e.g., at corners of the domain). This lets you measure the force between specific pairs without interference.
- **Compare particles:** Do particle 0 and particle 3 generate the same field? Place a "test" particle at the same distance from each and compare the resulting acceleration.
- **Vary distance:** For a given pair, try several separations to determine how force scales with distance.
- **Look for species:** If you find that some particles behave differently from others, figure out which particles belong to each group and quantify the difference.
- **Check symmetry:** Does particle A attract B with the same force that B attracts A? This tests whether the asymmetry is in field generation (source strength) vs. response (inertia/force coupling).

## Final Submission

Once confident, submit a single Python function in `<final_law>` tags.

**Requirements:**
1. Function name: `discovered_law`
2. Signature: `def discovered_law(positions, velocities, duration)`
   - `positions`: list of 6 `[x, y]` coords relative to center at `t=0`
   - `velocities`: list of 6 `[vx, vy]` at `t=0`
   - `duration`: float, time to simulate
3. Return: a list or array of 6 `[x, y]` final positions at `t=duration`
4. Define all constants as local variables inside the function body
5. Import any required libraries inside the function body

**Implementation note — time integration:**
The evaluator calls your `discovered_law` once per (case × measurement time) during scoring. **Do NOT write Python `for` loops with small fixed `dt`** (e.g. `for _ in range(int(duration/dt))` with `dt=1e-3`) — they dominate runtime and can make a single law evaluation take seconds instead of milliseconds. **Use `scipy.integrate.solve_ivp`** with adaptive RK45 instead; it picks step sizes automatically and reaches comparable accuracy with 10–100× fewer steps.

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
    Include the key force law and any per-particle constants.
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
Alongside your `<final_law>`, you MUST also include a separate `<explanation>` tag containing a 2–3 sentence prose description of the physical system you discovered. Describe the underlying field equation, how particles couple to it, and any species or coupling differences you identified — in plain English, not code. This is graded independently from the trajectory accuracy.

Example final-round response:
<final_law>
def discovered_law(positions, velocities, duration):
    ...
</final_law>
<explanation>
The six particles interact through a static Laplacian field with force minus the field gradient. They split into two hidden species: particles 0–2 source the field with coupling 1.0 while particles 3–5 source it with coupling 3.0, generating a field three times stronger.
</explanation>
