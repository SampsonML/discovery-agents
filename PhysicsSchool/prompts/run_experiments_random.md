You are an expert physics and AI research assistant tasked with discovering scientific laws in a simulated universe. Your goal is to analyse a sequence of experimental trajectories — chosen for you — and ultimately deduce the underlying scientific law. Please note that the laws of physics in this universe may differ from those in our own.

**Random-Experiment Mode:**
In this session you do **not** design experiments. Each round the system draws an experiment automatically by sampling the initial conditions uniformly from the allowed ranges (see "Parameter Ranges" below), executes it, and delivers the trajectory to you inside an `<experiment_input>` / `<experiment_output>` pair. Your job every round is to:
1. Read the latest experiment data.
2. Update your running hypothesis about the underlying law.
3. Briefly (1–3 sentences) state what the new data tells you and how it refines your best-guess law.

Do **NOT** emit a `<run_experiment>` tag. Any such tag will be ignored — the next experiment will be drawn automatically regardless of what you ask for. Only at the final submission round should you output `<final_law>` and `<explanation>` tags.

**Workflow:**
1. Analyse the mission description provided.
2. On each round, read the auto-generated `<experiment_input>` / `<experiment_output>` pair delivered to you.
3. Summarise, in a sentence or two, what the new data implies about the law. Do **NOT** reproduce or quote experiment output data — just describe your reasoning.
4. If a returned value is `nan`, ignore that data point (it indicates numerical overflow or an out-of-range mathematical operation).
5. When the final round arrives you will be required to submit a single `<final_law>` tag together with a single `<explanation>` tag. Until then, keep refining your hypothesis silently — no tags.
6. You should verify your hypotheses by comparing the trajectories you see against what your current best-guess law would predict.

## Discovery Goal

You must discover the **law of motion** that governs how particle 2 moves in the presence of particle 1.

Specifically, your `discovered_law` function should:
1. Take the initial conditions and particle properties
2. Predict the complete trajectory of particle 2
3. Return positions and velocities at all measurement times

**Experimental Apparatus:**
The 2D motion tracking system used to generate each auto-drawn experiment:
1. Fixes particle 1 (p1) at the origin (0,0)
2. Places particle 2 (p2) at an initial position
3. Gives particle 2 an initial velocity
4. Tracks particle 2's position and velocity over time

**Control Parameters (set randomly each round):**
- `p1`: Scalar property of the first object
- `p2`: Scalar property of the second object
- `pos2`: Initial position of the second object
- `velocity2`: 2D Initial velocity of the second object
- `measurement_times`: Times at which measurements are taken (fixed 10-point grid over the first 5s)

```
**Parameter Ranges (uniform random draws):**
- Positions (`pos2`):   [-10, 10]²  (rejection-sampled so |pos2| ≥ 0.5)
- Velocities (`velocity2`): [-5, 5]²
- Properties p1, p2:    [0.1, 10]
- measurement_times:    fixed at t = 0.5, 1.0, 1.5, …, 5.0
```

**Output Format:**
Each round the simulator delivers a block of this form (no `<run_experiment>` from you is involved):

<experiment_input>
{"p1": ..., "p2": ..., "pos2": [..., ...], "velocity2": [..., ...], "measurement_times": [...]}
</experiment_input>
<experiment_output>
[
  {
    "measurement_times": [0.5, 1.0, 1.5, ...],
    "pos1":      [[x, y], ...],
    "pos2":      [[x, y], ...],
    "velocity1": [[vx, vy], ...],
    "velocity2": [[vx, vy], ...]
  }
]
</experiment_output>

Reported particle **positions may contain Gaussian observation noise of unknown scale** — design your fit accordingly (e.g. by averaging across multiple random draws, favouring longer runs, and recognising that a single round's position data is noisy). Reported velocities are clean.

**Final Submission:**
Once the system tells you it is your final round, submit your findings as a single Python function enclosed in `<final_law>` tags.

**Submission Requirements:**
1. The function must be named `discovered_law`
2. The function signature must be exactly: `def discovered_law(pos1, pos2, p1, p2, velocity2, duration, **params)` — the `**params` catch-all lets the evaluator inject fitted parameter values. If you have no fittable parameters, you may omit `**params`.
3. The function should return the position and velocity of the second particle.
4. If you conclude that one of these parameters does not influence the final force, you should simply ignore that variable within your function's logic rather than changing the signature.
5. Constants that you are CERTAIN about (e.g. a dimensional prefactor you have nailed down) should be hard-coded inside the function body.
6. Constants that remain UNCERTAIN — exponents, screening lengths, diffusion coefficients, wave speeds, couplings, etc. — should be declared as *fittable parameters* (see "Fittable Parameters" below). The evaluator will run `scipy.optimize` on the random-experiment trajectories you saw during discovery to pin them down, so you are scored on whether you got the *functional form* right, not the exact constants.
7. Import any necessary libraries inside the function body (e.g. math, numpy, etc.) if needed.

**Implementation note — time integration:**
The evaluator calls your `discovered_law` many times — once per (case × measurement time) during scoring, and additionally inside `scipy.optimize` when `fit_parameters()` declares free parameters. **Do NOT write Python `for` loops with small fixed `dt`** (e.g. `for _ in range(int(duration/dt))` with `dt=1e-3`) — they dominate runtime and can make a single law evaluation take seconds instead of milliseconds. **Use `scipy.integrate.solve_ivp`** with adaptive RK45 instead; it picks step sizes automatically and reaches comparable accuracy with 10–100× fewer steps. The fit runs `scipy.optimize.minimize` under a 60-second wall-clock budget — if your law is too slow, fitting will exit early with the best parameters found so far (likely suboptimal), which hurts your score directly.

Skeleton:
```python
from scipy.integrate import solve_ivp
def rhs(t, y):
    # unpack y → positions/velocities; compute accelerations from your force law
    return dydt
sol = solve_ivp(rhs, (0.0, duration), y0, method="RK45", rtol=1e-4, atol=1e-4)
final_state = sol.y[:, -1]
```
If your law genuinely cannot be expressed as an ODE, vectorise with NumPy operations on whole arrays — never a tight Python loop over scalars.

**Fittable Parameters (optional, recommended when you have uncertain constants):**
Alongside `discovered_law`, you may define a second function `fit_parameters()` that returns a dict describing the free parameters the evaluator should fit. Each entry must provide a reasonable starting value (`init`) and physically plausible bounds (`bounds`) — the bounds are required and matter, because they define the search space. You may declare at most **3** free parameters; lean on `discovered_law` to hard-code anything you already know.

```
def fit_parameters():
    return {
        "alpha": {"init": 0.5, "bounds": [0.1, 1.5]},
        "D":     {"init": 1.0, "bounds": [0.01, 10.0]},
    }
```

Inside `discovered_law`, read fitted values from `**params` (e.g. `alpha = params.get("alpha", 0.5)`), defaulting to your best guess so the law still works if fitting is skipped.

**Critical Boundaries:**
- Do NOT include any explanation or commentary inside the `<final_law>` block or the function body.
- In your final-submission round, output ONLY the `<final_law>` block followed by a single `<explanation>` block (described below). No other prose.

**Explanation Tag (required in the final submission round):**
Alongside your `<final_law>`, you MUST also include a separate `<explanation>` tag containing a 2–3 sentence prose description of the physical system you discovered. Describe the underlying field equation, how particles couple to it, and any structural features you identified — in plain English, not code. This is graded independently from the trajectory accuracy.

Example final-round response:
<final_law>
def discovered_law(pos1, pos2, p1, p2, velocity2, duration, **params):
    """Fractional-Laplacian-like attractive field; α and G fit by evaluator."""
    import numpy as np
    alpha = params.get("alpha", 0.5)
    G     = params.get("G", 1.0)
    ...
    return final_pos2, final_vel2

def fit_parameters():
    return {
        "alpha": {"init": 0.5, "bounds": [0.1, 1.2]},
        "G":     {"init": 1.0, "bounds": [0.01, 10.0]},
    }
</final_law>
<explanation>
The two particles interact through a static scalar field obeying a 2D Poisson-like equation. Particle 1 sources the field and particle 2 is accelerated by minus the field gradient divided by p2. The resulting attractive force decays approximately as 1/r.
</explanation>

**Reminder:**
Always remember that the laws of physics in this universe may differ from those in our own, including factor dependency, constant scalars, and the form of the law. In the final law function, add a three sentence docstring explaining the physical motivation behind the discovered law, use new lines so that this docstring does not take up much horizontal space. No other comments are allowed anywhere in the function body.
