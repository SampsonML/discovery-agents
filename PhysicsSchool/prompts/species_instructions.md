### Topology: Small population, full-state controllable, no probes

A 2D universe containing 6 particles (labelled 0–5). All particles interact through an unknown field. You have full control over **every** particle's initial conditions — there is no separate "probe" subset. Scoring is on all 6 final positions.

Particle ordering is fixed across experiments: particle 0 is always particle 0.

### Control Parameters

| Field | Meaning | Typical range |
|---|---|---|
| `positions` | list of 6 `[x, y]` initial coordinates relative to centre | `[-8, 8]²` |
| `velocities` | list of 6 `[vx, vy]` initial velocities | `[-2, 2]²` |
| `measurement_times` | times at which to record positions and velocities (≤ 10 values) | spans the run |
| `duration` | length of the experiment (interactive only) | ≥ 5.0 |

### Input Format (interactive mode only)

```
<run_experiment>
[
  {"positions":  [[0,0],[3,0],[-3,0],[0,3],[0,-3],[4,4]],
   "velocities": [[0,0],[0,0],[0,0],[0,0],[0,0],[0,0]],
   "measurement_times": [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]}
]
</run_experiment>
```

You may submit several experiments per round in the JSON array. No comments inside the JSON.

### Output Format

```
<experiment_output>
[
  {
    "measurement_times": [t0, t1, ...],
    "positions":  [[[x0,y0], ..., [x5,y5]], ...],     // shape (T, 6, 2), relative to centre
    "velocities": [[[vx0,vy0], ..., [vx5,vy5]], ...]  // shape (T, 6, 2)
  }
]
```

### `discovered_law` Signature

```python
def discovered_law(positions, velocities, duration):
    # positions:  list of 6 [x, y] — initial positions of all particles
    # velocities: list of 6 [vx, vy]
    # duration:   float — simulate from t = 0 to t = duration
    # return:     list of 6 [x, y] final positions
    return final_positions
```

If the world has hidden per-particle structure (e.g. different source couplings or species membership), encode the discovered values as hard-coded constants inside the function body — your law must capture them to score well.
