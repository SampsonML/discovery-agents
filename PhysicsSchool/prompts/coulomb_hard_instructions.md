### Topology: 10-particle signed-charge system

A 2D universe containing 10 mobile particles. Each particle carries a scalar `charge` that you set; charges may be **positive or negative**. All 10 particles interact through a central pairwise force whose magnitude and sign depend on the product of the two charges.

You have full control over **every** particle's initial position, velocity, and charge — there is no separate "probe" subset. Scoring is on all 10 final positions.

### Control Parameters

| Field | Meaning | Typical range |
|---|---|---|
| `positions` | list of 10 `[x, y]` initial coordinates relative to centre | `[-10, 10]²` |
| `velocities` | list of 10 `[vx, vy]` initial velocities | `[-3, 3]²` |
| `charges` | list of 10 signed scalar charges | `[-3, 3]`, can be 0 |
| `measurement_times` | times at which to record positions and velocities (≤ 10 values) | spans the run |
| `duration` | length of the experiment (interactive only) | ≥ 5.0 |

### Input Format (interactive mode only)

```
<run_experiment>
[
  {"charges":    [1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0],
   "positions":  [[3,0],[-3,0],[0,3],[0,-3],[5,5],[-5,5],[5,-5],[-5,-5],[2,0],[0,2]],
   "velocities": [[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0]],
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
    "positions":  [[[x0,y0], ..., [x9,y9]], ...],     // shape (T, 10, 2), relative to centre
    "velocities": [[[vx0,vy0], ..., [vx9,vy9]], ...], // shape (T, 10, 2)
    "charges":    [q0, q1, ..., q9]                   // length-10 charge array (echoes input)
  }
]
```

### `discovered_law` Signature

```python
def discovered_law(positions, velocities, charges, duration):
    # positions:  list of 10 [x, y] — initial positions of all particles
    # velocities: list of 10 [vx, vy]
    # charges:    list of 10 signed scalar charges
    # duration:   float — simulate from t = 0 to t = duration
    # return:     list of 10 [x, y] final positions
    return final_positions
```

Use the sign and magnitude of each charge to determine the per-pair force. Probe systematically with mixed-sign and zero-charge configurations to confirm the sign convention (which combinations attract vs. repel) and the dependence on the charge product.
