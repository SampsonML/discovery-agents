"""
Named world configurations and mission descriptions.

Each world entry defines the FieldSampler operator config that is
*hidden* from the agent, plus a mission string that is shown to it.
"""

from scienceagent.executor import (
    SimulationExecutor,
    CircleExecutor,
    SpeciesExecutor,
    ThreeSpeciesExecutor,
    DarkMatterExecutor,
    NBodySimulationExecutor,
    NBodyCircleExecutor,
    NBodySpeciesExecutor,
    NBodyThreeSpeciesExecutor,
    NBodyDarkMatterExecutor,
    NBodyEtherExecutor,
    NBodyHubbleExecutor,
    NBodyOscillatorExecutor,
    NBodyExtraDimensionsExecutor,
)

_GENERAL_FORM = (
    r"$\dfrac{\partial^n \varphi}{\partial t^n} = L[\varphi] + S(\mathrm{particles})$"
)

_TRUE_LAW_GRAVITY = (
    _GENERAL_FORM + "\n\n"
    r"$n = 0$" + "\n"
    r"$L[\varphi] = \nabla^2\varphi$" + "\n"
    r"$\mathbf{F} = -\nabla\varphi\,/\,p_2$"
)

_TRUE_LAW_YUKAWA = (
    _GENERAL_FORM + "\n\n"
    r"$n = 0$" + "\n"
    r"$L[\varphi] = \nabla^2\varphi - \lambda^{-2}\varphi,\quad \lambda = 2$" + "\n"
    r"$\mathbf{F} = -\nabla\varphi\,/\,p_2$"
)

_TRUE_LAW_FRACTIONAL = (
    _GENERAL_FORM + "\n\n"
    r"$n = 0$" + "\n"
    r"$L[\varphi] = -(-\nabla^2)^{\alpha}\varphi,\quad \alpha = 0.75$" + "\n"
    r"$\mathbf{F} = -\nabla\varphi\,/\,p_2$"
)

_TRUE_LAW_DIFFUSION = (
    _GENERAL_FORM + "\n\n"
    r"$n = 1$" + "\n"
    r"$L[\varphi] = D\,\nabla^2\varphi,\quad D = 0.5$" + "\n"
    r"$\mathbf{F} = -\nabla\varphi\,/\,p_2$"
)

_TRUE_LAW_SPECIES = (
    _GENERAL_FORM + "\n\n"
    r"$n = 0$" + "\n"
    r"$L[\varphi] = \nabla^2\varphi$" + "\n"
    r"$\mathbf{F}_i = -\nabla\varphi_i$" + "\n"
    r"$\text{source\_coupling}_i = 1.0\;\text{(particles 0,1,2)},\; 3.0\;\text{(particles 3,4,5)}$"
)

_TRUE_LAW_CIRCLE = (
    _GENERAL_FORM + "\n\n"
    r"$n = 0$" + "\n"
    r"$L[\varphi] = -(-\nabla^2)^{\alpha}\varphi,\quad \alpha = 0.75$" + "\n"
    r"$\mathbf{F}_i = -\nabla\varphi_i$" + "\n"
    r"$\text{11 particles: 1 center} + \text{10 ring}$"
)

_TRUE_LAW_THREE_SPECIES = (
    _GENERAL_FORM + "\n\n"
    r"$n = 0$" + "\n"
    r"$L[\varphi] = \nabla^2\varphi$" + "\n"
    r"$\mathbf{F}_i = -\nabla\varphi_i$" + "\n"
    r"$\text{source\_coupling}_i = 1.0\;\text{(particles 0–9)},\; "
    r"3.0\;\text{(particles 10–19)},\; -2.0\;\text{(particles 20–29)}$" + "\n"
    r"$\text{Probes (30–34): source\_coupling} = 0$"
)

_TRUE_LAW_DARK_MATTER = (
    _GENERAL_FORM + "\n\n"
    r"$n = 0$" + "\n"
    r"$L[\varphi] = \nabla^2\varphi$" + "\n"
    r"$\mathbf{F}_i = -\nabla\varphi_i$" + "\n"
    r"$\text{Visible (0–19): source\_coupling} = 1.0$" + "\n"
    r"$\text{Dark matter (hidden, 10 particles): source\_coupling} = 5.0$" + "\n"
    r"$\text{Probes (20–24 in agent view): source\_coupling} = 0$"
)

_TRUE_LAW_ETHER = (
    _GENERAL_FORM + "\n\n"
    r"$n = 0$" + "\n"
    r"$L[\varphi] = \nabla^2\varphi$  (sourced only by the central anchor, $Q=50$)"
    + "\n"
    r"$\mathbf{F}_i = -\nabla\varphi_i + \alpha\,m_i\,\hat{\mathbf{y}},\quad \alpha = 0.05$"
    + "\n"
    r"$\text{20 ring orbiters with masses}\in\{1,2,4\};\;\text{5 probes (test particles)}$"
)


_TRUE_LAW_HUBBLE = (
    _GENERAL_FORM + "\n\n"
    r"$n = 0$" + "\n"
    r"$L[\varphi] = \nabla^2\varphi$  (sourced only by the central anchor, $Q=50$)"
    + "\n"
    r"$\mathbf{a}_i = -\nabla\varphi_i / m_i + H\,\mathbf{r}_i,\quad H = 0.05$" + "\n"
    r"$r_{\rm crit} = \sqrt{Q/(2\pi H)} \approx 12.6$" + "\n"
    r"$\text{20 ring orbiters with masses}\in\{1,2,4\};\;\text{5 probes (test particles)}$"
)


_TRUE_LAW_WAVE = (
    _GENERAL_FORM + "\n\n"
    r"$n = 2$" + "\n"
    r"$L[\varphi] = c^2\nabla^2\varphi,\quad c = 1$" + "\n"
    r"$\mathbf{F} = -\nabla\varphi\,/\,p_2$"
)


_TRUE_LAW_OSCILLATOR = (
    _GENERAL_FORM + "\n\n"
    r"$n = 0$" + "\n"
    r"$L[\varphi] = G(t)\,\nabla^2\varphi,\quad G(t) = G_0\cos(\omega t + \phi)$"
    + "\n"
    r"$G_0 = 5,\;\omega = \pi/2\;(T = 4),\;\phi = 0$" + "\n"
    r"$\mathbf{F}_2 = -\nabla\varphi\,/\,p_2$"
)


_TRUE_LAW_EXTRA_DIMENSIONS = (
    _GENERAL_FORM + "\n\n"
    r"$n = 0$" + "\n"
    r"$\mathbf{F}_2(r) = \dfrac{G\,L\,p_1}{4\pi}\sum_{n\in\mathbb{Z}}"
    r"\dfrac{r}{(r^2+(nL)^2)^{3/2}}\,(-\hat{\mathbf{r}}),"
    r"\quad L = 2\pi R_c$" + "\n"
    r"$G = 1,\;R_c = 0.5$" + "\n"
    r"$r \gg R_c:\; F \to p_1/(2\pi r)$ (2D Poisson — looks like gravity)"
    + "\n"
    r"$r \ll R_c:\; F \to G L p_1/(4\pi r^2) = R_c p_1/(2 r^2)$ (3D Newton)"
)


# -----------------------------------------------------------------------------
# Per-world scoring rubrics for the explanation judge. Each rubric is
# calibrated to what the agent can plausibly recover given the world's
# experimental capabilities: the top band requires the correct physical
# structure plus numeric parameters within a reasonable tolerance, not
# pinpoint accuracy.

_RUBRIC_GRAVITY = """\
10 — Identifies a static scalar field with a Laplacian/Poisson operator
     (∇²φ = source); states the resulting attractive force falls off as 1/r
     (not 1/r²) in 2D, or equivalently that the Green's function is
     logarithmic; correctly distinguishes p1 as source coupling and p2 as
     particle inertia.
 7–9 — Identifies a static attractive field with roughly correct decay, but
     asserts the 3D-Newtonian 1/r² falloff, muddles the p1/p2 roles, or
     describes the operator only qualitatively without naming it.
 4–6 — Recognises static attraction but commits to a clearly wrong decay law
     (exponential, inverse-cube), or fails to distinguish source coupling
     from inertia.
 1–3 — Wrong temporal character (time-evolving or wave-like) or a
     qualitatively wrong force law (repulsive, constant).
   0 — Empty, irrelevant, or no physical content."""

_RUBRIC_YUKAWA = """\
10 — Identifies a static screened / Helmholtz operator
     (∇²φ − φ/λ² = source, or equivalent Yukawa form); names or describes
     exponential suppression at long range with screening length within
     roughly 1–4 (ground truth λ = 2); correctly identifies p1 as source
     coupling and p2 as inertia.
 7–9 — Identifies screened / Yukawa-like behaviour qualitatively (short-range
     attraction, long-range suppression) but misses a quantitative detail
     (λ badly off or not estimated) or muddles the p1/p2 roles.
 4–6 — Recognises a static attractive field but treats it as a plain
     Laplacian or generic power-law and misses the screening structure, or
     asserts screening without any characterisation of its scale.
 1–3 — Wrong operator family (time-evolving, wave) or no distance-dependent
     suppression despite the clear experimental signature.
   0 — Empty or irrelevant."""

_RUBRIC_FRACTIONAL = """\
10 — Identifies a static non-local / fractional-Laplacian operator of the
     form −(−∇²)^α with α in a plausible range (roughly 0.3–0.8; ground
     truth α = 0.5); describes the force as decaying more slowly with
     distance than the standard 2D Laplacian case (enhanced long-range
     interaction); correctly names p1 as source and p2 as inertia.
 7–9 — Identifies an anomalous / non-local spatial operator with unusual
     (slower-than-standard) long-range behaviour, but misses α, places α
     outside a plausible range, or articulates the direction of the anomaly
     vaguely or incorrectly.
 4–6 — Recognises static attraction but assumes a standard Laplacian or
     ordinary power-law, missing the non-local / fractional character.
 1–3 — Wrong operator family (time-evolving or repulsive) or no
     characterisation of the spatial operator at all.
   0 — Empty or irrelevant."""

_RUBRIC_DIFFUSION = """\
10 — Identifies a first-order-in-time diffusion equation with Laplacian
     spatial operator (∂φ/∂t = D∇²φ); estimates D in a plausible range
     (roughly 0.1–2; ground truth D = 0.5); articulates that the force
     depends on the time-history of the source (the field builds up and
     spreads outward) rather than on the instantaneous configuration.
 7–9 — Identifies a time-evolving field with first-order time derivative and
     spatial smoothing, but misses D or fails to articulate the
     history-dependence.
 4–6 — Recognises time-dependence but identifies the wrong temporal order
     (treats as wave-like / second-order) or misses the spatial operator.
 1–3 — Treats the field as static, or proposes a qualitatively wrong
     dynamical picture.
   0 — Empty or irrelevant."""

_RUBRIC_WAVE = """\
10 — Identifies a second-order-in-time wave equation with Laplacian spatial
     operator (∂²φ/∂t² = c²∇²φ); estimates the propagation speed c in a
     plausible range (roughly 0.5–2; ground truth c = 1); articulates
     retarded-force / finite-propagation effects producing oscillatory,
     history-dependent dynamics.
 7–9 — Identifies wave-like, oscillatory, or retarded behaviour with finite
     propagation speed, but misses c or fails to clearly articulate
     second-order-in-time structure.
 4–6 — Recognises time-evolving dynamics but identifies the wrong temporal
     order (diffusive / first-order) or misses the oscillatory /
     retardation signature.
 1–3 — Treats the field as static, or proposes a qualitatively wrong
     dynamical picture.
   0 — Empty or irrelevant."""

_RUBRIC_EXTRA_DIMENSIONS = """\
10 — Identifies that the visible 2D world has *one* extra spatial dimension
     compactified at radius R; states the force is the Kaluza-Klein image
     sum over the compact dimension, ∝ Σ_n r/(r²+(nL)²)^(3/2) with
     L = 2π R; correctly identifies the two asymptotic regimes
     (r ≫ R: 2D Poisson F ∝ p1/(2π r); r ≲ R: 3D Newtonian
     F ∝ p1/(4π r²) becoming exponentially divergent at r → 0); estimates
     R within roughly a factor of 2 of 0.5 (i.e. R ∈ [0.2, 1.5]) and gets
     the long-range coupling within ~30 % of G = 1; correctly assigns p1
     as source coupling and p2 as inertia.
 7–9 — Recognises a *crossover* in the force law from 1/r at long range to
     a steeper power (≈ 1/r²) at short range, and offers a quantitative
     compactification scale within roughly an order of magnitude of 0.5,
     but misses the geometric image-sum structure or doesn't connect the
     short-range regime to a higher-dimensional Newtonian law.
 4–6 — Notices anomalous deviation from a pure 1/r force at short range
     (or anomalous deviation from pure 1/r² at long range) but proposes a
     wrong functional form (e.g. Yukawa screening, fractional Laplacian,
     pure power law of intermediate exponent) without identifying a true
     dimensional crossover.
 1–3 — Fits the data with a single canonical law (plain 2D gravity, plain
     3D gravity, fractional, Yukawa, …) without acknowledging that the
     same law fails at the opposite end of the distance range.
   0 — Empty or irrelevant."""


_RUBRIC_OSCILLATOR = """\
10 — Identifies the standard 2D Laplacian spatial form (∇²φ = source, force
     ∝ 1/r in 2D) but with a *time-modulated overall coupling* G(t) — i.e.
     the strength (and SIGN) of the interaction depends on absolute time,
     not just on geometry; explicitly notes that the same configuration is
     attractive at some phases and *repulsive* at others; identifies a
     sinusoidal modulation G(t) = G₀·cos(ω t + φ) (or sin equivalent) and
     estimates the period within roughly a factor of 2 of T = 4; correctly
     keeps p1 as source coupling and p2 as inertia.
 7–9 — Identifies time-varying coupling on top of a 2D-gravity-like spatial
     form, but misses the period quantitatively, omits the sign-flipping
     behaviour, or muddles the p1/p2 roles.
 4–6 — Recognises that the dynamics are time-dependent but treats the
     variation as transient / decaying or otherwise non-periodic, OR
     identifies the 1/r spatial form but treats the coupling as constant
     (missing the entire time modulation).
 1–3 — Models the system as static gravity, or proposes a qualitatively
     wrong mechanism (drag, noise, history-dependent diffusion, …) that
     does not include an explicit dependence on absolute time.
   0 — Empty or irrelevant."""

_RUBRIC_SPECIES = """\
10 — Identifies a static Laplacian operator with force −∇φ; identifies
     exactly two hidden species among the 6 particles; correctly partitions
     particles 0–2 vs. 3–5 (or equivalently marks the correct three "strong"
     sources); estimates the coupling ratio within a factor of ~2 of the
     true 3:1 (second species stronger).
 7–9 — Identifies Laplacian + two species, but gets the partition wrong,
     reverses which group is stronger, or puts the ratio far outside a
     1.5–5× window.
 4–6 — Identifies a static Laplacian-like field but treats all six particles
     as identical, missing the species structure entirely; OR proposes the
     right ratio but with a substantially wrong particle assignment.
 1–3 — Wrong operator, or fabricates spurious structure (e.g. claims 3+
     species when there are clearly two).
   0 — Empty or irrelevant."""

_RUBRIC_CIRCLE = """\
10 — Identifies a static non-local / fractional-Laplacian operator
     −(−∇²)^α with α in a plausible range (roughly 0.5–1.0; ground truth
     α = 0.75); states that the coupling is uniform across all 11 particles;
     describes force behaviour intermediate between the logarithmic 2D
     Laplacian case and pure long-range attraction.
 7–9 — Identifies a non-local / anomalous-decay operator, but estimates α
     outside a plausible range, omits the uniform-coupling claim, or
     muddles the direction of the anomaly.
 4–6 — Recognises attractive static interactions but assumes a standard
     Laplacian (or Newtonian 1/r) and misses the fractional character.
 1–3 — Wrong operator family (time-evolving, repulsive) or no spatial
     operator named.
   0 — Empty or irrelevant."""

_RUBRIC_THREE_SPECIES = """\
10 — Identifies a static Laplacian with force −∇φ; identifies three distinct
     source species among particles 0–29 plus 5 neutral probes (30–34);
     correctly identifies one species as repulsive (negative coupling);
     estimates coupling ratios approximately matching +1 : +3 : −2 (each
     within roughly a factor of 2, with signs correct); identifies the
     probes as having approximately zero source coupling.
 7–9 — Identifies the Laplacian and the three-species + probe structure
     with correct signs, but coupling magnitudes are off, probes are lumped
     with one of the species, or the particle-index partitioning is
     slightly wrong.
 4–6 — Identifies a Laplacian field and some species structure, but misses
     the repulsive species (all three treated as attractive), finds only
     two species, or fails to distinguish the neutral probes.
 1–3 — Treats all particles as identical, or posits a wrong operator family.
   0 — Empty or irrelevant."""

_RUBRIC_ETHER = """\
10 — Identifies a static 2D Laplacian central force sourced by the single
     anchor particle (index 0); identifies the 20 orbiters and 5 probes as
     test particles responding to that field; identifies a uniform northward
     drift acceleration on every particle (or, equivalently, a body-force
     proportional to mass producing a mass-independent acceleration);
     estimates the drift acceleration α within roughly a factor of 2 of
     0.05; recognises that orbiter masses ∈ {1, 2, 4} but that with the
     mass-proportional ether force the drift looks identical for all
     particles in absolute coordinates.
 7–9 — Identifies the central Laplacian + uniform northward drift, but
     misses the F ∝ m / a = const equivalence, gets α badly off, or fails
     to identify which particle is the anchor.
 4–6 — Identifies the central attraction OR the drift but not both, or
     mis-attributes the drift to a directional Laplacian / wind / repulsion
     between specific particles.
 1–3 — Wrong operator family, no drift identified, or no central anchor
     identified despite the obvious common parabolic envelope.
   0 — Empty or irrelevant."""


_RUBRIC_HUBBLE = """\
10 — Identifies a static 2D Laplacian central force sourced by the single
     anchor particle (index 0); identifies the 20 orbiters and 5 probes as
     test particles; identifies a *position-dependent* outward body-force
     that grows linearly with distance from the anchor (a = H · r,
     mass-independent), recognises a critical radius beyond which probes
     accelerate outward and orbits unbind, and estimates H within roughly
     a factor of 2 of 0.05.
 7–9 — Identifies the central Laplacian and an outward repulsive effect at
     large radii, but misses the linear-in-r structure (e.g. assumes a
     constant outward force), gets H badly off, or attributes the outward
     push to a particular particle rather than to space itself.
 4–6 — Identifies the central attraction but treats the outward effect
     qualitatively only — e.g. notes "orbits unbind at large r" without
     distinguishing a body-force from a missing-mass / dark-matter
     hypothesis.
 1–3 — Wrong operator family, or interprets the outward push as random
     noise, drag, or some pairwise repulsion between probes/orbiters
     despite the obvious radial-from-anchor pattern.
   0 — Empty or irrelevant."""


_RUBRIC_DARK_MATTER = """\
10 — Identifies a static Laplacian with force −∇φ; concludes that hidden /
     unseen sources exist based on visible particles accelerating toward
     apparently empty regions; estimates the hidden population roughly
     correctly (count in 5–15; ground truth 10) with coupling stronger than
     the visible population (roughly 3–8×; ground truth 5×); identifies the
     probes (agent indices 20–24) as neutral (non-sourcing but responsive).
 7–9 — Identifies the Laplacian and the existence of hidden sources, but
     gets their count or coupling strength badly wrong, or fails to
     characterise the probes as neutral.
 4–6 — Identifies a Laplacian field but attributes the visible particles'
     anomalous behaviour to noise, the probes, or measurement error rather
     than to hidden sources.
 1–3 — Wrong operator family, or a fundamentally wrong mechanism (e.g.
     dynamical instability with no hidden matter).
   0 — Empty or irrelevant."""


WORLDS = {
    "gravity": {
        "description": "Classic inverse-square-law-like attraction mediated by a 2D Laplacian field.",
        "mission": (
            "Two particles interact through an unknown field in a 2D universe. "
            "The field is generated by particle 1 and exerts a force on particle 2. "
            "Discover the law of motion governing particle 2."
        ),
        "executor_kwargs": {
            "operators": [{"type": "laplacian", "params": {"strength": 1.0}}],
            "temporal_order": 0,
        },
        "true_law": _TRUE_LAW_GRAVITY,
        "true_law_title": "True Laplacian",
        "optimal_explanation": (
            "Two particles interact through a static scalar field obeying the 2D Poisson "
            "equation, ∇²φ = source, where particle 1 sources the field with strength p1 "
            "and particle 2 is accelerated by -∇φ divided by its inertia p2. Because the "
            "Laplacian Green's function is logarithmic in 2D, the resulting attractive "
            "force falls off as 1/r rather than 1/r²."
        ),
        "explanation_rubric": _RUBRIC_GRAVITY,
    },
    "yukawa": {
        "description": "Screened (Yukawa) potential — exponentially suppressed at long range.",
        "mission": (
            "Two particles interact in a universe where forces may be screened at long distances. "
            "Discover the force law, including any distance-dependent suppression."
        ),
        "executor_kwargs": {
            "operators": [
                {
                    "type": "screening",
                    "params": {"strength": 1.0, "screening_length": 2.0},
                }
            ],
            "temporal_order": 0,
        },
        "true_law": _TRUE_LAW_YUKAWA,
        "true_law_title": "True Yukawa",
        "optimal_explanation": (
            "Two particles interact through a screened scalar field obeying a 2D Helmholtz "
            "equation, ∇²φ - φ/λ² = source, with screening length λ = 2. Particle 1 sources "
            "the field with strength p1 and particle 2 accelerates under -∇φ/p2. The force "
            "is attractive at short range but suppressed exponentially beyond ~2 length units."
        ),
        "explanation_rubric": _RUBRIC_YUKAWA,
    },
    "fractional": {
        "description": "Fractional Laplacian — anomalous power-law force.",
        "mission": (
            "Two particles interact through a field obeying an unusual spatial operator. "
            "The force may scale with distance in an unexpected way. "
            "Discover the law of motion."
        ),
        "executor_kwargs": {
            "operators": [
                {
                    "type": "fractional_laplacian",
                    "params": {"strength": 1.0, "alpha": 0.5},
                }
            ],
            "temporal_order": 0,
        },
        "true_law": _TRUE_LAW_FRACTIONAL,
        "true_law_title": "True Fractional Laplacian",
        "optimal_explanation": (
            "Two particles interact through a static field governed by a fractional Laplacian "
            "operator, -(-∇²)^α with α = 0.5, sourced by particle 1 with coupling p1. The "
            "non-local operator produces a force on particle 2 that decays more slowly with "
            "distance than the standard 2D Laplacian case — long-range interactions are "
            "enhanced. Particle 2 responds with inertia p2 and the field is time-independent."
        ),
        "explanation_rubric": _RUBRIC_FRACTIONAL,
    },
    "diffusion": {
        "description": "Diffusive (n=1) field — force from a time-dependent spreading field.",
        "mission": (
            "The field in this universe evolves over time by diffusion before exerting forces. "
            "Discover how particle 2 moves in the presence of particle 1."
        ),
        "executor_kwargs": {
            "operators": [{"type": "laplacian", "params": {"strength": 0.5}}],
            "temporal_order": 1,
        },
        "true_law": _TRUE_LAW_DIFFUSION,
        "true_law_title": "True Diffusion",
        "optimal_explanation": (
            "The scalar field obeys a 2D diffusion equation, ∂φ/∂t = D∇²φ with D = 0.5, "
            "slowly spreading the source deposited by particle 1. Particle 2 feels a force "
            "-∇φ/p2 from this evolving field, so its dynamics depend on the full time history "
            "of the source rather than the instantaneous configuration. Forces are weak at "
            "early times and strengthen as the field diffuses outward."
        ),
        "explanation_rubric": _RUBRIC_DIFFUSION,
    },
    "wave": {
        "description": "Wave (n=2) field — oscillatory, retarded forces.",
        "mission": (
            "The field propagates as a wave in this universe. "
            "Forces on particle 2 may depend on the history of the field. "
            "Discover the law of motion."
        ),
        "executor_kwargs": {
            "operators": [{"type": "laplacian", "params": {"strength": 1.0}}],
            "temporal_order": 2,
        },
        "true_law": _TRUE_LAW_WAVE,
        "true_law_title": "True Wave Equation",
        "optimal_explanation": (
            "The field obeys the 2D wave equation, ∂²φ/∂t² = c²∇²φ with c = 1, propagating "
            "disturbances sourced by particle 1 at finite speed. Particle 2 experiences "
            "retarded forces — its acceleration at time t depends on the field history, "
            "not the instantaneous source position — producing oscillatory, wave-like "
            "dynamics with force -∇φ/p2."
        ),
        "explanation_rubric": _RUBRIC_WAVE,
    },
    "oscillator": {
        "description": (
            "2D Poisson force (1/r in 2D) whose overall coupling oscillates "
            "sinusoidally with absolute time — the law of physics itself "
            "varies, with period T = 4 and a coupling that periodically "
            "changes sign."
        ),
        "mission": (
            "Two particles interact through an unknown field in a 2D universe. "
            "The field is generated by particle 1 and exerts a force on particle 2. "
            "However, the force law in this universe is *not* constant in time: "
            "the magnitude — and possibly even the sign — of the interaction may "
            "depend on *when* the experiment is performed.  Identical initial "
            "conditions can produce qualitatively different trajectories at "
            "different absolute times.  Discover the law of motion governing "
            "particle 2, including any temporal structure of the coupling.\n\n"
            "Each experiment runs from local time 0 to the requested duration. "
            "Optionally you may pass a `start_time` field, which sets the "
            "absolute clock t at which the experiment begins; this lets you "
            "probe the same initial conditions at different phases of any "
            "underlying time-dependence."
        ),
        "executor_class": "OscillatorExecutor",
        # Same {} kwargs as the other 2-particle worlds — operators and
        # temporal_order are accepted+ignored by the executor for API parity.
        "executor_kwargs": {},
        # The mapping below routes ``OscillatorExecutor`` to
        # ``NBodyOscillatorExecutor`` under engine='nbody'; this world has
        # no FieldSampler twin, so engine='field' raises.
        "true_law": _TRUE_LAW_OSCILLATOR,
        "true_law_title": "True Time-Modulated Laplacian",
        "optimal_explanation": (
            "Two particles interact through a 2D Poisson field, ∇²φ = source, with "
            "particle 1 the source (coupling p1) and particle 2 a test particle "
            "(inertia p2), so the spatial part of the force is the standard 2D "
            "1/r law, F = -∇φ/p2.  The overall coupling of this field is modulated "
            "in absolute time by a sinusoid G(t) = G₀·cos(ω t + φ) with G₀ = 5, "
            "ω = π/2 (period T = 4), and φ = 0, so the effective force is G(t) "
            "times the static 2D-Poisson force.  Because G(t) changes sign within "
            "each period, the same particle pair attracts for a quarter-period, "
            "exerts no force at the zero-crossings, and repels for the next "
            "quarter — identical configurations evolve into qualitatively "
            "different trajectories depending on the absolute time at which the "
            "experiment is performed."
        ),
        "explanation_rubric": _RUBRIC_OSCILLATOR,
    },
    "extra_dimensions": {
        "description": (
            "Extra-dimension force law: 2D-Poisson 1/r at long range, "
            "3D Newtonian 1/r² below the compactification scale "
            "R_c = 0.5.  Identical to the 'gravity' world for any "
            "experiment confined to r ≳ a few R_c."
        ),
        "mission": (
            "Two particles interact through an unknown field in a 2D universe. "
            "The field is generated by particle 1 and exerts a force on particle 2. "
            "At first glance the force law may look quite ordinary — but the "
            "underlying geometry of space in this universe might not be exactly "
            "what it appears.  Identical experimental setups can produce "
            "different effective force laws depending on the *length scale* "
            "you probe.  Discover the law of motion governing particle 2, "
            "including any hidden structure that becomes apparent only at "
            "particular distances.\n\n"
            "Tip: think carefully about which separations to test.  An "
            "experimental campaign that only spans a narrow range of "
            "inter-particle distances may miss the most important features "
            "of the physics."
        ),
        "executor_class": "KaluzaKleinExecutor",
        # Same {} kwargs as the other 2-particle worlds — operators and
        # temporal_order are accepted+ignored by the executor for API parity.
        "executor_kwargs": {},
        # KaluzaKleinExecutor is nbody-only — the image-sum kernel has no
        # equivalent in the FFT FieldSampler operator set.
        "true_law": _TRUE_LAW_EXTRA_DIMENSIONS,
        "true_law_title": "True Kaluza-Klein extra-dimension force",
        "optimal_explanation": (
            "Two particles interact through a static force law that comes "
            "from a 2D visible universe with one extra spatial dimension "
            "compactified on a circle of radius R_c = 0.5 (so the compact "
            "circumference is L = 2π R_c ≈ 3.14).  Particle 1 sits fixed at "
            "the origin with source coupling p1; particle 2 has inertia p2 "
            "and feels the pairwise force\n\n"
            "    F(r) = (G·L)/(4π) · p1 · Σ_n r / (r² + (nL)²)^(3/2)\n\n"
            "where the sum runs over the infinite tower of image charges "
            "of particle 1 along the compact dimension and G = 1.  Two "
            "asymptotic regimes:\n"
            "  • r ≫ R_c: the sum becomes an integral and the force "
            "reduces to F → p1 / (2π r) — the standard 2D Poisson 1/r "
            "law indistinguishable from the gravity world.\n"
            "  • r ≲ R_c: only the n = 0 image survives and the force "
            "becomes F → G·L·p1 / (4π r²) = R_c·p1 / (2 r²) — a 3D "
            "Newtonian inverse-square law.\n"
            "An agent that only probes typical separations (r ≈ 3–5) sees "
            "essentially pure 2D gravity; the extra dimension only reveals "
            "itself in experiments with r ≲ 1, where the force grows as "
            "1/r² rather than 1/r."
        ),
        "explanation_rubric": _RUBRIC_EXTRA_DIMENSIONS,
    },
    "species": {
        "description": "Standard Laplacian with 6 particles of 2 hidden species (different source strengths).",
        "mission": (
            "You are observing a system of 6 particles in a 2D universe. "
            "All particles interact through an unknown field. "
            "The particles may not all be identical — some may generate "
            "stronger or weaker fields than others. "
            "Your goal is to discover the law of motion governing all particles, "
            "including any differences between them."
        ),
        "executor_class": "SpeciesExecutor",
        "executor_kwargs": {},
        "true_law": _TRUE_LAW_SPECIES,
        "true_law_title": "True Laplacian (two species)",
        "optimal_explanation": (
            "Six particles interact through a static Laplacian field, ∇²φ = source, with "
            "each particle feeling a force -∇φ at its location. The particles belong to two "
            "hidden species: particles 0–2 source the field with coupling 1.0, while "
            "particles 3–5 source with coupling 3.0 — the second species generates a field "
            "three times stronger than the first. The asymmetry lives entirely in the source "
            "couplings; all particles respond identically to the resulting total field."
        ),
        "explanation_rubric": _RUBRIC_SPECIES,
        "system_prompt": "PhysicsSchool/prompts/run_experiments_species.md",
        "law_stub": (
            "def discovered_law(positions, velocities, duration):\n"
            "    # positions: list of 6 [x, y] coords relative to center\n"
            "    # velocities: list of 6 [vx, vy]\n"
            "    # duration: float, simulate from t=0 to t=duration\n"
            "    # return: list of 6 [x, y] final positions\n"
            "    return final_positions\n"
        ),
        "experiment_format": (
            '<run_experiment>[{"positions": '
            "[[0,0],[3,0],[-3,0],[0,3],[0,-3],[4,4]], "
            '"velocities": '
            "[[0,0],[0,0],[0,0],[0,0],[0,0],[0,0]], "
            '"measurement_times": [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]}]'
            "</run_experiment>"
        ),
    },
    "dark_matter": {
        "description": "Standard Laplacian with 20 visible particles, 10 invisible dark matter (source=5), and 5 probes.",
        "mission": (
            "You are observing a system of 25 particles in a 2D universe. "
            "20 background particles (indices 0–19) start in a fixed configuration "
            "and interact through an unknown field. They all appear identical. "
            "You control 5 neutral probe particles (indices 20–24) that feel "
            "forces but do not generate any field. "
            "However, the visible particles experience unexplained forces — "
            "they accelerate toward regions where no visible matter exists. "
            "Something invisible may be influencing the dynamics. "
            "Your goal is to discover the law of motion, determine whether "
            "hidden sources exist, and if so, characterize their strength "
            "and approximate location."
        ),
        "executor_class": "DarkMatterExecutor",
        "executor_kwargs": {},
        "true_law": _TRUE_LAW_DARK_MATTER,
        "true_law_title": "True Laplacian (dark matter)",
        "optimal_explanation": (
            "The system obeys a static 2D Laplacian field, ∇²φ = source, with force -∇φ on "
            "each particle, but contains hidden structure the agent cannot directly observe: "
            "10 dark-matter particles with source coupling 5.0 — five times stronger than the "
            "visible population — whose positions are concealed. The 20 visible particles all "
            "have source coupling 1.0 and the 5 probes are neutral (coupling 0). Visible "
            "particles appear to accelerate toward empty regions because those regions "
            "actually contain unseen dark-matter sources, and probes respond to the combined "
            "visible+dark field."
        ),
        "explanation_rubric": _RUBRIC_DARK_MATTER,
        "system_prompt": "PhysicsSchool/prompts/run_experiments_dark_matter.md",
        "law_stub": (
            "def discovered_law(positions, velocities, duration):\n"
            "    # positions: list of 25 [x, y] coords relative to center\n"
            "    #   indices 0-19: visible background, 20-24: probes\n"
            "    # velocities: list of 25 [vx, vy]\n"
            "    # duration: float, simulate from t=0 to t=duration\n"
            "    # return: list of 25 [x, y] final positions\n"
            "    # NOTE: you are scored on the 5 PROBE trajectories (indices 20-24)\n"
            "    return final_positions\n"
        ),
        "experiment_format": (
            '<run_experiment>[{"probe_positions": '
            "[[5,0],[0,5],[-5,0],[0,-5],[7,7]], "
            '"probe_velocities": '
            "[[0,0],[0,0],[0,0],[0,0],[0,0]], "
            '"measurement_times": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]}]'
            "</run_experiment>"
        ),
    },
    "three_species": {
        "description": "Standard Laplacian with 30 particles of 3 hidden species (source couplings 1, 3, -2) + 5 neutral probes.",
        "mission": (
            "You are observing a system of 35 particles in a 2D universe. "
            "30 background particles (indices 0–29) start in a fixed configuration "
            "and interact through an unknown field. They may belong to different "
            "species with different field-generation strengths — some may even "
            "repel rather than attract. "
            "You control 5 neutral probe particles (indices 30–34) that feel "
            "forces but do not generate any field. "
            "Your goal is to discover the law of motion, including how many "
            "species exist, which particles belong to each, and their coupling strengths."
        ),
        "executor_class": "ThreeSpeciesExecutor",
        "executor_kwargs": {},
        "true_law": _TRUE_LAW_THREE_SPECIES,
        "true_law_title": "True Laplacian (three species)",
        "optimal_explanation": (
            "Thirty-five particles interact through a static Laplacian field, ∇²φ = source, "
            "with acceleration -∇φ. The 30 background particles split into three hidden "
            "species: particles 0–9 with source coupling +1, particles 10–19 with +3 (strong "
            "attractors), and particles 20–29 with -2 (repulsive — they source a field that "
            "pushes other particles away rather than pulling them in). Particles 30–34 are "
            "neutral probes with zero coupling, feeling forces without sourcing the field."
        ),
        "explanation_rubric": _RUBRIC_THREE_SPECIES,
        "system_prompt": "PhysicsSchool/prompts/run_experiments_three_species.md",
        "law_stub": (
            "def discovered_law(positions, velocities, duration):\n"
            "    # positions: list of 35 [x, y] coords relative to center\n"
            "    # velocities: list of 35 [vx, vy]\n"
            "    # duration: float, simulate from t=0 to t=duration\n"
            "    # return: list of 35 [x, y] final positions\n"
            "    return final_positions\n"
        ),
        "experiment_format": (
            '<run_experiment>[{"probe_positions": '
            "[[5,0],[0,5],[-5,0],[0,-5],[7,7]], "
            '"probe_velocities": '
            "[[0,0],[0,0],[0,0],[0,0],[0,0]], "
            '"measurement_times": [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]}]'
            "</run_experiment>"
        ),
    },
    "ether": {
        "description": "2D Laplacian central anchor + 20 orbiters with masses {1,2,4} + 5 probes, with a uniform northward 'ether' drift acceleration on every particle.",
        "mission": (
            "You are observing a system of 26 particles in a 2D universe. "
            "One particle (index 0) sits near the centre and acts as a strong "
            "central source. 20 'background' particles (indices 1–20) start "
            "on a ring around it; their masses may not all be equal. You "
            "control 5 probe particles (indices 21–25) that you can place and "
            "give arbitrary masses. "
            "Some property of empty space appears to push every particle in a "
            "preferred direction. Your goal is to discover the full law of "
            "motion: the central force law, any per-particle differences "
            "(masses, couplings), and the nature of the background drift."
        ),
        "executor_class": "EtherExecutor",
        "executor_kwargs": {},
        "true_law": _TRUE_LAW_ETHER,
        "true_law_title": "True Laplacian + Ether drift",
        "optimal_explanation": (
            "Twenty-six particles interact through a static 2D Laplacian field, "
            "∇²φ = source, sourced only by the central anchor (index 0) with "
            "coupling 50. The 20 orbiters (masses cycled through 1, 2, 4) and 5 "
            "probes are test particles (zero source coupling) feeling -∇φ from "
            "the anchor. Layered on top is a uniform 'ether' field that exerts a "
            "body-force F = α·m·ŷ on every particle, with α ≈ 0.05; because the "
            "force is exactly proportional to mass, every particle picks up the "
            "same northward acceleration α regardless of mass — a parabolic drift "
            "common to anchor, orbiters, and probes alike, on top of the orbital "
            "motion."
        ),
        "explanation_rubric": _RUBRIC_ETHER,
        "system_prompt": "PhysicsSchool/prompts/run_experiments_ether.md",
        "law_stub": (
            "def discovered_law(positions, velocities, masses, duration):\n"
            "    # positions: list of 26 [x, y] coords relative to centre\n"
            "    # velocities: list of 26 [vx, vy]\n"
            "    # masses: list of 26 per-particle masses\n"
            "    # duration: float, simulate from t=0 to t=duration\n"
            "    # return: list of 26 [x, y] final positions\n"
            "    # NOTE: scoring focuses on the 5 PROBE trajectories (indices 21-25)\n"
            "    return final_positions\n"
        ),
        "experiment_format": (
            '<run_experiment>[{"probe_positions": '
            "[[8,0],[0,8],[-8,0],[0,-8],[10,10]], "
            '"probe_velocities": '
            "[[0,0],[0,0],[0,0],[0,0],[0,0]], "
            '"probe_masses": [1.0, 1.0, 2.0, 4.0, 1.0], '
            '"measurement_times": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]}]'
            "</run_experiment>"
        ),
    },
    "hubble": {
        "description": "2D Laplacian central anchor + 20 orbiters with masses {1,2,4} + 5 probes, with a position-dependent radial 'Hubble flow' that pushes every particle outward proportional to its distance from the centre.",
        "mission": (
            "You are observing a system of 26 particles in a 2D universe. "
            "One particle (index 0) sits at the centre and acts as a strong "
            "central source. 20 'background' particles (indices 1–20) start "
            "on a ring around it; their masses may not all be equal. You "
            "control 5 probe particles (indices 21–25) — you set their "
            "positions, velocities, AND masses. "
            "Particles far from the centre seem to behave anomalously — "
            "as if some property of empty space pushes them outward. The "
            "effect is hard to spot near the centre, where the central "
            "force dominates. Your goal is to discover the full law of "
            "motion: the central force law, the dependence on distance "
            "and mass, and the nature of any background effect."
        ),
        "executor_class": "HubbleExecutor",
        "executor_kwargs": {},
        "true_law": _TRUE_LAW_HUBBLE,
        "true_law_title": "True Laplacian + Hubble flow",
        "optimal_explanation": (
            "Twenty-six particles interact through a static 2D Laplacian field, "
            "∇²φ = source, sourced only by the central anchor (index 0) with "
            "coupling 50. The 20 orbiters (masses cycled through 1, 2, 4) and 5 "
            "probes are test particles (zero source coupling) feeling -∇φ from "
            "the anchor. Layered on top is a Hubble-flow body-force that gives "
            "every particle an additional outward radial acceleration "
            "a = H · r with H ≈ 0.05, where r is the displacement from the "
            "centre. Because the force is mass-independent, the same H acts on "
            "every particle. The critical radius where Hubble outward push "
            "balances central inward gravity is r_crit = √(Q/(2πH)) ≈ 12.6: "
            "probes at smaller r remain bound and orbit (with slightly reduced "
            "effective gravity), while probes outside r_crit accelerate outward "
            "and escape."
        ),
        "explanation_rubric": _RUBRIC_HUBBLE,
        "system_prompt": "PhysicsSchool/prompts/run_experiments_hubble.md",
        "law_stub": (
            "def discovered_law(positions, velocities, masses, duration):\n"
            "    # positions: list of 26 [x, y] coords relative to centre\n"
            "    # velocities: list of 26 [vx, vy]\n"
            "    # masses: list of 26 per-particle masses\n"
            "    # duration: float, simulate from t=0 to t=duration\n"
            "    # return: list of 26 [x, y] final positions\n"
            "    # NOTE: scoring focuses on the 5 PROBE trajectories (indices 21-25)\n"
            "    return final_positions\n"
        ),
        "experiment_format": (
            '<run_experiment>[{"probe_positions": '
            "[[5,0],[10,0],[15,0],[18,0],[0,12]], "
            '"probe_velocities": '
            "[[0,0],[0,0],[0,0],[0,0],[0,0]], "
            '"probe_masses": [1.0, 1.0, 2.0, 4.0, 1.0], '
            '"measurement_times": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]}]'
            "</run_experiment>"
        ),
    },
    "circle": {
        "description": "Fractional Laplacian (alpha=0.75) — 11 particles, 1 center + 10 ring.",
        "mission": (
            "You are observing a system of 11 particles in a 2D universe. "
            "One particle sits at the center; 10 others are arranged on a ring around it. "
            "All particles interact through an unknown field. "
            "Your goal is to discover the law of motion governing all particles."
        ),
        "executor_class": "CircleExecutor",
        "executor_kwargs": {},
        "true_law": _TRUE_LAW_CIRCLE,
        "true_law_title": "True Fractional Laplacian (circle)",
        "optimal_explanation": (
            "Eleven particles — one at the center plus ten arranged on a surrounding ring — "
            "interact through a static field governed by a fractional Laplacian operator "
            "-(-∇²)^α with α = 0.75. The force on each particle is -∇φ, where φ is sourced "
            "by all particles with uniform coupling. The non-local fractional operator "
            "produces a force-versus-distance law that is intermediate between the "
            "logarithmic 2D Laplacian and pure long-range behavior."
        ),
        "explanation_rubric": _RUBRIC_CIRCLE,
        "system_prompt": "PhysicsSchool/prompts/run_experiments_circle.md",
        "law_stub": (
            "def discovered_law(positions, velocities, duration):\n"
            "    # positions: list of 11 [x, y] coords relative to center\n"
            "    # velocities: list of 11 [vx, vy]\n"
            "    # duration: float, simulate from t=0 to t=duration\n"
            "    # return: list of 11 [x, y] final positions\n"
            "    return final_positions\n"
        ),
        "experiment_format": (
            '<run_experiment>[{"ring_radius": 5.0, '
            '"initial_tangential_velocity": 0.0, '
            '"measurement_times": [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]}]'
            "</run_experiment>"
        ),
    },
}


# Map FieldSampler executor names → NBody equivalents.  Worlds whose
# FieldSampler operator is not portable (diffusion / wave / temporal_order>0)
# have no entry here, and ``get_world(..., engine='nbody')`` will reject them.
_NBODY_EXECUTOR_CLASSES = {
    "SimulationExecutor": NBodySimulationExecutor,
    "CircleExecutor": NBodyCircleExecutor,
    "SpeciesExecutor": NBodySpeciesExecutor,
    "ThreeSpeciesExecutor": NBodyThreeSpeciesExecutor,
    "DarkMatterExecutor": NBodyDarkMatterExecutor,
    "EtherExecutor": NBodyEtherExecutor,
    "HubbleExecutor": NBodyHubbleExecutor,
    # OscillatorExecutor is nbody-only: the time-modulated coupling
    # G(t)·F_static(r) is implemented via the NBody integrator's
    # ``force_modulation`` hook, and has no FieldSampler equivalent.
    "OscillatorExecutor": NBodyOscillatorExecutor,
    # ExtraDimensionsExecutor is nbody-only: the image-charge sum from the
    # compactified extra dimension is a non-PDE pairwise kernel and has
    # no equivalent in the FFT FieldSampler operator set.
    "ExtraDimensionsExecutor": NBodyExtraDimensionsExecutor,
}

_FIELD_EXECUTOR_CLASSES = {
    "SimulationExecutor": SimulationExecutor,
    "CircleExecutor": CircleExecutor,
    "SpeciesExecutor": SpeciesExecutor,
    "ThreeSpeciesExecutor": ThreeSpeciesExecutor,
    "DarkMatterExecutor": DarkMatterExecutor,
    # NOTE: EtherExecutor, HubbleExecutor, and OscillatorExecutor are
    # nbody-only — their body-force / time-modulation terms have no
    # equivalent in the FFT FieldSampler operator set.
}


def get_world(name: str, engine: str = "field", **executor_overrides) -> dict:
    """
    Return a config dict for the named world with keys:
        executor, mission, true_law, true_law_title,
        system_prompt (path string), law_stub (function signature string)

    ``engine`` selects the simulation backend:

      * ``'field'`` (default) — FFT/CIC ``FieldSampler``: supports every world,
        including diffusion (n=1) and wave (n=2) physics.
      * ``'nbody'`` — direct O(N²) ``NBodySampler`` with a high-order
        symplectic integrator.  Available for the 7 worlds whose physics is
        a static (n=0) linear PDE; the diffusion and wave worlds have no
        instantaneous-pairwise-force equivalent and will raise.
    """
    if name not in WORLDS:
        raise ValueError(f"Unknown world '{name}'. Available: {list(WORLDS)}")

    if engine not in ("field", "nbody"):
        raise ValueError(f"engine must be 'field' or 'nbody', got {engine!r}")

    entry = WORLDS[name]
    kwargs = {**entry["executor_kwargs"], **executor_overrides}

    executor_class_name = entry.get("executor_class", "SimulationExecutor")

    if engine == "nbody":
        if executor_class_name not in _NBODY_EXECUTOR_CLASSES:
            raise ValueError(
                f"World {name!r} has no NBody twin (engine='nbody' "
                "supports only worlds with temporal_order=0 and a single "
                "static PDE operator).  Use engine='field' instead."
            )
        # Diffusion / wave have temporal_order != 0 → reject early.
        if kwargs.get("temporal_order", 0) != 0:
            raise ValueError(
                f"engine='nbody' cannot run world {name!r}: it requires a "
                "time-evolving field (temporal_order != 0)."
            )
        executor_cls = _NBODY_EXECUTOR_CLASSES[executor_class_name]
    else:
        if executor_class_name not in _FIELD_EXECUTOR_CLASSES:
            raise ValueError(
                f"World {name!r} has no FieldSampler implementation "
                "(its physics — e.g. uniform body-forces — has no equivalent "
                "in the FFT operator set). Use engine='nbody' instead."
            )
        executor_cls = _FIELD_EXECUTOR_CLASSES[executor_class_name]

    executor = executor_cls(**kwargs)

    default_law_stub = (
        "def discovered_law(pos1, pos2, p1, p2, velocity2, duration):\n"
        "    # your best implementation\n"
        "    return final_pos2, final_vel2\n"
    )
    default_system_prompt = "PhysicsSchool/prompts/run_experiments.md"
    default_experiment_format = (
        '<run_experiment>[{"p1": 1.0, "p2": 1.0, "pos2": [3.0, 0.0], '
        '"velocity2": [0.0, 0.0], "measurement_times": [0.5, 1.0, 2.0]}]</run_experiment>'
    )

    return {
        "executor": executor,
        "mission": entry["mission"],
        "true_law": entry["true_law"],
        "true_law_title": entry["true_law_title"],
        "optimal_explanation": entry.get("optimal_explanation", ""),
        "explanation_rubric": entry.get("explanation_rubric", ""),
        "system_prompt": entry.get("system_prompt", default_system_prompt),
        "law_stub": entry.get("law_stub", default_law_stub),
        "experiment_format": entry.get("experiment_format", default_experiment_format),
    }
