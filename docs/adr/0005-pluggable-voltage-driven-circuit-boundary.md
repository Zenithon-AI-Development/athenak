# Pluggable, voltage-driven circuit boundary for the load current

**Context.** Z-pinch/MagLIF is circuit-driven: a pulsed-power source delivers current to the liner,
setting `B_φ(R_out)=μ₀I(t)/2πR_out`. As the liner implodes its inductance changes → back-EMF couples
load and circuit. The verification anchor ([[maglif-verification-anchor]], arXiv:2504.10760) drives
benchmarks 1–4 with a **prescribed current** and 5–6 with a **coupled lumped-element circuit with
dynamic impedance feedback**, computing the load voltage by **Faraday's law (rate of change of magnetic
flux)**. The natural physical drive is a *voltage* source, not a current.

**Decision.** Put a **pluggable "drive source"** behind the `B_φ` outer boundary, with three modes:
- **A — prescribed `I(t)`**: tabulated/analytic current waveform, no feedback (covers benchmarks 1–4;
  also the path for replaying measured experimental current).
- **B — voltage + fixed RLC**: integrate a simple series-circuit ODE from an open-circuit voltage waveform.
- **C — voltage + coupled circuit with load feedback** (default target): integrate the lumped-element
  circuit ODE alongside the MHD; the **load voltage = d(flux)/dt via a global Kokkos reduction**
  (Faraday) feeds back; captures current loss and the stagnation voltage spike.

Architecture, consistent with AthenaK: the circuit is a small ODE advanced once per step (operator-split,
host-side scalars); the feedback uses the existing `history.cpp` global-reduction machinery; the `B_φ`
BC uses the existing time-dependent `MHDBoundaryFnPtr` user-hook, plus the **nocurrent** vacuum
`∂_r(r·B_φ)=0` extrapolation outside the load. The circuit current drives **both** the ideal CT `B_φ`
BC and the resistive `B_φ` diffusion BC (the special operator in [[adr-0004]]) — one shared boundary
current. Build staged: A first (validate the pinch end-to-end), then C; B falls out cheaply.

**Rejected.** Hardwiring a single circuit model — the verification suite itself needs both prescribed-I
and coupled-circuit, so the drive source must be pluggable.
