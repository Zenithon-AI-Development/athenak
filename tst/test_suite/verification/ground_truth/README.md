# Ground-truth experimental data (Ellison MagLIF benchmarks 1-4)

The **Layer-1 experimental oracle** for the Ellison benchmark replications (PRD #106 /
arXiv:2504.10760). Per [ADR-0008](../../../../docs/adr/0008-verification-provenance-policy.md),
the binding ground truth for these benchmarks is the **experiment** — digitized or stated
data from the primary Sandia papers — never another code. FLASH / LASNEX / HYDRA values, if
ever stored, go in a datum's optional `secondary_reference` block and are **never** the
oracle.

Loaded and served by [`ground_truth_oracle.py`](../ground_truth_oracle.py); exercised by
[`test_verify_ground_truth_oracle_cpu.py`](../test_verify_ground_truth_oracle_cpu.py).

## Files (one per benchmark)

| File | Benchmark | Experiment | Primary source |
|------|-----------|------------|----------------|
| `b1_single_mode_mrt_sinars_2011.json` | B1 | single-mode MRT, Al liner | Sinars et al., Phys. Plasmas **18**, 056301 (2011), [10.1063/1.3560911](https://doi.org/10.1063/1.3560911) |
| `b2_multimode_mrt_mcbride_2012.json` | B2 | multi-mode MRT, Be liner | McBride et al., Phys. Rev. Lett. **109**, 135004 (2012), [10.1103/PhysRevLett.109.135004](https://doi.org/10.1103/PhysRevLett.109.135004); companion PoP **20**, 056309 (2013), [10.1063/1.4803079](https://doi.org/10.1063/1.4803079) |
| `b3_convergent_rm_knapp_2020.json` | B3 | converging Richtmyer–Meshkov | Knapp et al., Phys. Plasmas **27**, 092707 (2020), [10.1063/5.0013194](https://doi.org/10.1063/5.0013194) |
| `b4_icf_confinement_knapp_2017.json` | B4 | ICF confinement time | Knapp et al., Phys. Plasmas **24**, 042708 (2017), [10.1063/1.4981206](https://doi.org/10.1063/1.4981206) |

## Datum format

Each file is `{ "benchmark", "ellison_id", "title", "description", "primary_source",
"setup", "data": [ ... ] }`. Every entry in `data` is one observable the simulation must
reproduce and **must** carry full provenance (validated on load — a missing field is a
`ProvenanceError`, not a silent pass):

```jsonc
{
  "id": "min_radius",                    // unique within the benchmark
  "observable": "minimum fuel radius at stagnation",
  "kind": "scalar" | "curve_point" | "curve",
  "value": 0.45,                         // omitted/null when pending digitization
  "unit": "mm",
  "confidence": "high" | "medium" | "low",
  "extraction_method": "stated scalar | digitized (WebPlotDigitizer) | pending_digitization (#NNN)",
  "oracle_kind": "experiment",           // MUST be the experiment (ADR-0008)
  "experimental_error":  { "value": 0.10, "kind": "relative", "basis": "..." },
  "digitization_error":  { "value": 0.0,  "kind": "absolute", "basis": "..." },
  "source": { "paper": "...", "figure": "...", "doi": "10.xxxx/..." },
  "secondary_reference": { "code": "FLASH", "value": 9.7 }   // optional, NEVER the oracle
}
```

* **Tolerance band** for a comparison is `max(experimental_error, digitization_error)`
  in absolute units (`kind: "relative"` is scaled by `|value|`). Each error carries a
  `basis` string saying where the bar came from (published value, inferred from
  significant figures, or a placeholder for a downstream issue) — **no fabricated
  tolerance without saying so.**
* **No fabricated data points.** A not-yet-digitized observable is committed as a
  `"status": "pending_digitization"` datum: full provenance (which paper/figure it will
  come from), `confidence: "low"`, and **no value**. Comparing it raises
  `PendingDatumError`. The Phase-C benchmark issues (#119–#122) digitize the curves and
  fill these in, flagging each digitized point's confidence.

## Currently extracted vs. pending

Faithfully extracted now: **B4** `min_radius` = 0.45 mm and `peak_density` ≈ 10 g/cc
(stated scalars). All MRT/RM growth curves and B4's confinement-time result are
`pending_digitization`, deferred to their Phase-C benchmark issues so the curves are
digitized at the point of use rather than guessed here.
