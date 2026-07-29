"""
B6 (MagLIF current-scaling study) baseline reference + similarity-scaling generator.

Pure-python transcription tests (#241): no build, no run.  The committed reference
JSON (verification/reference/b6_ruiz2023_current_scaling.json) transcribes the
anchor-load parameters the Ellison paper defers to Ruiz 2023 Paper II -- the
baseline load at I0 = 20 MA (Sec. III), the lumped-element circuit of Fig. 1
(element values stated in Sec. III), and a digitization of the Fig. 2 open-circuit
voltage trace (flagged for human QC per ADR-0008).  The generator module
(verification/b6_ruiz_scaling.py) loads it (provenance validated eagerly) and
produces similarity-scaled parameters at arbitrary peak current via the power-law
fits of Ruiz Eqs. (26)-(29) / Ellison Eqs. (18)-(23) plus the circuit rules of
Ruiz Eqs. (22)-(23) and the voltage rule of Ruiz Eq. (25).

Oracle: Layer 1 -- literature.  Stated anchor-load and circuit values from
D. E. Ruiz et al., "Exploring the parameter space of MagLIF implosions using
similarity scaling. II. Current scaling", Phys. Plasmas 30, 032708 (2023)
[10.1063/5.0126699, arXiv:2209.14911] Sec. III; power-law scaling exponents from
its Eqs. (26)-(29) (= Ellison et al. 2025, arXiv:2504.10760, Eqs. (18)-(23));
circuit-element scaling from its Eqs. (22)-(23); voltage scaling from Eq. (25);
preheat-radius scaling from Eq. (17).  The paper's own stated 60-MA values
(E_pre 34 kJ, rho_fuel 4.1 mg/cc, h 18.3 mm, Bz0 30 T, AR ~3.1) cross-check the
generator within the fit-vs-exact-prescription spread documented in the test.
The Fig. 2 voltage-trace points are a committed digitization (confidence low,
human QC pending -- ADR-0008).
"""

import math

import numpy as np

import test_suite.verification.b6_ruiz_scaling as b6

# Anchor-load values stated in Ruiz 2023 II, Sec. III (the "Table 3" the Ellison
# preprint cites is a dangling reference -- no such table exists; these text
# statements are the source of record).
BASELINE = {
    "I0_MA": 20.0,
    "h_mm": 10.0,
    "R_i_mm": 2.325,
    "R_o_mm": 2.79,
    "rho_fuel_mg_cc": 2.25,
    "Bz0_T": 14.0,
    "E_pre_kJ": 2.1,
    "R_pre_mm": 0.75,
}

# Lumped-circuit element values stated in Ruiz 2023 II, Sec. III (Fig. 1 topology).
CIRCUIT = {
    "Z0_ohm": 0.18,
    "L0_nH": 9.58,
    "C_nF": 0.1,
    "L1_nH": 5.0,
    "Rloss_i_ohm": 80.0,
    "Rloss_f_ohm": 0.25,
    "t_loss_ns": 0.0,
    "dt_loss_ns": 5.0,
}

# Power-law scaling exponents: Ruiz Eqs. (26)-(29) / Ellison Eqs. (18)-(23);
# R_pre from Ruiz Eq. (17) (geometric similarity with R_i); voltage from
# Eq. (25) (V ~ I*h -> 1 + 0.529); circuit R/L ~ h and C ~ 1/h from
# Eqs. (22)-(23).
EXPONENTS = {
    "R_i": 0.206,
    "R_o": 0.381,
    "Bz0": 0.647,
    "rho_fuel": 0.529,
    "E_pre": 2.529,
    "h": 0.529,
    "R_pre": 0.206,
    "voltage": 1.529,
    "circuit_RL": 0.529,
    "circuit_C": -0.529,
}

# Scalable parameter name -> (baseline key, exponent key).
SCALED_KEYS = {
    "R_i_mm": ("R_i_mm", "R_i"),
    "R_o_mm": ("R_o_mm", "R_o"),
    "Bz0_T": ("Bz0_T", "Bz0"),
    "rho_fuel_mg_cc": ("rho_fuel_mg_cc", "rho_fuel"),
    "E_pre_kJ": ("E_pre_kJ", "E_pre"),
    "h_mm": ("h_mm", "h"),
    "R_pre_mm": ("R_pre_mm", "R_pre"),
}


def test_reference_provenance_complete():
    """Each committed entry carries value/unit/confidence/extraction/source (ADR-0008)."""
    ref = b6.load_reference()
    assert ref["benchmark"] == "B6"
    src = ref["primary_source"]
    assert src["doi"] == "10.1063/5.0126699"
    assert src["arxiv"] == "2209.14911"
    assert src["year"] == 2023

    def check_entry(name, entry):
        for field in ("value", "unit", "confidence", "extraction_method", "source"):
            assert field in entry, f"{name}: missing {field!r}"
        assert entry["confidence"] in ("high", "medium", "low"), name
        for field in ("paper", "where", "doi"):
            assert field in entry["source"], f"{name}: source missing {field!r}"

    for name, entry in ref["baseline_parameters"].items():
        check_entry(f"baseline_parameters.{name}", entry)
    for name, entry in ref["circuit"]["elements"].items():
        check_entry(f"circuit.elements.{name}", entry)
    for name, entry in ref["scaling_exponents"].items():
        for field in ("value", "confidence", "source", "basis"):
            assert field in entry, f"scaling_exponents.{name}: missing {field!r}"

    trace = ref["voltage_trace"]
    assert "digitized" in trace["extraction_method"]
    assert trace["human_qc"] == "pending"
    assert trace["confidence"] == "low"
    for field in ("paper", "where", "doi"):
        assert field in trace["source"], f"voltage_trace: source missing {field!r}"
    assert "digitization_error" in trace


def test_baseline_values_match_ruiz_2023_text():
    """The 20-MA anchor load and circuit elements equal the Sec. III statements."""
    ref = b6.load_reference()
    for name, expect in BASELINE.items():
        got = ref["baseline_parameters"][name]["value"]
        assert got == expect, f"{name}: {got} != {expect}"
    for name, expect in CIRCUIT.items():
        got = ref["circuit"]["elements"][name]["value"]
        assert got == expect, f"{name}: {got} != {expect}"
    for name, expect in EXPONENTS.items():
        got = ref["scaling_exponents"][name]["value"]
        assert got == expect, f"exponent {name}: {got} != {expect}"


def test_generated_40_60_MA_reproduce_exponents():
    """Log-slope of generated 40/60-MA parameters recovers each committed exponent."""
    base = b6.scaled_parameters(20.0)
    for imax in (40.0, 60.0):
        scaled = b6.scaled_parameters(imax)
        ratio = imax / 20.0
        for out_key, (base_key, exp_key) in SCALED_KEYS.items():
            slope = math.log(scaled[out_key] / base[out_key]) / math.log(ratio)
            assert abs(slope - EXPONENTS[exp_key]) < 1.0e-12, (
                f"{out_key}@{imax} MA: slope {slope} != {EXPONENTS[exp_key]}")
        vslope = math.log(
            scaled["voltage_multiplier"] / base["voltage_multiplier"]
        ) / math.log(ratio)
        assert abs(vslope - EXPONENTS["voltage"]) < 1.0e-12
        for el in ("Z0_ohm", "L0_nH", "L1_nH", "Rloss_i_ohm", "Rloss_f_ohm"):
            slope = math.log(
                scaled["circuit"][el] / base["circuit"][el]) / math.log(ratio)
            assert abs(slope - EXPONENTS["circuit_RL"]) < 1.0e-12, el
        cslope = math.log(
            scaled["circuit"]["C_nF"] / base["circuit"]["C_nF"]) / math.log(ratio)
        assert abs(cslope - EXPONENTS["circuit_C"]) < 1.0e-12
        # The shunt-transition times ride the fixed characteristic time t_phi
        # (Ruiz Eq. (24)): unchanged under scaling.
        assert scaled["circuit"]["t_loss_ns"] == base["circuit"]["t_loss_ns"]
        assert scaled["circuit"]["dt_loss_ns"] == base["circuit"]["dt_loss_ns"]


def test_60MA_values_match_paper_statements():
    """Generated 60-MA values match the paper's own stated scaled load.

    Ruiz Sec. III states the 60-MA similarity-scaled load: E_pre 34 kJ, rho_fuel
    4.1 mg/cc, h 18.3 mm, Bz0 30 T, and AR "close to 3.1".  Those statements come
    from the exact scaling prescriptions (Eqs. (4)-(16)); the committed exponents
    are the paper's own power-law FITS, so the two agree only to the fit spread
    (largest for Bz0: 28.5 vs 30 T, ~5%).  A 6% band pins the generator to the
    published numbers without pretending the fit is exact.
    """
    p = b6.scaled_parameters(60.0)
    stated = {"E_pre_kJ": 34.0, "rho_fuel_mg_cc": 4.1, "h_mm": 18.3, "Bz0_T": 30.0}
    for key, expect in stated.items():
        rel = abs(p[key] - expect) / expect
        assert rel < 0.06, f"{key}: {p[key]:.3g} vs stated {expect} (rel {rel:.3f})"
    aspect = p["R_o_mm"] / (p["R_o_mm"] - p["R_i_mm"])
    assert abs(aspect - 3.1) / 3.1 < 0.05, f"AR@60MA {aspect:.3f} vs stated ~3.1"


def test_voltage_trace_digitized_shape_and_scaling():
    """The digitized Fig. 2 trace has the published shape and scales as I^1.529."""
    t_ns, phi_mv = b6.voltage_trace()
    assert len(t_ns) >= 30
    assert np.all(np.diff(t_ns) > 0.0)
    assert t_ns[0] <= -90.0 and t_ns[-1] >= 190.0
    # Characteristic voltage phi_0 = max(phi_oc) ~ 8 MV (Fig. 2 dashed line).
    phi0 = float(np.max(phi_mv))
    assert 7.5 <= phi0 <= 8.5, f"phi_0 {phi0:.2f} MV outside the Fig. 2 band"
    # Characteristic time t_phi = FWHM ~ 100 ns (Fig. 2 arrow spans 0..100 ns).
    half = 0.5 * phi0
    above = np.where(phi_mv >= half)[0]
    fwhm = float(t_ns[above[-1]] - t_ns[above[0]])
    assert 85.0 <= fwhm <= 115.0, f"t_phi (FWHM) {fwhm:.1f} ns outside 100 +/- 15"
    # Scaled trace: same time base (t_phi fixed), voltage multiplied by
    # (Imax/I0)^1.529 (Ruiz Eq. (25) with h ~ I^0.529).
    t60, phi60 = b6.voltage_trace(60.0)
    assert np.array_equal(t60, t_ns)
    expect = phi_mv * (60.0 / 20.0) ** 1.529
    assert np.allclose(phi60, expect, rtol=1.0e-12)
