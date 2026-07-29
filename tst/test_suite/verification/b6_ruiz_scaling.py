"""
B6 baseline reference loader + similarity-scaling parameter generator (#241).

The Ellison benchmark-6 replication (MagLIF current-scaling study, arXiv:2504.10760
Sec. 3.6) defers every baseline number to Ruiz et al., "Exploring the parameter
space of MagLIF implosions using similarity scaling. II. Current scaling",
Phys. Plasmas 30, 032708 (2023) [10.1063/5.0126699, arXiv:2209.14911].  The
Ellison preprint's "Table 3" pointer is a dangling reference (no such table
exists), so the source of record is Ruiz Sec. III itself: the 20-MA anchor load,
the Fig. 1 lumped-circuit element values, and the Fig. 2 open-circuit voltage
trace (committed as a digitization flagged for human QC per ADR-0008).

This module loads the committed reference JSON
(``reference/b6_ruiz2023_current_scaling.json``), validates provenance eagerly
(a missing citation/confidence field is a ``ProvenanceError``, not a silent
pass), and generates similarity-scaled deck parameters at arbitrary peak
current:

* target parameters via the paper's power-law fits, Ruiz Eqs. (26)-(29)
  (= Ellison Eqs. (18)-(23)), plus R_pre via geometric similarity with R_i
  (Ruiz Eq. (17));
* circuit elements via Ruiz Eqs. (22)-(23): R, L scale with the liner height h,
  C inversely with h; the shunt-transition times ride the fixed characteristic
  time t_phi (Eq. (24)) and are unchanged;
* the voltage drive via Ruiz Eq. (25): amplitude multiplied by
  (Imax/I0)^(1+0.529), time base unchanged.
"""

import json
import os

import numpy as np

from test_suite.verification.ground_truth_oracle import ProvenanceError

# Anchor the committed reference JSON to this module so it is found from any CWD.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REFERENCE_JSON = os.path.join(
    _THIS_DIR, "reference", "b6_ruiz2023_current_scaling.json")

_ENTRY_FIELDS = ("value", "unit", "confidence", "extraction_method", "source")
_EXPONENT_FIELDS = ("value", "confidence", "source", "basis")
_SOURCE_FIELDS = ("paper", "where", "doi")
_VALID_CONFIDENCE = ("high", "medium", "low")

# (output key, baseline-parameter key, exponent key) for the scaled target set.
_SCALED_PARAMETERS = (
    ("R_i_mm", "R_i_mm", "R_i"),
    ("R_o_mm", "R_o_mm", "R_o"),
    ("Bz0_T", "Bz0_T", "Bz0"),
    ("rho_fuel_mg_cc", "rho_fuel_mg_cc", "rho_fuel"),
    ("E_pre_kJ", "E_pre_kJ", "E_pre"),
    ("h_mm", "h_mm", "h"),
    ("R_pre_mm", "R_pre_mm", "R_pre"),
)
# Circuit elements that scale with the liner height h (Ruiz Eq. (22)).
_CIRCUIT_RL = ("Z0_ohm", "L0_nH", "L1_nH", "Rloss_i_ohm", "Rloss_f_ohm")
# Shunt-transition times ride the fixed characteristic time t_phi (Eq. (24)).
_CIRCUIT_FIXED = ("t_loss_ns", "dt_loss_ns")


def _check_source(name, src):
    for field in _SOURCE_FIELDS:
        if field not in src:
            raise ProvenanceError(f"{name}: source missing {field!r} (ADR-0008)")


def _check_entry(name, entry):
    for field in _ENTRY_FIELDS:
        if field not in entry:
            raise ProvenanceError(f"{name}: missing {field!r} (ADR-0008)")
    if entry["confidence"] not in _VALID_CONFIDENCE:
        raise ProvenanceError(
            f"{name}: confidence {entry['confidence']!r} not in {_VALID_CONFIDENCE}")
    _check_source(name, entry["source"])


def load_reference(path=REFERENCE_JSON):
    """Load and provenance-validate the committed B6 reference JSON."""
    with open(path, "r") as f:
        doc = json.load(f)
    if doc.get("benchmark") != "B6":
        raise ProvenanceError(f"{path}: benchmark is not 'B6'")
    src = doc.get("primary_source", {})
    for field in ("authors", "paper", "journal", "year", "doi", "arxiv"):
        if field not in src:
            raise ProvenanceError(f"primary_source: missing {field!r}")
    for name, entry in doc.get("baseline_parameters", {}).items():
        _check_entry(f"baseline_parameters.{name}", entry)
    for name, entry in doc.get("circuit", {}).get("elements", {}).items():
        _check_entry(f"circuit.elements.{name}", entry)
    for name, entry in doc.get("scaling_exponents", {}).items():
        for field in _EXPONENT_FIELDS:
            if field not in entry:
                raise ProvenanceError(f"scaling_exponents.{name}: missing {field!r}")
        _check_source(f"scaling_exponents.{name}", entry["source"])
    trace = doc.get("voltage_trace", {})
    for field in ("confidence", "human_qc", "extraction_method", "source",
                  "digitization_error", "points"):
        if field not in trace:
            raise ProvenanceError(f"voltage_trace: missing {field!r}")
    if "digitized" not in trace["extraction_method"]:
        raise ProvenanceError(
            "voltage_trace: a figure-derived curve must declare a digitized "
            "extraction_method (ADR-0008)")
    _check_source("voltage_trace", trace["source"])
    if not trace["points"]:
        raise ProvenanceError("voltage_trace: no points committed")
    for i, p in enumerate(trace["points"]):
        if "t_ns" not in p or "phi_oc_MV" not in p:
            raise ProvenanceError(f"voltage_trace.points[{i}]: missing t/phi field")
    return doc


def scaled_parameters(imax_MA, reference=None):
    """Similarity-scaled deck parameters at peak current ``imax_MA`` (in MA).

    Returns a dict of scaled target parameters (power-law fits, Ruiz
    Eqs. (26)-(29) + Eq. (17)), the scaled circuit elements (Eqs. (22)-(23)),
    and the voltage-amplitude multiplier (Eq. (25)).
    """
    ref = load_reference() if reference is None else reference
    baseline = {k: e["value"] for k, e in ref["baseline_parameters"].items()}
    exps = {k: e["value"] for k, e in ref["scaling_exponents"].items()}
    ratio = float(imax_MA) / baseline["I0_MA"]

    out = {"Imax_MA": float(imax_MA)}
    for out_key, base_key, exp_key in _SCALED_PARAMETERS:
        out[out_key] = baseline[base_key] * ratio ** exps[exp_key]
    out["voltage_multiplier"] = ratio ** exps["voltage"]

    elements = ref["circuit"]["elements"]
    circuit = {}
    for name in _CIRCUIT_RL:
        circuit[name] = elements[name]["value"] * ratio ** exps["circuit_RL"]
    circuit["C_nF"] = elements["C_nF"]["value"] * ratio ** exps["circuit_C"]
    for name in _CIRCUIT_FIXED:
        circuit[name] = elements[name]["value"]
    out["circuit"] = circuit
    return out


def voltage_trace(imax_MA=None, reference=None):
    """The digitized Fig. 2 open-circuit voltage trace, optionally scaled.

    Returns ``(t_ns, phi_oc_MV)`` arrays.  With ``imax_MA`` given, the
    amplitude is multiplied by ``(imax_MA/I0)**1.529`` (Ruiz Eq. (25)); the
    time base is unchanged because the characteristic time t_phi is held fixed.
    """
    ref = load_reference() if reference is None else reference
    points = ref["voltage_trace"]["points"]
    t_ns = np.array([p["t_ns"] for p in points], dtype=float)
    phi_mv = np.array([p["phi_oc_MV"] for p in points], dtype=float)
    if imax_MA is not None:
        i0 = ref["baseline_parameters"]["I0_MA"]["value"]
        phi_mv = phi_mv * (float(imax_MA) / i0) ** ref[
            "scaling_exponents"]["voltage"]["value"]
    return t_ns, phi_mv
