"""B1 deck faithfulness: the liner geometry must be the Sinars Series-1 target (#219).

The committed B1 deck modelled a *generic* "AR6" aluminum liner (inner 2.8875 mm, outer
3.4688 mm -- a 581 um wall) rather than the target the benchmark actually replicates.
From arXiv:2504.10760 section 3.1, describing the Series-1 aluminum liner of Sinars 2011
(Phys. Plasmas 18, 056301 -- the same reference the committed oracle is digitized from):

    "The inner radius of the target is 2.876 mm and the outer radius is 3.168 mm"

so the wall is 292 um, not 581 um, and the liner carries about half the mass the deck gave
it.  This matters quantitatively, not cosmetically: MRT growth is
``Gamma = int sqrt(k g) dt`` with the in-flight acceleration ``g ~ B_phi^2/(rho dr)``, so
halving the wall roughly doubles ``g`` and multiplies the growth exponent by ``~sqrt(2)``.

The same section pins the seeded mode (400 um wavelength, 20 um amplitude, imposed on the
OUTER radius as ``r(z) = r_o + delta sin(2 pi z / lambda)``), which the deck already had
right; asserting it here keeps it that way.

Pure-python deck audit: no build, no run.
"""

import os
import re

import test_suite.testutils as testutils

# arXiv:2504.10760 section 3.1 -- Series-1 aluminum liner, all in mm.
PAPER_R_IN_MM = 2.876
PAPER_R_OUT_MM = 3.168
PAPER_LAMBDA_UM = 400.0
PAPER_DELTA_UM = 20.0

DECK = os.path.join(
    testutils._repo_root(), "tst", "inputs", "maglif_b1_sinars.athinput"
)


def _deck_values():
    """Parse ``key = value`` pairs from the committed B1 athinput."""
    out = {}
    with open(DECK, "r") as fh:
        for line in fh:
            line = line.split("#", 1)[0].strip()
            m = re.match(r"^([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(\S+)\s*$", line)
            if m:
                out[m.group(1)] = m.group(2)
    return out


def test_liner_radii_match_the_series1_target():
    """Inner and outer radii are the paper's, to 1 um.

    The deck carries them in CGS cm (the <units> length_cgs is 0.1, so 1 code length is
    1 mm and the raw deck values are cm).
    """
    v = _deck_values()
    r_in_mm = float(v["r_fuel"]) * 10.0     # cm -> mm
    r_out_mm = float(v["r_liner"]) * 10.0
    assert abs(r_in_mm - PAPER_R_IN_MM) < 1.0e-3, (
        f"inner radius {r_in_mm:.4f} mm != paper's {PAPER_R_IN_MM} mm"
    )
    assert abs(r_out_mm - PAPER_R_OUT_MM) < 1.0e-3, (
        f"outer radius {r_out_mm:.4f} mm != paper's {PAPER_R_OUT_MM} mm"
    )


def test_wall_thickness_and_aspect_ratio_match_the_series1_target():
    """The wall -- the quantity the MRT feed is actually sensitive to -- is 292 um.

    Stated separately from the radii because it is the derived quantity that carries the
    physics: a 2x error here is a ~sqrt(2)x error in the MRT growth exponent.
    """
    v = _deck_values()
    r_in_mm = float(v["r_fuel"]) * 10.0
    r_out_mm = float(v["r_liner"]) * 10.0
    wall_um = (r_out_mm - r_in_mm) * 1.0e3
    paper_wall_um = (PAPER_R_OUT_MM - PAPER_R_IN_MM) * 1.0e3
    assert abs(wall_um - paper_wall_um) < 1.0, (
        f"liner wall {wall_um:.1f} um != paper's {paper_wall_um:.1f} um "
        f"(mass ratio {wall_um / paper_wall_um:.2f}x)"
    )
    aspect = r_out_mm / (r_out_mm - r_in_mm)
    paper_aspect = PAPER_R_OUT_MM / (PAPER_R_OUT_MM - PAPER_R_IN_MM)
    assert abs(aspect - paper_aspect) < 0.05, (
        f"aspect ratio {aspect:.2f} != paper's {paper_aspect:.2f}"
    )


def test_outer_boundary_still_encloses_the_liner():
    """The driven outer boundary must sit outside the (now smaller) liner, with a gap.

    Guards the obvious way to break the deck while fixing the radii.
    """
    v = _deck_values()
    r_out_mm = float(v["r_liner"]) * 10.0
    x1max_mm = float(v["x1max"])            # <mesh> x1max is already in code length = mm
    assert x1max_mm > r_out_mm, (
        f"domain outer radius {x1max_mm} mm does not enclose the liner {r_out_mm:.4f} mm"
    )
    assert x1max_mm - r_out_mm > 0.1, (
        f"vacuum gap {x1max_mm - r_out_mm:.4f} mm is too thin for the driven boundary"
    )


def test_seeded_mode_matches_the_paper():
    """400 um wavelength, 20 um amplitude, on the outer radius (already correct)."""
    v = _deck_values()
    lz_mm = float(v["x3max"]) - float(v["x3min"])
    lambda_um = lz_mm / float(v["pert_mode"]) * 1.0e3
    delta_um = float(v["pert_amp"]) * 1.0e4        # cm -> um
    assert abs(lambda_um - PAPER_LAMBDA_UM) < 1.0, (
        f"seeded wavelength {lambda_um:.1f} um != paper's {PAPER_LAMBDA_UM} um"
    )
    assert abs(delta_um - PAPER_DELTA_UM) < 0.1, (
        f"seeded amplitude {delta_um:.1f} um != paper's {PAPER_DELTA_UM} um"
    )
