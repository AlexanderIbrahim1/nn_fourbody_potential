"""
This module contains functions that return the coefficients for the pair-, triple-,
and quadruple-dipole dispersion interactions between parahydrogen molecules.
"""


def c6_parahydrogen() -> float:
    """
    The C_6 coefficient for the dipole-dipole dispersion interaction between two
    parahydrogen molecules.

    Units: [cm^{-1}] [Angstrom]^{-6}

    Taken from table 6 in:
        M. Schmidt et al. "Raman vibrational shifts of small clusters of
        hydrogen isotopologues." J. Phys. Chem. A. 119, p. 12551-12561 (2015).
    """
    return 58203.64


def c9_parahydrogen() -> float:
    """
    The C_9 coefficient for the triple-dipole dispersion interaction between three
    parahydrogen molecules (i.e. the Axilrod-Teller-Muto potential).

    Units: [cm^{-1}] [Angstrom]^{-9}

    Converted from the value given at the bottom of paragraph 1 in section 3
    (Results and discussion section) of:
        R. J. Hinde. "Three-body interactions in solid parahydrogen." Chem. Phys.
        Lett. 460, p. 141-145 (2008).
    """
    return 34336.220013464925


def b12_parahydrogen_midzuno_kihara() -> float:
    """
    The B_12 coefficient for the quadruple-dipole dispersion interaction between
    four parahydrogen molecules.

    Units: [cm^{-1}] [Angstrom]^{-12}

    This is an approximation of the B_12 coefficient using a the Midzuno-Kihara
    approximation. We use the C_6 and C_9 coefficients to estimate the B_12
    coefficient.
    """
    c6_coeff = c6_parahydrogen()
    c9_coeff = c9_parahydrogen()

    return (5.0 * c9_coeff**2) / (3.0 * c6_coeff)


def b12_parahydrogen_avdz_approx() -> float:
    """
    The B_12 coefficient for the quadruple-dipole dispersion interaction between
    four parahydrogen molecules.

    Units: [cm^{-1}] [Angstrom]^{-12}

    This is an approximation of the coefficient based on the preliminary electronic
    structure calculations done for the four-body parahydrogen interaction.

    We calculate the four-body interaction energy for a tetrahedron with a parahydrogen
    molecule at each corner. The energies are calculated at the CCSD(T) level, using
    an AVDZ basis set. The energies are averaged using the L=3 lebedev quadrature,
    aligned along the xyz axes.

    The estimate of B_12 is based on the value of the interaction energy when the
    tetrahedron's side length is 4.25 Angstroms. This value was chosen because it is
    the turning point in the curve when the interaction energy is multiplied by the
    sidelength to the 12th power.

    The estimate is given as a ratio of the Midzuno-Kihara estimate.
    """
    mk_to_avdz_ratio = 0.9380

    return mk_to_avdz_ratio * b12_parahydrogen_midzuno_kihara()


def b12_parahydrogen_avtz_approx() -> float:
    """
    The B_12 coefficient for the quadruple-dipole dispersion interaction between
    four parahydrogen molecules.

    Units: [cm^{-1}] [Angstrom]^{-12}

    We follow a similar approach to what was done in 'b12_parahydrogen_avdz_approx()'.
    However, we use the CCSD(T) energies calculated using the AVTZ basis instead of
    the AVDZ basis, and we base our estimate on the value of the interaction energy
    at the tetrahedron side length of 4.15 Angstroms (the turning point in the curve).

    The estimate is given as a ratio of the Midzuno-Kihara estimate.
    """
    mk_to_avtz_ratio = 0.8736

    return mk_to_avtz_ratio * b12_parahydrogen_midzuno_kihara()


def q12_parahydrogen_midzuno_kihara() -> float:
    """
    The full Bade dispersion interaction is composed of a pair component, a triplet
    component, and a quadruplet component. We are only interested in the quadruplet
    component.

    Recall that we are using the disperion interaction as an extension of the ab
    initio interaction energy. When we calculate the four-body ab initio interaction
    energy, we already subtract out the contributions from the pair and triplet
    interactions. Thus, we can do the same with the Bade potential.

    When we remove the pair and triplet components, we end up with a weaker interaction
    energy. One way to describe just how much weaker it is from the original Bade
    potential is by providing a new interaction coefficient.

    We choose the tetrahedron geometry. Then we calculate the quadruplet component of
    the Bade interaction, and compare it to the total Bade interaction. For this
    specific geometry, the ratio of the quadruplet component to the total interaction
    is 9 / 65.

    NOTE: this value is just used as a measure of the remaining interaction strength.
    When creating the interaction potential that only uses the quadruplet component
    of the Bade potential, use the proper B12 coefficient:
    ```
    b12_coeff = b12_parahydrogen_midzuno_kihara()
    pot = QuadrupletDispersionPotential(b12_coeff)
    ```
    """
    return (9.0 / 65.0) * b12_parahydrogen_midzuno_kihara()


def q12_parahydrogen_avdz_approx() -> float:
    """
    This is an estimate of the Q_12 coefficient for the quadruplet component of the
    Bade potential, calculated from the AVDZ electronic structure energies.

    The estimate is given as a ratio of the Midzuno-Kihara Q_12 coefficient.
    """
    mk_to_avdz_ratio = 0.9380

    return mk_to_avdz_ratio * q12_parahydrogen_midzuno_kihara()


def q12_parahydrogen_avtz_approx() -> float:
    """
    This is an estimate of the Q_12 coefficient for the quadruplet component of the
    Bade potential, calculated from the AVTZ electronic structure energies.

    The estimate is given as a ratio of the Midzuno-Kihara Q_12 coefficient.
    """
    mk_to_avtz_ratio = 0.8736

    return mk_to_avtz_ratio * q12_parahydrogen_midzuno_kihara()
