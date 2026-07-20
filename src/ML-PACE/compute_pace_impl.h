/* -*- c++ -*- ----------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/, Sandia National Laboratories
   LAMMPS development team: developers@lammps.org

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

// Shared with compute_pace_kokkos.cpp: the Kokkos TU needs to read
// acecimpl->basis_set, so ACECimpl must have one definition both TUs agree on
// (previously duplicated verbatim in each .cpp -- an ODR footgun if they drifted).

#ifndef LMP_COMPUTE_PACE_IMPL_H
#define LMP_COMPUTE_PACE_IMPL_H

#include "ace-evaluator/ace_c_basis.h"
#include "ace-evaluator/ace_evaluator.h"

namespace LAMMPS_NS {
struct ACECimpl {
  ACECimpl() : basis_set(nullptr), ace(nullptr) {}
  ~ACECimpl()
  {
    delete basis_set;
    delete ace;
  }
  ACECTildeBasisSet *basis_set;
  ACECTildeEvaluator *ace;
};
}    // namespace LAMMPS_NS

#endif
