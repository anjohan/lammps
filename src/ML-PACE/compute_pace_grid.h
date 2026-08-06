/* -*- c++ -*- ----------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/, Sandia National Laboratories
   LAMMPS Development team: developers@lammps.org

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

#ifdef COMPUTE_CLASS
// clang-format off
ComputeStyle(pace/grid,ComputePACEGrid<0>);
ComputeStyle(pace/grid/local,ComputePACEGrid<1>);
// clang-format on
#else

#ifndef LMP_COMPUTE_PACE_GRID_H
#define LMP_COMPUTE_PACE_GRID_H

#include "compute_grid.h"
#include "compute_grid_local.h"

#include <type_traits>

namespace LAMMPS_NS {

// LOCAL = 0 -> compute pace/grid       : global array (x,y,z + nvalues ACE descriptors)
// LOCAL = 1 -> compute pace/grid/local : local array (ix,iy,iz,x,y,z + nvalues descriptors)
// Both styles share the same brute-force per-grid-point ACE evaluation loop
// (eval_grid(), in the .cpp), mirroring the ComputePACE<PERATOM> if-constexpr
// pattern in compute_pace.{h,cpp} -- except here the divergent output storage
// (grid/gridall vs alocal) comes from a divergent BASE class, not always-
// present member fields, so every base-class member access (including ones
// with identical names/types in both bases, like nvalues/cutmax) needs
// `this->`: the base is a dependent type (std::conditional_t<LOCAL,...>), so
// ordinary unqualified lookup cannot see into it.

template <int LOCAL>
class ComputePACEGrid : public std::conditional_t<LOCAL, ComputeGridLocal, ComputeGrid> {
 public:
  ComputePACEGrid(class LAMMPS *, int, char **);
  ~ComputePACEGrid() override;
  void init() override;
  void compute_array() override;    // LOCAL = 0
  void compute_local() override;    // LOCAL = 1
  double memory_usage() override;

 protected:
  using Base = std::conditional_t<LOCAL, ComputeGridLocal, ComputeGrid>;

  int chunksize_arg;    // optional "chunksize N" keyword; used by the Kokkos
                        // subclass only, accepted-and-ignored here (same
                        // convention as compute_pace.{h,cpp})
  int elem_index;       // grid-point central species: 0-based index into the
                        // .yace potential's element list ("element X" keyword,
                        // default: element 0)

  struct ACECimpl *acecimpl;

  // extended per-invocation atom arrays (rebuilt once per compute_array()/
  // compute_local() call, not per grid point): entries [0,ntotal) alias
  // atom->x / atom->type, entry [ntotal] is the phantom grid-point "atom"
  // (see eval_grid() in the .cpp)
  double **xg;
  int *typeg;
  int *jlist;
  double xgrid_pt[3];
  int ntotalmax;    // allocated capacity (real atoms) backing xg/typeg/jlist

  void eval_grid();    // shared brute-force per-grid-point ACE evaluation loop
};

}    // namespace LAMMPS_NS

#endif
#endif
