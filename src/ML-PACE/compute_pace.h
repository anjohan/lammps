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
ComputeStyle(pace,ComputePACE<0>);
ComputeStyle(pace/atom,ComputePACE<1>);
// clang-format on
#else

#ifndef LMP_COMPUTE_PACE_H
#define LMP_COMPUTE_PACE_H

#include "compute.h"

namespace LAMMPS_NS {

// PERATOM = 0 -> compute pace      : global array (descriptors, forces, virial)
// PERATOM = 1 -> compute pace/atom : per-atom array of ACE descriptors B_{i,nu}
// Both styles share the same ACE evaluation kernel (eval_atom).

template <int PERATOM> class ComputePACE : public Compute {
 public:
  ComputePACE(class LAMMPS *, int, char **);
  ~ComputePACE() override;
  void init() override;
  void init_list(int, class NeighList *) override;
  void compute_array() override;
  void compute_peratom() override;
  double memory_usage() override;

 protected:
  int natoms, nmax, nmaxatom, size_peratom, lastcol;
  int nvalues, yoffset, zoffset;
  int ndims_peratom, ndims_force, ndims_virial;
  double **cutsq;
  class NeighList *list;
  double **pace, **paceall;
  double **pace_peratom;
  double **pace_atom;    // per-atom descriptor array (array_atom), PERATOM=1
  int *map;              // map types to [0,nelements)
  int bikflag, bik_rows, dgradflag, dgrad_rows;
  double cutmax;

  Compute *c_pe;
  Compute *c_virial;
  std::string id_virial;

  // shared per-atom ACE kernel: builds the evaluator for atom i and runs
  // compute_atom, leaving projections (and, for PERATOM=0, neighbours_dB)
  // available on acecimpl->ace.
  void eval_atom(int i, int max_jnum, int ntypes);
  void dbdotr_compute();
  struct ACECimpl *acecimpl;
};

}    // namespace LAMMPS_NS

#endif
#endif
