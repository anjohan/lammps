// clang-format off
/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/, Sandia National Laboratories
   LAMMPS Development team: developers@lammps.org
   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.
   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

#include "compute_pace_grid.h"
#include "compute_pace_impl.h"

#include "ace-evaluator/ace_types.h"

#include "atom.h"
#include "comm.h"
#include "domain.h"
#include "error.h"
#include "force.h"
#include "memory.h"
#include "modify.h"
#include "pair.h"
#include "update.h"

#include <cstring>

using namespace LAMMPS_NS;

/* ---------------------------------------------------------------------- */

template<int LOCAL>
ComputePACEGrid<LOCAL>::ComputePACEGrid(LAMMPS *lmp, int narg, char **arg) :
    Base(lmp, narg, arg), chunksize_arg(0), elem_index(0), acecimpl(nullptr),
    xg(nullptr), typeg(nullptr), jlist(nullptr), ntotalmax(0)
{
  // skip over arguments used by the base class ("grid nx ny nz") so that
  // argument positions line up with a regular per-atom-style compute
  // (see ComputeSNAGrid's constructor, compute_sna_grid.cpp:28-45)

  arg += this->nargbase;
  narg -= this->nargbase;

  int nargmin = 4;    // id/group/style-slots (unused, now hold nx ny nz) + file
  if (narg < nargmin) this->error->all(FLERR, "Illegal compute {} command", this->style);

  acecimpl = new ACECimpl;

  // read the .yace potential file -- stores pre-merged c-tilde (coupling x fit coefficients)

  auto potential_file_name = utils::get_potential_file_path(arg[3]);
  delete acecimpl->basis_set;
  acecimpl->basis_set = new ACECTildeBasisSet(potential_file_name);
  this->cutmax = acecimpl->basis_set->cutoffmax;

  // # of rank 1, rank > 1 functions

  int n_r1 = acecimpl->basis_set->total_basis_size_rank1[0];
  int n_rp = acecimpl->basis_set->total_basis_size[0];
  this->nvalues = n_r1 + n_rp;
  for (int mu = 1; mu < (int) acecimpl->basis_set->nelements; mu++) {
    if (acecimpl->basis_set->total_basis_size_rank1[mu] != n_r1 ||
        acecimpl->basis_set->total_basis_size[mu] != n_rp)
      this->error->all(FLERR, "compute {}: per-element basis sizes differ; only "
                 "equal-size potentials are supported", this->style);
  }

  // optional trailing keywords

  int iarg = nargmin;
  while (iarg < narg) {
    if (strcmp(arg[iarg], "element") == 0) {
      if (iarg + 1 >= narg)
        this->error->all(FLERR, "compute {}: element keyword needs a value", this->style);
      const std::string elem_name = arg[iarg + 1];
      int idx = -1;
      for (int e = 0; e < (int) acecimpl->basis_set->nelements; e++)
        if (acecimpl->basis_set->elements_name[e] == elem_name) {
          idx = e;
          break;
        }
      if (idx < 0)
        this->error->all(FLERR, "compute {}: unknown element '{}' in potential file", this->style,
                          elem_name);
      elem_index = idx;
      iarg += 2;
    } else if (strcmp(arg[iarg], "chunksize") == 0) {
      if (iarg + 1 >= narg)
        this->error->all(FLERR, "compute {}: chunksize keyword needs a value", this->style);
      chunksize_arg = utils::inumeric(FLERR, arg[iarg + 1], false, lmp);
      if (chunksize_arg <= 0)
        this->error->all(FLERR, "compute {}: chunksize must be positive", this->style);
      iarg += 2;
    } else {
      this->error->all(FLERR, "Unknown compute {} keyword: {}", this->style, arg[iarg]);
    }
  }

  if constexpr (LOCAL)
    this->size_local_cols = this->size_local_cols_base + this->nvalues;
  else
    this->size_array_cols = this->size_array_cols_base + this->nvalues;
}

/* ---------------------------------------------------------------------- */

template<int LOCAL>
ComputePACEGrid<LOCAL>::~ComputePACEGrid()
{
  // a Kokkos device copy must not free the original's memory
  if (this->copymode) return;

  delete acecimpl;
  delete[] xg;
  this->memory->destroy(typeg);
  this->memory->destroy(jlist);
}

/* ---------------------------------------------------------------------- */

template<int LOCAL>
void ComputePACEGrid<LOCAL>::init()
{
  if (this->force->pair == nullptr)
    this->error->all(FLERR, "Compute {} requires a pair style be defined", this->style);

  if (this->cutmax > this->force->pair->cutforce)
    this->error->all(FLERR, "Compute {} cutoff is longer than pairwise cutoff", this->style);

  // ComputeGrid/ComputeGridLocal's grid2x omits boxlo for an orthogonal box
  // (only the triclinic branch adds the origin, via domain->lamda2x) -- guard
  // here rather than patch the ML-SNAP base class

  if (!this->domain->triclinic) {
    double *boxlo = this->domain->boxlo;
    if (boxlo[0] != 0.0 || boxlo[1] != 0.0 || boxlo[2] != 0.0)
      this->error->all(FLERR, "Compute {} requires an orthogonal simulation box with boxlo = "
                 "(0,0,0): the grid is measured from the box origin and silently ignores boxlo "
                 "otherwise; shift the box to the origin, or use a triclinic box", this->style);
  }

  // anchored pattern: get_compute_by_style() substring-matches, so a bare
  // "pace/grid" would also count every "pace/grid/local" (cf. "^sna/grid$"
  // in compute_sna_grid.cpp)
  if (this->modify->get_compute_by_style(std::string("^") + this->style + "$").size() > 1 &&
      this->comm->me == 0)
    this->error->warning(FLERR, "More than one compute {}", this->style);

  // build the evaluator once per init, reused across all grid points and
  // timesteps; no gradients are needed here (descriptors only), so
  // compute_b_grad=false avoids allocating per-point gradient scratch

  delete acecimpl->ace;
  acecimpl->ace = new ACECTildeEvaluator(*acecimpl->basis_set);
  acecimpl->ace->compute_projections = true;
  acecimpl->ace->compute_b_grad = false;

  const int ntypes = this->atom->ntypes;
  if (ntypes > (int) acecimpl->basis_set->nelements)
    this->error->all(FLERR, "Compute {}: ntypes ({}) exceeds the number of elements "
               "in the potential file ({})", this->style, ntypes,
               (int) acecimpl->basis_set->nelements);

  // element_type_mapping maps a LAMMPS atom type -> internal species index;
  // slot ntypes+1 is our own phantom grid-point "atom" species, chosen by the
  // "element" keyword (default: element 0). Array1D is unchecked -- undersizing
  // this is silent garbage, hence +2 (valid indices 0..ntypes+1).

  acecimpl->ace->element_type_mapping.init(ntypes + 2);
  for (int ik = 1; ik <= ntypes; ik++)
    acecimpl->ace->element_type_mapping(ik) = ik - 1;
  acecimpl->ace->element_type_mapping(ntypes + 1) = elem_index;
}

/* ----------------------------------------------------------------------
   shared brute-force per-grid-point ACE evaluation loop (both styles).
   Loop skeleton mirrors ComputeSNAGrid::compute_array (compute_sna_grid.cpp)
   / ComputeSNAGridLocal::compute_local (compute_sna_grid_local.cpp): owned
   brick nzlo..nzhi/nylo..nyhi/nxlo..nxhi, iz outer / ix fastest, matching the
   row order ComputeGridLocal::assign_coords() used to fill the coordinate
   columns of alocal.
------------------------------------------------------------------------- */

template<int LOCAL>
void ComputePACEGrid<LOCAL>::eval_grid()
{
  double ** const x = this->atom->x;
  const int * const mask = this->atom->mask;
  int * const type = this->atom->type;
  const int ntypes = this->atom->ntypes;
  const int ntotal = this->atom->nlocal + this->atom->nghost;

  // refresh the extended atom arrays every invocation -- atom->x/atom->type
  // can be reallocated or reordered between calls even when ntotal is
  // unchanged (comm exchange, atom sorting, ...); xg is a plain array of
  // pointers (aliasing existing atom->x rows plus our own phantom grid-point
  // buffer), not a primitive array, so it is grown with new/delete rather
  // than memory->grow (which refuses 1d arrays of pointers, see memory.h)

  // "!xg" and not just growth: a rank whose subdomain+ghost shell holds no
  // atoms (ntotal == 0, e.g. a slab system with large vacuum under MPI) must
  // still allocate the phantom-slot entry written below
  if (!xg || ntotal > ntotalmax) {
    ntotalmax = ntotal;
    delete[] xg;
    xg = new double *[ntotalmax + 1];
    this->memory->grow(typeg, ntotalmax + 1, "pace/grid:typeg");
    this->memory->grow(jlist, ntotalmax, "pace/grid:jlist");
  }

  for (int j = 0; j < ntotal; j++) {
    xg[j] = x[j];
    typeg[j] = type[j];
  }
  xg[ntotal] = xgrid_pt;
  typeg[ntotal] = ntypes + 1;

  // the evaluator's per-neighbour scratch (Y_cache/DG_cache/R_cache/...) is
  // sized lazily by this call and otherwise stays at its (empty) default,
  // which compute_atom() indexes out of bounds; jnum for any grid point is
  // at most ntotal (every real atom+ghost), so that is a safe, if
  // conservative, upper bound -- same convention as compute_pace.cpp's
  // per-invocation resize_neighbours_cache(max_jnum) call
  acecimpl->ace->resize_neighbours_cache(ntotal);

  if constexpr (!LOCAL)
    memset(&this->grid[0][0], 0, sizeof(double) * this->size_array_rows * this->size_array_cols);

  [[maybe_unused]] int irow = 0;    // row counter, LOCAL output only (alocal rows are
                                    // enumerated in exactly this loop's nesting order by
                                    // assign_coords(); unused in the global instantiation)

  for (int iz = this->nzlo; iz <= this->nzhi; iz++)
    for (int iy = this->nylo; iy <= this->nyhi; iy++)
      for (int ix = this->nxlo; ix <= this->nxhi; ix++) {

        double xtmp[3];
        int igrid = 0;
        if constexpr (LOCAL) {
          this->grid2x(ix, iy, iz, xtmp);
        } else {
          igrid = iz * (this->nx * this->ny) + iy * this->nx + ix;
          this->grid2x(igrid, xtmp);
        }
        xgrid_pt[0] = xtmp[0];
        xgrid_pt[1] = xtmp[1];
        xgrid_pt[2] = xtmp[2];

        // brute-force neighbour search (no neighbor list for grid computes,
        // matching sna/grid): cutmax-only prefilter, per-pair cutoffs stay
        // the evaluator's own single source of truth; the r>1e-20 guard is
        // mandatory -- the evaluator divides by r and never coincidence-
        // guards itself

        int jnum = 0;
        for (int j = 0; j < ntotal; j++) {
          if (!(mask[j] & this->groupbit)) continue;
          const double dx = xgrid_pt[0] - x[j][0];
          const double dy = xgrid_pt[1] - x[j][1];
          const double dz = xgrid_pt[2] - x[j][2];
          const double rsq = dx * dx + dy * dy + dz * dz;
          if (rsq < this->cutmax * this->cutmax && rsq > 1e-20) jlist[jnum++] = j;
        }

        acecimpl->ace->compute_atom(ntotal, xg, typeg, jnum, jlist);
        const auto &Bs = acecimpl->ace->projections;

        // ONE writer for both styles -- only the output row/column offset
        // differs; coordinates are handled elsewhere (assign_coords_all()
        // below for LOCAL=0, the base's own assign_coords() in setup() for
        // LOCAL=1)

        if constexpr (LOCAL) {
          double *row = this->alocal[irow++];
          for (int k = 0; k < this->nvalues; k++) row[this->size_local_cols_base + k] = Bs(k);
        } else {
          double *row = this->grid[igrid];
          for (int k = 0; k < this->nvalues; k++) row[this->size_array_cols_base + k] = Bs(k);
        }
      }

  if constexpr (!LOCAL) {
    MPI_Allreduce(&this->grid[0][0], &this->gridall[0][0],
                  this->size_array_rows * this->size_array_cols, MPI_DOUBLE, MPI_SUM, this->world);
    this->assign_coords_all();
  }
}

/* ---------------------------------------------------------------------- */

template<int LOCAL>
void ComputePACEGrid<LOCAL>::compute_array()
{
  if constexpr (!LOCAL) {
    this->invoked_array = this->update->ntimestep;
    eval_grid();
  }
}

/* ---------------------------------------------------------------------- */

template<int LOCAL>
void ComputePACEGrid<LOCAL>::compute_local()
{
  if constexpr (LOCAL) {
    this->invoked_local = this->update->ntimestep;
    eval_grid();
  }
}

/* ----------------------------------------------------------------------
   memory usage
------------------------------------------------------------------------- */

template<int LOCAL>
double ComputePACEGrid<LOCAL>::memory_usage()
{
  double bytes = Base::memory_usage();
  bytes += (double) (ntotalmax + 1) * sizeof(double *);    // xg
  bytes += (double) (ntotalmax + 1) * sizeof(int);         // typeg
  bytes += (double) ntotalmax * sizeof(int);    // jlist
  return bytes;
}

/* ----------------------------------------------------------------------
   explicit instantiation of the two styles registered in compute_pace_grid.h
------------------------------------------------------------------------- */

namespace LAMMPS_NS {
template class ComputePACEGrid<0>;    // compute pace/grid       (global array)
template class ComputePACEGrid<1>;    // compute pace/grid/local (local array)
}
