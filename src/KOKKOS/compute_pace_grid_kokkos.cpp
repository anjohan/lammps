// clang-format off
/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/, Sandia National Laboratories
   LAMMPS development team: developers@lammps.org

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

#include "compute_pace_grid_kokkos.h"
#include "compute_pace_impl.h"

#include "ace-evaluator/ace_radial.h"

#include "atom_kokkos.h"
#include "atom_masks.h"
#include "domain.h"
#include "error.h"
#include "memory_kokkos.h"
#include "update.h"

using namespace LAMMPS_NS;

/* ---------------------------------------------------------------------- */

template<class DeviceType, int LOCAL>
ComputePACEGridKokkos<DeviceType, LOCAL>::ComputePACEGridKokkos(LAMMPS *lmp, int narg, char **arg) :
  ComputePACEGrid<LOCAL>(lmp, narg, arg)
{
  this->kokkosable = 1;
  this->atomKK = (AtomKokkos *) this->atom;
  this->execution_space = ExecutionSpaceFromDevice<DeviceType>::space;
  this->datamask_read = X_MASK | TYPE_MASK | MASK_MASK;
  this->datamask_modify = EMPTY_MASK;

  // Optional "chunksize N" keyword, mirroring compute_pace_kokkos's own
  // convention: parsed by the CPU base, Kokkos-only. No PERATOM=0-vs-1 memory
  // pressure distinction here (no per-function gradient scratch), so a single
  // default suffices.
  chunksize = 4096;
  if (this->chunksize_arg > 0) chunksize = this->chunksize_arg;
}

/* ---------------------------------------------------------------------- */

template<class DeviceType, int LOCAL>
ComputePACEGridKokkos<DeviceType, LOCAL>::~ComputePACEGridKokkos()
{
  if (this->copymode) return;

  if constexpr (LOCAL) {
    this->memoryKK->destroy_kokkos(k_pace_alocal, this->alocal);
  } else {
    this->memoryKK->destroy_kokkos(k_pace_grid, this->grid);
    this->memory->destroy(this->gridall);
  }

  deallocate_views_of_views();
}

/* ---------------------------------------------------------------------- */

template<class DeviceType, int LOCAL>
void ComputePACEGridKokkos<DeviceType, LOCAL>::deallocate_views_of_views()
{
  // deallocate views of views in serial to prevent race conditions
  // (duplicated from compute_pace_kokkos.cpp -- keep in sync)

  if (k_splines_gk.view_host().data()) {
    for (int i = 0; i < nelements; i++) {
      for (int j = 0; j < nelements; j++) {
        k_splines_gk.view_host()(i, j).deallocate();
        k_splines_rnl.view_host()(i, j).deallocate();
      }
    }
  }
}

/* ----------------------------------------------------------------------
   memory usage of device views
------------------------------------------------------------------------- */

template<class DeviceType, int LOCAL>
double ComputePACEGridKokkos<DeviceType, LOCAL>::memory_usage()
{
  double bytes = ComputePACEGrid<LOCAL>::memory_usage();

  if constexpr (LOCAL) bytes += MemKK::memory_usage(d_pace_alocal);
  else                 bytes += MemKK::memory_usage(d_pace_grid);

  bytes += MemKK::memory_usage(d_projections);
  bytes += MemKK::memory_usage(A_rank1);
  bytes += MemKK::memory_usage(A);
  bytes += MemKK::memory_usage(A_sph);
  bytes += MemKK::memory_usage(fr);
  bytes += MemKK::memory_usage(gr);
  bytes += MemKK::memory_usage(d_idx_sph);
  bytes += MemKK::memory_usage(alm);
  bytes += MemKK::memory_usage(blm);
  bytes += MemKK::memory_usage(cl);
  bytes += MemKK::memory_usage(dl);
  bytes += MemKK::memory_usage(d_ncount);
  bytes += MemKK::memory_usage(d_mu);
  bytes += MemKK::memory_usage(d_rnorms);
  bytes += MemKK::memory_usage(d_rhats);
  bytes += MemKK::memory_usage(d_cutsq);
  bytes += MemKK::memory_usage(d_idx_ms_combs_count);
  bytes += MemKK::memory_usage(d_rank);
  bytes += MemKK::memory_usage(d_idx_funcs);
  bytes += MemKK::memory_usage(d_mus);
  bytes += MemKK::memory_usage(d_ns);
  bytes += MemKK::memory_usage(d_func_base);
  bytes += MemKK::memory_usage(d_ms_combs);
  bytes += MemKK::memory_usage(d_ctildes);
  bytes += MemKK::memory_usage(d_map);

  if (k_splines_gk.view_host().data()) {
    for (int i = 0; i < nelements; i++) {
      for (int j = 0; j < nelements; j++) {
        bytes += k_splines_gk.view_host()(i, j).memory_usage();
        bytes += k_splines_rnl.view_host()(i, j).memory_usage();
      }
    }
  }

  return bytes;
}

/* ----------------------------------------------------------------------
   init: base init() (element_type_mapping, elem_index, ...), then build the
   device basis tables from this->acecimpl->basis_set. No neighbor list
   request (grid points use the same brute-force loop as the CPU base, see
   eval_grid_device()); no stack-array size guards (LMAXP1_MAX/... in
   compute_pace_kokkos) since this pipeline never uses thread-local spline
   storage -- every loop bound here is a runtime int reading a device view.
------------------------------------------------------------------------- */

template<class DeviceType, int LOCAL>
void ComputePACEGridKokkos<DeviceType, LOCAL>::init()
{
  ComputePACEGrid<LOCAL>::init();

  auto basis_set = this->acecimpl->basis_set;

  nelements = basis_set->nelements;
  lmax = basis_set->lmax;
  nradmax = basis_set->nradmax;
  nradbase = basis_set->nradbase;

  // real-atom type -> element (mu) map (mirrors compute_pace_kokkos::init())
  MemKK::realloc_kokkos(d_map, "pace/grid:map", this->atom->ntypes + 1);
  auto h_map = Kokkos::create_mirror_view(d_map);
  for (int ik = 1; ik <= this->atom->ntypes; ik++)
    h_map(ik) = ik - 1;
  Kokkos::deep_copy(d_map, h_map);

  // spherical harmonics
  MemKK::realloc_kokkos(d_idx_sph, "pace/grid:idx_sph", (lmax + 1) * (lmax + 1));
  MemKK::realloc_kokkos(alm, "pace/grid:alm", (lmax + 1) * (lmax + 1));
  MemKK::realloc_kokkos(blm, "pace/grid:blm", (lmax + 1) * (lmax + 1));
  MemKK::realloc_kokkos(cl, "pace/grid:cl", lmax + 1);
  MemKK::realloc_kokkos(dl, "pace/grid:dl", lmax + 1);

  pre_compute_harmonics(lmax);
  copy_pertype();
  copy_splines();
  copy_tilde();
}

/* ----------------------------------------------------------------------
   setup: override wholesale (the only virtual hook ComputeGrid/
   ComputeGridLocal expose -- allocate()/deallocate()/set_grid_global()/
   set_grid_local()/grid2x() are non-virtual, see compute_pace_grid.h).
   Calls the base grid-setters directly (pattern
   compute_sna_grid_kokkos_impl.h:150-169) instead of the base's full
   setup(), so our own device-visible buffers replace the base's plain
   memory->create() ones without a redundant double-allocation (the
   sna/gaussian grid/local precedents call the full base setup() THEN
   memoryKK->create_kokkos() on top, leaking the base's first allocation --
   not copied here). The CPU base's gridlocal array (ComputeGrid, LOCAL=0
   only) is dead code (never read by eval_grid()) and is not replicated.
------------------------------------------------------------------------- */

template<class DeviceType, int LOCAL>
void ComputePACEGridKokkos<DeviceType, LOCAL>::setup()
{
  if constexpr (LOCAL) {
    if (this->gridlocal_allocated) {
      this->gridlocal_allocated = 0;
      this->memoryKK->destroy_kokkos(k_pace_alocal, this->alocal);
    }
    this->array_local = nullptr;
  } else {
    this->memoryKK->destroy_kokkos(k_pace_grid, this->grid);
    this->memory->destroy(this->gridall);
    this->array = nullptr;
  }

  this->set_grid_global();
  this->set_grid_local();

  if constexpr (LOCAL) {
    ngrid_owned = this->size_local_rows;
    if (this->nxlo <= this->nxhi && this->nylo <= this->nyhi && this->nzlo <= this->nzhi) {
      this->gridlocal_allocated = 1;
      this->memoryKK->create_kokkos(k_pace_alocal, this->alocal, this->size_local_rows,
                                    this->size_local_cols, "pace/grid/local:alocal");
      this->array_local = this->alocal;
    }
    d_pace_alocal = k_pace_alocal.template view<DeviceType>();

    // coordinate columns (ix,iy,iz,x,y,z) are filled once here, on host, by
    // the CPU base's own (non-virtual) assign_coords(); the device pipeline
    // only ever touches the descriptor columns later. sync_device() pushes
    // them to the device copy too -- a no-op on host backends (device and
    // host views alias the same memory there) but required for correctness
    // on GPU backends (Task 6), where the device buffer would otherwise
    // start with undefined coordinate columns that a later
    // modify<DeviceType>()+sync_host() round trip would clobber the host
    // copy with.
    this->assign_coords();
    k_pace_alocal.modify_host();
    k_pace_alocal.sync_device();
  } else {
    ngrid_owned = this->ngridlocal;
    this->memoryKK->create_kokkos(k_pace_grid, this->grid, this->size_array_rows,
                                  this->size_array_cols, "pace/grid:grid");
    this->memory->create(this->gridall, this->size_array_rows, this->size_array_cols,
                         "pace/grid:gridall");
    this->array = this->gridall;
    d_pace_grid = k_pace_grid.template view<DeviceType>();
  }
}

/* ----------------------------------------------------------------------
   grow: (re)size the per-chunk scratch views. Never a fixed constant for
   maxneigh -- sized from the count-pass parallel_reduce in eval_grid_device().
------------------------------------------------------------------------- */

template<class DeviceType, int LOCAL>
void ComputePACEGridKokkos<DeviceType, LOCAL>::grow(int natom, int maxneigh_in)
{
  if ((int)A.extent(0) < natom) {
    MemKK::realloc_kokkos(A_sph, "pace/grid:A_sph", natom, nelements, idx_sph_max, nradmax + 1);
    MemKK::realloc_kokkos(A, "pace/grid:A", natom, nelements, (lmax + 1) * (lmax + 1), nradmax + 1);
    MemKK::realloc_kokkos(A_rank1, "pace/grid:A_rank1", natom, nelements, nradbase);
    MemKK::realloc_kokkos(d_projections, "pace/grid:projections", natom, this->nvalues);
  }

  const bool neigh_grow = ((int)d_mu.extent(0) < natom) || ((int)d_mu.extent(1) < maxneigh_in);
  const bool fr_grow = ((int)fr.extent(0) < natom) || ((int)fr.extent(1) < maxneigh_in);

  if (fr_grow || neigh_grow) {
    MemKK::realloc_kokkos(fr, "pace/grid:fr", natom, maxneigh_in, lmax + 1, nradmax);
    MemKK::realloc_kokkos(gr, "pace/grid:gr", natom, maxneigh_in, nradbase);

    MemKK::realloc_kokkos(d_ncount, "pace/grid:ncount", natom);
    MemKK::realloc_kokkos(d_mu, "pace/grid:mu", natom, maxneigh_in);
    MemKK::realloc_kokkos(d_rhats, "pace/grid:rhats", natom, maxneigh_in);
    MemKK::realloc_kokkos(d_rnorms, "pace/grid:rnorms", natom, maxneigh_in);
  }
}

/* ----------------------------------------------------------------------
   copy_pertype: per-pair cutoff-squared table (duplicated from
   compute_pace_kokkos.cpp -- keep in sync). compute pace/grid, like
   compute_pace, only ever evaluates descriptors -- never the FS/ZBL
   embedding energy -- so the same input-compatibility guards apply (no
   embedding-parameter table is read by any kernel here).
------------------------------------------------------------------------- */

template<class DeviceType, int LOCAL>
void ComputePACEGridKokkos<DeviceType, LOCAL>::copy_pertype()
{
  auto basis_set = this->acecimpl->basis_set;

  for (int n = 0; n < nelements; n++) {
    const std::string &npoti = basis_set->map_embedding_specifications.at(n).npoti;
    if (npoti != "FinnisSinclair" && npoti != "FinnisSinclairShiftedScaled")
      this->error->all(FLERR, "Compute {}: unsupported embedding type '{}'; "
                               "supported types: FinnisSinclair, FinnisSinclairShiftedScaled",
                       this->style, npoti);
  }

  if (basis_set->radial_functions->inner_cutoff_type == "zbl")
    this->error->all(FLERR, "Compute {}: ZBL inner cutoff is not supported; "
                             "use the non-Kokkos compute style instead", this->style);

  MemKK::realloc_kokkos(d_cutsq, "pace/grid:cutsq", nelements, nelements);
  auto h_cutsq = Kokkos::create_mirror_view(d_cutsq);
  for (int mu_i = 0; mu_i < nelements; ++mu_i) {
    for (int mu_j = 0; mu_j < nelements; ++mu_j) {
      const double rcut = basis_set->map_bond_specifications.at({mu_i, mu_j}).rcut;
      h_cutsq(mu_i, mu_j) = rcut * rcut;
    }
  }
  Kokkos::deep_copy(d_cutsq, h_cutsq);
}

/* ----------------------------------------------------------------------
   copy_splines (duplicated from compute_pace_kokkos.cpp -- keep in sync)
------------------------------------------------------------------------- */

template<class DeviceType, int LOCAL>
void ComputePACEGridKokkos<DeviceType, LOCAL>::copy_splines()
{
  auto basis_set = this->acecimpl->basis_set;

  deallocate_views_of_views();

  k_splines_gk = Kokkos::DualView<SplineInterpolatorKokkos**, DeviceType>("pace/grid:splines_gk", nelements, nelements);
  k_splines_rnl = Kokkos::DualView<SplineInterpolatorKokkos**, DeviceType>("pace/grid:splines_rnl", nelements, nelements);

  ACERadialFunctions* radial_functions = dynamic_cast<ACERadialFunctions*>(basis_set->radial_functions);

  if (radial_functions == nullptr)
    this->error->all(FLERR, "Chosen radial basis style not supported by compute {}", this->style);

  for (int i = 0; i < nelements; i++) {
    for (int j = 0; j < nelements; j++) {
      k_splines_gk.view_host()(i, j) = radial_functions->splines_gk(i, j);
      k_splines_rnl.view_host()(i, j) = radial_functions->splines_rnl(i, j);
    }
  }

  k_splines_gk.modify_host();
  k_splines_rnl.modify_host();

  k_splines_gk.sync_device();
  k_splines_rnl.sync_device();
}

/* ----------------------------------------------------------------------
   copy_tilde: flattened basis-function tables (duplicated-and-trimmed from
   compute_pace_kokkos.cpp -- keep in sync). Drops d_ls/d_tbs_r1/d_tbs and
   total_basis_size_rankgt1_max: those only feed the gradient pipeline
   (rho_one/weights_one/DerivativeDB), which this descriptor-only compute
   never runs.
------------------------------------------------------------------------- */

template<class DeviceType, int LOCAL>
void ComputePACEGridKokkos<DeviceType, LOCAL>::copy_tilde()
{
  auto basis_set = this->acecimpl->basis_set;

  idx_ms_combs_max = 0;
  int total_basis_size_max = 0;

  MemKK::realloc_kokkos(d_idx_ms_combs_count, "pace/grid:idx_ms_combs_count", nelements);
  auto h_idx_ms_combs_count = Kokkos::create_mirror_view(d_idx_ms_combs_count);

  for (int mu = 0; mu < nelements; mu++) {
    int idx_ms_combs = 0;
    const int total_basis_size_rank1 = basis_set->total_basis_size_rank1[mu];
    const int total_basis_size = basis_set->total_basis_size[mu];

    ACECTildeBasisFunction *basis = basis_set->basis[mu];

    for (int func_rank1_ind = 0; func_rank1_ind < total_basis_size_rank1; ++func_rank1_ind)
      idx_ms_combs++;

    for (int idx_func = 0; idx_func < total_basis_size; ++idx_func) {
      ACECTildeBasisFunction *func = &basis[idx_func];
      for (int ms_ind = 0; ms_ind < func->num_ms_combs; ++ms_ind)
        idx_ms_combs++;
    }
    h_idx_ms_combs_count(mu) = idx_ms_combs;
    idx_ms_combs_max = MAX(idx_ms_combs_max, idx_ms_combs);
    total_basis_size_max = MAX(total_basis_size_max, total_basis_size_rank1 + total_basis_size);
  }

  Kokkos::deep_copy(d_idx_ms_combs_count, h_idx_ms_combs_count);

  MemKK::realloc_kokkos(d_rank, "pace/grid:rank", nelements, total_basis_size_max);
  MemKK::realloc_kokkos(d_idx_funcs, "pace/grid:idx_func", nelements, idx_ms_combs_max);
  MemKK::realloc_kokkos(d_mus, "pace/grid:mus", nelements, total_basis_size_max, basis_set->rankmax);
  MemKK::realloc_kokkos(d_ns, "pace/grid:ns", nelements, total_basis_size_max, basis_set->rankmax);
  MemKK::realloc_kokkos(d_func_base, "pace/grid:func_base", nelements, total_basis_size_max, basis_set->rankmax);
  MemKK::realloc_kokkos(d_ms_combs, "pace/grid:ms_combs", nelements, idx_ms_combs_max, basis_set->rankmax);
  MemKK::realloc_kokkos(d_ctildes, "pace/grid:ctildes", nelements, idx_ms_combs_max, basis_set->ndensitymax);

  auto h_rank = Kokkos::create_mirror_view(d_rank);
  auto h_idx_funcs = Kokkos::create_mirror_view(d_idx_funcs);
  auto h_mus = Kokkos::create_mirror_view(d_mus);
  auto h_ns = Kokkos::create_mirror_view(d_ns);
  auto h_func_base = Kokkos::create_mirror_view(d_func_base);
  auto h_ms_combs = Kokkos::create_mirror_view(d_ms_combs);
  auto h_ctildes = Kokkos::create_mirror_view(d_ctildes);

  for (int mu = 0; mu < nelements; mu++) {
    const int total_basis_size_rank1 = basis_set->total_basis_size_rank1[mu];
    const int total_basis_size = basis_set->total_basis_size[mu];

    ACECTildeBasisFunction *basis_rank1 = basis_set->basis_rank1[mu];
    ACECTildeBasisFunction *basis = basis_set->basis[mu];

    const int ndensity = basis_set->map_embedding_specifications.at(mu).ndensity;

    int idx_ms_combs = 0;

    for (int idx_func = 0; idx_func < total_basis_size_rank1; ++idx_func) {
      ACECTildeBasisFunction *func = &basis_rank1[idx_func];
      h_rank(mu, idx_func) = 1;
      h_mus(mu, idx_func, 0) = func->mus[0];
      h_ns(mu, idx_func, 0) = func->ns[0];

      for (int p = 0; p < ndensity; ++p)
        h_ctildes(mu, idx_ms_combs, p) = func->ctildes[p];

      h_idx_funcs(mu, idx_ms_combs) = idx_func;
      idx_ms_combs++;
    }

    for (int idx_func = 0; idx_func < total_basis_size; ++idx_func) {
      ACECTildeBasisFunction *func = &basis[idx_func];
      const int idx_func_through = total_basis_size_rank1 + idx_func;

      const int rank = h_rank(mu, idx_func_through) = func->rank;
      for (int t = 0; t < rank; t++) {
        h_mus(mu, idx_func_through, t) = func->mus[t];
        h_ns(mu, idx_func_through, t) = func->ns[t];
        const int l_t = func->ls[t];
        h_func_base(mu, idx_func_through, t) = l_t * (l_t + 1);
      }

      for (int ms_ind = 0; ms_ind < func->num_ms_combs; ++ms_ind) {
        auto ms = &func->ms_combs[ms_ind * rank];
        for (int t = 0; t < rank; t++)
          h_ms_combs(mu, idx_ms_combs, t) = ms[t];

        for (int p = 0; p < ndensity; ++p)
          h_ctildes(mu, idx_ms_combs, p) = func->ctildes[ms_ind * ndensity + p];

        h_idx_funcs(mu, idx_ms_combs) = idx_func_through;
        idx_ms_combs++;
      }
    }
  }

  Kokkos::deep_copy(d_rank, h_rank);
  Kokkos::deep_copy(d_idx_funcs, h_idx_funcs);
  Kokkos::deep_copy(d_mus, h_mus);
  Kokkos::deep_copy(d_ns, h_ns);
  Kokkos::deep_copy(d_func_base, h_func_base);
  Kokkos::deep_copy(d_ms_combs, h_ms_combs);
  Kokkos::deep_copy(d_ctildes, h_ctildes);
}

/* ----------------------------------------------------------------------
   pre_compute_harmonics (duplicated from compute_pace_kokkos.cpp -- keep in sync)
------------------------------------------------------------------------- */

template<class DeviceType, int LOCAL>
void ComputePACEGridKokkos<DeviceType, LOCAL>::pre_compute_harmonics(int lmax_in)
{
  auto h_idx_sph = Kokkos::create_mirror_view(d_idx_sph);
  auto h_alm = Kokkos::create_mirror_view(alm);
  auto h_blm = Kokkos::create_mirror_view(blm);
  auto h_cl = Kokkos::create_mirror_view(cl);
  auto h_dl = Kokkos::create_mirror_view(dl);

  Kokkos::deep_copy(h_idx_sph, -1);

  int idx_sph = 0;
  for (int m = 0; m <= lmax_in; m++) {
    const double msq = m * m;
    for (int l = m; l <= lmax_in; l++) {
      const int idx = l * (l + 1) + m;
      h_idx_sph(idx) = idx_sph;

      double a = 0.0;
      double b = 0.0;

      if (l > 1 && l != m) {
        const double lsq = l * l;
        const double ld = 2 * l;
        const double l1 = (4 * lsq - 1);
        const double l2 = lsq - ld + 1;

        a = sqrt((double(l1)) / (double(lsq - msq)));
        b = -sqrt((double(l2 - msq)) / (double(4 * l2 - 1)));
      }
      h_alm(idx_sph) = a;
      h_blm(idx_sph) = b;
      idx_sph++;
    }
  }
  idx_sph_max = idx_sph;

  for (int l = 1; l <= lmax_in; l++) {
    h_cl(l) = -sqrt(1.0 + 0.5 / (double(l)));
    h_dl(l) = sqrt(double(2 * (l - 1) + 3));
  }

  Kokkos::deep_copy(d_idx_sph, h_idx_sph);
  Kokkos::deep_copy(alm, h_alm);
  Kokkos::deep_copy(blm, h_blm);
  Kokkos::deep_copy(cl, h_cl);
  Kokkos::deep_copy(dl, h_dl);
}

/* ----------------------------------------------------------------------
   evaluate_splines: always the no-derivative path (grid computes only ever
   need descriptors) -- unlike compute_pace_kokkos::evaluate_splines, no
   PERATOM branch.
------------------------------------------------------------------------- */

template<class DeviceType, int LOCAL>
// NOLINTNEXTLINE
KOKKOS_INLINE_FUNCTION
void ComputePACEGridKokkos<DeviceType, LOCAL>::evaluate_splines(const int ii, const int jj, KK_FLOAT r,
                                                                  int mu_i, int mu_j) const
{
  auto &spline_gk = k_splines_gk.template view<DeviceType>()(mu_i, mu_j);
  auto &spline_rnl = k_splines_rnl.template view<DeviceType>()(mu_i, mu_j);
  spline_gk.calcSplines(ii, jj, r, gr);
  spline_rnl.calcSplines(ii, jj, r, fr);
}

/* ---------------------------------------------------------------------- */

template<class DeviceType, int LOCAL>
void ComputePACEGridKokkos<DeviceType, LOCAL>::SplineInterpolatorKokkos::operator=(const SplineInterpolator &spline)
{
  cutoff = spline.cutoff;
  deltaSplineBins = spline.deltaSplineBins;
  ntot = spline.ntot;
  nlut = spline.nlut;
  invrscalelookup = spline.invrscalelookup;
  rscalelookup = spline.rscalelookup;
  num_of_functions = spline.num_of_functions;

  lookupTable = t_ace_3d4_lr("lookupTable", ntot + 1, num_of_functions);
  auto h_lookupTable = Kokkos::create_mirror_view(lookupTable);
  for (int i = 0; i < ntot + 1; i++)
    for (int j = 0; j < num_of_functions; j++)
      for (int k = 0; k < 4; k++)
        h_lookupTable(i, j, k) = spline.lookupTable(i, j, k);
  Kokkos::deep_copy(lookupTable, h_lookupTable);
}

/* ---------------------------------------------------------------------- */

template<class DeviceType, int LOCAL>
// NOLINTNEXTLINE
KOKKOS_INLINE_FUNCTION
void ComputePACEGridKokkos<DeviceType, LOCAL>::SplineInterpolatorKokkos::calcSplines(
  const int ii, const int jj, const KK_FLOAT r, const t_ace_3d &vals) const
{
  KK_FLOAT wl, wl2, wl3;
  KK_FLOAT c[4];
  KK_FLOAT x = r * rscalelookup;
  int nl = static_cast<int>(floor(x));

  if (nl <= 0) Kokkos::abort("Encountered very small distance. Stopping.");

  if (nl < nlut) {
    wl = x - KK_FLOAT(nl); wl2 = wl * wl; wl3 = wl2 * wl;
    for (int func_id = 0; func_id < num_of_functions; func_id++) {
      for (int idx = 0; idx < 4; idx++) c[idx] = lookupTable(nl, func_id, idx);
      vals(ii, jj, func_id) = c[0] + c[1] * wl + c[2] * wl2 + c[3] * wl3;
    }
  } else {
    for (int func_id = 0; func_id < num_of_functions; func_id++)
      vals(ii, jj, func_id) = 0.0;
  }
}

/* ---------------------------------------------------------------------- */

template<class DeviceType, int LOCAL>
// NOLINTNEXTLINE
KOKKOS_INLINE_FUNCTION
void ComputePACEGridKokkos<DeviceType, LOCAL>::SplineInterpolatorKokkos::calcSplines(
  const int ii, const int jj, const KK_FLOAT r, const t_ace_4d &vals4d) const
{
  KK_FLOAT wl, wl2, wl3;
  KK_FLOAT c[4];
  KK_FLOAT x = r * rscalelookup;
  int nl = static_cast<int>(floor(x));

  if (nl <= 0) Kokkos::abort("Encountered very small distance. Stopping.");

  const int nll = vals4d.extent(2);  // lmax+1
  const int nkk = vals4d.extent(3);  // nradmax

  if (nl < nlut) {
    wl = x - KK_FLOAT(nl); wl2 = wl * wl; wl3 = wl2 * wl;
    int func_id = 0;
    for (int kk = 0; kk < nkk; kk++) {
      for (int ll = 0; ll < nll; ll++, func_id++) {
        for (int idx = 0; idx < 4; idx++) c[idx] = lookupTable(nl, func_id, idx);
        vals4d(ii, jj, ll, kk) = c[0] + c[1] * wl + c[2] * wl2 + c[3] * wl3;
      }
    }
  } else {
    for (int kk = 0; kk < nkk; kk++)
      for (int ll = 0; ll < nll; ll++)
        vals4d(ii, jj, ll, kk) = 0.0;
  }
}

/* ----------------------------------------------------------------------
   decode_indices: chunk-local flat index -> this rank's owned-brick
   (ix,iy,iz), ix fastest.
------------------------------------------------------------------------- */

template<class DeviceType, int LOCAL>
// NOLINTNEXTLINE
KOKKOS_FORCEINLINE_FUNCTION
void ComputePACEGridKokkos<DeviceType, LOCAL>::decode_indices(int icnk, int &ix, int &iy, int &iz) const
{
  iz = icnk / (xlen * ylen);
  int i2 = icnk - iz * xlen * ylen;
  iy = i2 / xlen;
  ix = i2 % xlen;
  iz += this->nzlo;
  iy += this->nylo;
  ix += this->nxlo;
}

/* ----------------------------------------------------------------------
   grid2x_device: owned-brick indices -> grid point position. Orthogonal:
   x = ix*delx (no boxlo -- matches ComputeGrid::grid2x / init()'s boxlo==0
   guard). Triclinic: h-matrix transform (compute_sna_grid_kokkos_impl.h:
   404-420).
------------------------------------------------------------------------- */

template<class DeviceType, int LOCAL>
// NOLINTNEXTLINE
KOKKOS_FORCEINLINE_FUNCTION
void ComputePACEGridKokkos<DeviceType, LOCAL>::grid2x_device(int ix, int iy, int iz,
                                                               KK_FLOAT &xtmp, KK_FLOAT &ytmp, KK_FLOAT &ztmp) const
{
  const KK_FLOAT xg0 = ix * delx_kk;
  const KK_FLOAT xg1 = iy * dely_kk;
  const KK_FLOAT xg2 = iz * delz_kk;
  if (this->triclinic) {
    xtmp = h0 * xg0 + h5 * xg1 + h4 * xg2 + lo0;
    ytmp = h1 * xg1 + h3 * xg2 + lo1;
    ztmp = h2 * xg2 + lo2;
  } else {
    xtmp = xg0;
    ytmp = xg1;
    ztmp = xg2;
  }
}

/* ----------------------------------------------------------------------
   MaxNeigh: count-pass parallel_reduce, once per invocation, BEFORE
   chunking -- sizes maxneigh (never a fixed constant, unlike the sna/grid
   precedent's max_neighs=100). Superset criterion: cutmax only (per-pair
   cutoffs are always <= cutmax), so ncount computed here is a safe upper
   bound for the Neigh stage's per-pair-filtered short list.
------------------------------------------------------------------------- */

template<class DeviceType, int LOCAL>
// NOLINTNEXTLINE
KOKKOS_INLINE_FUNCTION
void ComputePACEGridKokkos<DeviceType, LOCAL>::operator()(TagComputePACEGridMaxNeigh, const int& icnk, int& count_max) const
{
  int ix, iy, iz;
  decode_indices(icnk, ix, iy, iz);
  KK_FLOAT xtmp, ytmp, ztmp;
  grid2x_device(ix, iy, iz, xtmp, ytmp, ztmp);

  int count = 0;
  for (int j = 0; j < ntotal; j++) {
    if (!(mask(j) & this->groupbit)) continue;
    const KK_FLOAT dx = xtmp - x(j, 0);
    const KK_FLOAT dy = ytmp - x(j, 1);
    const KK_FLOAT dz = ztmp - x(j, 2);
    const KK_FLOAT rsq = dx * dx + dy * dy + dz * dz;
    if (rsq < cutmaxsq && rsq > 1e-20) count++;
  }
  if (count_max < count) count_max = count;
}

/* ----------------------------------------------------------------------
   Neigh: build the per-grid-point short neighbour list, exactly filtered by
   the per-pair cutoff d_cutsq(mu0, mu_j) (mu0 fixed) + the mandatory r>~0
   guard (grid points can coincide exactly with an atom -- see the
   coincident-point harness case). One thread per grid point (no
   TeamPolicy/atomics): the whole pipeline below reuses this design.
------------------------------------------------------------------------- */

template<class DeviceType, int LOCAL>
// NOLINTNEXTLINE
KOKKOS_INLINE_FUNCTION
void ComputePACEGridKokkos<DeviceType, LOCAL>::operator()(TagComputePACEGridNeigh, const int& ii) const
{
  const int icnk = ii + chunk_offset;
  int ix, iy, iz;
  decode_indices(icnk, ix, iy, iz);
  KK_FLOAT xtmp, ytmp, ztmp;
  grid2x_device(ix, iy, iz, xtmp, ytmp, ztmp);

  int ncount = 0;
  for (int j = 0; j < ntotal; j++) {
    if (!(mask(j) & this->groupbit)) continue;
    const int mu_j = d_map(type(j));
    const KK_FLOAT delx = xtmp - x(j, 0);
    const KK_FLOAT dely = ytmp - x(j, 1);
    const KK_FLOAT delz = ztmp - x(j, 2);
    const KK_FLOAT rsq = delx * delx + dely * dely + delz * delz;
    if (rsq < d_cutsq(mu0, mu_j) && rsq > 1e-20) {
      const KK_FLOAT r = Kokkos::sqrt(rsq);
      const KK_FLOAT rinv = 1.0 / r;
      d_mu(ii, ncount) = mu_j;
      d_rnorms(ii, ncount) = r;
      d_rhats(ii, ncount, 0) = -delx * rinv;
      d_rhats(ii, ncount, 1) = -dely * rinv;
      d_rhats(ii, ncount, 2) = -delz * rinv;
      ncount++;
    }
  }
  d_ncount(ii) = ncount;
}

/* ----------------------------------------------------------------------
   Radial: radial functions for every short-list neighbour (mu_i = mu0).
------------------------------------------------------------------------- */

template<class DeviceType, int LOCAL>
// NOLINTNEXTLINE
KOKKOS_INLINE_FUNCTION
void ComputePACEGridKokkos<DeviceType, LOCAL>::operator()(TagComputePACEGridRadial, const int& ii) const
{
  const int ncount = d_ncount(ii);
  for (int jj = 0; jj < ncount; jj++) {
    const KK_FLOAT r_norm = d_rnorms(ii, jj);
    const int mu_j = d_mu(ii, jj);
    evaluate_splines(ii, jj, r_norm, mu0, mu_j);
  }
}

/* ----------------------------------------------------------------------
   ai_accumulate (duplicated from compute_pace_kokkos.cpp -- keep in sync;
   UseAtomic is always instantiated false here).
------------------------------------------------------------------------- */

template<class DeviceType, int LOCAL>
template<bool UseAtomic, class GrAcc, class FrAcc>
// NOLINTNEXTLINE
KOKKOS_FORCEINLINE_FUNCTION
void ComputePACEGridKokkos<DeviceType, LOCAL>::ai_accumulate(int ii, int jj, int mu_j,
                                                               const GrAcc& gracc, const FrAcc& fracc) const
{
  // rank = 1
  for (int n = 0; n < nradbase; n++) {
    const KK_FLOAT val = gracc(n) * Y00;
    if constexpr (UseAtomic) Kokkos::atomic_add(&A_rank1(ii, mu_j, n), val);
    else                     A_rank1(ii, mu_j, n) += val;
  }

  // rank > 1: plm/ylm recurrence, accumulate A_sph
  // requires |r_hat| = 1 and -1 <= rz <= 1

  complex ylm, phase;
  complex phasem;

  const KK_FLOAT rx = d_rhats(ii, jj, 0);
  const KK_FLOAT ry = d_rhats(ii, jj, 1);
  const KK_FLOAT rz = d_rhats(ii, jj, 2);

  phase.re = rx;
  phase.im = ry;

  KK_FLOAT plm_idx, plm_idx1, plm_idx2;
  plm_idx = plm_idx1 = plm_idx2 = 0.0;

  int idx_sph = 0;

  // m = 0
  for (int l = 0; l <= lmax; l++) {
    if (l == 0)      plm_idx = Y00;
    else if (l == 1) plm_idx = Y00 * sq3 * rz;
    else             plm_idx = alm(idx_sph) * (rz * plm_idx1 + blm(idx_sph) * plm_idx2);

    ylm.re = plm_idx; ylm.im = 0.0;

    for (int n = 0; n < nradmax; n++) {
      if constexpr (UseAtomic) {
        Kokkos::atomic_add(&A_sph(ii, mu_j, idx_sph, n).re, fracc(l, n) * ylm.re);
        Kokkos::atomic_add(&A_sph(ii, mu_j, idx_sph, n).im, fracc(l, n) * ylm.im);
      } else {
        A_sph(ii, mu_j, idx_sph, n).re += fracc(l, n) * ylm.re;
        A_sph(ii, mu_j, idx_sph, n).im += fracc(l, n) * ylm.im;
      }
    }
    plm_idx2 = plm_idx1; plm_idx1 = plm_idx; idx_sph++;
  }

  plm_idx = plm_idx1 = plm_idx2 = 0.0;

  // m = 1
  for (int l = 1; l <= lmax; l++) {
    if (l == 1)      plm_idx = -sq3o2 * Y00;
    else if (l == 2) { const KK_FLOAT t = dl(l) * plm_idx1; plm_idx = t * rz; }
    else             plm_idx = alm(idx_sph) * (rz * plm_idx1 + blm(idx_sph) * plm_idx2);

    ylm = phase * plm_idx;

    for (int n = 0; n < nradmax; n++) {
      if constexpr (UseAtomic) {
        Kokkos::atomic_add(&A_sph(ii, mu_j, idx_sph, n).re, fracc(l, n) * ylm.re);
        Kokkos::atomic_add(&A_sph(ii, mu_j, idx_sph, n).im, fracc(l, n) * ylm.im);
      } else {
        A_sph(ii, mu_j, idx_sph, n).re += fracc(l, n) * ylm.re;
        A_sph(ii, mu_j, idx_sph, n).im += fracc(l, n) * ylm.im;
      }
    }
    plm_idx2 = plm_idx1; plm_idx1 = plm_idx; idx_sph++;
  }

  plm_idx = plm_idx1 = plm_idx2 = 0.0;
  KK_FLOAT plm_mm1_mm1 = -sq3o2 * Y00;

  // m > 1
  phasem = phase;
  for (int m = 2; m <= lmax; m++) {
    phasem = phasem * phase;

    for (int l = m; l <= lmax; l++) {
      if (l == m)          { plm_idx = cl(l) * plm_mm1_mm1; plm_mm1_mm1 = plm_idx; }
      else if (l == (m+1)) { const KK_FLOAT t = dl(l) * plm_mm1_mm1; plm_idx = t * rz; }
      else                   plm_idx = alm(idx_sph) * (rz * plm_idx1 + blm(idx_sph) * plm_idx2);

      ylm.re = phasem.re * plm_idx;
      ylm.im = phasem.im * plm_idx;

      for (int n = 0; n < nradmax; n++) {
        if constexpr (UseAtomic) {
          Kokkos::atomic_add(&A_sph(ii, mu_j, idx_sph, n).re, fracc(l, n) * ylm.re);
          Kokkos::atomic_add(&A_sph(ii, mu_j, idx_sph, n).im, fracc(l, n) * ylm.im);
        } else {
          A_sph(ii, mu_j, idx_sph, n).re += fracc(l, n) * ylm.re;
          A_sph(ii, mu_j, idx_sph, n).im += fracc(l, n) * ylm.im;
        }
      }
      plm_idx2 = plm_idx1; plm_idx1 = plm_idx; idx_sph++;
    }
  }
}

/* ----------------------------------------------------------------------
   ai_one_neighbor (duplicated from compute_pace_kokkos.cpp -- keep in sync)
------------------------------------------------------------------------- */

template<class DeviceType, int LOCAL>
template<bool UseAtomic>
// NOLINTNEXTLINE
KOKKOS_INLINE_FUNCTION
void ComputePACEGridKokkos<DeviceType, LOCAL>::ai_one_neighbor(int ii, int jj) const
{
  const int mu_j = d_mu(ii, jj);

  ai_accumulate<UseAtomic>(ii, jj, mu_j,
    [&](const int n)              { return gr(ii, jj, n); },
    [&](const int l, const int n) { return fr(ii, jj, l, n); });
}

/* ----------------------------------------------------------------------
   Ai: one thread owns grid point ii, loops all of its neighbours.
------------------------------------------------------------------------- */

template<class DeviceType, int LOCAL>
// NOLINTNEXTLINE
KOKKOS_INLINE_FUNCTION
void ComputePACEGridKokkos<DeviceType, LOCAL>::operator()(TagComputePACEGridAi, const int& ii) const
{
  const int ncount = d_ncount(ii);
  for (int jj = 0; jj < ncount; jj++)
    ai_one_neighbor<false>(ii, jj);
}

/* ----------------------------------------------------------------------
   ConjugateAi: expand A_sph (half-triangle, m>=0) -> A (all m) (duplicated
   from compute_pace_kokkos.cpp -- keep in sync).
------------------------------------------------------------------------- */

template<class DeviceType, int LOCAL>
// NOLINTNEXTLINE
KOKKOS_INLINE_FUNCTION
void ComputePACEGridKokkos<DeviceType, LOCAL>::operator()(TagComputePACEGridConjugateAi, const int& ii) const
{
  for (int mu_j = 0; mu_j < nelements; mu_j++) {

    int idx_sph = 0;
    for (int m = 0; m <= lmax; m++) {
      for (int l = m; l <= lmax; l++) {
        const int idx = l * (l + 1) + m;
        for (int n = 0; n < nradmax; n++)
          A(ii, mu_j, idx, n) = A_sph(ii, mu_j, idx_sph, n);
        idx_sph++;
      }
    }

    for (int l = 0; l <= lmax; l++) {
      for (int m = 1; m <= l; m++) {
        const int idx = l * (l + 1) + m;   // (l, m)
        const int idxm = l * (l + 1) - m;  // (l, -m)
        const int idx_sph2 = d_idx_sph(idx);
        const int factor = m % 2 == 0 ? 1 : -1;
        for (int n = 0; n < nradmax; n++)
          A(ii, mu_j, idxm, n) = A_sph(ii, mu_j, idx_sph2, n).conj() * (KK_FLOAT)factor;
      }
    }
  }
}

/* ----------------------------------------------------------------------
   project_one (duplicated-and-trimmed from compute_pace_kokkos::project_one
   -- keep in sync; no PERATOM branch, always writes d_projections).
------------------------------------------------------------------------- */

template<class DeviceType, int LOCAL>
template<bool UseAtomic>
// NOLINTNEXTLINE
KOKKOS_INLINE_FUNCTION
void ComputePACEGridKokkos<DeviceType, LOCAL>::project_one(int ii, int mu_i, int idx_ms_combs) const
{
  const int idx_func = d_idx_funcs(mu_i, idx_ms_combs);
  const int rank = d_rank(mu_i, idx_func);

  KK_FLOAT val;
  if (rank == 1) {
    const int mu = d_mus(mu_i, idx_func, 0);
    const int n = d_ns(mu_i, idx_func, 0);
    val = d_ctildes(mu_i, idx_ms_combs, 0) * A_rank1(ii, mu, n - 1);
  } else {
    complex Bprod = complex::one();
    for (int t = 0; t < rank; t++) {
      const int mu = d_mus(mu_i, idx_func, t);
      const int n  = d_ns(mu_i, idx_func, t);
      const int lm = d_func_base(mu_i, idx_func, t);
      const int m  = d_ms_combs(mu_i, idx_ms_combs, t);
      Bprod = Bprod * A(ii, mu, lm + m, n - 1);
    }
    val = Bprod.re * d_ctildes(mu_i, idx_ms_combs, 0);
  }

  if constexpr (UseAtomic) Kokkos::atomic_add(&d_projections(ii, idx_func), val);
  else                     d_projections(ii, idx_func) += val;
}

/* ----------------------------------------------------------------------
   Projections: one thread per grid point, loops all of mu0's ms-combinations.
------------------------------------------------------------------------- */

template<class DeviceType, int LOCAL>
// NOLINTNEXTLINE
KOKKOS_INLINE_FUNCTION
void ComputePACEGridKokkos<DeviceType, LOCAL>::operator()(TagComputePACEGridProjections, const int& ii) const
{
  const int count = d_idx_ms_combs_count(mu0);
  for (int idx_ms_combs = 0; idx_ms_combs < count; idx_ms_combs++)
    project_one<false>(ii, mu0, idx_ms_combs);
}

/* ----------------------------------------------------------------------
   Fill: write this chunk's descriptors into the output DualView
   unconditionally (from the per-chunk zero-initialized d_projections) -- a
   zero-neighbour grid point must produce bitwise-zero descriptors even when
   a previous chunk left data in these reused scratch views.
------------------------------------------------------------------------- */

template<class DeviceType, int LOCAL>
// NOLINTNEXTLINE
KOKKOS_INLINE_FUNCTION
void ComputePACEGridKokkos<DeviceType, LOCAL>::operator()(TagComputePACEGridFill, const int& ii) const
{
  const int icnk = ii + chunk_offset;
  if constexpr (LOCAL) {
    for (int k = 0; k < this->nvalues; k++)
      d_pace_alocal(icnk, this->size_local_cols_base + k) = d_projections(ii, k);
  } else {
    int ix, iy, iz;
    decode_indices(icnk, ix, iy, iz);
    const int igrid = iz * (this->nx * this->ny) + iy * this->nx + ix;
    for (int k = 0; k < this->nvalues; k++)
      d_pace_grid(igrid, this->size_array_cols_base + k) = d_projections(ii, k);
  }
}

/* ----------------------------------------------------------------------
   eval_grid_device: shared chunked device pipeline (mirrors
   ComputePACEGrid<LOCAL>::eval_grid()'s brute-force-per-grid-point loop,
   with output-write divergence handled the same way, via if constexpr).
------------------------------------------------------------------------- */

template<class DeviceType, int LOCAL>
void ComputePACEGridKokkos<DeviceType, LOCAL>::eval_grid_device()
{
  mu0 = this->elem_index;

  this->atomKK->sync(this->execution_space, X_MASK | TYPE_MASK | MASK_MASK);
  x = this->atomKK->k_x.template view<DeviceType>();
  type = this->atomKK->k_type.template view<DeviceType>();
  mask = this->atomKK->k_mask.template view<DeviceType>();

  ntotal = this->atom->nlocal + this->atom->nghost;

  xlen = this->nxhi - this->nxlo + 1;
  ylen = this->nyhi - this->nylo + 1;

  delx_kk = static_cast<KK_FLOAT>(this->delx);
  dely_kk = static_cast<KK_FLOAT>(this->dely);
  delz_kk = static_cast<KK_FLOAT>(this->delz);

  if (this->triclinic) {
    h0 = static_cast<KK_FLOAT>(this->domain->h[0]);
    h1 = static_cast<KK_FLOAT>(this->domain->h[1]);
    h2 = static_cast<KK_FLOAT>(this->domain->h[2]);
    h3 = static_cast<KK_FLOAT>(this->domain->h[3]);
    h4 = static_cast<KK_FLOAT>(this->domain->h[4]);
    h5 = static_cast<KK_FLOAT>(this->domain->h[5]);
    lo0 = static_cast<KK_FLOAT>(this->domain->boxlo[0]);
    lo1 = static_cast<KK_FLOAT>(this->domain->boxlo[1]);
    lo2 = static_cast<KK_FLOAT>(this->domain->boxlo[2]);
  }

  cutmaxsq = static_cast<KK_FLOAT>(this->cutmax * this->cutmax);

  // full-array zero (matches the CPU eval_grid() memset): rows not owned by
  // this rank must stay exactly zero going into the MPI_Allreduce sum.
  if constexpr (!LOCAL) Kokkos::deep_copy(d_pace_grid, (KK_FLOAT) 0.0);

  this->copymode = 1;

  // ONCE per invocation, BEFORE chunking: size the chunk-local short-list
  // views from a count pass over every owned grid point.
  maxneigh = 0;
  if (ngrid_owned > 0) {
    Kokkos::parallel_reduce("pace/grid:maxneigh",
      Kokkos::RangePolicy<DeviceType, TagComputePACEGridMaxNeigh>(0, ngrid_owned),
      *this, Kokkos::Max<int>(maxneigh));
  }

  chunk_size = MIN(chunksize, ngrid_owned);
  chunk_offset = 0;
  grow(chunk_size, maxneigh);

  while (chunk_offset < ngrid_owned) {   // chunk up loop to bound memory

    if (chunk_size > ngrid_owned - chunk_offset)
      chunk_size = ngrid_owned - chunk_offset;

    Kokkos::deep_copy(A_rank1, 0.0);
    Kokkos::deep_copy(A_sph, 0.0);
    Kokkos::deep_copy(d_projections, 0.0);

    Kokkos::parallel_for("pace/grid:Neigh",
      Kokkos::RangePolicy<DeviceType, TagComputePACEGridNeigh>(0, chunk_size), *this);
    Kokkos::parallel_for("pace/grid:Radial",
      Kokkos::RangePolicy<DeviceType, TagComputePACEGridRadial>(0, chunk_size), *this);
    Kokkos::parallel_for("pace/grid:Ai",
      Kokkos::RangePolicy<DeviceType, TagComputePACEGridAi>(0, chunk_size), *this);
    Kokkos::parallel_for("pace/grid:ConjAi",
      Kokkos::RangePolicy<DeviceType, TagComputePACEGridConjugateAi>(0, chunk_size), *this);
    Kokkos::parallel_for("pace/grid:Projections",
      Kokkos::RangePolicy<DeviceType, TagComputePACEGridProjections>(0, chunk_size), *this);
    Kokkos::parallel_for("pace/grid:Fill",
      Kokkos::RangePolicy<DeviceType, TagComputePACEGridFill>(0, chunk_size), *this);

    chunk_offset += chunk_size;
  }

  this->copymode = 0;

  if constexpr (LOCAL) {
    k_pace_alocal.template modify<DeviceType>();
    k_pace_alocal.sync_host();
  } else {
    k_pace_grid.template modify<DeviceType>();
    k_pace_grid.sync_host();

    MPI_Allreduce(&this->grid[0][0], &this->gridall[0][0],
                  this->size_array_rows * this->size_array_cols, MPI_DOUBLE, MPI_SUM, this->world);
    this->assign_coords_all();
  }
}

/* ---------------------------------------------------------------------- */

template<class DeviceType, int LOCAL>
void ComputePACEGridKokkos<DeviceType, LOCAL>::compute_array()
{
  if constexpr (!LOCAL) {
    this->invoked_array = this->update->ntimestep;
    eval_grid_device();
  }
}

/* ---------------------------------------------------------------------- */

template<class DeviceType, int LOCAL>
void ComputePACEGridKokkos<DeviceType, LOCAL>::compute_local()
{
  if constexpr (LOCAL) {
    this->invoked_local = this->update->ntimestep;
    eval_grid_device();
  }
}

/* ----------------------------------------------------------------------
   explicit instantiation
------------------------------------------------------------------------- */

namespace LAMMPS_NS {
template class ComputePACEGridKokkos<LMPDeviceType, 0>;
template class ComputePACEGridKokkos<LMPDeviceType, 1>;
template class ComputePACEGridKokkosGlobal<LMPDeviceType>;
template class ComputePACEGridKokkosLocal<LMPDeviceType>;
#ifdef LMP_KOKKOS_GPU
template class ComputePACEGridKokkos<LMPHostType, 0>;
template class ComputePACEGridKokkos<LMPHostType, 1>;
template class ComputePACEGridKokkosGlobal<LMPHostType>;
template class ComputePACEGridKokkosLocal<LMPHostType>;
#endif
}
