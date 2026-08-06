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

#ifdef COMPUTE_CLASS
// clang-format off
// The real class ComputePACEGridKokkos<DeviceType,LOCAL> carries a comma the
// ComputeStyle macro cannot take, so each style is registered through a
// single-DeviceType-parameter wrapper (ComputePACEGridKokkosGlobal/-Local),
// mirroring compute_pace_kokkos.h.
ComputeStyle(pace/grid/kk,ComputePACEGridKokkosGlobal<LMPDeviceType>);
ComputeStyle(pace/grid/kk/device,ComputePACEGridKokkosGlobal<LMPDeviceType>);
ComputeStyle(pace/grid/kk/host,ComputePACEGridKokkosGlobal<LMPHostType>);
ComputeStyle(pace/grid/local/kk,ComputePACEGridKokkosLocal<LMPDeviceType>);
ComputeStyle(pace/grid/local/kk/device,ComputePACEGridKokkosLocal<LMPDeviceType>);
ComputeStyle(pace/grid/local/kk/host,ComputePACEGridKokkosLocal<LMPHostType>);
// clang-format on
#else

// clang-format off
#ifndef LMP_COMPUTE_PACE_GRID_KOKKOS_H
#define LMP_COMPUTE_PACE_GRID_KOKKOS_H

#include "compute_pace_grid.h"
#include "kokkos_type.h"

class SplineInterpolator;

namespace LAMMPS_NS {

// LOCAL = 0 -> compute pace/grid/kk       : global array (x,y,z + descriptors)
// LOCAL = 1 -> compute pace/grid/local/kk : local array (ix,iy,iz,x,y,z + descriptors)
//
// Descriptor-only device pipeline: no gradient/force/virial machinery (unlike
// compute_pace_kokkos, which additionally assembles RhoDB/WeightsDB/
// DerivativeDB/AssembleForce/AssembleVirial for PERATOM=0). The init-time
// device basis upload (copy_pertype/copy_splines/copy_tilde/
// pre_compute_harmonics/SplineInterpolatorKokkos) and the descriptor-only
// device helpers (evaluate_splines/ai_one_neighbor/ai_accumulate/
// project_one) are duplicated-and-adapted from compute_pace_kokkos.{h,cpp}
// (kept in sync via comments there); this file stays self-contained.
//
// Unlike compute_pace_kokkos, there is no neighbor list and no host_flag
// dispatch: every grid point is independent (no shared accumulators across
// grid points), so a single RangePolicy(chunk_size), one thread per grid
// point, races-free on every backend (host and device alike) -- matching
// the CPU eval_grid() brute-force loop in compute_pace_grid.cpp.

template<class DeviceType, int LOCAL>
class ComputePACEGridKokkos : public ComputePACEGrid<LOCAL> {
 public:
  typedef DeviceType device_type;
  typedef ArrayTypes<DeviceType> AT;
  using complex = SNAComplex<KK_FLOAT>;

  struct TagComputePACEGridMaxNeigh{};    // count-pass parallel_reduce (once per invocation)
  struct TagComputePACEGridNeigh{};
  struct TagComputePACEGridRadial{};
  struct TagComputePACEGridAi{};
  struct TagComputePACEGridConjugateAi{};
  struct TagComputePACEGridProjections{};
  struct TagComputePACEGridFill{};        // write the chunk's descriptors into the output DualView

  ComputePACEGridKokkos(class LAMMPS *, int, char **);
  ~ComputePACEGridKokkos() override;

  void init() override;
  void setup() override;
  void compute_array() override;    // LOCAL = 0
  void compute_local() override;    // LOCAL = 1
  double memory_usage() override;

// NOLINTNEXTLINE
  KOKKOS_INLINE_FUNCTION
  void operator()(TagComputePACEGridMaxNeigh, const int&, int&) const;

// NOLINTNEXTLINE
  KOKKOS_INLINE_FUNCTION
  void operator()(TagComputePACEGridNeigh, const int&) const;

// NOLINTNEXTLINE
  KOKKOS_INLINE_FUNCTION
  void operator()(TagComputePACEGridRadial, const int&) const;

// NOLINTNEXTLINE
  KOKKOS_INLINE_FUNCTION
  void operator()(TagComputePACEGridAi, const int&) const;

// NOLINTNEXTLINE
  KOKKOS_INLINE_FUNCTION
  void operator()(TagComputePACEGridConjugateAi, const int&) const;

// NOLINTNEXTLINE
  KOKKOS_INLINE_FUNCTION
  void operator()(TagComputePACEGridProjections, const int&) const;

// NOLINTNEXTLINE
  KOKKOS_INLINE_FUNCTION
  void operator()(TagComputePACEGridFill, const int&) const;

 protected:
  int chunksize, chunk_size, chunk_offset;
  int ngrid_owned;      // owned grid points on this rank (ComputeGrid::ngridlocal /
                        // ComputeGridLocal::size_local_rows -- different base member
                        // names per LOCAL, see the note in compute_pace_grid.h)
  int maxneigh;
  int nelements, lmax, nradmax, nradbase;
  int idx_ms_combs_max, idx_sph_max;
  int mu0;              // fixed grid-point species = this->elem_index
  int ntotal;            // atom->nlocal + atom->nghost, brute-force loop bound
  int xlen, ylen;        // owned-brick extents (nxhi-nxlo+1, nyhi-nylo+1), cached each call

  // box geometry, cached as KK_FLOAT scalars before kernel launch (host doubles
  // cannot be dereferenced on device) -- 10-line pattern from
  // compute_sna_grid_kokkos_impl.h:404-420
  KK_FLOAT delx_kk, dely_kk, delz_kk;
  KK_FLOAT h0, h1, h2, h3, h4, h5;
  KK_FLOAT lo0, lo1, lo2;
  KK_FLOAT cutmaxsq;

  // atom data (brute-force loop over ntotal, no neighbor list -- matches the
  // CPU eval_grid() / compute_sna_grid_kokkos precedent for grid computes)
  typename AT::t_kkfloat_1d_3_lr_randomread x;
  typename AT::t_int_1d_randomread type;
  typename AT::t_int_1d mask;

  // ---- ACE view typedefs (duplicated from compute_pace_kokkos.h) ----
  typedef Kokkos::View<int*, DeviceType> t_ace_1i;
  typedef Kokkos::View<int**, DeviceType> t_ace_2i;
  typedef Kokkos::View<int**, Kokkos::LayoutRight, DeviceType> t_ace_2i_lr;
  typedef Kokkos::View<int***, Kokkos::LayoutRight, DeviceType> t_ace_3i_lr;
  typedef Kokkos::View<KK_FLOAT*, DeviceType> t_ace_1d;
  typedef Kokkos::View<KK_FLOAT**, DeviceType> t_ace_2d;
  typedef Kokkos::View<KK_FLOAT***, DeviceType> t_ace_3d;
  typedef Kokkos::View<KK_FLOAT**[3], DeviceType> t_ace_3d3;
  typedef Kokkos::View<KK_FLOAT**[4], Kokkos::LayoutRight, DeviceType> t_ace_3d4_lr;
  typedef Kokkos::View<KK_FLOAT****, DeviceType> t_ace_4d;
  typedef Kokkos::View<complex****, DeviceType> t_ace_4c;

  // ---- output (descriptor columns only; coordinate columns are handled by
  // the CPU base, see setup()/eval_grid_device()) ----
  // LOCAL=1: alocal descriptor columns.  LOCAL=0: grid descriptor columns
  // (pre-Allreduce).  Both members always exist (single template covers both
  // LOCAL values); only the one matching LOCAL is ever allocated/used.
  DAT::ttransform_kkfloat_2d k_pace_alocal;
  typename AT::t_kkfloat_2d d_pace_alocal;
  DAT::ttransform_kkfloat_2d k_pace_grid;
  typename AT::t_kkfloat_2d d_pace_grid;

  // per-chunk descriptor accumulator (chunk_size, nvalues)
  t_ace_2d d_projections;

  // ---- A-arrays (per chunk) ----
  t_ace_3d A_rank1;
  t_ace_4c A;
  t_ace_4c A_sph;

  // ---- radial functions (per chunk); no derivatives -- grid computes only
  // ever need descriptors, never gradients ----
  t_ace_4d fr;
  t_ace_3d gr;

  // ---- spherical harmonics prefactors ----
  t_ace_1d d_idx_sph;
  t_ace_1d alm;
  t_ace_1d blm;
  t_ace_1d cl;
  t_ace_1d dl;

  // ---- short neigh list (per chunk); no d_nearest -- FillGrid never needs
  // the neighbour atom index, only mu_j/r/rhat (Radial/Ai inputs) ----
  t_ace_1i d_ncount;
  t_ace_2i d_mu;
  t_ace_2d d_rnorms;
  t_ace_3d3 d_rhats;

  // per-pair cutoff squared, indexed (mu0, mu_j); mu0 is fixed, so only row
  // mu0 is ever read, but the full (nelements, nelements) table is built
  // (matches copy_pertype's basis_set->map_bond_specifications source).
  t_ace_2d d_cutsq;

  // ---- flattened (tilde) basis tables, indexed [mu][...]; only the mu0
  // slice is ever read (mu_i is always mu0 for a grid point) ----
  t_ace_1i d_idx_ms_combs_count;
  t_ace_2i_lr d_rank;
  t_ace_2i_lr d_idx_funcs;
  t_ace_3i_lr d_mus;
  t_ace_3i_lr d_ns;
  // Precomputed l*(l+1) per (mu, func, t); see compute_pace_kokkos.h's d_func_base.
  t_ace_3i_lr d_func_base;
  t_ace_3i_lr d_ms_combs;
  t_ace_3d d_ctildes;

  // real-atom type -> element (mu) map (for neighbour atoms only; the grid
  // point's own species is the fixed scalar mu0)
  t_ace_1i d_map;

  // table builders (duplicated-and-adapted from pair_pace_kokkos /
  // compute_pace_kokkos)
  void grow(int, int);
  void copy_pertype();
  void copy_splines();
  void copy_tilde();
  void pre_compute_harmonics(int);
  void deallocate_views_of_views();

  // shared eval loop (mirrors ComputePACEGrid<LOCAL>::eval_grid()): chunked
  // device pipeline, LOCAL-specific output write handled by if constexpr
  void eval_grid_device();

// NOLINTNEXTLINE
  KOKKOS_INLINE_FUNCTION
  void evaluate_splines(const int, const int, KK_FLOAT, int, int) const;

  // decode a chunk-local flat index into this rank's owned-brick (ix,iy,iz),
  // ix fastest -- matches the CPU eval_grid() loop nesting (iz outer, iy
  // middle, ix fastest) and the row order ComputeGridLocal::assign_coords()
  // used for alocal's coordinate columns.
  KOKKOS_FORCEINLINE_FUNCTION
  void decode_indices(int, int&, int&, int&) const;

  // grid point position from owned-brick indices: orthogonal x = ix*delx,
  // triclinic via the h-matrix transform (10-line pattern from
  // compute_sna_grid_kokkos_impl.h:404-420).
  KOKKOS_FORCEINLINE_FUNCTION
  void grid2x_device(int, int, int, KK_FLOAT&, KK_FLOAT&, KK_FLOAT&) const;

  // Shared A-accumulation recursion for one bond jj (duplicated from
  // compute_pace_kokkos::ai_accumulate). UseAtomic is always false here (one
  // thread per grid point owns all of its neighbours, race-free); the
  // template parameter is kept for fidelity with the original so the two
  // copies stay easy to diff.
  template<bool UseAtomic, class GrAcc, class FrAcc>
  KOKKOS_FORCEINLINE_FUNCTION
  void ai_accumulate(int ii, int jj, int mu_j, const GrAcc& gracc, const FrAcc& fracc) const;

// NOLINTNEXTLINE
  template<bool UseAtomic>
  KOKKOS_INLINE_FUNCTION
  void ai_one_neighbor(int ii, int jj) const;

  // Accumulate one ms-combination's ctilde product into d_projections(ii,:)
  // (duplicated-and-trimmed from compute_pace_kokkos::project_one -- no
  // PERATOM branch, always writes d_projections).
  template<bool UseAtomic>
  KOKKOS_INLINE_FUNCTION
  void project_one(int ii, int mu_i, int idx_ms_combs) const;

 public:
  struct SplineInterpolatorKokkos {
    int ntot, nlut, num_of_functions;
    KK_FLOAT cutoff, deltaSplineBins, invrscalelookup, rscalelookup;

    t_ace_3d4_lr lookupTable;

    void operator=(const SplineInterpolator &spline);

    void deallocate() {
      lookupTable = t_ace_3d4_lr();
    }

    KK_FLOAT memory_usage() {
      return lookupTable.span() * sizeof(typename decltype(lookupTable)::value_type);
    }

// Vals-only variant for gr (skips derivatives -- grid computes never need them).
// NOLINTNEXTLINE
    KOKKOS_INLINE_FUNCTION
    void calcSplines(const int ii, const int jj, const KK_FLOAT r, const t_ace_3d &vals) const;

// Vals-only variant for fr (skips derivatives).
// NOLINTNEXTLINE
    KOKKOS_INLINE_FUNCTION
    void calcSplines(const int ii, const int jj, const KK_FLOAT r, const t_ace_4d &vals) const;
  };

  Kokkos::DualView<SplineInterpolatorKokkos**, DeviceType> k_splines_gk;
  Kokkos::DualView<SplineInterpolatorKokkos**, DeviceType> k_splines_rnl;
};

// ---- single-parameter registration wrappers (see note in COMPUTE_CLASS block) ----

template<class DeviceType>
class ComputePACEGridKokkosGlobal : public ComputePACEGridKokkos<DeviceType, 0> {
 public:
  ComputePACEGridKokkosGlobal(class LAMMPS *lmp, int narg, char **arg) :
      ComputePACEGridKokkos<DeviceType, 0>(lmp, narg, arg) {}
};

template<class DeviceType>
class ComputePACEGridKokkosLocal : public ComputePACEGridKokkos<DeviceType, 1> {
 public:
  ComputePACEGridKokkosLocal(class LAMMPS *lmp, int narg, char **arg) :
      ComputePACEGridKokkos<DeviceType, 1>(lmp, narg, arg) {}
};

}    // namespace LAMMPS_NS

#endif
#endif
