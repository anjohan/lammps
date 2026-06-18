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
// The real class ComputePACEKokkos<DeviceType,PERATOM> carries a comma the
// ComputeStyle macro cannot take, so each style is registered through a
// single-DeviceType-parameter wrapper (ComputePACEKokkosGlobal/-Atom).
ComputeStyle(pace/kk,ComputePACEKokkosGlobal<LMPDeviceType>);
ComputeStyle(pace/kk/device,ComputePACEKokkosGlobal<LMPDeviceType>);
ComputeStyle(pace/kk/host,ComputePACEKokkosGlobal<LMPHostType>);
ComputeStyle(pace/atom/kk,ComputePACEKokkosAtom<LMPDeviceType>);
ComputeStyle(pace/atom/kk/device,ComputePACEKokkosAtom<LMPDeviceType>);
ComputeStyle(pace/atom/kk/host,ComputePACEKokkosAtom<LMPHostType>);
// clang-format on
#else

// clang-format off
#ifndef LMP_COMPUTE_PACE_KOKKOS_H
#define LMP_COMPUTE_PACE_KOKKOS_H

#include "compute_pace.h"
#include "kokkos_type.h"

class SplineInterpolator;

namespace LAMMPS_NS {

// PERATOM = 0 -> compute pace/kk      : global array (descriptors, forces, virial)
// PERATOM = 1 -> compute pace/atom/kk : per-atom array of ACE descriptors B_{i,nu}
// Both share the on-device ACE pipeline; output assembly diverges by if constexpr.
//
// The device basis tables and the radial/harmonic scratch views are duplicated
// (and adapted) from pair_pace_kokkos, which stays untouched. See the Kokkos
// porting reference in CLAUDE.md / .github/copilot-instructions.md.
//
// CPU (host_flag=true): RangePolicy(chunk_size) — one thread per atom, serial
//   inner loops over neighbours/ms-combs, no atomics needed.
// GPU (host_flag=false): TeamPolicy over (atom,neighbour) for Radial/Ai (fills
//   CUDA warps), flat RangePolicy(chunk_size*idx_ms_combs_max) for Projections;
//   both use Kokkos::atomic_add. Dispatch selects the right policy at runtime.

template<class DeviceType, int PERATOM>
class ComputePACEKokkos : public ComputePACE<PERATOM> {
 public:
  typedef DeviceType device_type;
  typedef ArrayTypes<DeviceType> AT;
  using complex = SNAComplex<KK_FLOAT>;

  struct TagComputePACENeigh{};
  struct TagComputePACERadial{};
  struct TagComputePACEAi{};
  struct TagComputePACEConjugateAi{};
  struct TagComputePACEProjections{};       // CPU: RangePolicy(chunk_size), x=ii, no atomics
  struct TagComputePACEProjectionsFlat{};  // GPU: RangePolicy(chunk_size*idx_ms_combs_max), atomics
  struct TagComputePACECopyProjections{};
  struct TagComputePACEAiFused{};         // GPU: fused Radial+Ai for PERATOM=1 (no global fr/gr)
  // B-gradient pipeline (PERATOM=0 only): neighbours_dB = dB_{i,nu}/dr_j
  struct TagComputePACERhoDB{};           // CPU: RangePolicy(chunk_size), x=ii
  struct TagComputePACERhoDBFlat{};       // GPU: RangePolicy(chunk_size*idx_ms_combs_max), no atomics
  struct TagComputePACEWeightsDB{};        // CPU: RangePolicy(chunk_size), x=ii, no atomics
  struct TagComputePACEWeightsDBFlat{};   // GPU: RangePolicy(chunk_size*idx_ms_combs_max), atomics
  struct TagComputePACEDerivativeDB{};
  // GPU !dgradflag force-gradient Newton scatter (eliminates deep_copy(h_neighbours_dB))
  struct TagComputePACEAssembleForce{};   // GPU: TeamPolicy(atoms x neigh), atomics into d_pace_peratom

  ComputePACEKokkos(class LAMMPS *, int, char **);
  ~ComputePACEKokkos() override;

  void init() override;
  void compute_peratom() override;
  void compute_array() override;
  void setup_device_pipeline();

// NOLINTNEXTLINE
  KOKKOS_INLINE_FUNCTION
  void operator()(TagComputePACENeigh, const int&) const;

// NOLINTNEXTLINE
  KOKKOS_INLINE_FUNCTION
  void operator()(TagComputePACERadial, const int&) const;            // CPU path
// NOLINTNEXTLINE
  KOKKOS_INLINE_FUNCTION
  void operator()(TagComputePACERadial, const typename Kokkos::TeamPolicy<DeviceType,TagComputePACERadial>::member_type&) const; // GPU path

// NOLINTNEXTLINE
  KOKKOS_INLINE_FUNCTION
  void operator()(TagComputePACEAi, const int&) const;               // CPU path
// NOLINTNEXTLINE
  KOKKOS_INLINE_FUNCTION
  void operator()(TagComputePACEAi, const typename Kokkos::TeamPolicy<DeviceType,TagComputePACEAi>::member_type&) const; // GPU path

// NOLINTNEXTLINE
  KOKKOS_INLINE_FUNCTION
  void operator()(TagComputePACEConjugateAi, const int&) const;

// NOLINTNEXTLINE
  KOKKOS_INLINE_FUNCTION
  void operator()(TagComputePACEProjections, const int&) const;      // CPU: x = ii
// NOLINTNEXTLINE
  KOKKOS_INLINE_FUNCTION
  void operator()(TagComputePACEProjectionsFlat, const int&) const;  // GPU: x = flat iter

// NOLINTNEXTLINE
  KOKKOS_INLINE_FUNCTION
  void operator()(TagComputePACECopyProjections, const int&) const;

// NOLINTNEXTLINE
  KOKKOS_INLINE_FUNCTION
  void operator()(TagComputePACERhoDB, const int&) const;
// NOLINTNEXTLINE
  KOKKOS_INLINE_FUNCTION
  void operator()(TagComputePACERhoDBFlat, const int&) const;

// NOLINTNEXTLINE
  KOKKOS_INLINE_FUNCTION
  void operator()(TagComputePACEWeightsDB, const int&) const;
// NOLINTNEXTLINE
  KOKKOS_INLINE_FUNCTION
  void operator()(TagComputePACEWeightsDBFlat, const int&) const;

// NOLINTNEXTLINE
  KOKKOS_INLINE_FUNCTION
  void operator()(TagComputePACEDerivativeDB, const int&) const;                            // CPU path
// NOLINTNEXTLINE
  KOKKOS_INLINE_FUNCTION
  void operator()(TagComputePACEDerivativeDB, const typename Kokkos::TeamPolicy<DeviceType,TagComputePACEDerivativeDB>::member_type&) const; // GPU path

// NOLINTNEXTLINE
  KOKKOS_INLINE_FUNCTION
  void operator()(TagComputePACEAssembleForce, const typename Kokkos::TeamPolicy<DeviceType,TagComputePACEAssembleForce>::member_type&) const; // GPU: Newton scatter into d_pace_peratom

// NOLINTNEXTLINE
  KOKKOS_INLINE_FUNCTION
  void operator()(TagComputePACEAiFused, const typename Kokkos::TeamPolicy<DeviceType,TagComputePACEAiFused>::member_type&) const; // GPU: fused Radial+Ai for PERATOM=1

 protected:
  // Stack-array upper bounds for ai_one_neighbor_fused local spline storage.
  // init() asserts that runtime lmax+1/nradmax/nradbase fit within these.
  static constexpr int LMAXP1_MAX   = 8;
  static constexpr int NRADMAX_MAX  = 16;
  static constexpr int NRADBASE_MAX = 16;

  int host_flag;
  int chunksize, chunk_size, chunk_offset;
  int inum, maxneigh;
  int nelements, lmax, nradmax, nradbase;
  int idx_ms_combs_max, idx_sph_max;
  int total_basis_size_rankgt1_max;   // max rank>1 function count (weights_dB dim)

  // atom / neighbor data
  typename AT::t_kkfloat_1d_3_lr_randomread x;
  typename AT::t_int_1d_randomread type;
  typename AT::t_int_1d mask;
  typename AT::t_neighbors_2d d_neighbors;
  typename AT::t_int_1d_randomread d_ilist;
  typename AT::t_int_1d_randomread d_numneigh;

  // ---- ACE view typedefs (mirrors pair_pace_kokkos) ----
  typedef Kokkos::View<int*, DeviceType> t_ace_1i;
  typedef Kokkos::View<int**, DeviceType> t_ace_2i;
  typedef Kokkos::View<int**, Kokkos::LayoutRight, DeviceType> t_ace_2i_lr;
  typedef Kokkos::View<int***, Kokkos::LayoutRight, DeviceType> t_ace_3i_lr;
  typedef Kokkos::View<KK_FLOAT*, DeviceType> t_ace_1d;
  typedef Kokkos::View<KK_FLOAT**, DeviceType> t_ace_2d;
  typedef Kokkos::View<KK_FLOAT**, Kokkos::LayoutRight, DeviceType> t_ace_2d_lr;
  typedef Kokkos::View<KK_FLOAT*[3], DeviceType> t_ace_2d3;
  typedef Kokkos::View<KK_FLOAT***, DeviceType> t_ace_3d;
  typedef Kokkos::View<KK_FLOAT**[3], DeviceType> t_ace_3d3;
  typedef Kokkos::View<KK_FLOAT**[4], Kokkos::LayoutRight, DeviceType> t_ace_3d4_lr;
  typedef Kokkos::View<KK_FLOAT****, DeviceType> t_ace_4d;
  typedef Kokkos::View<complex***, DeviceType> t_ace_3c;
  typedef Kokkos::View<complex****, DeviceType> t_ace_4c;
  typedef Kokkos::View<complex*****, DeviceType> t_ace_5c;

  // ---- per-atom descriptor output (array_atom) ----
  DAT::ttransform_kkfloat_2d k_pace_atom;
  typename AT::t_kkfloat_2d d_pace_atom;

  // descriptors B_{i,nu} per chunk atom: (chunk_size, nvalues)
  t_ace_2d d_projections;

  // ---- A-arrays (per chunk) ----
  t_ace_3d A_rank1;
  t_ace_4c A;
  t_ace_4c A_sph;
  t_ace_3c A_list;
  t_ace_3c A_forward_prod;

  // ---- B-gradient scratch / output (per chunk, PERATOM=0 only) ----
  t_ace_3c dB_flatten;                 // (chunk, idx_ms_combs, rank): leave-one-out products
  t_ace_5c weights_dB;                 // (chunk, func_rankgt1, mu, idx_sph, nradmax+1)
  t_ace_4d d_neighbours_dB;            // (chunk, nvalues, maxneigh, 3): dB_{i,nu}/dr_j

  // GPU-only Newton scatter accumulator (nmax, size_peratom): replaces h_neighbours_dB
  // deep_copy + host scatter for the !dgradflag path. LayoutRight matches pace_peratom.
  t_ace_2d_lr d_pace_peratom;

  // ---- radial functions (per chunk) ----
  // fr/gr: read by Ai; allocated for both PERATOM values.
  // dfr/dgr: only read by DerivativeDB (PERATOM=0); left empty for PERATOM=1.
  // d_values/d_derivatives: flat intermediate for rnl spline (PERATOM=0 only).
  t_ace_4d fr;
  t_ace_4d dfr;
  t_ace_3d gr;
  t_ace_3d dgr;
  t_ace_3d d_values;
  t_ace_3d d_derivatives;

  // ---- spherical harmonics prefactors ----
  t_ace_1d d_idx_sph;
  t_ace_1d alm;
  t_ace_1d blm;
  t_ace_1d cl;
  t_ace_1d dl;

  // ---- short neigh list (per chunk) ----
  t_ace_1i d_ncount;
  t_ace_2d d_mu;
  t_ace_2d d_rnorms;
  t_ace_3d3 d_rhats;
  t_ace_2i d_nearest;

  // ---- per-type tables ----
  t_ace_2d d_cutsq;            // (mu_i, mu_j) pair cutoff squared, from basis_set rcut
  t_ace_1i d_ndensity;
  t_ace_1i d_npoti;
  t_ace_1d d_rho_core_cutoff;
  t_ace_1d d_drho_core_cutoff;
  t_ace_1d d_E0vals;
  t_ace_2d_lr d_wpre;
  t_ace_2d_lr d_mexp;
  t_ace_2d d_cut_in;
  t_ace_2d d_dcut_in;
  bool is_zbl;

  // ---- flattened (tilde) basis tables ----
  t_ace_1i d_idx_ms_combs_count;
  t_ace_2i_lr d_rank;
  t_ace_2i_lr d_num_ms_combs;
  t_ace_2i_lr d_idx_funcs;
  t_ace_3i_lr d_mus;
  t_ace_3i_lr d_ns;
  t_ace_3i_lr d_ls;
  // Precomputed l*(l+1) per (mu, func, t), stored in place of d_ls which gives l.
  // Saves one runtime multiply in project_one; view accessor A(ii,mu,lm+m,n-1) stays
  // layout-independent (no flat-pointer arithmetic that would break on CUDA LayoutLeft).
  t_ace_3i_lr d_func_base;
  t_ace_3i_lr d_ms_combs;
  t_ace_3d d_ctildes;
  t_ace_1i d_tbs_r1;   // per-element total_basis_size_rank1
  t_ace_1i d_tbs;      // per-element total_basis_size (rank>1)

  // type -> element (mu) map
  t_ace_1i d_map;

  // shared device pipeline: fill k_pace_atom (host-synced) with the per-local-atom
  // descriptors B_{i,nu}. Both styles use it; compute_array then assembles the
  // global bik rows on the host from the synced buffer.
  void compute_descriptors_device();

  // table builders (duplicated from pair_pace_kokkos)
  void grow(int, int);
  void copy_pertype();
  void copy_splines();
  void copy_tilde();
  void pre_compute_harmonics(int);

// NOLINTNEXTLINE
  KOKKOS_INLINE_FUNCTION
  void evaluate_splines(const int, const int, KK_FLOAT, int, int, int, int) const;

  template<class TagStyle>
  void check_team_size_for(int, int&, int);

  // Shared single-neighbour kernel body: reads fr/gr from global views.
  // Called by CPU PERATOM=0 (UseAtomic=false) and GPU PERATOM=0 (UseAtomic=true).
  template<bool UseAtomic>
  KOKKOS_INLINE_FUNCTION
  void ai_one_neighbor(int ii, int jj) const;

  // Fused variant: computes splines into thread-local storage, no fr/gr global I/O.
  // Called by CPU PERATOM=1 (UseAtomic=false) and GPU AiFused (UseAtomic=true).
  template<bool UseAtomic>
  KOKKOS_INLINE_FUNCTION
  void ai_one_neighbor_fused(int ii, int jj, int mu_i) const;

  // Shared single-ms-comb kernel body called by both the host (UseAtomic=false)
  // loop and the device (UseAtomic=true) flat-decomposition Projections overload.
  template<bool UseAtomic>
  KOKKOS_INLINE_FUNCTION
  void project_one(int ii, int mu_i, int idx_ms_combs) const;

  // Shared single-ms-comb kernel body for WeightsDB flat GPU decomposition.
  // UseAtomic=true (GPU): atomic_add into weights_dB (multiple idx_ms_combs can share
  // the same (ii, func_local, mu_t, idx_sph, n_t-1) slot). CPU keeps its inline body.
  template<bool UseAtomic>
  KOKKOS_INLINE_FUNCTION
  void weights_one(int ii, int mu_i, int tbs_r1, int idx_ms_combs) const;

  // Shared single-neighbour body for PERATOM=0 descriptor-gradient computation.
  // Writes to d_neighbours_dB(ii,...,jj,...) are disjoint per (ii,jj) so no atomics
  // are required; UseAtomic is templated for idiom symmetry only — always call <false>.
  template<bool UseAtomic>
  KOKKOS_INLINE_FUNCTION
  void derivative_one_neighbor(int ii, int jj) const;

  void deallocate_views_of_views();

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

// Existing: writes flat func_id into a 3D view (used for gr/dgr).
// NOLINTNEXTLINE
    KOKKOS_INLINE_FUNCTION
    void calcSplines(const int ii, const int jj, const KK_FLOAT r, const t_ace_3d &vals, const t_ace_3d &derivs) const;

// Vals-only variant for gr (skips writing derivatives).
// NOLINTNEXTLINE
    KOKKOS_INLINE_FUNCTION
    void calcSplines(const int ii, const int jj, const KK_FLOAT r, const t_ace_3d &vals) const;

// New: writes directly to the 4D (fr/dfr) layout, eliminating the d_values→fr reshape.
// NOLINTNEXTLINE
    KOKKOS_INLINE_FUNCTION
    void calcSplines(const int ii, const int jj, const KK_FLOAT r, const t_ace_4d &vals, const t_ace_4d &derivs) const;

// Vals-only variant for fr (skips writing dfr).
// NOLINTNEXTLINE
    KOKKOS_INLINE_FUNCTION
    void calcSplines(const int ii, const int jj, const KK_FLOAT r, const t_ace_4d &vals) const;

// Local-storage variants: write directly into raw KK_FLOAT* (no global view needed).
// Used by ai_one_neighbor_fused to avoid fr/gr global memory round-trips.
// gr_local[n] matches gr(ii,jj,n); fr_local[kk*nll+ll] matches fr(ii,jj,ll,kk).
// NOLINTNEXTLINE
    KOKKOS_INLINE_FUNCTION
    void calcSplines_local(const KK_FLOAT r, KK_FLOAT* gr_local) const;
// NOLINTNEXTLINE
    KOKKOS_INLINE_FUNCTION
    void calcSplines_local(const KK_FLOAT r, KK_FLOAT* fr_local, int nll) const;
  };

  Kokkos::DualView<SplineInterpolatorKokkos**, DeviceType> k_splines_gk;
  Kokkos::DualView<SplineInterpolatorKokkos**, DeviceType> k_splines_rnl;
  Kokkos::DualView<SplineInterpolatorKokkos**, DeviceType> k_splines_hc;
};

// ---- single-parameter registration wrappers (see note in COMPUTE_CLASS block) ----

template<class DeviceType>
class ComputePACEKokkosGlobal : public ComputePACEKokkos<DeviceType, 0> {
 public:
  ComputePACEKokkosGlobal(class LAMMPS *lmp, int narg, char **arg) :
      ComputePACEKokkos<DeviceType, 0>(lmp, narg, arg) {}
};

template<class DeviceType>
class ComputePACEKokkosAtom : public ComputePACEKokkos<DeviceType, 1> {
 public:
  ComputePACEKokkosAtom(class LAMMPS *lmp, int narg, char **arg) :
      ComputePACEKokkos<DeviceType, 1>(lmp, narg, arg) {}
};

}    // namespace LAMMPS_NS

#endif
#endif
