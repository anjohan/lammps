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
  // GPU !dgradflag fused: accumulates gradient in stack registers and scatters directly
  // into d_pace_peratom, eliminating the d_neighbours_dB round-trip.
  struct TagComputePACEDerivativeDBFused{};

  // Device-side global-array assembly (PERATOM=0, !host_flag && !dgradflag).
  // Replaces the host loops in compute_array() that assemble bik rows, force rows,
  // and virial rows from pace_peratom after the chunk loop.
  struct TagComputePACEAssembleBik{};     // bik rows: per-chunk after Projections
  struct TagComputePACEAssembleForce{};   // force rows + last-col forces: post-loop
  struct TagComputePACEAssembleVirial{};  // virial rows: post-loop (replaces dbdotr_compute)

  ComputePACEKokkos(class LAMMPS *, int, char **);
  ~ComputePACEKokkos() override;

  void init() override;
  void compute_peratom() override;
  void compute_array() override;
  void setup_device_pipeline();
  double memory_usage() override;

// NOLINTNEXTLINE
  KOKKOS_INLINE_FUNCTION
  void operator()(TagComputePACENeigh, const int&) const;             // CPU path
// NOLINTNEXTLINE
  KOKKOS_INLINE_FUNCTION
  void operator()(TagComputePACENeigh, const typename Kokkos::TeamPolicy<DeviceType,TagComputePACENeigh>::member_type&) const; // GPU path

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
  void operator()(TagComputePACEDerivativeDBFused, const typename Kokkos::TeamPolicy<DeviceType,TagComputePACEDerivativeDBFused>::member_type&) const; // GPU fused path

// NOLINTNEXTLINE
  KOKKOS_INLINE_FUNCTION
  void operator()(TagComputePACEAiFused, const typename Kokkos::TeamPolicy<DeviceType,TagComputePACEAiFused>::member_type&) const; // GPU: fused Radial+Ai for PERATOM=1

// NOLINTNEXTLINE
  KOKKOS_INLINE_FUNCTION
  void operator()(TagComputePACEAssembleBik, const int&) const;
// NOLINTNEXTLINE
  KOKKOS_INLINE_FUNCTION
  void operator()(TagComputePACEAssembleForce, const int&) const;
// NOLINTNEXTLINE
  KOKKOS_INLINE_FUNCTION
  void operator()(TagComputePACEAssembleVirial, const int&) const;

 protected:
  // Stack-array upper bounds for ai_one_neighbor_fused local spline storage and
  // the per-function cache in the CPU Projections operator.
  // init() asserts that runtime values fit within these.
  static constexpr int LMAXP1_MAX   = 8;
  static constexpr int NRADMAX_MAX  = 16;
  static constexpr int NRADBASE_MAX = 20;  // covers typical production potentials;
                                            // only sizes gr_local, so this is cheap to grow
  static constexpr int RANK_MAX     = 9;

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
  typename t_ace_2d::host_mirror_type h_projections;   // persistent; grown with d_projections

  // ---- A-arrays (per chunk) ----
  t_ace_3d A_rank1;
  t_ace_4c A;
  t_ace_4c A_sph;
  t_ace_3c A_list;
  t_ace_3c A_forward_prod;

  // ---- B-gradient scratch / output (per chunk, PERATOM=0 only) ----
  t_ace_3c dB_flatten;                 // (chunk, idx_ms_combs, rank): leave-one-out products
  t_ace_5c weights_dB;                 // (chunk, mu, idx_sph, nradmax+1, func_rankgt1)
  t_ace_4d d_neighbours_dB;            // (chunk, maxneigh, nvalues, 3): dB_{i,nu}/dr_j
  typename t_ace_4d::host_mirror_type h_neighbours_dB; // persistent; grown with d_neighbours_dB

  // GPU-only Newton scatter accumulator (nmax, size_peratom): replaces h_neighbours_dB
  // deep_copy + host scatter for the !dgradflag path. LayoutRight matches pace_peratom.
  t_ace_2d_lr d_pace_peratom;

  // Device mirror of the full global pace array (size_array_rows, size_array_cols).
  // LayoutRight matches the row-major host pace[][] so a single deep_copy suffices.
  // Only allocated/used on the !host_flag && !dgradflag path.
  t_ace_2d_lr d_pace;
  typename t_ace_2d_lr::host_mirror_type h_pace;   // persistent; grown with d_pace

  // Per-atom tag and force views for the device-assembly path.
  // Use AT:: not DAT:: so the kk/host instantiation maps to the right memory space.
  typename AT::t_tagint_1d d_tag;
  typename AT::t_kkacc_1d_3 d_f;

  // Scalars cached before kernel launches so device operators can read them via this->.
  int ntypes;    // atom->ntypes
  int nlocal;    // atom->nlocal (for last-column force writes, local atoms only)
  int ntotal;    // nlocal + nghost (for AssembleForce + AssembleVirial range)

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
  typename t_ace_1i::host_mirror_type h_ncount;        // persistent; grown with d_ncount
  t_ace_2i d_mu;       // neighbour element ids (pair_pace_kokkos stores these
                       // in a KK_FLOAT view; int is exact and half the bytes)
  t_ace_2d d_rnorms;
  t_ace_3d3 d_rhats;
  t_ace_2i d_nearest;
  typename t_ace_2i::host_mirror_type h_nearest;       // persistent; grown with d_nearest

  // ---- per-type tables ----
  // d_cutsq is the only per-type table any ACE-descriptor kernel reads (Neigh
  // short-list build). The FS/ZBL embedding tables the pair style needs are not
  // used here -- compute pace evaluates descriptors, not the embedding energy.
  t_ace_2d d_cutsq;            // (mu_i, mu_j) pair cutoff squared, from basis_set rcut

  // ---- flattened (tilde) basis tables ----
  t_ace_1i d_idx_ms_combs_count;
  t_ace_2i_lr d_rank;
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

  // PERATOM=1 device pipeline: fill k_pace_atom (host-synced) with the per-local-atom
  // descriptors B_{i,nu}, called from compute_peratom(). compute_array() (PERATOM=0)
  // runs its own separate device-assembly pipeline; this function is a no-op there.
  void compute_descriptors_device();

  // table builders (duplicated from pair_pace_kokkos)
  void grow(int, int);
  void copy_pertype();
  void copy_splines();
  void copy_tilde();
  void pre_compute_harmonics(int);

// NOLINTNEXTLINE
  KOKKOS_INLINE_FUNCTION
  void evaluate_splines(const int, const int, KK_FLOAT, int, int) const;

  template<class TagStyle>
  void check_team_size_for(int, int&, int);

  template<typename scratch_type>
  int scratch_size_helper(int values_per_team);

  // Shared A-accumulation recursion for one bond jj, with the radial values
  // supplied by accessor functors gracc(n) [g_k] and fracc(l,n) [R_nl] so the
  // global-view caller (ai_one_neighbor) and the thread-local-spline caller
  // (ai_one_neighbor_fused) share a single copy of the plm/ylm recurrence.
  template<bool UseAtomic, class GrAcc, class FrAcc>
  KOKKOS_FORCEINLINE_FUNCTION
  void ai_accumulate(int ii, int jj, int mu_j, const GrAcc& gracc, const FrAcc& fracc) const;

  // Single-neighbour kernel body: reads fr/gr from the global spline views
  // (Radial must have run first).
  template<bool UseAtomic>
  KOKKOS_INLINE_FUNCTION
  void ai_one_neighbor(int ii, int jj) const;

  // Fused variant: computes splines into thread-local storage, no fr/gr global I/O.
  // Only dispatched from the GPU AiFused TeamPolicy (UseAtomic=true); the host
  // PERATOM=1 path runs Radial+Ai through the global fr/gr views instead.
  template<bool UseAtomic>
  KOKKOS_INLINE_FUNCTION
  void ai_one_neighbor_fused(int ii, int jj, int mu_i) const;

  // Shared single-ms-comb kernel body called by both the host (UseAtomic=false)
  // loop and the device (UseAtomic=true) flat-decomposition Projections overload.
  template<bool UseAtomic>
  KOKKOS_INLINE_FUNCTION
  void project_one(int ii, int mu_i, int idx_ms_combs) const;

  // Shared single-ms-comb kernel body for RhoDB (CPU loop and GPU flat
  // decomposition). Writes to A_list / A_forward_prod / dB_flatten are
  // disjoint per (ii, idx_ms_combs) on both paths — no atomics needed.
  KOKKOS_FORCEINLINE_FUNCTION
  void rho_one(int ii, int mu_i, int idx_ms_combs) const;

  // Shared single-ms-comb kernel body for WeightsDB (CPU loop and GPU flat
  // decomposition). UseAtomic=true (GPU): atomic_add into weights_dB (multiple
  // idx_ms_combs can share the same (ii, func_local, mu_t, idx_sph, n_t-1)
  // slot). UseAtomic=false (CPU): one thread owns atom ii, direct +=.
  template<bool UseAtomic>
  KOKKOS_FORCEINLINE_FUNCTION
  void weights_one(int ii, int mu_i, int tbs_r1, int idx_ms_combs) const;

  // Shared single-neighbour body for PERATOM=0 descriptor-gradient computation.
  // Writes to d_neighbours_dB(ii,...,jj,...) are disjoint per (ii,jj) so no
  // atomics are required.
  // FuseScatter=true: acc_x/y/z must point to nvalues-sized PerThread L1 scratch;
  // accumulates gradient there and atomic-scatters directly into d_pace_peratom.
  // FuseScatter=false: acc_x/y/z are unused (pass nullptr); i/j/typeoffset unused too.
  template<bool FuseScatter>
  KOKKOS_INLINE_FUNCTION
  void derivative_one_neighbor(int ii, int jj, int i, int j, int typeoffset,
                                KK_FLOAT* acc_x, KK_FLOAT* acc_y, KK_FLOAT* acc_z) const;

  // Decode the (atom ii, neighbour jj) pair owned by this member of a per-pair
  // TeamPolicy (league = ceil(chunk_size/team_size) * maxneigh): consecutive
  // team members own consecutive atoms, leagues step the neighbour index.
  // Returns false for padding slots (ii >= chunk_size, jj >= ncount).
  template<class TeamMember>
  KOKKOS_INLINE_FUNCTION
  bool decode_pair(const TeamMember& team, int& ii, int& jj) const;

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
