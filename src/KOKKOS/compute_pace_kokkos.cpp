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

#include "compute_pace_kokkos.h"

#include "atom_kokkos.h"
#include "atom_masks.h"
#include "comm.h"
#include "error.h"
#include "memory_kokkos.h"
#include "neigh_list.h"
#include "neigh_request.h"
#include "neighbor.h"
#include "neighbor_kokkos.h"
#include "update.h"

#include "ace-evaluator/ace_c_basis.h"
#include "ace-evaluator/ace_evaluator.h"
#include "ace-evaluator/ace_radial.h"

using namespace LAMMPS_NS;

enum{FS,FS_SHIFTEDSCALED};

// repeated from compute_pace.cpp so this TU can read acecimpl->basis_set.
// Layout must match the definition there (ODR).
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

/* ---------------------------------------------------------------------- */

template<class DeviceType, int PERATOM>
ComputePACEKokkos<DeviceType, PERATOM>::ComputePACEKokkos(LAMMPS *lmp, int narg, char **arg) :
  ComputePACE<PERATOM>(lmp, narg, arg)
{
  this->kokkosable = 1;
  this->atomKK = (AtomKokkos *) this->atom;
  this->execution_space = ExecutionSpaceFromDevice<DeviceType>::space;
  this->datamask_read = EMPTY_MASK;
  this->datamask_modify = EMPTY_MASK;

  host_flag = (this->execution_space == HostKK);
  // the global (PERATOM=0) path also allocates the per-function weights_dB and
  // neighbours_dB scratch, so it uses a smaller chunk to bound device memory.
  chunksize = PERATOM ? 4096 : 256;
}

/* ---------------------------------------------------------------------- */

template<class DeviceType, int PERATOM>
ComputePACEKokkos<DeviceType, PERATOM>::~ComputePACEKokkos()
{
  if (this->copymode) return;

  this->memoryKK->destroy_kokkos(k_pace_atom, this->pace_atom);

  deallocate_views_of_views();
}

/* ---------------------------------------------------------------------- */

template<class DeviceType, int PERATOM>
void ComputePACEKokkos<DeviceType, PERATOM>::deallocate_views_of_views()
{
  // deallocate views of views in serial to prevent race conditions

  if (k_splines_gk.view_host().data()) {
    for (int i = 0; i < nelements; i++) {
      for (int j = 0; j < nelements; j++) {
        k_splines_gk.view_host()(i, j).deallocate();
        k_splines_rnl.view_host()(i, j).deallocate();
        k_splines_hc.view_host()(i, j).deallocate();
      }
    }
  }
}

/* ----------------------------------------------------------------------
   init: build the device basis tables from this->acecimpl->basis_set
------------------------------------------------------------------------- */

template<class DeviceType, int PERATOM>
void ComputePACEKokkos<DeviceType, PERATOM>::init()
{
  ComputePACE<PERATOM>::init();

  // Flip the (full, occasional) neighbor request to a Kokkos list. Both styles
  // run the device descriptor pipeline (compute_descriptors_device), which
  // consumes d_numneigh/d_neighbors from the NeighListKokkos.

  auto request = this->neighbor->find_request(this);
  request->set_kokkos_host(std::is_same_v<DeviceType,LMPHostType> &&
                           !std::is_same_v<DeviceType,LMPDeviceType>);
  request->set_kokkos_device(std::is_same_v<DeviceType,LMPDeviceType>);

  auto basis_set = this->acecimpl->basis_set;

  nelements = basis_set->nelements;
  lmax = basis_set->lmax;
  nradmax = basis_set->nradmax;
  nradbase = basis_set->nradbase;

  // guard the fused kernel's stack-allocated local arrays
  if (lmax + 1 > LMAXP1_MAX)
    this->error->all(FLERR,"compute pace/kk: lmax+1 exceeds LMAXP1_MAX; increase the constant");
  if (nradmax > NRADMAX_MAX)
    this->error->all(FLERR,"compute pace/kk: nradmax exceeds NRADMAX_MAX; increase the constant");
  if (nradbase > NRADBASE_MAX)
    this->error->all(FLERR,"compute pace/kk: nradbase exceeds NRADBASE_MAX; increase the constant");

  // type -> element (mu) map

  MemKK::realloc_kokkos(d_map, "pace:map", this->atom->ntypes + 1);
  auto h_map = Kokkos::create_mirror_view(d_map);
  for (int ik = 1; ik <= this->atom->ntypes; ik++) {
    for (int mu = 0; mu < nelements; mu++)
      if (mu == ik - 1) h_map(ik) = mu;
  }
  Kokkos::deep_copy(d_map, h_map);

  // spherical harmonics

  MemKK::realloc_kokkos(d_idx_sph, "pace:idx_sph", (lmax + 1) * (lmax + 1));
  MemKK::realloc_kokkos(alm, "pace:alm", (lmax + 1) * (lmax + 1));
  MemKK::realloc_kokkos(blm, "pace:blm", (lmax + 1) * (lmax + 1));
  MemKK::realloc_kokkos(cl, "pace:cl", lmax + 1);
  MemKK::realloc_kokkos(dl, "pace:dl", lmax + 1);

  pre_compute_harmonics(lmax);
  copy_pertype();
  copy_splines();
  copy_tilde();
}

/* ----------------------------------------------------------------------
   shared device pipeline: compute per-local-atom descriptors B_{i,nu} on
   device and sync them to the host k_pace_atom buffer (group atoms only;
   out-of-group atoms are left zero).
------------------------------------------------------------------------- */

template<class DeviceType, int PERATOM>
void ComputePACEKokkos<DeviceType, PERATOM>::setup_device_pipeline()
{
  // invoke full neighbor list (will copy or build if necessary)

  this->neighbor->build_one(this->list);

  inum = this->list->inum;
  NeighListKokkos<DeviceType>* k_list = static_cast<NeighListKokkos<DeviceType>*>(this->list);
  d_numneigh = k_list->d_numneigh;
  d_neighbors = k_list->d_neighbors;
  d_ilist = k_list->d_ilist;

  this->atomKK->sync(this->execution_space, X_MASK | TYPE_MASK | MASK_MASK);
  x = this->atomKK->k_x.template view<DeviceType>();
  type = this->atomKK->k_type.template view<DeviceType>();
  mask = this->atomKK->k_mask.template view<DeviceType>();

  // determine the maximum number of neighbours

  maxneigh = 0;
  {
    auto l_ilist = d_ilist;
    auto l_numneigh = d_numneigh;
    Kokkos::parallel_reduce("pace:maxneigh", inum,
      KOKKOS_LAMBDA(const int ii, int& m) {
        const int i = l_ilist[ii];
        const int n = l_numneigh[i];
        if (m < n) m = n;
      }, Kokkos::Max<int>(maxneigh));
  }
}

/* ---------------------------------------------------------------------- */

template<class DeviceType, int PERATOM>
void ComputePACEKokkos<DeviceType, PERATOM>::compute_descriptors_device()
{
  // grow per-atom descriptor buffer if necessary

  if (this->atom->nmax > this->nmaxatom) {
    this->memoryKK->destroy_kokkos(k_pace_atom, this->pace_atom);
    this->nmaxatom = this->atom->nmax;
    this->memoryKK->create_kokkos(k_pace_atom, this->pace_atom, this->nmaxatom,
                                  this->nvalues, "pace:pace_atom");
    if constexpr (PERATOM) this->array_atom = this->pace_atom;
    d_pace_atom = k_pace_atom.template view<DeviceType>();
  }

  setup_device_pipeline();

  this->copymode = 1;

  // zero the full output; out-of-group atoms remain zero

  Kokkos::deep_copy(d_pace_atom, 0.0);

  chunk_size = MIN(chunksize, inum);
  chunk_offset = 0;
  grow(chunk_size, maxneigh);

  while (chunk_offset < inum) {   // chunk up loop to bound memory

    if (chunk_size > inum - chunk_offset)
      chunk_size = inum - chunk_offset;

    Kokkos::deep_copy(A_rank1, 0.0);
    Kokkos::deep_copy(A_sph, 0.0);
    Kokkos::deep_copy(d_projections, 0.0);

    Kokkos::parallel_for("pace:Neigh",
      Kokkos::RangePolicy<DeviceType,TagComputePACENeigh>(0,chunk_size), *this);
    if (host_flag) {
      Kokkos::parallel_for("pace:Radial",
        Kokkos::RangePolicy<DeviceType,TagComputePACERadial>(0,chunk_size), *this);
      Kokkos::parallel_for("pace:Ai",
        Kokkos::RangePolicy<DeviceType,TagComputePACEAi>(0,chunk_size), *this);
    } else {
      const int league = ((chunk_size+32-1)/32)*maxneigh;
      if constexpr (PERATOM) {
        // GPU PERATOM=1: one fused kernel replaces Radial + Ai
        int team_size = 32, vector_length = 1;
        check_team_size_for<TagComputePACEAiFused>(league, team_size, vector_length);
        Kokkos::parallel_for("pace:AiFused",
          Kokkos::TeamPolicy<DeviceType,TagComputePACEAiFused>(league,team_size,vector_length), *this);
      } else {
        int team_size = 32, vector_length = 1;
        check_team_size_for<TagComputePACERadial>(league, team_size, vector_length);
        Kokkos::parallel_for("pace:Radial",
          Kokkos::TeamPolicy<DeviceType,TagComputePACERadial>(league,team_size,vector_length), *this);
        team_size = 32; vector_length = 1;
        check_team_size_for<TagComputePACEAi>(league, team_size, vector_length);
        Kokkos::parallel_for("pace:Ai",
          Kokkos::TeamPolicy<DeviceType,TagComputePACEAi>(league,team_size,vector_length), *this);
      }
    }
    Kokkos::parallel_for("pace:ConjAi",
      Kokkos::RangePolicy<DeviceType,TagComputePACEConjugateAi>(0,chunk_size), *this);
    if (host_flag)
      Kokkos::parallel_for("pace:Projections",
        Kokkos::RangePolicy<DeviceType,TagComputePACEProjections>(0,chunk_size), *this);
    else
      Kokkos::parallel_for("pace:Projections",
        Kokkos::RangePolicy<DeviceType,TagComputePACEProjectionsFlat>(0,chunk_size*idx_ms_combs_max), *this);
    Kokkos::parallel_for("pace:CopyProjections",
      Kokkos::RangePolicy<DeviceType,TagComputePACECopyProjections>(0,chunk_size), *this);

    chunk_offset += chunk_size;
  }

  this->copymode = 0;

  k_pace_atom.template modify<DeviceType>();
  k_pace_atom.sync_host();
}

/* ----------------------------------------------------------------------
   per-atom ACE descriptors B_{i,nu} on device (PERATOM=1)
------------------------------------------------------------------------- */

template<class DeviceType, int PERATOM>
void ComputePACEKokkos<DeviceType, PERATOM>::compute_peratom()
{
  if constexpr (PERATOM) {
    this->invoked_peratom = this->update->ntimestep;
    compute_descriptors_device();
  }
}

/* ----------------------------------------------------------------------
   global array (PERATOM=0): run the device descriptor + B-gradient pipeline
   per chunk, sync the per-chunk descriptors (projections) and gradients
   (neighbours_dB) to the host, and feed them into the same global-array
   assembly the CPU compute_array uses (bik rows, force rows, virial via
   dbdotr_compute, and the dgradflag=1 N^2 layout).
------------------------------------------------------------------------- */

template<class DeviceType, int PERATOM>
void ComputePACEKokkos<DeviceType, PERATOM>::compute_array()
{
  if constexpr (!PERATOM) {
    this->invoked_array = this->update->ntimestep;

    Atom *atom = this->atom;
    const int ntotal = atom->nlocal + atom->nghost;
    const int nvalues = this->nvalues;
    const int natoms = this->natoms;
    const int bik_rows = this->bik_rows;
    const int yoffset = this->yoffset;
    const int zoffset = this->zoffset;
    const int ndims_peratom = this->ndims_peratom;
    const int lastcol = this->lastcol;
    const int dgradflag = this->dgradflag;
    const int bikflag = this->bikflag;

    // grow per-proc dB-accumulation array if necessary

    if (atom->nmax > this->nmax) {
      this->memory->destroy(this->pace_peratom);
      this->nmax = atom->nmax;
      this->memory->create(this->pace_peratom, this->nmax, this->size_peratom, "pace:pace_peratom");
    }

    // clear global array and the per-proc dB accumulation array

    for (int irow = 0; irow < this->size_array_rows; irow++)
      for (int icoeff = 0; icoeff < this->size_array_cols; icoeff++)
        this->pace[irow][icoeff] = 0.0;

    for (int i = 0; i < ntotal; i++)
      for (int icoeff = 0; icoeff < this->size_peratom; icoeff++)
        this->pace_peratom[i][icoeff] = 0.0;

    // GPU !dgradflag path: allocate and zero d_pace_peratom (Newton scatter accumulator).
    // This replaces the per-chunk deep_copy(h_neighbours_dB) + host scatter.
    const bool use_device_assembly = (!host_flag && !dgradflag);
    if (use_device_assembly) {
      if (d_pace_peratom.extent(0) < (size_t)this->nmax ||
          d_pace_peratom.extent(1) < (size_t)this->size_peratom)
        d_pace_peratom = t_ace_2d_lr("pace:d_pace_peratom", this->nmax, this->size_peratom);
      Kokkos::deep_copy(d_pace_peratom, (KK_FLOAT)0.0);
    }

    // device pipeline setup (neighbor list, atom data, maxneigh)

    setup_device_pipeline();

    // tag/type/mask + positions/forces needed on the host for the global-array
    // assembly (force rows read atom->f, dbdotr_compute reads atom->x).
    this->atomKK->sync(Host, TAG_MASK | TYPE_MASK | MASK_MASK | X_MASK | F_MASK);
    int *tag = atom->tag;
    int *atype = atom->type;
    int *amask = atom->mask;

    this->copymode = 1;

    chunk_size = MIN(chunksize, inum);
    chunk_offset = 0;
    grow(chunk_size, maxneigh);

    // host mirrors of the per-chunk device outputs
    auto h_projections = Kokkos::create_mirror_view(d_projections);
    auto h_neighbours_dB = Kokkos::create_mirror_view(d_neighbours_dB);
    auto h_nearest = Kokkos::create_mirror_view(d_nearest);
    auto h_ncount = Kokkos::create_mirror_view(d_ncount);

    // device neighbor-list ordering, needed on the host so chunk-local ii maps
    // to the same atom i the device kernels used (d_ilist[ii + chunk_offset]).
    Kokkos::View<int*, DeviceType> d_ilist_copy("pace:ilist_copy", inum);
    Kokkos::deep_copy(d_ilist_copy, d_ilist);
    auto h_ilist = Kokkos::create_mirror_view(d_ilist_copy);
    Kokkos::deep_copy(h_ilist, d_ilist_copy);

    while (chunk_offset < inum) {   // chunk up loop to bound memory

      if (chunk_size > inum - chunk_offset)
        chunk_size = inum - chunk_offset;

      Kokkos::deep_copy(A_rank1, 0.0);
      Kokkos::deep_copy(A_sph, 0.0);
      Kokkos::deep_copy(d_projections, 0.0);
      Kokkos::deep_copy(d_neighbours_dB, 0.0);
      Kokkos::deep_copy(weights_dB, complex(0.0, 0.0));

      Kokkos::parallel_for("pace:Neigh",
        Kokkos::RangePolicy<DeviceType,TagComputePACENeigh>(0,chunk_size), *this);
      if (host_flag) {
        Kokkos::parallel_for("pace:Radial",
          Kokkos::RangePolicy<DeviceType,TagComputePACERadial>(0,chunk_size), *this);
        Kokkos::parallel_for("pace:Ai",
          Kokkos::RangePolicy<DeviceType,TagComputePACEAi>(0,chunk_size), *this);
      } else {
        int team_size = 32, vector_length = 1;
        const int league = ((chunk_size+team_size-1)/team_size)*maxneigh;
        check_team_size_for<TagComputePACERadial>(league, team_size, vector_length);
        Kokkos::parallel_for("pace:Radial",
          Kokkos::TeamPolicy<DeviceType,TagComputePACERadial>(league,team_size,vector_length), *this);
        team_size = 32; vector_length = 1;
        check_team_size_for<TagComputePACEAi>(league, team_size, vector_length);
        Kokkos::parallel_for("pace:Ai",
          Kokkos::TeamPolicy<DeviceType,TagComputePACEAi>(league,team_size,vector_length), *this);
      }
      Kokkos::parallel_for("pace:ConjAi",
        Kokkos::RangePolicy<DeviceType,TagComputePACEConjugateAi>(0,chunk_size), *this);
      if (host_flag)
        Kokkos::parallel_for("pace:Projections",
          Kokkos::RangePolicy<DeviceType,TagComputePACEProjections>(0,chunk_size), *this);
      else
        Kokkos::parallel_for("pace:Projections",
          Kokkos::RangePolicy<DeviceType,TagComputePACEProjectionsFlat>(0,chunk_size*idx_ms_combs_max), *this);
      if (host_flag)
        Kokkos::parallel_for("pace:RhoDB",
          Kokkos::RangePolicy<DeviceType,TagComputePACERhoDB>(0,chunk_size), *this);
      else
        Kokkos::parallel_for("pace:RhoDB",
          Kokkos::RangePolicy<DeviceType,TagComputePACERhoDBFlat>(0,chunk_size*idx_ms_combs_max), *this);
      if (host_flag)
        Kokkos::parallel_for("pace:WeightsDB",
          Kokkos::RangePolicy<DeviceType,TagComputePACEWeightsDB>(0,chunk_size), *this);
      else
        Kokkos::parallel_for("pace:WeightsDB",
          Kokkos::RangePolicy<DeviceType,TagComputePACEWeightsDBFlat>(0,chunk_size*idx_ms_combs_max), *this);
      if (host_flag)
        Kokkos::parallel_for("pace:DerivativeDB",
          Kokkos::RangePolicy<DeviceType,TagComputePACEDerivativeDB>(0,chunk_size), *this);
      else {
        int team_size = 32, vector_length = 1;
        const int league = ((chunk_size+team_size-1)/team_size)*maxneigh;
        check_team_size_for<TagComputePACEDerivativeDB>(league, team_size, vector_length);
        Kokkos::parallel_for("pace:DerivativeDB",
          Kokkos::TeamPolicy<DeviceType,TagComputePACEDerivativeDB>(league,team_size,vector_length), *this);
      }

      // GPU !dgradflag: scatter d_neighbours_dB into d_pace_peratom on device, avoiding
      // the 44 MB/chunk deep_copy(h_neighbours_dB) + host Newton scatter.
      if (use_device_assembly) {
        int team_size = 32, vector_length = 1;
        const int league = ((chunk_size+team_size-1)/team_size)*maxneigh;
        check_team_size_for<TagComputePACEAssembleForce>(league, team_size, vector_length);
        Kokkos::parallel_for("pace:AssembleForce",
          Kokkos::TeamPolicy<DeviceType,TagComputePACEAssembleForce>(league,team_size,vector_length), *this);
      }

      Kokkos::deep_copy(h_projections, d_projections);
      // h_neighbours_dB, h_nearest, h_ncount are only needed for the host scatter path
      // (CPU/OMP backends or dgradflag=1 on GPU). Skip the 44MB deep_copy otherwise.
      if (!use_device_assembly) {
        Kokkos::deep_copy(h_neighbours_dB, d_neighbours_dB);
        Kokkos::deep_copy(h_nearest, d_nearest);
        Kokkos::deep_copy(h_ncount, d_ncount);
      }

      // host-side global-array assembly for this chunk (mirrors compute_pace.cpp)

      for (int ii = 0; ii < chunk_size; ii++) {
        const int i = h_ilist(ii + chunk_offset);
        if (!(amask[i] & this->groupbit)) continue;

        const int itype = atype[i];
        const int typeoffset_local = ndims_peratom * nvalues * (itype - 1);
        const int typeoffset_global = nvalues * (itype - 1);
        const int irow = bikflag ? (tag[i] - 1) : 0;

        // dB contributions: host path (CPU/OMP or dgradflag=1 on GPU)
        // GPU !dgradflag path: handled by TagComputePACEAssembleForce kernel above
        if (!use_device_assembly) {
          const int ncount = h_ncount(ii);

          if (dgradflag) {
            // dBi/dRi and dBi/dRj index tags
            const int ti = tag[i] - 1;
            for (int d = 0; d < 3; d++) {
              this->pace[bik_rows + (ti*3*natoms) + 3*ti + d][0] = ti;
              this->pace[bik_rows + (ti*3*natoms) + 3*ti + d][1] = ti;
              this->pace[bik_rows + (ti*3*natoms) + 3*ti + d][2] = d;
            }
            for (int j = 0; j < natoms; j++)
              for (int d = 0; d < 3; d++) {
                this->pace[bik_rows + (j*3*natoms) + 3*ti + d][0] = ti;
                this->pace[bik_rows + (j*3*natoms) + 3*ti + d][1] = j;
                this->pace[bik_rows + (j*3*natoms) + 3*ti + d][2] = d;
              }
          }

          for (int jj = 0; jj < ncount; jj++) {
            const int j = h_nearest(ii, jj);
            if (!dgradflag) {
              double *pacedi = this->pace_peratom[i] + typeoffset_local;
              double *pacedj = this->pace_peratom[j] + typeoffset_local;
              for (int func = 0; func < nvalues; func++) {
                const double fx_dB = h_neighbours_dB(ii, func, jj, 0);
                const double fy_dB = h_neighbours_dB(ii, func, jj, 1);
                const double fz_dB = h_neighbours_dB(ii, func, jj, 2);
                pacedi[func] += fx_dB;
                pacedi[func + yoffset] += fy_dB;
                pacedi[func + zoffset] += fz_dB;
                pacedj[func] -= fx_dB;
                pacedj[func + yoffset] -= fy_dB;
                pacedj[func + zoffset] -= fz_dB;
              }
            } else {
              const int ti = tag[i] - 1;
              const int tj = tag[j] - 1;
              for (int icoeff = 0; icoeff < nvalues; icoeff++) {
                const double fx_dB = h_neighbours_dB(ii, icoeff, jj, 0);
                const double fy_dB = h_neighbours_dB(ii, icoeff, jj, 1);
                const double fz_dB = h_neighbours_dB(ii, icoeff, jj, 2);
                this->pace[bik_rows + (tj*3*natoms) + 3*ti + 0][icoeff+3] -= fx_dB;
                this->pace[bik_rows + (tj*3*natoms) + 3*ti + 1][icoeff+3] -= fy_dB;
                this->pace[bik_rows + (tj*3*natoms) + 3*ti + 2][icoeff+3] -= fz_dB;
                this->pace[bik_rows + (ti*3*natoms) + 3*ti + 0][icoeff+3] += fx_dB;
                this->pace[bik_rows + (ti*3*natoms) + 3*ti + 1][icoeff+3] += fy_dB;
                this->pace[bik_rows + (ti*3*natoms) + 3*ti + 2][icoeff+3] += fz_dB;
              }
            }
          }
        }

        // bik row (descriptors)
        int k = dgradflag ? 3 : typeoffset_global;
        for (int icoeff = 0; icoeff < nvalues; icoeff++)
          this->pace[irow][k++] += h_projections(ii, icoeff);
      }

      chunk_offset += chunk_size;
    }

    this->copymode = 0;

    // GPU !dgradflag: copy d_pace_peratom back to pace_peratom for the force-row assembly.
    // d_pace_peratom holds the Newton-scattered dB contributions from all chunks.
    if (use_device_assembly) {
      auto h_pace_peratom = Kokkos::create_mirror_view(d_pace_peratom);
      Kokkos::deep_copy(h_pace_peratom, d_pace_peratom);
      for (int i = 0; i < ntotal; i++)
        for (int icoeff = 0; icoeff < this->size_peratom; icoeff++)
          this->pace_peratom[i][icoeff] = h_pace_peratom(i, icoeff);
    }

    // accumulate force contributions to global array (forces rows), !dgradflag

    if (!dgradflag) {
      for (int itype = 0; itype < atom->ntypes; itype++) {
        const int typeoffset_local = ndims_peratom * nvalues * itype;
        const int typeoffset_global = nvalues * itype;
        for (int icoeff = 0; icoeff < nvalues; icoeff++) {
          for (int i = 0; i < ntotal; i++) {
            double *pacedi = this->pace_peratom[i] + typeoffset_local;
            int iglobal = atom->tag[i];
            int irow = 3 * (iglobal - 1) + bik_rows;
            this->pace[irow++][icoeff + typeoffset_global] += pacedi[icoeff];
            this->pace[irow++][icoeff + typeoffset_global] += pacedi[icoeff + yoffset];
            this->pace[irow][icoeff + typeoffset_global] += pacedi[icoeff + zoffset];
          }
        }
      }

      // actual forces in the last column
      for (int i = 0; i < atom->nlocal; i++) {
        int iglobal = atom->tag[i];
        int irow = 3 * (iglobal - 1) + bik_rows;
        this->pace[irow++][lastcol] = atom->f[i][0];
        this->pace[irow++][lastcol] = atom->f[i][1];
        this->pace[irow][lastcol] = atom->f[i][2];
      }
    } else {
      // for dgradflag=1, put forces at first 3 columns of bik rows
      for (int i = 0; i < atom->nlocal; i++) {
        int iglobal = atom->tag[i];
        this->pace[iglobal - 1][0] = atom->f[i][0];
        this->pace[iglobal - 1][1] = atom->f[i][1];
        this->pace[iglobal - 1][2] = atom->f[i][2];
      }
    }

    this->dbdotr_compute();

    // sum over all processes
    MPI_Allreduce(&this->pace[0][0], &this->paceall[0][0],
                  this->size_array_rows * this->size_array_cols, MPI_DOUBLE, MPI_SUM, this->world);

    // reference energy
    if (!dgradflag) {
      for (int i = 0; i < bik_rows; i++) this->paceall[i][lastcol] = 0.0;
      this->paceall[0][lastcol] = this->c_pe->compute_scalar();
    } else {
      int irow = bik_rows + 3 * natoms * natoms;
      this->paceall[irow][0] = this->c_pe->compute_scalar();
    }

    // virial stress in the last column (Voigt), !dgradflag
    if (!dgradflag) {
      this->c_virial->compute_vector();
      int irow = 3 * natoms + bik_rows;
      this->paceall[irow++][lastcol] = this->c_virial->vector[0];
      this->paceall[irow++][lastcol] = this->c_virial->vector[1];
      this->paceall[irow++][lastcol] = this->c_virial->vector[2];
      this->paceall[irow++][lastcol] = this->c_virial->vector[5];
      this->paceall[irow++][lastcol] = this->c_virial->vector[4];
      this->paceall[irow][lastcol] = this->c_virial->vector[3];
    }
  }
}

/* ----------------------------------------------------------------------
   ComputeNeigh: build the per-atom short neighbour list (within ACE cutoff)
------------------------------------------------------------------------- */

template<class DeviceType, int PERATOM>
// NOLINTNEXTLINE
KOKKOS_INLINE_FUNCTION
void ComputePACEKokkos<DeviceType, PERATOM>::operator()(TagComputePACENeigh, const int& ii) const
{
  const int i = d_ilist[ii + chunk_offset];
  const int mu_i = d_map(type(i));
  const KK_FLOAT xtmp = x(i,0);
  const KK_FLOAT ytmp = x(i,1);
  const KK_FLOAT ztmp = x(i,2);
  const int jnum = d_numneigh[i];

  int ncount = 0;
  for (int jj = 0; jj < jnum; jj++) {
    int j = d_neighbors(i,jj);
    j &= NEIGHMASK;
    const int mu_j = d_map(type(j));
    const KK_FLOAT delx = xtmp - x(j,0);
    const KK_FLOAT dely = ytmp - x(j,1);
    const KK_FLOAT delz = ztmp - x(j,2);
    const KK_FLOAT rsq = delx*delx + dely*dely + delz*delz;
    if (rsq < d_cutsq(mu_i, mu_j)) {
      const KK_FLOAT r = Kokkos::sqrt(rsq);
      const KK_FLOAT rinv = 1.0/r;
      d_mu(ii,ncount) = mu_j;
      d_rnorms(ii,ncount) = r;
      d_rhats(ii,ncount,0) = -delx*rinv;
      d_rhats(ii,ncount,1) = -dely*rinv;
      d_rhats(ii,ncount,2) = -delz*rinv;
      d_nearest(ii,ncount) = j;
      ncount++;
    }
  }
  d_ncount(ii) = ncount;
}

/* ----------------------------------------------------------------------
   ComputeRadial: radial functions for every short-list neighbour
------------------------------------------------------------------------- */

template<class DeviceType, int PERATOM>
// NOLINTNEXTLINE
KOKKOS_INLINE_FUNCTION
void ComputePACEKokkos<DeviceType, PERATOM>::operator()(TagComputePACERadial, const int& ii) const
{
  const int i = d_ilist[ii + chunk_offset];
  const int mu_i = d_map(type(i));
  const int ncount = d_ncount(ii);
  for (int jj = 0; jj < ncount; jj++) {
    const KK_FLOAT r_norm = d_rnorms(ii, jj);
    const int mu_j = d_mu(ii, jj);
    evaluate_splines(ii, jj, r_norm, nradbase, nradmax, mu_i, mu_j);
  }
}

/* ----------------------------------------------------------------------
   ComputeRadial GPU path: one (atom, neighbour) pair per team member
------------------------------------------------------------------------- */

template<class DeviceType, int PERATOM>
// NOLINTNEXTLINE
KOKKOS_INLINE_FUNCTION
void ComputePACEKokkos<DeviceType, PERATOM>::operator()(TagComputePACERadial, const typename Kokkos::TeamPolicy<DeviceType,TagComputePACERadial>::member_type& team) const
{
  int ii = team.team_rank() + team.team_size() * (team.league_rank() %
           ((chunk_size+team.team_size()-1)/team.team_size()));
  if (ii >= chunk_size) return;
  const int i = d_ilist[ii + chunk_offset];

  const int jj = team.league_rank() / ((chunk_size+team.team_size()-1)/team.team_size());
  if (jj >= d_ncount(ii)) return;

  evaluate_splines(ii, jj, d_rnorms(ii,jj), nradbase, nradmax, d_map(type(i)), d_mu(ii,jj));
}

/* ----------------------------------------------------------------------
   ComputeAi CPU path: one thread owns atom ii, loops all neighbours.
   ComputeAi GPU path: one (atom, neighbour) pair per team member.
   Both delegate per-neighbour math to ai_one_neighbor<UseAtomic>.
------------------------------------------------------------------------- */

template<class DeviceType, int PERATOM>
// NOLINTNEXTLINE
KOKKOS_INLINE_FUNCTION
void ComputePACEKokkos<DeviceType, PERATOM>::operator()(TagComputePACEAi, const int& ii) const
{
  // Radial has written fr/gr; accumulate A from the global spline arrays.
  for (int jj = 0; jj < d_ncount(ii); jj++)
    ai_one_neighbor<false>(ii, jj);
}

template<class DeviceType, int PERATOM>
// NOLINTNEXTLINE
KOKKOS_INLINE_FUNCTION
void ComputePACEKokkos<DeviceType, PERATOM>::operator()(TagComputePACEAi, const typename Kokkos::TeamPolicy<DeviceType,TagComputePACEAi>::member_type& team) const
{
  int ii = team.team_rank() + team.team_size() * (team.league_rank() %
           ((chunk_size+team.team_size()-1)/team.team_size()));
  if (ii >= chunk_size) return;

  const int jj = team.league_rank() / ((chunk_size+team.team_size()-1)/team.team_size());
  if (jj >= d_ncount(ii)) return;

  ai_one_neighbor<true>(ii, jj);
}

/* ----------------------------------------------------------------------
   AiFused GPU: fused Radial+Ai for PERATOM=1 — computes splines into
   thread-local storage and accumulates A, without the fr/gr global views.
   One (atom, neighbour) pair per team member, same league sizing as Ai.
------------------------------------------------------------------------- */

template<class DeviceType, int PERATOM>
// NOLINTNEXTLINE
KOKKOS_INLINE_FUNCTION
void ComputePACEKokkos<DeviceType, PERATOM>::operator()(TagComputePACEAiFused, const typename Kokkos::TeamPolicy<DeviceType,TagComputePACEAiFused>::member_type& team) const
{
  int ii = team.team_rank() + team.team_size() * (team.league_rank() %
           ((chunk_size+team.team_size()-1)/team.team_size()));
  if (ii >= chunk_size) return;
  const int i = d_ilist[ii + chunk_offset];
  const int mu_i = d_map(type(i));

  const int jj = team.league_rank() / ((chunk_size+team.team_size()-1)/team.team_size());
  if (jj >= d_ncount(ii)) return;

  ai_one_neighbor_fused<true>(ii, jj, mu_i);
}

/* ----------------------------------------------------------------------
   ai_one_neighbor: Legendre/ylm recurrence for one neighbour jj of atom ii.
   UseAtomic=false (CPU path): direct +=; UseAtomic=true (GPU path): atomic_add.
------------------------------------------------------------------------- */

template<class DeviceType, int PERATOM>
template<bool UseAtomic>
// NOLINTNEXTLINE
KOKKOS_INLINE_FUNCTION
void ComputePACEKokkos<DeviceType, PERATOM>::ai_one_neighbor(int ii, int jj) const
{
  const int mu_j = d_mu(ii, jj);

  // rank = 1
  for (int n = 0; n < nradbase; n++) {
    const KK_FLOAT val = gr(ii, jj, n) * Y00;
    if constexpr (UseAtomic) Kokkos::atomic_add(&A_rank1(ii, mu_j, n), val);
    else                     A_rank1(ii, mu_j, n) += val;
  }

  // rank > 1: plm/ylm recurrence, accumulate A_sph
  // requires |r_hat| = 1 and -1 <= rz <= 1

  complex ylm, phase;
  complex phasem, mphasem1;

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
        Kokkos::atomic_add(&A_sph(ii, mu_j, idx_sph, n).re, fr(ii, jj, l, n) * ylm.re);
        Kokkos::atomic_add(&A_sph(ii, mu_j, idx_sph, n).im, fr(ii, jj, l, n) * ylm.im);
      } else {
        A_sph(ii, mu_j, idx_sph, n).re += fr(ii, jj, l, n) * ylm.re;
        A_sph(ii, mu_j, idx_sph, n).im += fr(ii, jj, l, n) * ylm.im;
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
        Kokkos::atomic_add(&A_sph(ii, mu_j, idx_sph, n).re, fr(ii, jj, l, n) * ylm.re);
        Kokkos::atomic_add(&A_sph(ii, mu_j, idx_sph, n).im, fr(ii, jj, l, n) * ylm.im);
      } else {
        A_sph(ii, mu_j, idx_sph, n).re += fr(ii, jj, l, n) * ylm.re;
        A_sph(ii, mu_j, idx_sph, n).im += fr(ii, jj, l, n) * ylm.im;
      }
    }
    plm_idx2 = plm_idx1; plm_idx1 = plm_idx; idx_sph++;
  }

  plm_idx = plm_idx1 = plm_idx2 = 0.0;
  KK_FLOAT plm_mm1_mm1 = -sq3o2 * Y00;

  // m > 1
  phasem = phase;
  for (int m = 2; m <= lmax; m++) {
    mphasem1.re = phasem.re * KK_FLOAT(m);
    mphasem1.im = phasem.im * KK_FLOAT(m);
    phasem = phasem * phase;

    for (int l = m; l <= lmax; l++) {
      if (l == m)          { plm_idx = cl(l) * plm_mm1_mm1; plm_mm1_mm1 = plm_idx; }
      else if (l == (m+1)) { const KK_FLOAT t = dl(l) * plm_mm1_mm1; plm_idx = t * rz; }
      else                   plm_idx = alm(idx_sph) * (rz * plm_idx1 + blm(idx_sph) * plm_idx2);

      ylm.re = phasem.re * plm_idx;
      ylm.im = phasem.im * plm_idx;

      for (int n = 0; n < nradmax; n++) {
        if constexpr (UseAtomic) {
          Kokkos::atomic_add(&A_sph(ii, mu_j, idx_sph, n).re, fr(ii, jj, l, n) * ylm.re);
          Kokkos::atomic_add(&A_sph(ii, mu_j, idx_sph, n).im, fr(ii, jj, l, n) * ylm.im);
        } else {
          A_sph(ii, mu_j, idx_sph, n).re += fr(ii, jj, l, n) * ylm.re;
          A_sph(ii, mu_j, idx_sph, n).im += fr(ii, jj, l, n) * ylm.im;
        }
      }
      plm_idx2 = plm_idx1; plm_idx1 = plm_idx; idx_sph++;
    }
  }
}

/* ----------------------------------------------------------------------
   ai_one_neighbor_fused: like ai_one_neighbor but computes splines into
   thread-local arrays (no fr/gr global I/O). Used for PERATOM=1 on both
   CPU (UseAtomic=false) and GPU (UseAtomic=true, via AiFused TeamPolicy).
   mu_i is passed in so the CPU loop computes it once before the jj loop.
------------------------------------------------------------------------- */

template<class DeviceType, int PERATOM>
template<bool UseAtomic>
// NOLINTNEXTLINE
KOKKOS_INLINE_FUNCTION
void ComputePACEKokkos<DeviceType, PERATOM>::ai_one_neighbor_fused(int ii, int jj, int mu_i) const
{
  const int mu_j = d_mu(ii, jj);
  const KK_FLOAT r = d_rnorms(ii, jj);
  const int nll = lmax + 1;  // lmax+1, stride for fr_local[l + nll*n]

  KK_FLOAT gr_local[NRADBASE_MAX];
  KK_FLOAT fr_local[NRADMAX_MAX * LMAXP1_MAX];

  auto &spline_gk  = k_splines_gk.template  view<DeviceType>()(mu_i, mu_j);
  auto &spline_rnl = k_splines_rnl.template view<DeviceType>()(mu_i, mu_j);
  spline_gk.calcSplines_local(r, gr_local);
  spline_rnl.calcSplines_local(r, fr_local, nll);

  // rank = 1
  for (int n = 0; n < nradbase; n++) {
    const KK_FLOAT val = gr_local[n] * Y00;
    if constexpr (UseAtomic) Kokkos::atomic_add(&A_rank1(ii, mu_j, n), val);
    else                     A_rank1(ii, mu_j, n) += val;
  }

  // rank > 1: plm/ylm recurrence (identical to ai_one_neighbor with fr->fr_local)

  complex ylm, phase;
  complex phasem, mphasem1;

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
        Kokkos::atomic_add(&A_sph(ii, mu_j, idx_sph, n).re, fr_local[l + nll*n] * ylm.re);
        Kokkos::atomic_add(&A_sph(ii, mu_j, idx_sph, n).im, fr_local[l + nll*n] * ylm.im);
      } else {
        A_sph(ii, mu_j, idx_sph, n).re += fr_local[l + nll*n] * ylm.re;
        A_sph(ii, mu_j, idx_sph, n).im += fr_local[l + nll*n] * ylm.im;
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
        Kokkos::atomic_add(&A_sph(ii, mu_j, idx_sph, n).re, fr_local[l + nll*n] * ylm.re);
        Kokkos::atomic_add(&A_sph(ii, mu_j, idx_sph, n).im, fr_local[l + nll*n] * ylm.im);
      } else {
        A_sph(ii, mu_j, idx_sph, n).re += fr_local[l + nll*n] * ylm.re;
        A_sph(ii, mu_j, idx_sph, n).im += fr_local[l + nll*n] * ylm.im;
      }
    }
    plm_idx2 = plm_idx1; plm_idx1 = plm_idx; idx_sph++;
  }

  plm_idx = plm_idx1 = plm_idx2 = 0.0;
  KK_FLOAT plm_mm1_mm1 = -sq3o2 * Y00;

  // m > 1
  phasem = phase;
  for (int m = 2; m <= lmax; m++) {
    mphasem1.re = phasem.re * KK_FLOAT(m);
    mphasem1.im = phasem.im * KK_FLOAT(m);
    phasem = phasem * phase;

    for (int l = m; l <= lmax; l++) {
      if (l == m)          { plm_idx = cl(l) * plm_mm1_mm1; plm_mm1_mm1 = plm_idx; }
      else if (l == (m+1)) { const KK_FLOAT t = dl(l) * plm_mm1_mm1; plm_idx = t * rz; }
      else                   plm_idx = alm(idx_sph) * (rz * plm_idx1 + blm(idx_sph) * plm_idx2);

      ylm.re = phasem.re * plm_idx;
      ylm.im = phasem.im * plm_idx;

      for (int n = 0; n < nradmax; n++) {
        if constexpr (UseAtomic) {
          Kokkos::atomic_add(&A_sph(ii, mu_j, idx_sph, n).re, fr_local[l + nll*n] * ylm.re);
          Kokkos::atomic_add(&A_sph(ii, mu_j, idx_sph, n).im, fr_local[l + nll*n] * ylm.im);
        } else {
          A_sph(ii, mu_j, idx_sph, n).re += fr_local[l + nll*n] * ylm.re;
          A_sph(ii, mu_j, idx_sph, n).im += fr_local[l + nll*n] * ylm.im;
        }
      }
      plm_idx2 = plm_idx1; plm_idx1 = plm_idx; idx_sph++;
    }
  }
}

/* ----------------------------------------------------------------------
   ConjugateAi: transpose A_sph -> A and fill the negative-m entries
------------------------------------------------------------------------- */

template<class DeviceType, int PERATOM>
// NOLINTNEXTLINE
KOKKOS_INLINE_FUNCTION
void ComputePACEKokkos<DeviceType, PERATOM>::operator()(TagComputePACEConjugateAi, const int& ii) const
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

    // complex conjugate A's (for the negative -m terms), rank > 1
    for (int l = 0; l <= lmax; l++) {
      for (int m = 1; m <= l; m++) {
        const int idx = l * (l + 1) + m;  // (l, m)
        const int idxm = l * (l + 1) - m; // (l, -m)
        const int idx_sph = d_idx_sph(idx);
        const int factor = m % 2 == 0 ? 1 : -1;
        for (int n = 0; n < nradmax; n++)
          A(ii, mu_j, idxm, n) = A_sph(ii, mu_j, idx_sph, n).conj() * (KK_FLOAT)factor;
      }
    }
  }
}

/* ----------------------------------------------------------------------
   Projections CPU path (TagComputePACEProjections): one thread per atom ii.
   Inlined with per-function hoisting: reloads rank/mus/ns/lm stack arrays
   only when idx_func changes, cutting redundant index-table traffic for
   rank>1 functions with many ms-combinations.
------------------------------------------------------------------------- */

template<class DeviceType, int PERATOM>
// NOLINTNEXTLINE
KOKKOS_INLINE_FUNCTION
void ComputePACEKokkos<DeviceType, PERATOM>::operator()(TagComputePACEProjections, const int& ii) const
{
  const int i = d_ilist[ii + chunk_offset];
  const int mu_i = d_map(type(i));
  const int count = d_idx_ms_combs_count(mu_i);

  constexpr int RANK_MAX = 9;
  int cached_idx_func = -1;
  int cached_rank = 0;
  int cached_mus[RANK_MAX], cached_ns[RANK_MAX], cached_lm[RANK_MAX];

  for (int idx_ms_combs = 0; idx_ms_combs < count; idx_ms_combs++) {
    const int idx_func = d_idx_funcs(mu_i, idx_ms_combs);

    if (idx_func != cached_idx_func) {
      cached_idx_func = idx_func;
      cached_rank = d_rank(mu_i, idx_func);
      for (int t = 0; t < cached_rank; t++) {
        cached_mus[t] = d_mus(mu_i, idx_func, t);
        cached_ns[t]  = d_ns(mu_i, idx_func, t);
        cached_lm[t]  = d_func_base(mu_i, idx_func, t);
      }
    }

    KK_FLOAT val;
    if (cached_rank == 1) {
      val = d_ctildes(mu_i, idx_ms_combs, 0) * A_rank1(ii, cached_mus[0], cached_ns[0] - 1);
    } else {
      complex Bprod = complex::one();
      for (int t = 0; t < cached_rank; t++) {
        const int m = d_ms_combs(mu_i, idx_ms_combs, t);
        Bprod = Bprod * A(ii, cached_mus[t], cached_lm[t] + m, cached_ns[t] - 1);
      }
      val = Bprod.real_part_product(d_ctildes(mu_i, idx_ms_combs, 0));
    }
    d_projections(ii, idx_func) += val;
  }
}

/* ----------------------------------------------------------------------
   Projections GPU path (TagComputePACEProjectionsFlat): flat (atom,ms-comb)
   decomposition — iter encodes (idx_ms_combs, ii). Uses project_one<true>.
------------------------------------------------------------------------- */

template<class DeviceType, int PERATOM>
// NOLINTNEXTLINE
KOKKOS_INLINE_FUNCTION
void ComputePACEKokkos<DeviceType, PERATOM>::operator()(TagComputePACEProjectionsFlat, const int& iter) const
{
  const int idx_ms_combs = iter / chunk_size;
  const int ii = iter % chunk_size;
  const int i = d_ilist[ii + chunk_offset];
  const int mu_i = d_map(type(i));
  if (idx_ms_combs >= d_idx_ms_combs_count(mu_i)) return;
  project_one<true>(ii, mu_i, idx_ms_combs);
}

/* ----------------------------------------------------------------------
   project_one: accumulate one ms-combination's ctilde product into
   d_projections. UseAtomic=false: direct += (CPU); true: atomic_add (GPU).
------------------------------------------------------------------------- */

template<class DeviceType, int PERATOM>
template<bool UseAtomic>
// NOLINTNEXTLINE
KOKKOS_INLINE_FUNCTION
void ComputePACEKokkos<DeviceType, PERATOM>::project_one(int ii, int mu_i, int idx_ms_combs) const
{
  const int idx_func = d_idx_funcs(mu_i, idx_ms_combs);
  const int rank = d_rank(mu_i, idx_func);

  KK_FLOAT val;
  if (rank == 1) {
    const int mu = d_mus(mu_i, idx_func, 0);
    const int n = d_ns(mu_i, idx_func, 0);
    val = d_ctildes(mu_i, idx_ms_combs, 0) * A_rank1(ii, mu, n - 1);
  } else {
    // d_func_base stores precomputed l*(l+1) per (mu,func,t), saving one runtime
    // multiply. View accessor A(ii,mu,lm+m,n-1) is layout-independent (correct on
    // both CPU LayoutRight and CUDA LayoutLeft — a flat-pointer approach would not be).
    complex Bprod = complex::one();
    for (int t = 0; t < rank; t++) {
      const int mu = d_mus(mu_i, idx_func, t);
      const int n  = d_ns(mu_i, idx_func, t);
      const int lm = d_func_base(mu_i, idx_func, t);  // precomputed l*(l+1)
      const int m  = d_ms_combs(mu_i, idx_ms_combs, t);
      Bprod = Bprod * A(ii, mu, lm + m, n - 1);
    }
    val = Bprod.real_part_product(d_ctildes(mu_i, idx_ms_combs, 0));
  }

  if constexpr (UseAtomic) Kokkos::atomic_add(&d_projections(ii, idx_func), val);
  else                     d_projections(ii, idx_func) += val;
}

/* ----------------------------------------------------------------------
   CopyProjections: write descriptors of in-group atoms to array_atom
------------------------------------------------------------------------- */

template<class DeviceType, int PERATOM>
// NOLINTNEXTLINE
KOKKOS_INLINE_FUNCTION
void ComputePACEKokkos<DeviceType, PERATOM>::operator()(TagComputePACECopyProjections, const int& ii) const
{
  const int i = d_ilist[ii + chunk_offset];
  if (!(mask[i] & this->groupbit)) return;   // out-of-group atoms stay zero
  const int ncols = (int)d_projections.extent(1);
  for (int nu = 0; nu < ncols; nu++)
    d_pace_atom(i, nu) = d_projections(ii, nu);
}

/* ----------------------------------------------------------------------
   RhoDB: forward/backward A-product triangle -> dB_flatten (leave-one-out
   products) for the rank>1 functions. One thread per central atom owns its
   ii-slice, so A_list / A_forward_prod / dB_flatten are written race-free.
------------------------------------------------------------------------- */

template<class DeviceType, int PERATOM>
// NOLINTNEXTLINE
KOKKOS_INLINE_FUNCTION
void ComputePACEKokkos<DeviceType, PERATOM>::operator()(TagComputePACERhoDB, const int& ii) const
{
  const int i = d_ilist[ii + chunk_offset];
  const int mu_i = d_map(type(i));
  const int count = d_idx_ms_combs_count(mu_i);

  for (int idx_ms_combs = 0; idx_ms_combs < count; idx_ms_combs++) {
    const int idx_func = d_idx_funcs(mu_i, idx_ms_combs);
    const int rank = d_rank(mu_i, idx_func);
    if (rank == 1) continue;
    const int r = rank - 1;

    A_forward_prod(ii, idx_ms_combs, 0) = complex::one();
    for (int t = 0; t < rank; t++) {
      const int mu = d_mus(mu_i, idx_func, t);
      const int n = d_ns(mu_i, idx_func, t);
      const int l = d_ls(mu_i, idx_func, t);
      const int m = d_ms_combs(mu_i, idx_ms_combs, t);
      const int idx = l * (l + 1) + m;
      A_list(ii, idx_ms_combs, t) = A(ii, mu, idx, n - 1);
      A_forward_prod(ii, idx_ms_combs, t + 1) = A_forward_prod(ii, idx_ms_combs, t) * A_list(ii, idx_ms_combs, t);
    }

    complex A_backward_prod = complex::one();
    for (int t = r; t >= 1; t--) {
      const complex dB = A_forward_prod(ii, idx_ms_combs, t) * A_backward_prod;
      dB_flatten(ii, idx_ms_combs, t) = dB;
      A_backward_prod = A_backward_prod * A_list(ii, idx_ms_combs, t);
    }
    dB_flatten(ii, idx_ms_combs, 0) = A_forward_prod(ii, idx_ms_combs, 0) * A_backward_prod;
  }
}

/* ----------------------------------------------------------------------
   RhoDBFlat: GPU flat decomposition of RhoDB over (idx_ms_combs, ii) pairs.
   Writes to A_list / A_forward_prod / dB_flatten are disjoint per (ii,
   idx_ms_combs) — no atomics required.
------------------------------------------------------------------------- */

template<class DeviceType, int PERATOM>
// NOLINTNEXTLINE
KOKKOS_INLINE_FUNCTION
void ComputePACEKokkos<DeviceType, PERATOM>::operator()(TagComputePACERhoDBFlat, const int& iter) const
{
  const int idx_ms_combs = iter / chunk_size;
  const int ii = iter % chunk_size;
  const int i = d_ilist[ii + chunk_offset];
  const int mu_i = d_map(type(i));
  if (idx_ms_combs >= d_idx_ms_combs_count(mu_i)) return;

  const int idx_func = d_idx_funcs(mu_i, idx_ms_combs);
  const int rank = d_rank(mu_i, idx_func);
  if (rank == 1) return;
  const int r = rank - 1;

  A_forward_prod(ii, idx_ms_combs, 0) = complex::one();
  for (int t = 0; t < rank; t++) {
    const int mu = d_mus(mu_i, idx_func, t);
    const int n = d_ns(mu_i, idx_func, t);
    const int l = d_ls(mu_i, idx_func, t);
    const int m = d_ms_combs(mu_i, idx_ms_combs, t);
    const int idx = l * (l + 1) + m;
    A_list(ii, idx_ms_combs, t) = A(ii, mu, idx, n - 1);
    A_forward_prod(ii, idx_ms_combs, t + 1) = A_forward_prod(ii, idx_ms_combs, t) * A_list(ii, idx_ms_combs, t);
  }

  complex A_backward_prod = complex::one();
  for (int t = r; t >= 1; t--) {
    const complex dB = A_forward_prod(ii, idx_ms_combs, t) * A_backward_prod;
    dB_flatten(ii, idx_ms_combs, t) = dB;
    A_backward_prod = A_backward_prod * A_list(ii, idx_ms_combs, t);
  }
  dB_flatten(ii, idx_ms_combs, 0) = A_forward_prod(ii, idx_ms_combs, 0) * A_backward_prod;
}

/* ----------------------------------------------------------------------
   WeightsDB: per-function adjoint weights for the descriptor gradients.
   Unlike the pair-style energy weights (which fold dF_drho and sum over all
   functions), these keep the function index separate and use theta_dB =
   0.5 * ctilde[density 0]. Half-basis packing: +m at d_idx_sph(idx), the -m
   partner at d_idx_sph(idxm). One thread per atom -> race-free accumulation.
------------------------------------------------------------------------- */

template<class DeviceType, int PERATOM>
// NOLINTNEXTLINE
KOKKOS_INLINE_FUNCTION
void ComputePACEKokkos<DeviceType, PERATOM>::operator()(TagComputePACEWeightsDB, const int& ii) const
{
  const int i = d_ilist[ii + chunk_offset];
  const int mu_i = d_map(type(i));
  const int count = d_idx_ms_combs_count(mu_i);
  const int tbs_r1 = d_tbs_r1(mu_i);

  for (int idx_ms_combs = 0; idx_ms_combs < count; idx_ms_combs++) {
    const int idx_func = d_idx_funcs(mu_i, idx_ms_combs);
    const int rank = d_rank(mu_i, idx_func);
    if (rank == 1) continue;
    const int func_local = idx_func - tbs_r1;

    const KK_FLOAT theta_dB = d_ctildes(mu_i, idx_ms_combs, 0) * 0.5;

    for (int t = 0; t < rank; t++) {
      const int m_t = d_ms_combs(mu_i, idx_ms_combs, t);
      const int factor = (m_t % 2 == 0 ? 1 : -1);
      const complex dB = dB_flatten(ii, idx_ms_combs, t);
      const int mu_t = d_mus(mu_i, idx_func, t);
      const int n_t = d_ns(mu_i, idx_func, t);
      const int l_t = d_ls(mu_i, idx_func, t);

      const int idx = l_t * (l_t + 1) + m_t;  // (l, m)
      const int idx_sph = d_idx_sph(idx);
      if (idx_sph >= 0) {
        const complex value = theta_dB * dB;
        weights_dB(ii, func_local, mu_t, idx_sph, n_t - 1).re += value.re;
        weights_dB(ii, func_local, mu_t, idx_sph, n_t - 1).im += value.im;
      }
      const int idxm = l_t * (l_t + 1) - m_t;  // (l, -m)
      const int idxm_sph = d_idx_sph(idxm);
      if (idxm_sph >= 0) {
        const complex valuem = theta_dB * dB.conj() * (KK_FLOAT)factor;
        weights_dB(ii, func_local, mu_t, idxm_sph, n_t - 1).re += valuem.re;
        weights_dB(ii, func_local, mu_t, idxm_sph, n_t - 1).im += valuem.im;
      }
    }
  }
}

/* ----------------------------------------------------------------------
   weights_one: shared per-ms-comb body for the WeightsDB flat GPU path.
   UseAtomic=true (GPU): atomic_add into weights_dB .re/.im.
   UseAtomic=false: direct += (not called from CPU — CPU keeps its inline body).
------------------------------------------------------------------------- */

template<class DeviceType, int PERATOM>
template<bool UseAtomic>
// NOLINTNEXTLINE
KOKKOS_INLINE_FUNCTION
void ComputePACEKokkos<DeviceType, PERATOM>::weights_one(int ii, int mu_i, int tbs_r1, int idx_ms_combs) const
{
  const int idx_func = d_idx_funcs(mu_i, idx_ms_combs);
  const int rank = d_rank(mu_i, idx_func);
  if (rank == 1) return;
  const int func_local = idx_func - tbs_r1;

  const KK_FLOAT theta_dB = d_ctildes(mu_i, idx_ms_combs, 0) * 0.5;

  for (int t = 0; t < rank; t++) {
    const int m_t = d_ms_combs(mu_i, idx_ms_combs, t);
    const int factor = (m_t % 2 == 0 ? 1 : -1);
    const complex dB = dB_flatten(ii, idx_ms_combs, t);
    const int mu_t = d_mus(mu_i, idx_func, t);
    const int n_t = d_ns(mu_i, idx_func, t);
    const int l_t = d_ls(mu_i, idx_func, t);

    const int idx = l_t * (l_t + 1) + m_t;
    const int idx_sph = d_idx_sph(idx);
    if (idx_sph >= 0) {
      const complex value = theta_dB * dB;
      if constexpr (UseAtomic) {
        Kokkos::atomic_add(&weights_dB(ii, func_local, mu_t, idx_sph, n_t - 1).re, value.re);
        Kokkos::atomic_add(&weights_dB(ii, func_local, mu_t, idx_sph, n_t - 1).im, value.im);
      } else {
        weights_dB(ii, func_local, mu_t, idx_sph, n_t - 1).re += value.re;
        weights_dB(ii, func_local, mu_t, idx_sph, n_t - 1).im += value.im;
      }
    }
    const int idxm = l_t * (l_t + 1) - m_t;
    const int idxm_sph = d_idx_sph(idxm);
    if (idxm_sph >= 0) {
      const complex valuem = theta_dB * dB.conj() * (KK_FLOAT)factor;
      if constexpr (UseAtomic) {
        Kokkos::atomic_add(&weights_dB(ii, func_local, mu_t, idxm_sph, n_t - 1).re, valuem.re);
        Kokkos::atomic_add(&weights_dB(ii, func_local, mu_t, idxm_sph, n_t - 1).im, valuem.im);
      } else {
        weights_dB(ii, func_local, mu_t, idxm_sph, n_t - 1).re += valuem.re;
        weights_dB(ii, func_local, mu_t, idxm_sph, n_t - 1).im += valuem.im;
      }
    }
  }
}

/* ----------------------------------------------------------------------
   WeightsDBFlat: GPU flat decomposition over (idx_ms_combs, ii) pairs.
   Writes to weights_dB share (ii, func_local, mu_t, idx_sph, n_t-1)
   slots across idx_ms_combs -> atomic_add required.
------------------------------------------------------------------------- */

template<class DeviceType, int PERATOM>
// NOLINTNEXTLINE
KOKKOS_INLINE_FUNCTION
void ComputePACEKokkos<DeviceType, PERATOM>::operator()(TagComputePACEWeightsDBFlat, const int& iter) const
{
  const int idx_ms_combs = iter / chunk_size;
  const int ii = iter % chunk_size;
  const int i = d_ilist[ii + chunk_offset];
  const int mu_i = d_map(type(i));
  if (idx_ms_combs >= d_idx_ms_combs_count(mu_i)) return;
  const int tbs_r1 = d_tbs_r1(mu_i);
  weights_one<true>(ii, mu_i, tbs_r1, idx_ms_combs);
}

/* ----------------------------------------------------------------------
   AssembleForce: GPU-only Newton scatter of d_neighbours_dB into
   d_pace_peratom. Replaces the per-chunk deep_copy(h_neighbours_dB) + host
   scatter for the !dgradflag path. Same (atom,neighbour) TeamPolicy as
   DerivativeDB. Writes are NOT disjoint (j is shared) -> atomic_add required.
------------------------------------------------------------------------- */

template<class DeviceType, int PERATOM>
// NOLINTNEXTLINE
KOKKOS_INLINE_FUNCTION
void ComputePACEKokkos<DeviceType, PERATOM>::operator()(
    TagComputePACEAssembleForce,
    const typename Kokkos::TeamPolicy<DeviceType,TagComputePACEAssembleForce>::member_type& team) const
{
  const int atoms_per_team = (chunk_size + team.team_size() - 1) / team.team_size();
  const int ii = team.team_rank() + team.team_size() * (team.league_rank() % atoms_per_team);
  const int jj = team.league_rank() / atoms_per_team;
  if (ii >= chunk_size) return;
  if (jj >= d_ncount(ii)) return;

  const int i = d_ilist[ii + chunk_offset];
  if (!(mask(i) & this->groupbit)) return;

  const int j = d_nearest(ii, jj);
  const int itype = type(i) - 1;
  const int typeoffset = this->ndims_peratom * this->nvalues * itype;
  const int nv = this->nvalues;

  for (int func = 0; func < nv; func++) {
    const KK_FLOAT dx = d_neighbours_dB(ii, func, jj, 0);
    const KK_FLOAT dy = d_neighbours_dB(ii, func, jj, 1);
    const KK_FLOAT dz = d_neighbours_dB(ii, func, jj, 2);
    Kokkos::atomic_add(&d_pace_peratom(i, typeoffset + func),        dx);
    Kokkos::atomic_add(&d_pace_peratom(i, typeoffset + func + nv),   dy);
    Kokkos::atomic_add(&d_pace_peratom(i, typeoffset + func + 2*nv), dz);
    Kokkos::atomic_add(&d_pace_peratom(j, typeoffset + func),        -dx);
    Kokkos::atomic_add(&d_pace_peratom(j, typeoffset + func + nv),   -dy);
    Kokkos::atomic_add(&d_pace_peratom(j, typeoffset + func + 2*nv), -dz);
  }
}

/* ----------------------------------------------------------------------
   DerivativeDB: contract the per-function weights with grad_phi_nlm to fill
   neighbours_dB(ii, nu, jj, xyz) = dB_{i,nu}/dr_j. plm/dplm recursion is
   identical to the pair-style force kernel; one thread per atom loops its
   own neighbours, so neighbours_dB is written race-free.
------------------------------------------------------------------------- */

template<class DeviceType, int PERATOM>
// NOLINTNEXTLINE
KOKKOS_INLINE_FUNCTION
void ComputePACEKokkos<DeviceType, PERATOM>::operator()(TagComputePACEDerivativeDB, const int& ii) const
{
  const int i = d_ilist[ii + chunk_offset];
  const int mu_i = d_map(type(i));
  const int tbs_r1 = d_tbs_r1(mu_i);
  const int tbs = d_tbs(mu_i);
  const int ncount = d_ncount(ii);

  for (int jj = 0; jj < ncount; jj++) {
    const int mu_j = d_mu(ii, jj);
    KK_FLOAT r_hat[3];
    r_hat[0] = d_rhats(ii, jj, 0);
    r_hat[1] = d_rhats(ii, jj, 1);
    r_hat[2] = d_rhats(ii, jj, 2);
    const KK_FLOAT r = d_rnorms(ii, jj);
    const KK_FLOAT rinv = 1.0 / r;

    // rank = 1: neighbours_dB(func) = DG*Y00*ctilde * r_hat, only when mu_j matches
    for (int func_ind = 0; func_ind < tbs_r1; func_ind++) {
      if (d_mus(mu_i, func_ind, 0) != mu_j) continue;
      const int n = d_ns(mu_i, func_ind, 0) - 1;
      const KK_FLOAT ctilde = d_ctildes(mu_i, func_ind, 0);
      const KK_FLOAT DGR = dgr(ii, jj, n) * Y00 * ctilde;
      d_neighbours_dB(ii, func_ind, jj, 0) += DGR * r_hat[0];
      d_neighbours_dB(ii, func_ind, jj, 1) += DGR * r_hat[1];
      d_neighbours_dB(ii, func_ind, jj, 2) += DGR * r_hat[2];
    }

    // rank > 1: plm, dplm, ylm, dylm recursion (requires |r_hat| = 1)
    complex ylm, dylm[3];
    complex phase, phasem, mphasem1;
    complex dyx, dyy, dyz, rdy;

    const KK_FLOAT rx = r_hat[0];
    const KK_FLOAT ry = r_hat[1];
    const KK_FLOAT rz = r_hat[2];

    phase.re = rx;
    phase.im = ry;

    KK_FLOAT plm_idx, plm_idx1, plm_idx2;
    KK_FLOAT dplm_idx, dplm_idx1, dplm_idx2;

    plm_idx = plm_idx1 = plm_idx2 = 0.0;
    dplm_idx = dplm_idx1 = dplm_idx2 = 0.0;

    int idx_sph = 0;

    // m = 0
    for (int l = 0; l <= lmax; l++) {
      if (l == 0)      { plm_idx = Y00; dplm_idx = 0.0; }
      else if (l == 1) { plm_idx = Y00 * sq3 * rz; dplm_idx = Y00 * sq3; }
      else {
        plm_idx = alm(idx_sph) * (rz * plm_idx1 + blm(idx_sph) * plm_idx2);
        dplm_idx = alm(idx_sph) * (plm_idx1 + rz * dplm_idx1 + blm(idx_sph) * dplm_idx2);
      }

      ylm.re = plm_idx;
      ylm.im = 0.0;

      dyz.re = dplm_idx;
      rdy.re = dyz.re * rz;

      dylm[0].re = -rdy.re * rx; dylm[0].im = 0.0;
      dylm[1].re = -rdy.re * ry; dylm[1].im = 0.0;
      dylm[2].re = dyz.re - rdy.re * rz; dylm[2].im = 0.0;

      for (int n = 0; n < nradmax; n++) {
        const KK_FLOAT R_over_r = fr(ii, jj, l, n) * rinv;
        const KK_FLOAT DR = dfr(ii, jj, l, n);
        const complex Y_DR = ylm * DR;

        complex grad_phi[3];
        grad_phi[0] = Y_DR * r_hat[0] + dylm[0] * R_over_r;
        grad_phi[1] = Y_DR * r_hat[1] + dylm[1] * R_over_r;
        grad_phi[2] = Y_DR * r_hat[2] + dylm[2] * R_over_r;

        for (int func_local = 0; func_local < tbs; func_local++) {
          complex w_dB = weights_dB(ii, func_local, mu_j, idx_sph, n);
          if (w_dB.re == 0.0 && w_dB.im == 0.0) continue;
          const int func_ind = tbs_r1 + func_local;
          d_neighbours_dB(ii, func_ind, jj, 0) += w_dB.real_part_product(grad_phi[0]);
          d_neighbours_dB(ii, func_ind, jj, 1) += w_dB.real_part_product(grad_phi[1]);
          d_neighbours_dB(ii, func_ind, jj, 2) += w_dB.real_part_product(grad_phi[2]);
        }
      }

      plm_idx2 = plm_idx1; dplm_idx2 = dplm_idx1;
      plm_idx1 = plm_idx;  dplm_idx1 = dplm_idx;
      idx_sph++;
    }

    plm_idx = plm_idx1 = plm_idx2 = 0.0;
    dplm_idx = dplm_idx1 = dplm_idx2 = 0.0;

    // m = 1
    for (int l = 1; l <= lmax; l++) {
      if (l == 1)      { plm_idx = -sq3o2 * Y00; dplm_idx = 0.0; }
      else if (l == 2) { const KK_FLOAT t = dl(l) * plm_idx1; plm_idx = t * rz; dplm_idx = t; }
      else {
        plm_idx = alm(idx_sph) * (rz * plm_idx1 + blm(idx_sph) * plm_idx2);
        dplm_idx = alm(idx_sph) * (plm_idx1 + rz * dplm_idx1 + blm(idx_sph) * dplm_idx2);
      }

      ylm = phase * plm_idx;

      dyx.re = plm_idx; dyx.im = 0.0;
      dyy.re = 0.0;     dyy.im = plm_idx;
      dyz.re = phase.re * dplm_idx; dyz.im = phase.im * dplm_idx;

      rdy.re = rx * dyx.re + rz * dyz.re;
      rdy.im = ry * dyy.im + rz * dyz.im;

      dylm[0].re = dyx.re - rdy.re * rx; dylm[0].im = -rdy.im * rx;
      dylm[1].re = -rdy.re * ry;         dylm[1].im = dyy.im - rdy.im * ry;
      dylm[2].re = dyz.re - rdy.re * rz; dylm[2].im = dyz.im - rdy.im * rz;

      for (int n = 0; n < nradmax; n++) {
        const KK_FLOAT R_over_r = fr(ii, jj, l, n) * rinv;
        const KK_FLOAT DR = dfr(ii, jj, l, n);
        const complex Y_DR = ylm * DR;

        complex grad_phi[3];
        grad_phi[0] = Y_DR * r_hat[0] + dylm[0] * R_over_r;
        grad_phi[1] = Y_DR * r_hat[1] + dylm[1] * R_over_r;
        grad_phi[2] = Y_DR * r_hat[2] + dylm[2] * R_over_r;

        for (int func_local = 0; func_local < tbs; func_local++) {
          complex w_dB = weights_dB(ii, func_local, mu_j, idx_sph, n);
          if (w_dB.re == 0.0 && w_dB.im == 0.0) continue;
          w_dB.re *= 2.0; w_dB.im *= 2.0;  // count -m partner
          const int func_ind = tbs_r1 + func_local;
          d_neighbours_dB(ii, func_ind, jj, 0) += w_dB.real_part_product(grad_phi[0]);
          d_neighbours_dB(ii, func_ind, jj, 1) += w_dB.real_part_product(grad_phi[1]);
          d_neighbours_dB(ii, func_ind, jj, 2) += w_dB.real_part_product(grad_phi[2]);
        }
      }

      plm_idx2 = plm_idx1; dplm_idx2 = dplm_idx1;
      plm_idx1 = plm_idx;  dplm_idx1 = dplm_idx;
      idx_sph++;
    }

    plm_idx = plm_idx1 = plm_idx2 = 0.0;
    dplm_idx = dplm_idx1 = dplm_idx2 = 0.0;

    KK_FLOAT plm_mm1_mm1 = -sq3o2 * Y00; // (1, 1)

    // m > 1
    phasem = phase;
    for (int m = 2; m <= lmax; m++) {
      mphasem1.re = phasem.re * KK_FLOAT(m);
      mphasem1.im = phasem.im * KK_FLOAT(m);
      phasem = phasem * phase;

      for (int l = m; l <= lmax; l++) {
        if (l == m)          { plm_idx = cl(l) * plm_mm1_mm1; dplm_idx = 0.0; plm_mm1_mm1 = plm_idx; }
        else if (l == (m+1)) { const KK_FLOAT t = dl(l) * plm_mm1_mm1; plm_idx = t * rz; dplm_idx = t; }
        else {
          plm_idx = alm(idx_sph) * (rz * plm_idx1 + blm(idx_sph) * plm_idx2);
          dplm_idx = alm(idx_sph) * (plm_idx1 + rz * dplm_idx1 + blm(idx_sph) * dplm_idx2);
        }

        ylm.re = phasem.re * plm_idx;
        ylm.im = phasem.im * plm_idx;

        dyx = mphasem1 * plm_idx;
        dyy.re = -dyx.im; dyy.im = dyx.re;
        dyz = phasem * dplm_idx;

        rdy.re = rx * dyx.re + ry * dyy.re + rz * dyz.re;
        rdy.im = rx * dyx.im + ry * dyy.im + rz * dyz.im;

        dylm[0].re = dyx.re - rdy.re * rx; dylm[0].im = dyx.im - rdy.im * rx;
        dylm[1].re = dyy.re - rdy.re * ry; dylm[1].im = dyy.im - rdy.im * ry;
        dylm[2].re = dyz.re - rdy.re * rz; dylm[2].im = dyz.im - rdy.im * rz;

        for (int n = 0; n < nradmax; n++) {
          const KK_FLOAT R_over_r = fr(ii, jj, l, n) * rinv;
          const KK_FLOAT DR = dfr(ii, jj, l, n);
          const complex Y_DR = ylm * DR;

          complex grad_phi[3];
          grad_phi[0] = Y_DR * r_hat[0] + dylm[0] * R_over_r;
          grad_phi[1] = Y_DR * r_hat[1] + dylm[1] * R_over_r;
          grad_phi[2] = Y_DR * r_hat[2] + dylm[2] * R_over_r;

          for (int func_local = 0; func_local < tbs; func_local++) {
            complex w_dB = weights_dB(ii, func_local, mu_j, idx_sph, n);
            if (w_dB.re == 0.0 && w_dB.im == 0.0) continue;
            w_dB.re *= 2.0; w_dB.im *= 2.0;  // count -m partner
            const int func_ind = tbs_r1 + func_local;
            d_neighbours_dB(ii, func_ind, jj, 0) += w_dB.real_part_product(grad_phi[0]);
            d_neighbours_dB(ii, func_ind, jj, 1) += w_dB.real_part_product(grad_phi[1]);
            d_neighbours_dB(ii, func_ind, jj, 2) += w_dB.real_part_product(grad_phi[2]);
          }
        }

        plm_idx2 = plm_idx1; dplm_idx2 = dplm_idx1;
        plm_idx1 = plm_idx;  dplm_idx1 = dplm_idx;
        idx_sph++;
      }
    }
  } // jj
}

/* ----------------------------------------------------------------------
   derivative_one_neighbor: descriptor-gradient accumulation for one
   neighbour jj of atom ii (PERATOM=0 only).
   Writes to d_neighbours_dB(ii,...,jj,...) are disjoint per (ii,jj) so
   no atomics are required; UseAtomic is templated for symmetry only.
   Called by the CPU RangePolicy operator (UseAtomic=false, jj looped
   inside) and the GPU TeamPolicy operator (UseAtomic=false, one jj per
   team member).
------------------------------------------------------------------------- */

template<class DeviceType, int PERATOM>
template<bool UseAtomic>
// NOLINTNEXTLINE
KOKKOS_INLINE_FUNCTION
void ComputePACEKokkos<DeviceType, PERATOM>::derivative_one_neighbor(int ii, int jj) const
{
  const int i = d_ilist[ii + chunk_offset];
  const int mu_i = d_map(type(i));
  const int tbs_r1 = d_tbs_r1(mu_i);
  const int tbs = d_tbs(mu_i);

  const int mu_j = d_mu(ii, jj);
  KK_FLOAT r_hat[3];
  r_hat[0] = d_rhats(ii, jj, 0);
  r_hat[1] = d_rhats(ii, jj, 1);
  r_hat[2] = d_rhats(ii, jj, 2);
  const KK_FLOAT r = d_rnorms(ii, jj);
  const KK_FLOAT rinv = 1.0 / r;

  // rank = 1: neighbours_dB(func) = DG*Y00*ctilde * r_hat, only when mu_j matches
  for (int func_ind = 0; func_ind < tbs_r1; func_ind++) {
    if (d_mus(mu_i, func_ind, 0) != mu_j) continue;
    const int n = d_ns(mu_i, func_ind, 0) - 1;
    const KK_FLOAT ctilde = d_ctildes(mu_i, func_ind, 0);
    const KK_FLOAT DGR = dgr(ii, jj, n) * Y00 * ctilde;
    d_neighbours_dB(ii, func_ind, jj, 0) += DGR * r_hat[0];
    d_neighbours_dB(ii, func_ind, jj, 1) += DGR * r_hat[1];
    d_neighbours_dB(ii, func_ind, jj, 2) += DGR * r_hat[2];
  }

  // rank > 1: plm, dplm, ylm, dylm recursion (requires |r_hat| = 1)
  complex ylm, dylm[3];
  complex phase, phasem, mphasem1;
  complex dyx, dyy, dyz, rdy;

  const KK_FLOAT rx = r_hat[0];
  const KK_FLOAT ry = r_hat[1];
  const KK_FLOAT rz = r_hat[2];

  phase.re = rx;
  phase.im = ry;

  KK_FLOAT plm_idx, plm_idx1, plm_idx2;
  KK_FLOAT dplm_idx, dplm_idx1, dplm_idx2;

  plm_idx = plm_idx1 = plm_idx2 = 0.0;
  dplm_idx = dplm_idx1 = dplm_idx2 = 0.0;

  int idx_sph = 0;

  // m = 0
  for (int l = 0; l <= lmax; l++) {
    if (l == 0)      { plm_idx = Y00; dplm_idx = 0.0; }
    else if (l == 1) { plm_idx = Y00 * sq3 * rz; dplm_idx = Y00 * sq3; }
    else {
      plm_idx = alm(idx_sph) * (rz * plm_idx1 + blm(idx_sph) * plm_idx2);
      dplm_idx = alm(idx_sph) * (plm_idx1 + rz * dplm_idx1 + blm(idx_sph) * dplm_idx2);
    }

    ylm.re = plm_idx;
    ylm.im = 0.0;

    dyz.re = dplm_idx;
    rdy.re = dyz.re * rz;

    dylm[0].re = -rdy.re * rx; dylm[0].im = 0.0;
    dylm[1].re = -rdy.re * ry; dylm[1].im = 0.0;
    dylm[2].re = dyz.re - rdy.re * rz; dylm[2].im = 0.0;

    for (int n = 0; n < nradmax; n++) {
      const KK_FLOAT R_over_r = fr(ii, jj, l, n) * rinv;
      const KK_FLOAT DR = dfr(ii, jj, l, n);
      const complex Y_DR = ylm * DR;

      complex grad_phi[3];
      grad_phi[0] = Y_DR * r_hat[0] + dylm[0] * R_over_r;
      grad_phi[1] = Y_DR * r_hat[1] + dylm[1] * R_over_r;
      grad_phi[2] = Y_DR * r_hat[2] + dylm[2] * R_over_r;

      for (int func_local = 0; func_local < tbs; func_local++) {
        complex w_dB = weights_dB(ii, func_local, mu_j, idx_sph, n);
        if (w_dB.re == 0.0 && w_dB.im == 0.0) continue;
        const int func_ind = tbs_r1 + func_local;
        d_neighbours_dB(ii, func_ind, jj, 0) += w_dB.real_part_product(grad_phi[0]);
        d_neighbours_dB(ii, func_ind, jj, 1) += w_dB.real_part_product(grad_phi[1]);
        d_neighbours_dB(ii, func_ind, jj, 2) += w_dB.real_part_product(grad_phi[2]);
      }
    }

    plm_idx2 = plm_idx1; dplm_idx2 = dplm_idx1;
    plm_idx1 = plm_idx;  dplm_idx1 = dplm_idx;
    idx_sph++;
  }

  plm_idx = plm_idx1 = plm_idx2 = 0.0;
  dplm_idx = dplm_idx1 = dplm_idx2 = 0.0;

  // m = 1
  for (int l = 1; l <= lmax; l++) {
    if (l == 1)      { plm_idx = -sq3o2 * Y00; dplm_idx = 0.0; }
    else if (l == 2) { const KK_FLOAT t = dl(l) * plm_idx1; plm_idx = t * rz; dplm_idx = t; }
    else {
      plm_idx = alm(idx_sph) * (rz * plm_idx1 + blm(idx_sph) * plm_idx2);
      dplm_idx = alm(idx_sph) * (plm_idx1 + rz * dplm_idx1 + blm(idx_sph) * dplm_idx2);
    }

    ylm = phase * plm_idx;

    dyx.re = plm_idx; dyx.im = 0.0;
    dyy.re = 0.0;     dyy.im = plm_idx;
    dyz.re = phase.re * dplm_idx; dyz.im = phase.im * dplm_idx;

    rdy.re = rx * dyx.re + rz * dyz.re;
    rdy.im = ry * dyy.im + rz * dyz.im;

    dylm[0].re = dyx.re - rdy.re * rx; dylm[0].im = -rdy.im * rx;
    dylm[1].re = -rdy.re * ry;         dylm[1].im = dyy.im - rdy.im * ry;
    dylm[2].re = dyz.re - rdy.re * rz; dylm[2].im = dyz.im - rdy.im * rz;

    for (int n = 0; n < nradmax; n++) {
      const KK_FLOAT R_over_r = fr(ii, jj, l, n) * rinv;
      const KK_FLOAT DR = dfr(ii, jj, l, n);
      const complex Y_DR = ylm * DR;

      complex grad_phi[3];
      grad_phi[0] = Y_DR * r_hat[0] + dylm[0] * R_over_r;
      grad_phi[1] = Y_DR * r_hat[1] + dylm[1] * R_over_r;
      grad_phi[2] = Y_DR * r_hat[2] + dylm[2] * R_over_r;

      for (int func_local = 0; func_local < tbs; func_local++) {
        complex w_dB = weights_dB(ii, func_local, mu_j, idx_sph, n);
        if (w_dB.re == 0.0 && w_dB.im == 0.0) continue;
        w_dB.re *= 2.0; w_dB.im *= 2.0;  // count -m partner
        const int func_ind = tbs_r1 + func_local;
        d_neighbours_dB(ii, func_ind, jj, 0) += w_dB.real_part_product(grad_phi[0]);
        d_neighbours_dB(ii, func_ind, jj, 1) += w_dB.real_part_product(grad_phi[1]);
        d_neighbours_dB(ii, func_ind, jj, 2) += w_dB.real_part_product(grad_phi[2]);
      }
    }

    plm_idx2 = plm_idx1; dplm_idx2 = dplm_idx1;
    plm_idx1 = plm_idx;  dplm_idx1 = dplm_idx;
    idx_sph++;
  }

  plm_idx = plm_idx1 = plm_idx2 = 0.0;
  dplm_idx = dplm_idx1 = dplm_idx2 = 0.0;

  KK_FLOAT plm_mm1_mm1 = -sq3o2 * Y00; // (1, 1)

  // m > 1
  phasem = phase;
  for (int m = 2; m <= lmax; m++) {
    mphasem1.re = phasem.re * KK_FLOAT(m);
    mphasem1.im = phasem.im * KK_FLOAT(m);
    phasem = phasem * phase;

    for (int l = m; l <= lmax; l++) {
      if (l == m)          { plm_idx = cl(l) * plm_mm1_mm1; dplm_idx = 0.0; plm_mm1_mm1 = plm_idx; }
      else if (l == (m+1)) { const KK_FLOAT t = dl(l) * plm_mm1_mm1; plm_idx = t * rz; dplm_idx = t; }
      else {
        plm_idx = alm(idx_sph) * (rz * plm_idx1 + blm(idx_sph) * plm_idx2);
        dplm_idx = alm(idx_sph) * (plm_idx1 + rz * dplm_idx1 + blm(idx_sph) * dplm_idx2);
      }

      ylm.re = phasem.re * plm_idx;
      ylm.im = phasem.im * plm_idx;

      dyx = mphasem1 * plm_idx;
      dyy.re = -dyx.im; dyy.im = dyx.re;
      dyz = phasem * dplm_idx;

      rdy.re = rx * dyx.re + ry * dyy.re + rz * dyz.re;
      rdy.im = rx * dyx.im + ry * dyy.im + rz * dyz.im;

      dylm[0].re = dyx.re - rdy.re * rx; dylm[0].im = dyx.im - rdy.im * rx;
      dylm[1].re = dyy.re - rdy.re * ry; dylm[1].im = dyy.im - rdy.im * ry;
      dylm[2].re = dyz.re - rdy.re * rz; dylm[2].im = dyz.im - rdy.im * rz;

      for (int n = 0; n < nradmax; n++) {
        const KK_FLOAT R_over_r = fr(ii, jj, l, n) * rinv;
        const KK_FLOAT DR = dfr(ii, jj, l, n);
        const complex Y_DR = ylm * DR;

        complex grad_phi[3];
        grad_phi[0] = Y_DR * r_hat[0] + dylm[0] * R_over_r;
        grad_phi[1] = Y_DR * r_hat[1] + dylm[1] * R_over_r;
        grad_phi[2] = Y_DR * r_hat[2] + dylm[2] * R_over_r;

        for (int func_local = 0; func_local < tbs; func_local++) {
          complex w_dB = weights_dB(ii, func_local, mu_j, idx_sph, n);
          if (w_dB.re == 0.0 && w_dB.im == 0.0) continue;
          w_dB.re *= 2.0; w_dB.im *= 2.0;  // count -m partner
          const int func_ind = tbs_r1 + func_local;
          d_neighbours_dB(ii, func_ind, jj, 0) += w_dB.real_part_product(grad_phi[0]);
          d_neighbours_dB(ii, func_ind, jj, 1) += w_dB.real_part_product(grad_phi[1]);
          d_neighbours_dB(ii, func_ind, jj, 2) += w_dB.real_part_product(grad_phi[2]);
        }
      }

      plm_idx2 = plm_idx1; dplm_idx2 = dplm_idx1;
      plm_idx1 = plm_idx;  dplm_idx1 = dplm_idx;
      idx_sph++;
    }
  }
}

/* ----------------------------------------------------------------------
   DerivativeDB GPU: TeamPolicy (atom, neighbour) decomposition for PERATOM=0.
   One team member per (ii, jj) pair; league_size = ceil(chunk/32)*maxneigh.
   Writes to d_neighbours_dB are disjoint per (ii,jj) — no atomics needed.
------------------------------------------------------------------------- */

template<class DeviceType, int PERATOM>
// NOLINTNEXTLINE
KOKKOS_INLINE_FUNCTION
void ComputePACEKokkos<DeviceType, PERATOM>::operator()(TagComputePACEDerivativeDB, const typename Kokkos::TeamPolicy<DeviceType,TagComputePACEDerivativeDB>::member_type& team) const
{
  int ii = team.team_rank() + team.team_size() * (team.league_rank() %
           ((chunk_size+team.team_size()-1)/team.team_size()));
  if (ii >= chunk_size) return;

  const int jj = team.league_rank() / ((chunk_size+team.team_size()-1)/team.team_size());
  if (jj >= d_ncount(ii)) return;

  derivative_one_neighbor<false>(ii, jj);
}

/* ---------------------------------------------------------------------- */

template<class DeviceType, int PERATOM>
void ComputePACEKokkos<DeviceType, PERATOM>::grow(int natom, int maxneigh)
{
  if ((int)A.extent(0) < natom) {
    MemKK::realloc_kokkos(A_sph, "pace:A_sph", natom, nelements, idx_sph_max, nradmax + 1);
    MemKK::realloc_kokkos(A, "pace:A", natom, nelements, (lmax + 1) * (lmax + 1), nradmax + 1);
    MemKK::realloc_kokkos(A_rank1, "pace:A_rank1", natom, nelements, nradbase);

    auto basis_set = this->acecimpl->basis_set;
    MemKK::realloc_kokkos(d_projections, "pace:projections", natom, this->nvalues);

    if constexpr (!PERATOM) {
      // A_list/A_forward_prod are the leave-one-out product temporaries for RhoDB;
      // dB_flatten/weights_dB are subsequent gradient scratch — all PERATOM=0 only.
      MemKK::realloc_kokkos(A_list, "pace:A_list", natom, idx_ms_combs_max, basis_set->rankmax);
      // +1 to avoid out-of-boundary access in the double-triangular accumulation scheme
      MemKK::realloc_kokkos(A_forward_prod, "pace:A_forward_prod", natom, idx_ms_combs_max, basis_set->rankmax + 1);
      MemKK::realloc_kokkos(dB_flatten, "pace:dB_flatten", natom, idx_ms_combs_max, basis_set->rankmax);
      MemKK::realloc_kokkos(weights_dB, "pace:weights_dB", natom,
                            total_basis_size_rankgt1_max, nelements, idx_sph_max, nradmax + 1);
    }
  }

  // fr/gr are needed by Radial+Ai whenever running on host (both PERATOM values);
  // for PERATOM=1 GPU the fused kernel uses thread-local storage, so fr/gr stay
  // unallocated there. Use d_mu extent as the condition proxy for the neigh-list block.
  const bool need_fr    = !PERATOM || host_flag;
  const bool neigh_grow = ((int)d_mu.extent(0) < natom) || ((int)d_mu.extent(1) < maxneigh);
  const bool fr_grow    = need_fr && (((int)fr.extent(0) < natom) || ((int)fr.extent(1) < maxneigh));

  if (fr_grow || neigh_grow) {
    if constexpr (!PERATOM) {
      MemKK::realloc_kokkos(d_neighbours_dB, "pace:neighbours_dB", natom, this->nvalues, maxneigh, 3);
      // dfr/dgr + d_values/d_derivatives: read by DerivativeDB (PERATOM=0 only)
      MemKK::realloc_kokkos(dfr, "pace:dfr", natom, maxneigh, lmax + 1, nradmax);
      MemKK::realloc_kokkos(dgr, "pace:dgr", natom, maxneigh, nradbase);
      const int max_num_functions = MAX(nradbase, nradmax*(lmax + 1));
      MemKK::realloc_kokkos(d_values, "pace:d_values", natom, maxneigh, max_num_functions);
      MemKK::realloc_kokkos(d_derivatives, "pace:d_derivatives", natom, maxneigh, max_num_functions);
    }
    if (need_fr) {
      MemKK::realloc_kokkos(fr, "pace:fr", natom, maxneigh, lmax + 1, nradmax);
      MemKK::realloc_kokkos(gr, "pace:gr", natom, maxneigh, nradbase);
    }

    // short neigh list (always)
    MemKK::realloc_kokkos(d_ncount, "pace:ncount", natom);
    MemKK::realloc_kokkos(d_mu, "pace:mu", natom, maxneigh);
    MemKK::realloc_kokkos(d_rhats, "pace:rhats", natom, maxneigh);
    MemKK::realloc_kokkos(d_rnorms, "pace:rnorms", natom, maxneigh);
    MemKK::realloc_kokkos(d_nearest, "pace:nearest", natom, maxneigh);
  }
}

/* ---------------------------------------------------------------------- */

template<class DeviceType, int PERATOM>
void ComputePACEKokkos<DeviceType, PERATOM>::copy_pertype()
{
  auto basis_set = this->acecimpl->basis_set;

  MemKK::realloc_kokkos(d_rho_core_cutoff, "pace:rho_core_cutoff", nelements);
  MemKK::realloc_kokkos(d_drho_core_cutoff, "pace:drho_core_cutoff", nelements);
  MemKK::realloc_kokkos(d_E0vals, "pace:E0vals", nelements);
  MemKK::realloc_kokkos(d_ndensity, "pace:ndensity", nelements);
  MemKK::realloc_kokkos(d_npoti, "pace:npoti", nelements);

  auto h_rho_core_cutoff = Kokkos::create_mirror_view(d_rho_core_cutoff);
  auto h_drho_core_cutoff = Kokkos::create_mirror_view(d_drho_core_cutoff);
  auto h_E0vals = Kokkos::create_mirror_view(d_E0vals);
  auto h_ndensity = Kokkos::create_mirror_view(d_ndensity);
  auto h_npoti = Kokkos::create_mirror_view(d_npoti);

  for (int n = 0; n < nelements; n++) {
    h_rho_core_cutoff[n] = basis_set->map_embedding_specifications.at(n).rho_core_cutoff;
    h_drho_core_cutoff[n] = basis_set->map_embedding_specifications.at(n).drho_core_cutoff;

    h_E0vals(n) = basis_set->E0vals(n);

    h_ndensity(n) = basis_set->map_embedding_specifications.at(n).ndensity;

    string npoti = basis_set->map_embedding_specifications.at(n).npoti;
    if (npoti == "FinnisSinclair")
      h_npoti(n) = FS;
    else if (npoti == "FinnisSinclairShiftedScaled")
      h_npoti(n) = FS_SHIFTEDSCALED;
    else
      this->error->all(FLERR, "compute pace/kk: unsupported embedding type '{}'; "
                               "supported types: FinnisSinclair, FinnisSinclairShiftedScaled", npoti);
  }

  Kokkos::deep_copy(d_rho_core_cutoff, h_rho_core_cutoff);
  Kokkos::deep_copy(d_drho_core_cutoff, h_drho_core_cutoff);
  Kokkos::deep_copy(d_E0vals, h_E0vals);
  Kokkos::deep_copy(d_ndensity, h_ndensity);
  Kokkos::deep_copy(d_npoti, h_npoti);

  MemKK::realloc_kokkos(d_wpre, "pace:wpre", nelements, basis_set->ndensitymax);
  MemKK::realloc_kokkos(d_mexp, "pace:mexp", nelements, basis_set->ndensitymax);

  auto h_wpre = Kokkos::create_mirror_view(d_wpre);
  auto h_mexp = Kokkos::create_mirror_view(d_mexp);

  for (int n = 0; n < nelements; n++) {
    const int ndensity = basis_set->map_embedding_specifications.at(n).ndensity;
    for (int p = 0; p < ndensity; p++) {
      h_wpre(n, p) = basis_set->map_embedding_specifications.at(n).FS_parameters.at(p * 2 + 0);
      h_mexp(n, p) = basis_set->map_embedding_specifications.at(n).FS_parameters.at(p * 2 + 1);
    }
  }

  Kokkos::deep_copy(d_wpre, h_wpre);
  Kokkos::deep_copy(d_mexp, h_mexp);

  // ZBL core-rep
  MemKK::realloc_kokkos(d_cut_in, "pace:d_cut_in", nelements, nelements);
  MemKK::realloc_kokkos(d_dcut_in, "pace:d_dcut_in", nelements, nelements);
  auto h_cut_in = Kokkos::create_mirror_view(d_cut_in);
  auto h_dcut_in = Kokkos::create_mirror_view(d_dcut_in);

  for (int mu_i = 0; mu_i < nelements; ++mu_i) {
    for (int mu_j = 0; mu_j < nelements; ++mu_j) {
      h_cut_in(mu_i,mu_j) = basis_set->map_bond_specifications.at({mu_i,mu_j}).rcut_in;
      h_dcut_in(mu_i,mu_j) = basis_set->map_bond_specifications.at({mu_i,mu_j}).dcut_in;
    }
  }
  Kokkos::deep_copy(d_cut_in, h_cut_in);
  Kokkos::deep_copy(d_dcut_in, h_dcut_in);

  is_zbl = basis_set->radial_functions->inner_cutoff_type == "zbl";
  if (is_zbl)
    this->error->all(FLERR, "compute pace/kk: ZBL inner cutoff is not supported; use compute pace (CPU) instead");

  // per-pair cutoff squared, indexed by (mu_i, mu_j) element type
  MemKK::realloc_kokkos(d_cutsq, "pace:cutsq", nelements, nelements);
  auto h_cutsq = Kokkos::create_mirror_view(d_cutsq);
  for (int mu_i = 0; mu_i < nelements; ++mu_i) {
    for (int mu_j = 0; mu_j < nelements; ++mu_j) {
      const double rcut = basis_set->map_bond_specifications.at({mu_i,mu_j}).rcut;
      h_cutsq(mu_i, mu_j) = rcut * rcut;
    }
  }
  Kokkos::deep_copy(d_cutsq, h_cutsq);
}

/* ---------------------------------------------------------------------- */

template<class DeviceType, int PERATOM>
void ComputePACEKokkos<DeviceType, PERATOM>::copy_splines()
{
  auto basis_set = this->acecimpl->basis_set;

  deallocate_views_of_views();

  k_splines_gk = Kokkos::DualView<SplineInterpolatorKokkos**, DeviceType>("pace:splines_gk", nelements, nelements);
  k_splines_rnl = Kokkos::DualView<SplineInterpolatorKokkos**, DeviceType>("pace:splines_rnl", nelements, nelements);
  k_splines_hc = Kokkos::DualView<SplineInterpolatorKokkos**, DeviceType>("pace:splines_hc", nelements, nelements);

  ACERadialFunctions* radial_functions = dynamic_cast<ACERadialFunctions*>(basis_set->radial_functions);

  if (radial_functions == nullptr)
    this->error->all(FLERR,"Chosen radial basis style not supported by compute pace/kk");

  for (int i = 0; i < nelements; i++) {
    for (int j = 0; j < nelements; j++) {
      k_splines_gk.view_host()(i, j) = radial_functions->splines_gk(i, j);
      k_splines_rnl.view_host()(i, j) = radial_functions->splines_rnl(i, j);
      k_splines_hc.view_host()(i, j) = radial_functions->splines_hc(i, j);
    }
  }

  k_splines_gk.modify_host();
  k_splines_rnl.modify_host();
  k_splines_hc.modify_host();

  k_splines_gk.sync_device();
  k_splines_rnl.sync_device();
  k_splines_hc.sync_device();
}

/* ---------------------------------------------------------------------- */

template<class DeviceType, int PERATOM>
void ComputePACEKokkos<DeviceType, PERATOM>::copy_tilde()
{
  auto basis_set = this->acecimpl->basis_set;

  // flatten loops, get per-element count and max

  idx_ms_combs_max = 0;
  int total_basis_size_max = 0;
  total_basis_size_rankgt1_max = 0;

  MemKK::realloc_kokkos(d_idx_ms_combs_count, "pace:idx_ms_combs_count", nelements);
  MemKK::realloc_kokkos(d_tbs_r1, "pace:tbs_r1", nelements);
  MemKK::realloc_kokkos(d_tbs, "pace:tbs", nelements);
  auto h_idx_ms_combs_count = Kokkos::create_mirror_view(d_idx_ms_combs_count);
  auto h_tbs_r1 = Kokkos::create_mirror_view(d_tbs_r1);
  auto h_tbs = Kokkos::create_mirror_view(d_tbs);

  for (int mu = 0; mu < nelements; mu++) {
    int idx_ms_combs = 0;
    const int total_basis_size_rank1 = basis_set->total_basis_size_rank1[mu];
    const int total_basis_size = basis_set->total_basis_size[mu];

    ACECTildeBasisFunction *basis = basis_set->basis[mu];

    // rank=1
    for (int func_rank1_ind = 0; func_rank1_ind < total_basis_size_rank1; ++func_rank1_ind)
      idx_ms_combs++;

    // rank > 1
    for (int idx_func = 0; idx_func < total_basis_size; ++idx_func) {
      ACECTildeBasisFunction *func = &basis[idx_func];

      // loop over {ms} combinations in sum
      for (int ms_ind = 0; ms_ind < func->num_ms_combs; ++ms_ind)
        idx_ms_combs++;
    }
    h_idx_ms_combs_count(mu) = idx_ms_combs;
    h_tbs_r1(mu) = total_basis_size_rank1;
    h_tbs(mu) = total_basis_size;
    idx_ms_combs_max = MAX(idx_ms_combs_max, idx_ms_combs);
    total_basis_size_max = MAX(total_basis_size_max, total_basis_size_rank1 + total_basis_size);
    total_basis_size_rankgt1_max = MAX(total_basis_size_rankgt1_max, total_basis_size);
  }

  Kokkos::deep_copy(d_idx_ms_combs_count, h_idx_ms_combs_count);
  Kokkos::deep_copy(d_tbs_r1, h_tbs_r1);
  Kokkos::deep_copy(d_tbs, h_tbs);

  MemKK::realloc_kokkos(d_rank, "pace:rank", nelements, total_basis_size_max);
  MemKK::realloc_kokkos(d_num_ms_combs, "pace:num_ms_combs", nelements, total_basis_size_max);
  MemKK::realloc_kokkos(d_idx_funcs, "pace:idx_func", nelements, idx_ms_combs_max);
  MemKK::realloc_kokkos(d_mus, "pace:mus", nelements, total_basis_size_max, basis_set->rankmax);
  MemKK::realloc_kokkos(d_ns, "pace:ns", nelements, total_basis_size_max, basis_set->rankmax);
  MemKK::realloc_kokkos(d_ls, "pace:ls", nelements, total_basis_size_max, basis_set->rankmax);
  MemKK::realloc_kokkos(d_func_base, "pace:func_base", nelements, total_basis_size_max, basis_set->rankmax);
  MemKK::realloc_kokkos(d_ms_combs, "pace:ms_combs", nelements, idx_ms_combs_max, basis_set->rankmax);
  MemKK::realloc_kokkos(d_ctildes, "pace:ctildes", nelements, idx_ms_combs_max, basis_set->ndensitymax);

  auto h_rank = Kokkos::create_mirror_view(d_rank);
  auto h_num_ms_combs = Kokkos::create_mirror_view(d_num_ms_combs);
  auto h_idx_funcs = Kokkos::create_mirror_view(d_idx_funcs);
  auto h_mus = Kokkos::create_mirror_view(d_mus);
  auto h_ns = Kokkos::create_mirror_view(d_ns);
  auto h_ls = Kokkos::create_mirror_view(d_ls);
  auto h_func_base = Kokkos::create_mirror_view(d_func_base);
  auto h_ms_combs = Kokkos::create_mirror_view(d_ms_combs);
  auto h_ctildes = Kokkos::create_mirror_view(d_ctildes);

  // copy values on host

  for (int mu = 0; mu < nelements; mu++) {
    const int total_basis_size_rank1 = basis_set->total_basis_size_rank1[mu];
    const int total_basis_size = basis_set->total_basis_size[mu];

    ACECTildeBasisFunction *basis_rank1 = basis_set->basis_rank1[mu];
    ACECTildeBasisFunction *basis = basis_set->basis[mu];

    const int ndensity = basis_set->map_embedding_specifications.at(mu).ndensity;

    int idx_ms_combs = 0;

    // rank=1
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

    // rank > 1
    for (int idx_func = 0; idx_func < total_basis_size; ++idx_func) {
      ACECTildeBasisFunction *func = &basis[idx_func];
      // TODO: check if func->ctildes are zero, then skip

      const int idx_func_through = total_basis_size_rank1 + idx_func;

      const int rank = h_rank(mu, idx_func_through) = func->rank;
      h_num_ms_combs(mu, idx_func_through) = func->num_ms_combs;
      for (int t = 0; t < rank; t++) {
        h_mus(mu, idx_func_through, t) = func->mus[t];
        h_ns(mu, idx_func_through, t) = func->ns[t];
        h_ls(mu, idx_func_through, t) = func->ls[t];
        const int l_t = func->ls[t];
        h_func_base(mu, idx_func_through, t) = l_t * (l_t + 1);
      }

      // loop over {ms} combinations in sum
      for (int ms_ind = 0; ms_ind < func->num_ms_combs; ++ms_ind) {
        auto ms = &func->ms_combs[ms_ind * rank]; // current ms-combination (of length = rank)
        for (int t = 0; t < rank; t++)
          h_ms_combs(mu, idx_ms_combs, t) = ms[t];

        for (int p = 0; p < ndensity; ++p) {
          // real-part only multiplication
          h_ctildes(mu, idx_ms_combs, p) = func->ctildes[ms_ind * ndensity + p];
        }

        h_idx_funcs(mu, idx_ms_combs) = idx_func_through;
        idx_ms_combs++;
      }
    }
  }

  Kokkos::deep_copy(d_rank, h_rank);
  Kokkos::deep_copy(d_num_ms_combs, h_num_ms_combs);
  Kokkos::deep_copy(d_idx_funcs, h_idx_funcs);
  Kokkos::deep_copy(d_mus, h_mus);
  Kokkos::deep_copy(d_ns, h_ns);
  Kokkos::deep_copy(d_ls, h_ls);
  Kokkos::deep_copy(d_func_base, h_func_base);
  Kokkos::deep_copy(d_ms_combs, h_ms_combs);
  Kokkos::deep_copy(d_ctildes, h_ctildes);
}

/* ---------------------------------------------------------------------- */

template<class DeviceType, int PERATOM>
void ComputePACEKokkos<DeviceType, PERATOM>::pre_compute_harmonics(int lmax)
{
  auto h_idx_sph = Kokkos::create_mirror_view(d_idx_sph);
  auto h_alm = Kokkos::create_mirror_view(alm);
  auto h_blm = Kokkos::create_mirror_view(blm);
  auto h_cl = Kokkos::create_mirror_view(cl);
  auto h_dl = Kokkos::create_mirror_view(dl);

  Kokkos::deep_copy(h_idx_sph,-1);

  int idx_sph = 0;
  for (int m = 0; m <= lmax; m++) {
    const double msq = m * m;
    for (int l = m; l <= lmax; l++) {
      const int idx = l * (l + 1) + m; // (l, m)
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

  for (int l = 1; l <= lmax; l++) {
    h_cl(l) = -sqrt(1.0 + 0.5 / (double(l)));
    h_dl(l) = sqrt(double(2 * (l - 1) + 3));
  }

  Kokkos::deep_copy(d_idx_sph, h_idx_sph);
  Kokkos::deep_copy(alm, h_alm);
  Kokkos::deep_copy(blm, h_blm);
  Kokkos::deep_copy(cl, h_cl);
  Kokkos::deep_copy(dl, h_dl);
}

/* ---------------------------------------------------------------------- */

template<class DeviceType, int PERATOM>
// NOLINTNEXTLINE
KOKKOS_INLINE_FUNCTION
void ComputePACEKokkos<DeviceType, PERATOM>::evaluate_splines(const int ii, const int jj, KK_FLOAT r,
                                                  int /*nradbase_c*/, int /*nradial_c*/,
                                                  int mu_i, int mu_j) const
{
  auto &spline_gk = k_splines_gk.template view<DeviceType>()(mu_i, mu_j);
  auto &spline_rnl = k_splines_rnl.template view<DeviceType>()(mu_i, mu_j);

  if constexpr (PERATOM) {
    // PERATOM=1: no derivatives needed; write directly to fr/gr (skip d_values)
    spline_gk.calcSplines(ii, jj, r, gr);
    spline_rnl.calcSplines(ii, jj, r, fr);
  } else {
    // PERATOM=0: derivatives needed for DerivativeDB; use proven old path
    spline_gk.calcSplines(ii, jj, r, gr, dgr);
    spline_rnl.calcSplines(ii, jj, r, d_values, d_derivatives);
    for (int ll = 0; ll < (int)fr.extent(2); ll++) {
      for (int kk = 0; kk < (int)fr.extent(3); kk++) {
        const int flatten = kk*fr.extent(2) + ll;
        fr(ii, jj, ll, kk)  = d_values(ii, jj, flatten);
        dfr(ii, jj, ll, kk) = d_derivatives(ii, jj, flatten);
      }
    }
  }
}

/* ---------------------------------------------------------------------- */

template<class DeviceType, int PERATOM>
void ComputePACEKokkos<DeviceType, PERATOM>::SplineInterpolatorKokkos::operator=(const SplineInterpolator &spline) {
    cutoff = spline.cutoff;
    deltaSplineBins = spline.deltaSplineBins;
    ntot = spline.ntot;
    nlut = spline.nlut;
    invrscalelookup = spline.invrscalelookup;
    rscalelookup = spline.rscalelookup;
    num_of_functions = spline.num_of_functions;

    lookupTable = t_ace_3d4_lr("lookupTable", ntot+1, num_of_functions);
    auto h_lookupTable = Kokkos::create_mirror_view(lookupTable);
    for (int i = 0; i < ntot+1; i++)
        for (int j = 0; j < num_of_functions; j++)
            for (int k = 0; k < 4; k++)
                h_lookupTable(i, j, k) = spline.lookupTable(i, j, k);
    Kokkos::deep_copy(lookupTable, h_lookupTable);
}

/* ---------------------------------------------------------------------- */

template<class DeviceType, int PERATOM>
// NOLINTNEXTLINE
KOKKOS_INLINE_FUNCTION
void ComputePACEKokkos<DeviceType, PERATOM>::SplineInterpolatorKokkos::calcSplines(const int ii, const int jj, const KK_FLOAT r, const t_ace_3d &d_values, const t_ace_3d &d_derivatives) const
{
  KK_FLOAT wl, wl2, wl3, w2l1, w3l2;
  KK_FLOAT c[4];
  KK_FLOAT x = r * rscalelookup;
  int nl = static_cast<int>(floor(x));

  if (nl <= 0)
    Kokkos::abort("Encountered very small distance. Stopping.");

  if (nl < nlut) {
    wl = x - KK_FLOAT(nl);
    wl2 = wl * wl;
    wl3 = wl2 * wl;
    w2l1 = 2.0 * wl;
    w3l2 = 3.0 * wl2;
    for (int func_id = 0; func_id < num_of_functions; func_id++) {
      for (int idx = 0; idx < 4; idx++)
        c[idx] = lookupTable(nl, func_id, idx);
      d_values(ii, jj, func_id) = c[0] + c[1] * wl + c[2] * wl2 + c[3] * wl3;
      d_derivatives(ii, jj, func_id) = (c[1] + c[2] * w2l1 + c[3] * w3l2) * rscalelookup;
    }
  } else { // fill with zeroes
    for (int func_id = 0; func_id < num_of_functions; func_id++) {
      d_values(ii, jj, func_id) = 0.0;
      d_derivatives(ii, jj, func_id) = 0.0;
    }
  }
}

/* ---------------------------------------------------------------------- */

// Vals-only variant for 3D (gr without dgr — used in PERATOM=1 path)
template<class DeviceType, int PERATOM>
// NOLINTNEXTLINE
KOKKOS_INLINE_FUNCTION
void ComputePACEKokkos<DeviceType, PERATOM>::SplineInterpolatorKokkos::calcSplines(
  const int ii, const int jj, const KK_FLOAT r, const t_ace_3d &vals) const
{
  KK_FLOAT wl, wl2, wl3;
  KK_FLOAT c[4];
  KK_FLOAT x = r * rscalelookup;
  int nl = static_cast<int>(floor(x));

  if (nl <= 0) Kokkos::abort("Encountered very small distance. Stopping.");

  if (nl < nlut) {
    wl = x - KK_FLOAT(nl); wl2 = wl*wl; wl3 = wl2*wl;
    for (int func_id = 0; func_id < num_of_functions; func_id++) {
      for (int idx = 0; idx < 4; idx++) c[idx] = lookupTable(nl, func_id, idx);
      vals(ii, jj, func_id) = c[0] + c[1]*wl + c[2]*wl2 + c[3]*wl3;
    }
  } else {
    for (int func_id = 0; func_id < num_of_functions; func_id++)
      vals(ii, jj, func_id) = 0.0;
  }
}

/* ---------------------------------------------------------------------- */

// 4D vals+derivs: writes directly to fr/dfr, eliminating the d_values→fr reshape.
// func_id linearisation: func_id = kk*(lmax+1) + ll  (matches flatten in old code).
template<class DeviceType, int PERATOM>
// NOLINTNEXTLINE
KOKKOS_INLINE_FUNCTION
void ComputePACEKokkos<DeviceType, PERATOM>::SplineInterpolatorKokkos::calcSplines(
  const int ii, const int jj, const KK_FLOAT r,
  const t_ace_4d &vals4d, const t_ace_4d &derivs4d) const
{
  KK_FLOAT wl, wl2, wl3, w2l1, w3l2;
  KK_FLOAT c[4];
  KK_FLOAT x = r * rscalelookup;
  int nl = static_cast<int>(floor(x));

  if (nl <= 0) Kokkos::abort("Encountered very small distance. Stopping.");

  const int nll = vals4d.extent(2);  // lmax+1
  const int nkk = vals4d.extent(3);  // nradmax

  if (nl < nlut) {
    wl = x - KK_FLOAT(nl); wl2 = wl*wl; wl3 = wl2*wl;
    w2l1 = 2.0*wl; w3l2 = 3.0*wl2;
    int func_id = 0;
    for (int kk = 0; kk < nkk; kk++) {
      for (int ll = 0; ll < nll; ll++, func_id++) {
        for (int idx = 0; idx < 4; idx++) c[idx] = lookupTable(nl, func_id, idx);
        vals4d(ii, jj, ll, kk)   = c[0] + c[1]*wl + c[2]*wl2 + c[3]*wl3;
        derivs4d(ii, jj, ll, kk) = (c[1] + c[2]*w2l1 + c[3]*w3l2) * rscalelookup;
      }
    }
  } else {
    for (int kk = 0; kk < nkk; kk++)
      for (int ll = 0; ll < nll; ll++)
        vals4d(ii, jj, ll, kk) = derivs4d(ii, jj, ll, kk) = 0.0;
  }
}

/* ---------------------------------------------------------------------- */

// 4D vals-only: writes directly to fr, skips dfr (used in PERATOM=1 path).
template<class DeviceType, int PERATOM>
// NOLINTNEXTLINE
KOKKOS_INLINE_FUNCTION
void ComputePACEKokkos<DeviceType, PERATOM>::SplineInterpolatorKokkos::calcSplines(
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
    wl = x - KK_FLOAT(nl); wl2 = wl*wl; wl3 = wl2*wl;
    int func_id = 0;
    for (int kk = 0; kk < nkk; kk++) {
      for (int ll = 0; ll < nll; ll++, func_id++) {
        for (int idx = 0; idx < 4; idx++) c[idx] = lookupTable(nl, func_id, idx);
        vals4d(ii, jj, ll, kk) = c[0] + c[1]*wl + c[2]*wl2 + c[3]*wl3;
      }
    }
  } else {
    for (int kk = 0; kk < nkk; kk++)
      for (int ll = 0; ll < nll; ll++)
        vals4d(ii, jj, ll, kk) = 0.0;
  }
}

/* ---------------------------------------------------------------------- */

// Local-storage gr: writes gr_local[func_id], same values as calcSplines(ii,jj,r,gr).
template<class DeviceType, int PERATOM>
// NOLINTNEXTLINE
KOKKOS_INLINE_FUNCTION
void ComputePACEKokkos<DeviceType, PERATOM>::SplineInterpolatorKokkos::calcSplines_local(
  const KK_FLOAT r, KK_FLOAT* gr_local) const
{
  KK_FLOAT wl, wl2, wl3;
  KK_FLOAT c[4];
  KK_FLOAT x = r * rscalelookup;
  int nl = static_cast<int>(floor(x));

  if (nl <= 0) Kokkos::abort("Encountered very small distance. Stopping.");

  if (nl < nlut) {
    wl = x - KK_FLOAT(nl); wl2 = wl*wl; wl3 = wl2*wl;
    for (int func_id = 0; func_id < num_of_functions; func_id++) {
      for (int idx = 0; idx < 4; idx++) c[idx] = lookupTable(nl, func_id, idx);
      gr_local[func_id] = c[0] + c[1]*wl + c[2]*wl2 + c[3]*wl3;
    }
  } else {
    for (int func_id = 0; func_id < num_of_functions; func_id++)
      gr_local[func_id] = 0.0;
  }
}

/* ---------------------------------------------------------------------- */

// Local-storage fr: writes fr_local[kk*nll + ll] = fr(ii,jj,ll,kk), same math
// as calcSplines(ii,jj,r,fr) but into a raw stack array.
template<class DeviceType, int PERATOM>
// NOLINTNEXTLINE
KOKKOS_INLINE_FUNCTION
void ComputePACEKokkos<DeviceType, PERATOM>::SplineInterpolatorKokkos::calcSplines_local(
  const KK_FLOAT r, KK_FLOAT* fr_local, int nll) const
{
  KK_FLOAT wl, wl2, wl3;
  KK_FLOAT c[4];
  KK_FLOAT x = r * rscalelookup;
  int nl = static_cast<int>(floor(x));

  if (nl <= 0) Kokkos::abort("Encountered very small distance. Stopping.");

  const int nkk = num_of_functions / nll;  // nradmax

  if (nl < nlut) {
    wl = x - KK_FLOAT(nl); wl2 = wl*wl; wl3 = wl2*wl;
    int func_id = 0;
    for (int kk = 0; kk < nkk; kk++) {
      for (int ll = 0; ll < nll; ll++, func_id++) {
        for (int idx = 0; idx < 4; idx++) c[idx] = lookupTable(nl, func_id, idx);
        fr_local[kk*nll + ll] = c[0] + c[1]*wl + c[2]*wl2 + c[3]*wl3;
      }
    }
  } else {
    for (int kk = 0; kk < nkk; kk++)
      for (int ll = 0; ll < nll; ll++)
        fr_local[kk*nll + ll] = 0.0;
  }
}

/* ---------------------------------------------------------------------- */

template<class DeviceType, int PERATOM>
template<class TagStyle>
void ComputePACEKokkos<DeviceType, PERATOM>::check_team_size_for(int inum, int &team_size, int vector_length) {
  int team_size_max = Kokkos::TeamPolicy<DeviceType,TagStyle>(inum,Kokkos::AUTO).team_size_max(*this,Kokkos::ParallelForTag());
  if (team_size * vector_length > team_size_max)
    team_size = team_size_max / vector_length;
}

/* ---------------------------------------------------------------------- */

namespace LAMMPS_NS {
template class ComputePACEKokkos<LMPDeviceType, 0>;
template class ComputePACEKokkos<LMPDeviceType, 1>;
template class ComputePACEKokkosGlobal<LMPDeviceType>;
template class ComputePACEKokkosAtom<LMPDeviceType>;
#ifdef LMP_KOKKOS_GPU
template class ComputePACEKokkos<LMPHostType, 0>;
template class ComputePACEKokkos<LMPHostType, 1>;
template class ComputePACEKokkosGlobal<LMPHostType>;
template class ComputePACEKokkosAtom<LMPHostType>;
#endif
}
