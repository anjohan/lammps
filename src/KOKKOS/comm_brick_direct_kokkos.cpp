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

#include "comm_brick_direct_kokkos.h"

#include "atom_kokkos.h"
#include "atom_masks.h"
#include "atom_vec_kokkos.h"
#include "domain.h"
#include "error.h"
#include "fix.h"
#include "kokkos.h"
#include "kokkos_base.h"
#include "memory_kokkos.h"
#include "modify.h"
#include "neighbor.h"

// NOTES:
// still need cutoff calculation for nonuniform layout
// need forward_comm_array to test molecular systems
// test msg tags with individual procs as multiple neighbors via big stencil
// test when cutoffs >> box length
// test with triclinic
// doc msg tag logic in code
// doc stencil data structs and logic in code
// CommBrick could use local maxsend in its borders() check for sendlist realloc
//   instead of indexing the swap for each atom

using namespace LAMMPS_NS;

static constexpr double BUFFACTOR = 1.5;
static constexpr int BUFMIN = 1024;
static constexpr int BUFEXTRA = 1024;

/* ---------------------------------------------------------------------- */

CommBrickDirectKokkos::CommBrickDirectKokkos(LAMMPS *lmp) : CommBrickDirect(lmp)
{
}

/* ---------------------------------------------------------------------- */

CommBrickDirectKokkos::~CommBrickDirectKokkos()
{
  buf_send = nullptr;
  buf_recv = nullptr;
  buf_send_direct = nullptr;
  buf_recv_direct = nullptr;
}

/* ---------------------------------------------------------------------- */
//IMPORTANT: we *MUST* pass "*oldcomm" to the Comm initializer here, as
//           the code below *requires* that the (implicit) copy constructor
//           for Comm is run and thus creating a shallow copy of "oldcomm".
//           The call to Comm::copy_arrays() then converts the shallow copy
//           into a deep copy of the class with the new layout.

CommBrickDirectKokkos::CommBrickDirectKokkos(LAMMPS *lmp, Comm *oldcomm) : CommBrickDirect(lmp, oldcomm)
{
}

/* ----------------------------------------------------------------------
   create stencil of direct swaps this procs make with each proc in stencil
   direct swap = send and recv
     same proc can appear multiple times in stencil, self proc can also appear
   stencil is used for border and forward and reverse comm
------------------------------------------------------------------------- */

void CommBrickDirectKokkos::setup()
{
  CommBrickDirect::setup();

  MemKK::realloc_kokkos(k_swap2list,"comm_direct:swap2list",ndirect);
  MemKK::realloc_kokkos(k_pbc_flag_direct,"comm_direct:pbc_flag",ndirect);
  MemKK::realloc_kokkos(k_pbc_direct,"comm_direct:pbc",ndirect,6);
  MemKK::realloc_kokkos(k_self_flag,"comm_direct:pbc",ndirect);

  for (int iswap = 0; iswap < ndirect; iswap++) {
    k_swap2list.h_view[iswap] = swap2list[iswap];
    k_pbc_flag_direct.h_view[iswap] = pbc_flag_direct[iswap];
    k_pbc_direct.h_view(iswap,0) = pbc_direct[iswap][0];
    k_pbc_direct.h_view(iswap,1) = pbc_direct[iswap][1];
    k_pbc_direct.h_view(iswap,2) = pbc_direct[iswap][2];
    k_pbc_direct.h_view(iswap,3) = pbc_direct[iswap][3];
    k_pbc_direct.h_view(iswap,4) = pbc_direct[iswap][4];
    k_pbc_direct.h_view(iswap,5) = pbc_direct[iswap][5];
    k_self_flag.h_view(iswap) = proc_direct[iswap] == me;
  }

  k_swap2list.modify_host();
  k_pbc_flag_direct.modify_host();
  k_pbc_direct.modify_host();
  k_self_flag.modify_host();

  k_exchange_sendlist = DAT::tdual_int_1d("comm:k_exchange_sendlist",100);
  k_exchange_copylist = DAT::tdual_int_1d("comm:k_exchange_copylist",100);
  k_count = DAT::tdual_int_scalar("comm:k_count");

  maxsend = BUFMIN;
  maxrecv = BUFMIN;

  grow_send_kokkos(maxsend+bufextra,0,Host);
  grow_recv_kokkos(maxrecv,Host);
}

/* ----------------------------------------------------------------------
   forward communication of atom coords every timestep
   other per-atom attributes may also be sent via pack/unpack routines
   exchange owned atoms directly with all neighbor procs,
     not via CommBrick 6-way stencil
------------------------------------------------------------------------- */

void CommBrickDirectKokkos::forward_comm(int dummy)
{
  int forward_comm_classic = 0;
  int forward_comm_on_host = lmp->kokkos->forward_comm_on_host;

  if (!forward_comm_classic) {
    if (forward_comm_on_host) forward_comm_device<LMPHostType>();
    else forward_comm_device<LMPDeviceType>();
    return;
  }

  if (comm_x_only) {
    atomKK->sync(Host,X_MASK);
    atomKK->modified(Host,X_MASK);
  } else if (ghost_velocity) {
    atomKK->sync(Host,X_MASK | V_MASK);
    atomKK->modified(Host,X_MASK | V_MASK);
  } else {
    atomKK->sync(Host,ALL_MASK);
    atomKK->modified(Host,ALL_MASK);
  }

  CommBrickDirect::forward_comm(dummy);
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
void CommBrickDirectKokkos::forward_comm_device()
{
  double *buf;

  // post all receives for ghost atoms
  // except for self copies

  int offset;

  int npost = 0;
  for (int iswap = 0; iswap < ndirect; iswap++) {
    if (proc_direct[iswap] == me) continue;
    if (size_forward_recv_direct[iswap]) {
      if (comm_x_only) {
        buf = atomKK->k_x.view<DeviceType>().data() + firstrecv_direct[iswap]*atomKK->k_x.view<DeviceType>().extent(1);
      } else {
        offset = recv_offset_forward_direct[iswap];
        buf = k_buf_recv_direct.view<DeviceType>().data() + offset;
      }
      MPI_Irecv(buf,size_forward_recv_direct[iswap],MPI_DOUBLE,
                proc_direct[iswap],recvtag[iswap],world,&requests[npost++]);
    }
  }

  // pack all atom data at once, including copying self data

  k_sendatoms_list.sync<DeviceType>();
  k_swap2list.sync<DeviceType>();
  k_pbc_flag_direct.sync<DeviceType>();
  k_pbc_direct.sync<DeviceType>();
  k_self_flag.sync<DeviceType>();
  k_sendatoms_list.sync<DeviceType>();
  k_sendnum_scan_direct.sync<DeviceType>();
  k_firstrecv_direct.sync<DeviceType>();

  if (ghost_velocity) {
    //atomKK->avecKK->pack_comm_vel_direct(totalsend,k_sendatoms_list,
    //                    k_firstrecv,k_pbc_flag_direct,k_pbc_direct,
    //                    k_swap2list,k_buf_send_direct);
  } else {
    atomKK->avecKK->pack_comm_direct(totalsend,k_sendatoms_list,
                        k_sendnum_scan_direct,k_firstrecv_direct,
                        k_pbc_flag_direct,k_pbc_direct,
                        k_swap2list,k_buf_send_direct,k_self_flag);
  }
  DeviceType().fence();

  // send all owned atoms to receiving procs
  // except for self copies

  offset = 0;
  for (int iswap = 0; iswap < ndirect; iswap++) {
    if (sendnum_direct[iswap]) {
      int n = sendnum_direct[iswap]*atomKK->avecKK->size_forward;
      if (proc_direct[iswap] != me)
        MPI_Send(k_buf_send_direct.view<DeviceType>().data() + offset,n,MPI_DOUBLE,proc_direct[iswap],sendtag[iswap],world);
      offset += n;
    }
  }

  // wait on incoming messages with ghost atoms
  // unpack all messages at once

  if (npost == 0) return;

  MPI_Waitall(npost,requests,MPI_STATUS_IGNORE);

  if (comm_x_only) return;

  if (ghost_velocity) {
    //atomKK->avecKK->unpack_comm_vel_direct(recvnum_direct,firstrecv_direct,buf_recv_direct);
  } else {
    //atomKK->avecKK->unpack_comm_direct(recvnum_direct,firstrecv_direct,buf_recv_direct);
  }
  DeviceType().fence();
}

/* ----------------------------------------------------------------------
   reverse communication of forces on atoms every timestep
   other per-atom attributes may also be sent via pack/unpack routines
------------------------------------------------------------------------- */

void CommBrickDirectKokkos::reverse_comm()
{
  if (comm_f_only)
    atomKK->sync(Host,F_MASK);
  else
    atomKK->sync(Host,ALL_MASK);

  CommBrickDirect::reverse_comm();

  if (comm_f_only)
    atomKK->modified(Host,F_MASK);
  else
    atomKK->modified(Host,ALL_MASK);
}

/* ----------------------------------------------------------------------
   exchange: move atoms to correct processors
   atoms exchanged with all 6 stencil neighbors
   send out atoms that have left my box, receive ones entering my box
   atoms will be lost if not inside some proc's box
     can happen if atom moves outside of non-periodic boundary
     or if atom moves more than one proc away
   this routine called before every reneighboring
   for triclinic, atoms must be in lamda coords (0-1) before exchange is called
------------------------------------------------------------------------- */

void CommBrickDirectKokkos::exchange()
{
  if (lmp->kokkos->exchange_comm_classic) {
    atomKK->sync(Host,ALL_MASK);
    CommBrickDirect::exchange();
    atomKK->modified(Host,ALL_MASK);
    return;
  }

  if (atom->nextra_grow)
    error->all(FLERR, "Fixes with extra communication not supported with comm/brick/direct/kk");

  if (lmp->kokkos->exchange_comm_on_host)
    exchange_device<LMPHostType>();
  else
    exchange_device<LMPDeviceType>();
}

template<class DeviceType>
struct BuildExchangeListFunctor {
  typedef DeviceType device_type;
  typedef ArrayTypes<DeviceType> AT;
  X_FLOAT _lo,_hi;
  typename AT::t_x_array _x;

  int _nlocal,_dim;
  typename AT::t_int_scalar _nsend;
  typename AT::t_int_1d _sendlist;


  BuildExchangeListFunctor(
      const typename AT::tdual_x_array x,
      const typename AT::tdual_int_1d sendlist,
      typename AT::tdual_int_scalar nsend,
      int nlocal, int dim,
      X_FLOAT lo, X_FLOAT hi):
                _lo(lo),_hi(hi),
                _x(x.template view<DeviceType>()),
                _nlocal(nlocal),_dim(dim),
                _nsend(nsend.template view<DeviceType>()),
                _sendlist(sendlist.template view<DeviceType>()) { }

  KOKKOS_INLINE_FUNCTION
  void operator() (int i) const {
    if (_x(i,_dim) < _lo || _x(i,_dim) >= _hi) {
      const int mysend = Kokkos::atomic_fetch_add(&_nsend(),1);
      if (mysend < (int)_sendlist.extent(0))
        _sendlist(mysend) = i;
    }
  }
};

/* ---------------------------------------------------------------------- */

template<class DeviceType>
void CommBrickDirectKokkos::exchange_device()
{
  int nsend,nrecv,nrecv1,nrecv2,nlocal;
  double *sublo,*subhi;
  double lo,hi;
  MPI_Request request;

  // clear global->local map for owned and ghost atoms
  // b/c atoms migrate to new procs in exchange() and
  //   new ghosts are created in borders()
  // map_set() is done at end of borders()

  if (lmp->kokkos->atom_map_classic)
    if (map_style != Atom::MAP_NONE) atom->map_clear();

  // clear ghost count and any ghost bonus data internal to AtomVec

  atom->nghost = 0;
  atom->avec->clear_bonus();

  if (comm->nprocs > 1) { // otherwise no-op

    // subbox bounds for orthogonal or triclinic

    if (triclinic == 0) {
      sublo = domain->sublo;
      subhi = domain->subhi;
    } else {
      sublo = domain->sublo_lamda;
      subhi = domain->subhi_lamda;
    }

    atomKK->sync(ExecutionSpaceFromDevice<DeviceType>::space,ALL_MASK);

    // loop over dimensions
    for (int dim = 0; dim < 3; dim++) {

      lo = sublo[dim];
      hi = subhi[dim];
      nlocal = atom->nlocal;
      nsend = 0;

      // fill buffer with atoms leaving my box, using < and >=

      k_count.h_view() = k_exchange_sendlist.h_view.extent(0);
      while (k_count.h_view() >= (int)k_exchange_sendlist.h_view.extent(0)) {
        k_count.h_view() = 0;
        k_count.modify<LMPHostType>();
        k_count.sync<DeviceType>();

        BuildExchangeListFunctor<DeviceType>
          f(atomKK->k_x,k_exchange_sendlist,k_count,
            nlocal,dim,lo,hi);
        Kokkos::parallel_for(nlocal,f);
        k_exchange_sendlist.modify<DeviceType>();
        k_count.modify<DeviceType>();

        k_count.sync<LMPHostType>();
        int count = k_count.h_view();
        if (count >= (int)k_exchange_sendlist.h_view.extent(0)) {
          MemKK::realloc_kokkos(k_exchange_sendlist,"comm:k_exchange_sendlist",count*1.1);
          MemKK::realloc_kokkos(k_exchange_copylist,"comm:k_exchange_copylist",count*1.1);
          k_count.h_view() = k_exchange_sendlist.h_view.extent(0);
        }
      }
      int count = k_count.h_view();

      // sort exchange_sendlist

      auto d_exchange_sendlist = Kokkos::subview(k_exchange_sendlist.view<DeviceType>(),std::make_pair(0,count));
      Kokkos::sort(DeviceType(), d_exchange_sendlist);
      k_exchange_sendlist.sync<LMPHostType>();

      // when atom is deleted, fill it in with last atom

      int sendpos = count-1;
      int icopy = nlocal-1;
      nlocal -= count;
      for (int recvpos = 0; recvpos < count; recvpos++) {
        int irecv = k_exchange_sendlist.h_view(recvpos);
        if (irecv < nlocal) {
          if (icopy == k_exchange_sendlist.h_view(sendpos)) icopy--;
          while (sendpos > 0 && icopy <= k_exchange_sendlist.h_view(sendpos-1)) {
            sendpos--;
            icopy = k_exchange_sendlist.h_view(sendpos) - 1;
          }
          k_exchange_copylist.h_view(recvpos) = icopy;
          icopy--;
        } else
          k_exchange_copylist.h_view(recvpos) = -1;
      }

      k_exchange_copylist.modify<LMPHostType>();
      k_exchange_copylist.sync<DeviceType>();
      nsend = count;
      if (nsend > maxsend) grow_send_kokkos(nsend,0);
      nsend =
        atomKK->avecKK->pack_exchange_kokkos(count,k_buf_send,
                                   k_exchange_sendlist,k_exchange_copylist,
                                   ExecutionSpaceFromDevice<DeviceType>::space);
      DeviceType().fence();
      atom->nlocal = nlocal;

      // send/recv atoms in both directions
      // send size of message first so receiver can realloc buf_recv if needed
      // if 1 proc in dimension, no send/recv
      //   set nrecv = 0 so buf_send atoms will be lost
      // if 2 procs in dimension, single send/recv
      // if more than 2 procs in dimension, send/recv to both neighbors

      const int data_size = atomKK->avecKK->size_exchange;

      if (procgrid[dim] == 1) nrecv = 0;
      else {
        MPI_Sendrecv(&nsend,1,MPI_INT,procneigh[dim][0],0,
                     &nrecv1,1,MPI_INT,procneigh[dim][1],0,world,MPI_STATUS_IGNORE);
        nrecv = nrecv1;
        if (procgrid[dim] > 2) {
          MPI_Sendrecv(&nsend,1,MPI_INT,procneigh[dim][1],0,
                       &nrecv2,1,MPI_INT,procneigh[dim][0],0,world,MPI_STATUS_IGNORE);
          nrecv += nrecv2;
        }
        if (nrecv > maxrecv) grow_recv_kokkos(nrecv);

        MPI_Irecv(k_buf_recv.view<DeviceType>().data(),nrecv1,
                  MPI_DOUBLE,procneigh[dim][1],0,
                  world,&request);
        MPI_Send(k_buf_send.view<DeviceType>().data(),nsend,
                 MPI_DOUBLE,procneigh[dim][0],0,world);
        MPI_Wait(&request,MPI_STATUS_IGNORE);

        if (procgrid[dim] > 2) {
          MPI_Irecv(k_buf_recv.view<DeviceType>().data()+nrecv1,
                    nrecv2,MPI_DOUBLE,procneigh[dim][0],0,
                    world,&request);
          MPI_Send(k_buf_send.view<DeviceType>().data(),nsend,
                   MPI_DOUBLE,procneigh[dim][1],0,world);
          MPI_Wait(&request,MPI_STATUS_IGNORE);
        }

        if (nrecv) {

          if (atom->nextra_grow) {
            if ((int) k_indices.extent(0) < nrecv/data_size)
              MemoryKokkos::realloc_kokkos(k_indices,"comm:indices",nrecv/data_size);
          } else if (k_indices.h_view.data())
           k_indices = DAT::tdual_int_1d();


          atom->nlocal = atomKK->avecKK->
            unpack_exchange_kokkos(k_buf_recv,nrecv,atom->nlocal,dim,lo,hi,
                                     ExecutionSpaceFromDevice<DeviceType>::space,k_indices);

          DeviceType().fence();
        }
      }

      if (atom->nextra_grow) {
        for (int iextra = 0; iextra < atom->nextra_grow; iextra++) {
          auto fix_iextra = modify->fix[atom->extra_grow[iextra]];
          KokkosBase *kkbase = dynamic_cast<KokkosBase*>(fix_iextra);
          int nextrasend = 0;
          nsend = count;
          if (nsend) {
            if (nsend*fix_iextra->maxexchange > maxsend)
              grow_send_kokkos(nsend*fix_iextra->maxexchange,0);
            nextrasend = kkbase->pack_exchange_kokkos(
              count,k_buf_send,k_exchange_sendlist,k_exchange_copylist,
              ExecutionSpaceFromDevice<DeviceType>::space);
            DeviceType().fence();
          }

          int nextrarecv,nextrarecv1,nextrarecv2;
          if (procgrid[dim] == 1) nextrarecv = 0;
          else {
            MPI_Sendrecv(&nextrasend,1,MPI_INT,procneigh[dim][0],0,
                         &nextrarecv1,1,MPI_INT,procneigh[dim][1],0,
                         world,MPI_STATUS_IGNORE);

            nextrarecv = nextrarecv1;

            if (procgrid[dim] > 2) {
              MPI_Sendrecv(&nextrasend,1,MPI_INT,procneigh[dim][1],0,
                           &nextrarecv2,1,MPI_INT,procneigh[dim][0],0,
                           world,MPI_STATUS_IGNORE);

              nextrarecv += nextrarecv2;
            }

            if (nextrarecv > maxrecv) grow_recv_kokkos(nextrarecv);

            MPI_Irecv(k_buf_recv.view<DeviceType>().data(),nextrarecv1,
                      MPI_DOUBLE,procneigh[dim][1],0,
                      world,&request);
            MPI_Send(k_buf_send.view<DeviceType>().data(),nextrasend,
                     MPI_DOUBLE,procneigh[dim][0],0,world);
            MPI_Wait(&request,MPI_STATUS_IGNORE);

            if (procgrid[dim] > 2) {
              MPI_Irecv(k_buf_recv.view<DeviceType>().data()+nextrarecv1,
                        nextrarecv2,MPI_DOUBLE,procneigh[dim][0],0,
                        world,&request);
              MPI_Send(k_buf_send.view<DeviceType>().data(),nextrasend,
                       MPI_DOUBLE,procneigh[dim][1],0,world);
              MPI_Wait(&request,MPI_STATUS_IGNORE);
            }

            if (nextrarecv) {
              kkbase->unpack_exchange_kokkos(
                k_buf_recv,k_indices,nrecv/data_size,
                nrecv1/data_size,nextrarecv1,
                ExecutionSpaceFromDevice<DeviceType>::space);
              DeviceType().fence();
            }
          }
        }
      }
    }
    atomKK->modified(ExecutionSpaceFromDevice<DeviceType>::space,ALL_MASK);
  }

  if (atom->firstgroupname) {
    /* this is not yet implemented with Kokkos */
    atomKK->sync(Host,ALL_MASK);
    atom->first_reorder();
    atomKK->modified(Host,ALL_MASK);
  }
}


/* ----------------------------------------------------------------------
   borders: list nearby atoms to send to neighboring procs at every timestep
   one list is created for every swap that will be made
   as list is made, actually do swaps
   this does equivalent of a forward_comm(), so don't need to explicitly
     call forward_comm() on reneighboring timestep
   this routine is called before every reneighboring
   for triclinic, atoms must be in lamda coords (0-1) before borders is called
  // loop over conventional 6-way BRICK swaps in 3 dimensions
  // construct BRICK_DIRECT swaps from them
  // unlike borders() in CommBrick, cannot perform borders comm until end
  // this is b/c the swaps take place simultaneously in all dimensions
  //   and thus cannot contain ghost atoms in the forward comm
------------------------------------------------------------------------- */

void CommBrickDirectKokkos::borders()
{
  atomKK->sync(Host,ALL_MASK);
  int prev_auto_sync = lmp->kokkos->auto_sync;
  lmp->kokkos->auto_sync = 1;
  CommBrickDirect::borders();
  lmp->kokkos->auto_sync = prev_auto_sync;
  atomKK->modified(Host,ALL_MASK);

  int maxsend = 0;
  for (int ilist = 0; ilist < maxlist; ilist++)
    maxsend = MAX(maxsend,maxsendatoms_list[ilist]);

  if (k_sendatoms_list.d_view.extent(1) < maxsend)
    MemKK::realloc_kokkos(k_sendatoms_list,"comm_direct:sendatoms_list",maxlist,maxsend);

  if(k_sendnum_scan_direct.extent(0) < ndirect) {
    MemKK::realloc_kokkos(k_sendnum_scan_direct,"comm_direct:sendnum_scan",ndirect);
    MemKK::realloc_kokkos(k_firstrecv_direct,"comm_direct:firstrecv",ndirect);
  }

  for (int ilist = 0; ilist < maxlist; ilist++) {
    if (!active_list[ilist]) continue;
    const int nsend = sendnum_list[ilist];
    for (int i = 0; i < nsend; i++)
      k_sendatoms_list.h_view(ilist,i) = sendatoms_list[ilist][i];
  }

  int scan = 0;
  for (int iswap = 0; iswap < ndirect; iswap++) {
    scan += sendnum_direct[iswap];
    k_sendnum_scan_direct.h_view[iswap] = scan;
    k_firstrecv_direct.h_view[iswap] = firstrecv_direct[iswap];
  }
  totalsend = scan;

  // grow send and recv buffers

  if (totalsend*size_forward > k_buf_send_direct.d_view.extent(0))
    grow_send_direct(totalsend*size_forward,0);

  k_sendatoms_list.modify_host();
  k_sendnum_scan_direct.modify_host();
  k_firstrecv_direct.modify_host();
}

/* ----------------------------------------------------------------------
   realloc the size of the send_direct buffer as needed with BUFFACTOR
   do not use bufextra as in CommBrick, b/c not using buf_send_direct for exchange()
   flag = 0, don't need to realloc with copy, just free/malloc w/ BUFFACTOR
   flag = 1, realloc with BUFFACTOR
   flag = 2, free/malloc w/out BUFFACTOR
------------------------------------------------------------------------- */

void CommBrickDirectKokkos::grow_send_direct(int n, int flag)
{
  if (flag == 0) {
    maxsend_direct = static_cast<int> (BUFFACTOR * n);
    MemKK::realloc_kokkos(k_buf_send_direct,"comm:buf_send_direct",maxsend_direct);
  } else if (flag == 1) {
    maxsend_direct = static_cast<int> (BUFFACTOR * n);
    k_buf_send_direct.resize(maxsend_direct);
  } else {
    MemKK::realloc_kokkos(k_buf_send_direct,"comm:buf_send_direct",maxsend_direct);
  }

  buf_send_direct = k_buf_send_direct.h_view.data();
}

/* ----------------------------------------------------------------------
   free/malloc the size of the recv_direct buffer as needed with BUFFACTOR
------------------------------------------------------------------------- */

void CommBrickDirectKokkos::grow_recv_direct(int n)
{
  maxrecv_direct = static_cast<int> (BUFFACTOR * n);
  MemKK::realloc_kokkos(k_buf_recv_direct,"comm:buf_recv_direct",maxrecv_direct);
  buf_recv_direct = k_buf_recv_direct.h_view.data();
}

/* ----------------------------------------------------------------------
   realloc the size of the send buffer as needed with BUFFACTOR & BUFEXTRA
   if flag = 1, realloc
   if flag = 0, don't need to realloc with copy, just free/malloc
------------------------------------------------------------------------- */

void CommBrickDirectKokkos::grow_send_kokkos(int n, int flag, ExecutionSpace space)
{

  maxsend = static_cast<int> (BUFFACTOR * n);
  int maxsend_border = (maxsend+BUFEXTRA)/atomKK->avecKK->size_border;
  if (flag) {
    if (space == Device)
      k_buf_send.modify<LMPDeviceType>();
    else
      k_buf_send.modify<LMPHostType>();

    if (ghost_velocity)
      k_buf_send.resize(maxsend_border,
                        atomKK->avecKK->size_border + atomKK->avecKK->size_velocity);
    else
      k_buf_send.resize(maxsend_border,atomKK->avecKK->size_border);
    buf_send = k_buf_send.view<LMPHostType>().data();
  } else {
    if (ghost_velocity)
      MemoryKokkos::realloc_kokkos(k_buf_send,"comm:k_buf_send",maxsend_border,
                        atomKK->avecKK->size_border + atomKK->avecKK->size_velocity);
    else
      MemoryKokkos::realloc_kokkos(k_buf_send,"comm:k_buf_send",maxsend_border,
                        atomKK->avecKK->size_border);
    buf_send = k_buf_send.view<LMPHostType>().data();
  }
}

/* ----------------------------------------------------------------------
   free/malloc the size of the recv buffer as needed with BUFFACTOR
------------------------------------------------------------------------- */

void CommBrickDirectKokkos::grow_recv_kokkos(int n, ExecutionSpace /*space*/)
{
  maxrecv = static_cast<int> (BUFFACTOR * n);
  int maxrecv_border = (maxrecv+BUFEXTRA)/atomKK->avecKK->size_border;

  MemoryKokkos::realloc_kokkos(k_buf_recv,"comm:k_buf_recv",maxrecv_border,
    atomKK->avecKK->size_border);
  buf_recv = k_buf_recv.view<LMPHostType>().data();
}
