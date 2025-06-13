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

/* ----------------------------------------------------------------------
   Contributing author: Paul Crozier (SNL)
------------------------------------------------------------------------- */

#include "pair_neighprint.h"

#include "atom.h"
#include "comm.h"
#include "error.h"
#include "force.h"
#include "math_const.h"
#include "memory.h"
#include "neigh_list.h"
#include "neighbor.h"
#include "respa.h"
#include "update.h"

#include <cmath>
#include <cstring>

using namespace LAMMPS_NS;
using namespace MathConst;

/* ---------------------------------------------------------------------- */

PairNeighPrint::PairNeighPrint(LAMMPS *lmp) : Pair(lmp)
{
  allocated = 1;
}

/* ---------------------------------------------------------------------- */

PairNeighPrint::~PairNeighPrint()
{
  if (copymode) return;
  memory->destroy(setflag);
  memory->destroy(cutsq);
}

/* ---------------------------------------------------------------------- */

void PairNeighPrint::compute(int eflag, int vflag)
{
  int i, j, ii, jj, inum, jnum, itype, jtype;
  int *ilist, *jlist, *numneigh, **firstneigh;

  double **x = atom->x;

  inum = list->inum;
  ilist = list->ilist;
  numneigh = list->numneigh;
  firstneigh = list->firstneigh;

  if (comm->me == 0) printf("NEIGH INFO:\n");

  for (ii = 0; ii < inum; ii++) {
    i = ilist[ii];
    jlist = firstneigh[i];
    jnum = numneigh[i];

    if (comm->me == 0) {
      printf("atom at x=%.1f has neighbors at:\n", x[i][0]);
    }

    for (jj = 0; jj < jnum; jj++) {
      j = jlist[jj];
      j &= NEIGHMASK;
      if (comm->me == 0) {
        printf("   x=%.1f, with neighbors at", x[j][0]);
        int *klist = firstneigh[j];
        int knum = numneigh[j];
        for (int kk = 0; kk < knum; kk++) {
          int k = klist[kk];
          //k &= NEIGHMASK;
          printf(" x=%.1f", x[k][0]);
        }
        printf("\n");
      }
    }
    if (comm->me == 0) printf("\n");
  }
}


/* ----------------------------------------------------------------------
   global settings
------------------------------------------------------------------------- */

void PairNeighPrint::settings(int narg, char **arg)
{
  if (narg != 1) error->all(FLERR, "Illegal pair_style command");

  cut = utils::numeric(FLERR, arg[0], false, lmp);
}

/* ----------------------------------------------------------------------
   set coeffs for one or more type pairs
------------------------------------------------------------------------- */

void PairNeighPrint::coeff(int narg, char **arg)
{

  int ilo, ihi, jlo, jhi;
  utils::bounds(FLERR, arg[0], 1, atom->ntypes, ilo, ihi, error);
  utils::bounds(FLERR, arg[1], 1, atom->ntypes, jlo, jhi, error);
  memory->create(setflag, 2, 2, "pair:setflag");
  memory->create(cutsq, 2, 2, "pair:cutsq");

  for (int i = ilo; i <= ihi; i++) {
    for (int j = MAX(jlo, i); j <= jhi; j++) {
      setflag[i][j] = 1;
    }
  }
}

/* ----------------------------------------------------------------------
   init specific to this pair style
------------------------------------------------------------------------- */

void PairNeighPrint::init_style()
{
  int list_style = NeighConst::REQ_FULL | NeighConst::REQ_GHOST;

  neighbor->add_request(this, list_style);
}

double PairNeighPrint::init_one(int i, int j) {
  return cut;
}
