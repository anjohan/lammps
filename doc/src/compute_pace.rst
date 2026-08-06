.. index:: compute pace
.. index:: compute pace/kk
.. index:: compute pace/atom
.. index:: compute pace/atom/kk
.. index:: compute pace/grid
.. index:: compute pace/grid/kk
.. index:: compute pace/grid/local
.. index:: compute pace/grid/local/kk

compute pace command
========================

Accelerator Variants: *pace/kk*

compute pace/atom command
=========================

Accelerator Variants: *pace/atom/kk*

compute pace/grid command
=========================

compute pace/grid/local command
===============================

Accelerator Variants: *pace/grid/kk*, *pace/grid/local/kk*

Syntax
""""""

.. code-block:: LAMMPS

   compute ID group-ID pace ace_potential_filename bikflag dgradflag keyword values ...
   compute ID group-ID pace/atom ace_potential_filename keyword values ...
   compute ID group-ID pace/grid grid nx ny nz ace_potential_filename keyword values ...
   compute ID group-ID pace/grid/local grid nx ny nz ace_potential_filename keyword values ...

* ID, group-ID are documented in :doc:`compute <compute>` command
* pace or pace/atom or pace/grid or pace/grid/local = style name of this compute command
* ace_potential_filename = file name (in the .yace or .ace format from :doc:`pace pair_style <pair_pace>`) including ACE hyper-parameters, bonds, and generalized coupling coefficients
* bikflag = *0* or *1* (style *pace* only)

  .. parsed-literal::

       *0* = descriptors are summed over atoms of each type
       *1* = descriptors are listed separately for each atom

* dgradflag = *0* or *1* (style *pace* only)

  .. parsed-literal::

       *0* = descriptor gradients are summed over atoms of each type
       *1* = descriptor gradients are listed separately for each atom pair

* *grid* values = nx, ny, nz, number of grid points in the x, y, and z directions (positive integers, styles *pace/grid* and *pace/grid/local* only)
* zero or more keyword/value pairs may be appended
* keyword = *chunksize* or *element*

  .. parsed-literal::

       *chunksize* value = number of atoms, or grid points for styles *pace/grid* and *pace/grid/local*, processed in each pass (Kokkos accelerator variants only)
       *element* value = name of an element in the *ace_potential_filename* file (styles *pace/grid* and *pace/grid/local* only)
          default = first element listed in the *ace_potential_filename* file

Examples
""""""""

.. code-block:: LAMMPS

   compute pace all pace coupling_coefficients.yace 0 0
   compute pace all pace coupling_coefficients.yace 1 0
   compute pace all pace coupling_coefficients.yace 1 1 chunksize 512
   compute bik all pace/atom coupling_coefficients.yace
   compute bik all pace/atom coupling_coefficients.yace chunksize 8192
   compute bgrid all pace/grid grid 20 20 20 coupling_coefficients.yace
   compute bgrid all pace/grid grid 20 20 20 coupling_coefficients.yace element O
   compute bgridloc all pace/grid/local grid 20 20 20 coupling_coefficients.yace chunksize 512

Description
"""""""""""

.. versionadded:: 7Feb2024

This compute calculates a set of quantities related to the atomic
cluster expansion (ACE) descriptors of the atoms in a group.  ACE
descriptors are highly general atomic descriptors, encoding the radial
and angular distribution of neighbor atoms, up to arbitrary bond order
(rank).  The detailed mathematical definition is given in the paper by
:ref:`(Drautz) <Drautz19>`.  These descriptors are used in the
:doc:`pace pair_style <pair_pace>`.  Quantities obtained from `compute
pace` are related to those used in :doc:`pace pair_style <pair_pace>` to
evaluate atomic energies, forces, and stresses for linear ACE models.

For example, the energy for a linear ACE model is calculated as:
:math:`E=\sum_i^{N\_atoms} \sum_{\boldsymbol{\nu}} c_{\boldsymbol{\nu}}
B_{i,\boldsymbol{\boldsymbol{\nu}}}`.  The ACE descriptors for atom `i`
:math:`B_{i,\boldsymbol{\nu}}`, and :math:`c_{\nu}` are linear model
parameters.  The detailed definition and indexing convention for ACE
descriptors is given in :ref:`(Drautz) <Drautz19>`.  In short, body
order :math:`N`, angular character, radial character, and chemical
elements in the *N-body* descriptor are encoded by :math:`\nu`.  In the
:doc:`pace pair_style <pair_pace>`, the linear model parameters and the
ACE descriptors are combined for efficient evaluation of energies and
forces.  The details and benefits of this efficient implementation are
given in :ref:`(Lysogorskiy) <Lysogorskiy21>`, but the combined
descriptors and linear model parameters for the purposes of `compute
pace` may be expressed in terms of the ACE descriptors mentioned above.

:math:`c_{\boldsymbol{\nu}} B_{i,\boldsymbol{\nu}}= \sum_{\boldsymbol{\nu}' \in \boldsymbol{\nu} } \big[ c_{\boldsymbol{\nu}} C(\boldsymbol{\nu}') \big] A_{i,\boldsymbol{\nu}'}`

where the bracketed terms on the right-hand side are the combined functions
with linear model parameters typically provided in the `<name>.yace` potential
file for `pace pair_style`. When these bracketed terms are multiplied by the
products of the atomic base from :ref:`(Drautz) <Drautz19>`,
:math:`A_{i,\boldsymbol{\nu'}}`, the ACE descriptors are recovered but they
are also scaled by linear model parameters. The generalized coupling coefficients,
written in short-hand here as :math:`C(\boldsymbol{\nu}')`, are the generalized
Clebsch-Gordan or generalized Wigner symbols. It may be desirable to reverse the
combination of these descriptors and the linear model parameters so that the
ACE descriptors themselves may be used. The ACE descriptors and their gradients
are often used when training ACE models, performing custom data analysis,
generalizing ACE model forms, and other tasks that involve direct computation of
descriptors. The key utility of `compute pace` is that it can compute the ACE
descriptors and gradients so that these tasks can be performed during a LAMMPS
simulation or so that LAMMPS can be used as a driver for tasks like ACE model
parameterization. To see how this command can be used within a Python workflow
to train ACE potentials, see the examples in
`FitSNAP <https://github.com/FitSNAP/FitSNAP>`_. Examples on using outputs from
this compute to construct general ACE potential forms are demonstrated in
:ref:`(Goff) <Goff23>`. The various keywords and inputs to `compute pace`
determine what ACE descriptors and related quantities are returned in a compute
array.

The coefficient file, `<name>.yace`, ultimately defines the number of ACE
descriptors to be computed, their maximum body-order, the degree of angular
character they have, the degree of radial character they have, the chemical
character (which element-element interactions are encoded by descriptors),
and other hyper-parameters defined in :ref:`(Drautz) <Drautz19>`. These may
be modeled after the potential files in :doc:`pace pair_style <pair_pace>`,
and have the same format. Details on how to generate the coefficient files
to train ACE models may be found in `FitSNAP <https://github.com/FitSNAP/FitSNAP>`_.

.. versionadded:: TBD

Style *pace/atom* computes only the per-atom ACE descriptors
:math:`B_{i,\boldsymbol{\nu}}` and exposes them as a per-atom array with one
row per atom and one column per descriptor, analogous to :doc:`compute
sna/atom <compute_sna_atom>`.  The values are identical to the per-atom
descriptor rows of the global array produced by style *pace* with *bikflag* =
1, but as per-atom data they can be written out with a :doc:`dump custom
<dump>` command, time-averaged with :doc:`fix ave/atom <fix_ave_atom>`, or
processed by :doc:`compute reduce <compute_reduce>`.  Rows of atoms outside
the compute group are set to zero.  Style *pace/atom* takes no *bikflag* or
*dgradflag* arguments and computes no descriptor gradients, energies, or
virials.

The keyword *bikflag* determines whether or not to list the descriptors of
each atom separately, or sum them together and list in a single row. If
*bikflag* is set to *0* then a single descriptor row is used, which contains
the per-atom ACE descriptors :math:`B_{i,\boldsymbol{\nu}}` summed over all
atoms *i* to produce :math:`B_{\boldsymbol{\nu}}`. If *bikflag* is set to
*1* this is replaced by a separate per-atom ACE descriptor row for each atom.
In this case, the entries in the final column for these rows are set to zero.

The keyword *dgradflag* determines whether to sum atom gradients or list
them separately. If *dgradflag* is set to 0, the ACE descriptor gradients
with respect to atom *j* are summed over all atoms *i*, which may be useful
when training linear ACE models on atomic forces.
If *dgradflag* is set to 1, gradients are listed separately for each pair of atoms.
Each row corresponds
to a single term :math:`\frac{\partial {B_{i,\boldsymbol{\nu}}}}{\partial {r}^a_j}`
where :math:`{r}^a_j` is the *a-th* position coordinate of the atom with global
index *j*. This also changes the number of columns to be equal to the number of
ACE descriptors, with 3 additional columns representing the indices :math:`i`,
:math:`j`, and :math:`a`, as explained more in the Output info section below.
The option *dgradflag=1* requires that *bikflag=1*.

.. versionadded:: TBD

The keyword *chunksize* is only applicable when using the Kokkos accelerator
variants of these styles (*pace/kk*, *pace/atom/kk*); the plain styles accept
and ignore it, so the same input script runs with and without the
:doc:`-suffix command-line switch <Run_options>`.  It controls the number of
atoms processed in each pass of the device descriptor calculation, bounding
the size of the per-chunk scratch arrays.  For example, if there are 8192
atoms on a GPU and *chunksize* is set to 4096, the calculation is broken into
two passes.  Larger chunks amortize kernel launch overhead at the cost of
device memory; style *pace* uses a smaller default than *pace/atom* because
its descriptor-gradient scratch arrays are much larger per atom.

.. versionadded:: TBD

The compute *pace/grid* and *pace/grid/local* commands calculate ACE
descriptors :math:`B_{\boldsymbol{\nu}}` for a regular grid of points.
These are calculated from the local density of nearby atoms *i'* around
each grid point, as if there was a probe atom of a user-selected chemical
element located at the grid point.  This is useful for characterizing
fine-scale structure in a configuration of atoms, analogous to :doc:`compute
sna/grid <compute_sna_atom>`.  Neighbor atoms not in the compute group do
not contribute to the descriptors of a grid point.  Note that this is the
opposite sense of the group filter used by style *pace/atom*: for
*pace/atom* the group selects which atoms receive descriptors, while for
*pace/grid* and *pace/grid/local* the group instead selects which atoms are
allowed to contribute to the environment of every grid point; descriptors
are always computed for every grid point, regardless of group membership.
An atom located exactly at a grid point (separation :math:`r \approx 0`) is
excluded from that point's environment.  Both computes can be hardware
accelerated with Kokkos by using the *pace/grid/kk* and *pace/grid/local/kk*
commands, respectively.

Compute *pace/grid* calculates a global array containing ACE descriptors
for a regular grid of points.  The grid is aligned with the current box
dimensions, with the first point at the box origin, and forming a regular
3d array with *nx*, *ny*, and *nz* points in the x, y, and z directions.
For triclinic boxes, the array is congruent with the periodic lattice
vectors a, b, and c.  The array contains one row for each of the
:math:`nx \times ny \times nz` grid points, looping over the index for *ix*
fastest, then *iy*, and *iz* slowest.  Each row of the array contains the
*x*, *y*, and *z* coordinates of the grid point, followed by the ACE
descriptors.  See section below on output for a detailed explanation of the
data layout in the global array.

Compute *pace/grid/local* calculates ACE descriptors of a regular grid of
points similarly to compute *pace/grid* described above.  However, because
the array is local, it contains only rows for grid points that are local to
the processor subdomain.  The global grid of :math:`nx \times ny \times nz`
points is still laid out in space the same as for *pace/grid*, but grid
points are strictly partitioned, so that every grid point appears in one
and only one local array.  The array contains one row for each of the local
grid points, looping over the global index *ix* fastest, then *iy*, and
*iz* slowest.  Each row of the array contains the global indexes *ix*,
*iy*, and *iz* first, followed by the *x*, *y*, and *z* coordinates of the
grid point, followed by the ACE descriptors.  See section below on output
for a detailed explanation of the data layout in the local array.

.. note::

   Compute *pace/grid* allocates the entire
   :math:`nx \times ny \times nz \times (3 + K)` global array (and a second
   copy used during the reduction) on **every** MPI rank, and sums the array
   across all ranks with a global reduction on every invocation.  Its memory
   and communication cost therefore grow with the total grid size no matter
   how many processors are used.  For large grids -- for example the
   millions of grid points typical of grid-based machine-learning workflows
   -- use compute *pace/grid/local* instead: it stores only the rows for the
   grid points owned by each processor and performs no global reduction.

The *grid* keyword defines *nx*, *ny*, and *nz*, the number of grid points
spanning the box in each of the x, y, and z directions.

The keyword *element* selects the chemical element, from those listed in
the *ace_potential_filename* coupling-coefficient file, used as the central
(probe) species for every grid point.  If unspecified, the first element
listed in the file is used.  This selection applies uniformly to the entire
grid; a single invocation of *pace/grid* or *pace/grid/local* cannot mix
elements across grid points.

The keyword *chunksize* is only applicable when using the Kokkos accelerator
variants of these styles (*pace/grid/kk*, *pace/grid/local/kk*); the plain
styles accept and ignore it, so the same input script runs with and without
the :doc:`-suffix command-line switch <Run_options>`.  It controls the
number of grid points processed in each pass of the device descriptor
calculation, bounding the size of the per-chunk scratch arrays, in the same
way that *chunksize* bounds the number of atoms per pass for styles *pace*
and *pace/atom*.

.. note::

    It is noted here that in contrast to :doc:`pace pair_style <pair_pace>`,
    the *.yace* file for `compute pace` typically should not contain linear
    parameters for an ACE potential. If :math:`c_{\nu}` are included,
    the value of the descriptor will not be returned in the `compute` array,
    but instead, the energy contribution from that descriptor will be returned.
    Do not do this unless it is the desired behavior.
    *In short, you should not plug in a '.yace' for a pace potential into this
    compute to evaluate descriptors.*

.. note::

    *Generalized Clebsch-Gordan or Generalized Wigner symbols (with appropriate
    factors) must be used to evaluate ACE descriptors with this compute.* There
    are multiple ways to define the generalized coupling coefficients. Because
    of this, this compute will not revert your potential file to a coupling
    coefficient file. Instead this compute allows the user to supply coupling
    coefficients that follow any convention.

.. note::

   Using *dgradflag* = 1 produces a global array with :math:`N + 3N^2 + 1` rows
   which becomes expensive for systems with more than 1000 atoms.

.. note::

   If you have a bonded system, then the settings of :doc:`special_bonds
   <special_bonds>` command can remove pairwise interactions between
   atoms in the same bond, angle, or dihedral.  This is the default
   setting for the :doc:`special_bonds <special_bonds>` command, and
   means those pairwise interactions do not appear in the neighbor list.
   Because this compute uses the neighbor list, it also means those pairs
   will not be included in the calculation.  One way to get around this,
   is to write a dump file, and use the :doc:`rerun <rerun>` command to
   compute the ACE descriptors for snapshots in the dump file.
   The rerun script can use a :doc:`special_bonds <special_bonds>`
   command that includes all pairs in the neighbor list.

----------

Output info
"""""""""""

Compute *pace* evaluates a global array.  The columns are arranged into
*ntypes* blocks, listed in order of atom type *I*\ . Each block contains
one column for each ACE descriptor, the same as for compute
*sna/atom*\ in :doc:`compute snap <compute_sna_atom>`. A final column contains the corresponding energy, force
component on an atom, or virial stress component. The rows of the array
appear in the following order:

* 1 row: *pace* average descriptor values for all atoms of type *I*
* 3\*\ *n* force rows: quantities, with derivatives w.r.t. x, y, and z coordinate of atom *i* appearing in consecutive rows. The atoms are sorted based on atom ID and run up to the total number of atoms, *n*.
* 6 rows: *virial* quantities summed for all atoms of type *I*

For example, if :math:`\# \; B_{i, \boldsymbol{\nu}}` =30 and ntypes=1, the
number of columns in the global array generated by *pace* is 31, while the
number of rows is 1+3\*\ *n*\ +6, where *n* is the total number of atoms.

If the *bik* keyword is set to 1, the structure of the pace array is expanded.
The first :math:`N` rows of the pace array
correspond to :math:`\# \; B_{i,\boldsymbol{\nu}}` instead of a single row summed over atoms :math:`i`.
In this case, the entries in the final column for these rows
are set to zero. Also, each row contains only non-zero entries for the
columns corresponding to the type of that atom. This is not true in the case
of *dgradflag* keyword = 1 (see below).

If the *dgradflag* keyword is set to 1, this changes the structure of the
global array completely.
Here the per-atom quantities are replaced with rows corresponding to
descriptor gradient components on single atoms:

.. math::

  \frac{\partial {B_{i,\boldsymbol{\nu}}  }}{\partial {r}^a_j}

where :math:`{r}^a_j` is the *a-th* position coordinate of the atom with global
index *j*. The rows are
organized in chunks, where each chunk corresponds to an atom with global index
:math:`j`. The rows in an atom :math:`j` chunk correspond to
atoms with global index :math:`i`. The total number of rows for
these descriptor gradients is therefore :math:`3N^2`.
The number of columns is equal to the number of ACE descriptors,
plus 3 additional left-most columns representing the global atom indices
:math:`i`, :math:`j`,
and Cartesian direction :math:`a`  (0, 1, 2, for x, y, z).
The first 3 columns of the first :math:`N` rows belong to the reference
potential force components. The remaining K columns contain the
:math:`B_{i,\boldsymbol{\nu}}` per-atom descriptors corresponding to the non-zero entries
obtained when *bikflag* = 1.
The first column of the last row, after the first
:math:`N + 3N^2` rows, contains the reference potential
energy. The virial components are not used with this option. The total number of
rows is therefore :math:`N + 3N^2 + 1` and the number of columns is :math:`K + 3`.

Compute *pace/atom* instead evaluates a per-atom array.  The number of
columns is equal to the number of ACE descriptors; each row contains the
descriptors :math:`B_{i,\boldsymbol{\nu}}` of one atom, in the same
descriptor order as the columns of one type block of the *pace* global
array.  Rows of atoms outside the compute group are zero.

Compute *pace/grid* evaluates a global array.  The array contains one row
for each of the :math:`nx \times ny \times nz` grid points, looping over
the index for *ix* fastest, then *iy*, and *iz* slowest.  Each row of the
array contains the *x*, *y*, and *z* coordinates of the grid point,
followed by the *nvalues* ACE descriptors, in the same descriptor order as
one type block of the *pace* global array.  For an orthogonal box of edge
lengths :math:`L_x`, :math:`L_y`, :math:`L_z`, grid point (*ix*, *iy*,
*iz*) is located at :math:`(ix\,L_x/nx,\ iy\,L_y/ny,\ iz\,L_z/nz)`; for
triclinic boxes the grid is congruent with the periodic lattice vectors
instead.

Compute *pace/grid/local* evaluates a local array.  The array contains one
row for each of the local grid points, looping over the global index *ix*
fastest, then *iy*, and *iz* slowest.  Each row of the array contains the
global indexes *ix*, *iy*, and *iz* first, followed by the *x*, *y*, and
*z* coordinates of the grid point, followed by the *nvalues* ACE
descriptors.

The global array values of *pace* and *pace/grid* can be accessed by any
command that uses global values from a compute as input, the local array
of *pace/grid/local* by any command that uses local values from a compute
as input, and the per-atom array of *pace/atom* by any command that uses
per-atom values from a compute as input.  See the :doc:`Howto output
<Howto_output>` doc page for an overview of LAMMPS output options.

----------

.. include:: accel_styles.rst

----------

Restrictions
""""""""""""

These computes are part of the ML-PACE package.  They are only enabled
if LAMMPS was built with that package.  See the :doc:`Build package
<Build_package>` page for more info.

The global array of style *pace* is sized using the number of atoms present
when the compute is defined.  Define the compute after all atoms have been
created, and do not add or remove atoms afterwards.

The Kokkos accelerator variants support only the spline-based radial
functions and the *FinnisSinclair* / *FinnisSinclairShiftedScaled* embedding
forms of standard *.yace* potential files, and do not support a ZBL core
repulsion inner cutoff.  Unsupported potential files produce an error, in
which case the plain CPU styles can be used instead.

Styles *pace/grid* and *pace/grid/local* build on the grid infrastructure
of the ML-SNAP package (shared with :doc:`compute sna/grid
<compute_sna_atom>`), so the ML-SNAP package must also be installed;
enabling the ML-PACE package with CMake automatically enables ML-SNAP.

A pair style with a force cutoff at least as large as the cutoff of the
*ace_potential_filename* potential must be defined, exactly as for style
*pace* above; this guarantees that the ghost atoms needed to fill each grid
point's environment are available.

For orthogonal simulation boxes, the lower corner of the box must be
located at the origin (boxlo = 0), because grid point positions are
measured from the origin and the restriction is not otherwise enforced.
Triclinic boxes have no such restriction, because grid point positions
are computed from fractional (lamda) coordinates.

For all of these compute styles, LAMMPS atom types are mapped to the
elements of the *ace_potential_filename* file by order: atom type 1
corresponds to the first element listed in the file, type 2 to the second,
and so on.  The number of atom types may not exceed the number of elements
in the file.

A grid point of styles *pace/grid* and *pace/grid/local* may coincide
exactly with an atom position: any atom closer than :math:`10^{-10}`
distance units to a grid point is excluded from that grid point's
environment, so the computed descriptors are those of the remaining
neighborhood.  However, a grid point at a very small but nonzero distance
from an atom -- closer than roughly the innermost bin of the tabulated
radial functions, on the order of :math:`10^{-3}` distance units -- stops
the run with an "Encountered very small distance" error from the ACE
evaluator.  Dense grids in dense systems can occasionally place a grid
point this close to a nucleus; choosing slightly different grid dimensions
*nx*, *ny*, *nz* moves the grid points and avoids the error.

Related commands
""""""""""""""""

:doc:`pair_style pace <pair_pace>`
:doc:`pair_style snap <pair_snap>`
:doc:`compute snap <compute_sna_atom>`
:doc:`compute sna/grid <compute_sna_atom>`

Default
"""""""

The keyword default is *chunksize* = 4096 for *pace/atom/kk* and 256 for
*pace/kk*, and 4096 for *pace/grid/kk* and *pace/grid/local/kk*.  The
keyword default for *element* is the first element listed in the
*ace_potential_filename* file.

----------

.. _Drautz19:

**(Drautz)** Drautz, Phys Rev B, 99, 014104 (2019).

.. _Lysogorskiy21:

**(Lysogorskiy)** Lysogorskiy, van der Oord, Bochkarev, Menon, Rinaldi, Hammerschmidt, Mrovec, Thompson, Csanyi, Ortner, Drautz, npj Comp Mat, 7, 97 (2021).

.. _Goff23:

**(Goff)** Goff, Zhang, Negre, Rohskopf, Niklasson, Journal of Chemical Theory and Computation 19, no. 13 (2023).
