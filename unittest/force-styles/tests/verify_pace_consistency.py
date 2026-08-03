#!/usr/bin/env python3
"""Cross-verify the compute pace / pace/atom output-style reference data.

The four YAML reference files in unittest/force-styles/tests/ are generated
independently by the tester, yet they must agree because they are different
views of the same ACE descriptors B_{i,nu} and their gradients dB_i/dr_j on the
same geometry:

  compute-pace_atom.yaml   pace/atom : per-atom B_{i,nu}           (peratom_data)
  compute-pace.yaml        pace 1 0  : bikflag=1 dgradflag=0        (global_array)
  compute-pace_sum.yaml    pace 0 0  : bikflag=0 dgradflag=0        (global_array)
  compute-pace_dgrad.yaml  pace 1 1  : bikflag=1 dgradflag=1        (global_array)

Global-array layouts (from src/ML-PACE/compute_pace.cpp):

  dgradflag=0, cols = nvalues*ntypes + 1
    rows 0 .. bik_rows-1            descriptor rows (bik_rows = natoms if bikflag
                                    else 1); atom of type it -> block it-1,
                                    i.e. col nvalues*(it-1)+nu
    rows bik_rows .. +3*natoms-1    dB/dr rows: atom tag t, component d ->
                                    row 3*(t-1)+bik_rows+d; column binned by the
                                    *centre* atom's type block
    rows +3*natoms .. +6           virial rows (dbdotr, Voigt)
    last column                    reference energy / forces / virial (0 here)

  dgradflag=1, cols = nvalues + 3
    rows 0 .. natoms-1             descriptor rows: cols 0..2 forces, cols 3..
                                    B_{t,nu}
    gradient rows                  meta cols (0,1,2) = (centre I, wrt-atom J,
                                    component d); cols 3.. = V(I,J,d,nu), which
                                    equals -dB_{I,nu}/dr_{J,d} for every J (the
                                    diagonal J==I carries sum over neighbours).

This script recomputes each quantity from the others and checks they match.  It
is standalone (only needs the four YAML files + the data file for atom types) so
it can be re-run whenever the references are regenerated.

Usage:  python3 verify_pace_consistency.py [tests_dir]
"""
import sys, os, re

TESTS = sys.argv[1] if len(sys.argv) > 1 else os.path.dirname(os.path.abspath(__file__))
TOL = 1e-9

def load_block(path, key):
    rows = []
    with open(path) as fh:
        lines = fh.read().splitlines()
    i = next(n for n, l in enumerate(lines) if l.startswith(key))
    for l in lines[i + 1:]:
        if l and l[0] == ' ':
            rows.append([float(x) for x in l.split()])
        else:
            break
    return rows

def atom_types(datafile):
    lines = open(datafile).read().splitlines()
    i = next(n for n, l in enumerate(lines) if l.strip().startswith("Atoms"))
    types = {}
    for l in lines[i + 1:]:
        s = l.split()
        if not s:
            continue
        if not s[0].lstrip("-").isdigit():
            break
        types[int(s[0])] = int(s[1])   # tag -> type
    return types

# ---- load everything -------------------------------------------------------
types = atom_types(os.path.join(TESTS, "data.pace-hno"))
natoms = len(types)
peratom = load_block(os.path.join(TESTS, "compute-pace_atom.yaml"),  "peratom_data")
g_b1d0  = load_block(os.path.join(TESTS, "compute-pace.yaml"),       "global_array")
g_b0d0  = load_block(os.path.join(TESTS, "compute-pace_sum.yaml"),   "global_array")
g_b1d1  = load_block(os.path.join(TESTS, "compute-pace_dgrad.yaml"), "global_array")

nvalues = len(peratom[0]) - 1          # first per-atom column is the tag
ntypes  = max(types.values())
assert len(g_b1d1[0]) == nvalues + 3, "dgrad=1 width mismatch"
assert len(g_b1d0[0]) == nvalues * ntypes + 1, "dgrad=0 width mismatch"

# per-atom descriptors keyed by tag
B_atom = {int(r[0]): r[1:] for r in peratom}
assert set(B_atom) == set(types), "per-atom tags != data-file atoms"

fails = []
def check(name, cond, detail=""):
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f"  {detail}" if detail and not cond else ""))
    if not cond:
        fails.append(name)

def close(a, b):
    return abs(a - b) <= TOL + 1e-7 * max(abs(a), abs(b))

def maxdiff(pairs):
    return max((abs(a - b) for a, b in pairs), default=0.0)

print(f"System: natoms={natoms}, ntypes={ntypes}, nvalues={nvalues}, tol={TOL}")

# ---- 1. descriptor equivalence: pace/atom == dgrad0 blocks == dgrad1 cols3.. ----
d = []
for t in range(1, natoms + 1):
    it = types[t]
    for nu in range(nvalues):
        b_atom  = B_atom[t][nu]
        b_dgrad0 = g_b1d0[t - 1][nvalues * (it - 1) + nu]     # atom t's own type block
        b_dgrad1 = g_b1d1[t - 1][3 + nu]
        d.append((b_atom, b_dgrad0)); d.append((b_atom, b_dgrad1))
check("descriptors  pace/atom == pace(bik1,dgrad0) == pace(bik1,dgrad1)",
      all(close(a, b) for a, b in d), f"maxdiff={maxdiff(d):.2e}")

# off-block descriptor columns of dgrad0 must be exactly zero (type routing)
offblock = []
for t in range(1, natoms + 1):
    it = types[t]
    for T in range(ntypes):
        if T == it - 1:
            continue
        offblock += [g_b1d0[t - 1][nvalues * T + nu] for nu in range(nvalues)]
check("dgrad0 off-type descriptor columns are exactly zero (type routing)",
      all(v == 0.0 for v in offblock), f"max|v|={max(map(abs,offblock),default=0):.2e}")

# ---- 2. summed descriptors: bik0 row0 == sum over atoms of each type ----
s = []
for T in range(ntypes):
    for nu in range(nvalues):
        summed_ref = g_b0d0[0][nvalues * T + nu]
        summed_atom = sum(B_atom[t][nu] for t in types if types[t] == T + 1)
        summed_b1   = sum(g_b1d0[t - 1][nvalues * T + nu] for t in range(1, natoms + 1))
        s.append((summed_ref, summed_atom)); s.append((summed_ref, summed_b1))
check("summed descriptors  pace(bik0) row0 == Sigma_atoms pace/atom == Sigma rows pace(bik1)",
      all(close(a, b) for a, b in s), f"maxdiff={maxdiff(s):.2e}")

# ---- 3. gradients: bik0 dB/dr rows == bik1 dB/dr rows (identical) ----
g = []
for t in range(1, natoms + 1):
    for dcmp in range(3):
        r1 = g_b1d0[3 * (t - 1) + natoms + dcmp]        # bik_rows = natoms
        r0 = g_b0d0[3 * (t - 1) + 1 + dcmp]             # bik_rows = 1
        g += list(zip(r1[:nvalues * ntypes], r0[:nvalues * ntypes]))
check("dB/dr rows identical between bikflag=1 and bikflag=0",
      all(close(a, b) for a, b in g), f"maxdiff={maxdiff(g):.2e}")

# ---- 3b. gradients: reconstruct dgrad0 dB/dr rows from the dgrad1 per-pair data ----
# G0[g, block T, d, nu] = Sigma_{centre i : type(i)=T+1} V(i, g, d, nu)
V = {}                                   # (I0, J0, d) -> [values over nu]
ngrad = 3 * natoms * natoms              # gradient rows only; the trailing row
grad_rows = g_b1d1[natoms:natoms + ngrad]  # (energy) must be excluded
assert len(grad_rows) == ngrad, "unexpected dgrad=1 row count"
for r in grad_rows:
    I0, J0, dcmp = int(round(r[0])), int(round(r[1])), int(round(r[2]))
    V[(I0, J0, dcmp)] = r[3:3 + nvalues]
rec = []
for t in range(1, natoms + 1):           # g = atom tag t  (0-based t-1)
    for dcmp in range(3):
        row = g_b1d0[3 * (t - 1) + natoms + dcmp]
        for T in range(ntypes):
            for nu in range(nvalues):
                lhs = row[nvalues * T + nu]
                rhs = sum(V.get((i - 1, t - 1, dcmp), [0.0] * nvalues)[nu]
                          for i in range(1, natoms + 1) if types[i] == T + 1)
                rec.append((lhs, rhs))
check("dgrad0 dB/dr rows == reconstruction from dgrad1 per-pair gradients",
      all(close(a, b) for a, b in rec), f"maxdiff={maxdiff(rec):.2e}")

# ---- 4. translational invariance of the dgrad1 gradients: Sigma_J V(I,J) == 0 ----
ti_sum = []
for I in range(natoms):
    for dcmp in range(3):
        for nu in range(nvalues):
            tot = sum(V.get((I, J, dcmp), [0.0] * nvalues)[nu] for J in range(natoms))
            ti_sum.append(tot)
check("dgrad1 translational invariance  Sigma_J dB_I/dr_J == 0",
      all(abs(v) <= 1e-8 for v in ti_sum), f"max|Sigma|={max(map(abs,ti_sum),default=0):.2e}")

# ---- 4b. momentum sum on dgrad0 dB/dr rows: Sigma_atoms G0 == 0 per column ----
mom = []
for dcmp in range(3):
    for c in range(nvalues * ntypes):
        tot = sum(g_b1d0[3 * (t - 1) + natoms + dcmp][c] for t in range(1, natoms + 1))
        mom.append(tot)
check("dgrad0 dB/dr rows sum to zero over atoms (per column)",
      all(abs(v) <= 1e-8 for v in mom), f"max|Sigma|={max(map(abs,mom),default=0):.2e}")

# ---- 5. virial rows identical between bikflag=0 and bikflag=1 ----
vir = []
for v in range(6):
    r1 = g_b1d0[natoms + 3 * natoms + v]
    r0 = g_b0d0[1 + 3 * natoms + v]
    vir += list(zip(r1[:nvalues * ntypes], r0[:nvalues * ntypes]))
check("virial rows identical between bikflag=1 and bikflag=0 (dbdotr)",
      all(close(a, b) for a, b in vir), f"maxdiff={maxdiff(vir):.2e}")
check("virial rows are non-trivial (ghost r.dB/dr path exercised)",
      any(abs(a) > 1e-6 for a, _ in vir))

# ---- 6. pair zero => energy/force/virial last column and dgrad1 force cols are zero ----
lastcol0 = [g_b1d0[r][-1] for r in range(len(g_b1d0))] + [g_b0d0[r][-1] for r in range(len(g_b0d0))]
check("dgrad0 last column (energy/force/virial under pair zero) is zero",
      all(v == 0.0 for v in lastcol0), f"max|v|={max(map(abs,lastcol0)):.2e}")
force_cols = [g_b1d1[t][c] for t in range(natoms) for c in range(3)]
check("dgrad1 descriptor-row force columns (0..2) are zero under pair zero",
      all(v == 0.0 for v in force_cols), f"max|v|={max(map(abs,force_cols),default=0):.2e}")

print()
if fails:
    print(f"FAILED {len(fails)} check(s): " + "; ".join(fails))
    sys.exit(1)
print("All cross-consistency checks passed.")
