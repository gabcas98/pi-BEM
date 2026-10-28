#!/usr/bin/env python3
"""Regression test of the Galerkin and collocation solvers on small meshes.

    python3 run_regression.py            run every case and compare
    python3 run_regression.py --update   run every case and store the values
                                         as the new reference
    python3 run_regression.py CASE ...   run only the named cases

Each case writes parameters_bem_3.prm (parameters_bem_2.prm in 2D) from a
frozen base file in base/, runs the solver built in ../build on 8 processes,
and reads a few quantities from the log. They are compared with reference.json: the load vector and the
Frobenius norm of V, printed with sixteen digits, to 1e-12 relative; the
errors, printed with six, to 1e-5 relative; Gauss's identity of the mixed
problem, a difference of nearly equal sums, to 1e-3; the peak memory of a
process, where checked, may exceed its reference by at most a quarter. The
checks also cover what went wrong before: an executable in a run folder
older than the library, and runs that fail or hang instead of finishing.

The mixed problem runs on the NACA wing of the report, whose coarse mesh and
CAD files are frozen in naca/. The Robin and complex problems of upstream
pi-BEM run on the sphere of its samples, frozen in sphere/, and in 2D on
the decagon of circle/; their errors are those of the boundary conditions,
material by material.

The base files are copies, so later edits to the study files in RUN_EXP do
not move the reference. The run folder sits next to RUN_EXP because the
program finds the mesh and the CAD files by relative path.
"""
import filecmp
import json
import os
import re
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
EXECUTABLE = os.path.join(ROOT, 'build', 'bem_fma_3d')
REFERENCE = os.path.join(HERE, 'reference.json')

FORMAT = {'sixteen': 1e-12, 'six': 1e-5, 'loose': 1e-3}
# A measured value may exceed its reference by this factor.
CEILING = 1.25

# quantity: (pattern, precision class)
QUANTITIES = {
    'rhs_norm': (r'Galerkin assembly check: \|b\| (\S+),', 'sixteen'),
    'V_frobenius': (r'\|V\|_F (\S+)', 'sixteen'),
    'flux': (r'dPhidN true relative L_2 error: (\S+)', 'six'),
    'gradient': (r'Phi Cells Gradient  error L_2 norm: (\S+)', 'six'),
    'energy': (r'dPhidN energy norm error: (\S+)', 'six'),
    'best_ratio': (r'Galerkin error / best-projection error: (\S+)', 'six'),
    'gradient_pointwise': (r'Phi Gradient with the flux as normal '
                           r'component: L_2 (\S+);', 'six'),
    'iterations': (r'(?:CG|GMRES) iterations: (\d+)', 'exact'),
    'manufactured_norm': (r'\|b_manufactured\| (\S+),', 'six'),
    'rhs_difference': (r'\|b_manufactured \+ b_regularized\| (\S+),', 'six'),
    'mixed_flux': (r'flux on the Dirichlet part, relative L2 error (\S+);',
                   'six'),
    'mixed_potential': (r'potential on the Neumann part, relative L2 error '
                        r'(\S+)', 'six'),
    'mixed_gauss': (r'\|\(1/2 M \+ K\) 1\| / \|1/2 M 1\| = (\S+),', 'loose'),
    'memory_kb': (r'RSS (\d+) kB', 'ceiling'),
    # The errors by material of the Robin and complex problems.
    'phi_nodes': (r'Phi Nodes error L_inf norm: (\S+)', 'six'),
    'robin_residual': (r'Robin Nodes Residual error L_2 norm: (\S+)', 'six'),
    'steps': (r'Solve terminated at step (\d+)', 'exact'),
    'mixed_robin_flux': (r'flux on the Robin part, relative L2 error (\S+)',
                         'six'),
    # The real part of a complex mixed problem (the quantities above take
    # the last line, of the imaginary part).
    'mixed_flux_real': (r'real part: flux on the Dirichlet part, relative L2 '
                        r'error (\S+);', 'six'),
    'mixed_potential_real': (r'real part: flux on the Dirichlet part, '
                             r'relative L2 error \S+; potential on the '
                             r'Neumann part, relative L2 error (\S+)', 'six'),
}

# The mixed problem on the NACA wing: flux on the lower surface, potential
# on the upper surface and the tips, data on the exact surface.
MIXED = ['mixed_flux', 'mixed_potential', 'mixed_gauss', 'iterations',
         'memory_kb']

# The complex sample sphere in the Galerkin formulation.
GALERKIN_COMPLEX = {'Boundary integral formulation': 'galerkin',
                    'Linear solver': 'CG',
                    'Preconditioner': 'BlockJacobi'}

# name: (base file, settings in BEMProblem<dim>, quantities to compare
#        [, dim, 3 if absent])
CASES = {
    'dg_laplace_p1': ('g_p1_bal0_weak.prm',
                      {'Preconditioner': 'BlockJacobi',
                       'Compute energy norm error': 'true'},
                      ['rhs_norm', 'V_frobenius', 'flux', 'gradient',
                       'energy', 'best_ratio', 'gradient_pointwise',
                       'iterations']),
    'dg_laplace_p1_library_path': ('g_p1_bal0_weak.prm',
                                   {'Preconditioner': 'Jacobi',
                                    'Galerkin fused right-hand side': 'false',
                                    'Galerkin light geometry': 'false'},
                                   ['rhs_norm', 'V_frobenius', 'flux',
                                    'gradient', 'iterations']),
    'dg_screened_p2': ('screened_p2_bal0_weak.prm',
                       {'Preconditioner': 'BlockJacobi'},
                       ['rhs_norm', 'V_frobenius', 'flux', 'gradient',
                        'iterations']),
    'continuous_laplace_q2': ('g_p2_bal0_weak.prm',
                              {'Preconditioner': 'BlockJacobi',
                               'Finite Element type': 'FEQ',
                               'Compute energy norm error': 'true'},
                              ['rhs_norm', 'V_frobenius', 'flux', 'gradient',
                               'energy', 'best_ratio', 'iterations']),
    'representation_p2': ('g_p2_bal0_weak.prm',
                          {'Preconditioner': 'BlockJacobi',
                           'Gradient recovery': 'hypersingular_weak'},
                          ['flux', 'gradient']),
    'far_field_tolerance_p2': ('g_p2_bal0_weak.prm',
                               {'Preconditioner': 'BlockJacobi',
                                'Galerkin far field tolerance': '1e-8'},
                               ['rhs_norm', 'flux', 'gradient']),
    'dg_laplace_p1_fmm': ('g_p1_bal0_weak.prm',
                          {'Preconditioner': 'BlockJacobi',
                           'Solution method': 'FMA',
                           'FMA Truncation Order': '10'},
                          ['rhs_norm', 'flux', 'gradient', 'best_ratio',
                           'iterations']),
    'dg_laplace_p1_fmm_patch': ('g_p1_bal0_weak.prm',
                                {'Preconditioner': 'PatchSchwarz',
                                 'Solution method': 'FMA',
                                 'FMA Truncation Order': '10'},
                                ['flux', 'gradient', 'best_ratio',
                                 'iterations']),
    'continuous_laplace_q2_fmm': ('g_p2_bal0_weak.prm',
                                  {'Preconditioner': 'PatchSchwarz',
                                   'Finite Element type': 'FEQ',
                                   'Solution method': 'FMA',
                                   'FMA Truncation Order': '10'},
                                  ['rhs_norm', 'flux', 'gradient',
                                   'best_ratio', 'iterations']),
    'dg_screened_p2_fmm': ('screened_p2_bal0_weak.prm',
                           {'Preconditioner': 'PatchSchwarz',
                            'Solution method': 'FMA',
                            'FMA Truncation Order': '10'},
                           ['rhs_norm', 'flux', 'gradient', 'iterations']),
    # The consistent right-hand side through the FMM, integrated in the
    # traversal of the single layer and the double layer, compared with the
    # regularised one.
    'dg_screened_p2_fmm_manufactured': ('screened_p2_bal0_weak.prm',
                                        {'Preconditioner': 'PatchSchwarz',
                                         'Solution method': 'FMA',
                                         'FMA Truncation Order': '10',
                                         'Dirichlet RHS treatment':
                                           'manufactured_flux'},
                                        ['manufactured_norm',
                                         'rhs_difference', 'flux',
                                         'gradient', 'iterations']),
    # kappa = 10 plane wave with exponential refinement towards the corner of
    # the box, screened FMM of order 8 (the base file sets it all).
    'dg_screened_plane_p1_fmm': ('screened_plane_exponential_p1.prm', {},
                                 ['rhs_norm', 'flux', 'gradient',
                                  'iterations']),
    'collocation_q1': ('g_p1_bal0_weak.prm',
                       {'Boundary integral formulation': 'collocation',
                        'Finite Element type': 'FEQ',
                        'Linear solver': 'GMRES',
                        'Preconditioner': 'ILU',
                        'Dirichlet RHS treatment': 'nodal',
                        'Gradient recovery': 'projection'},
                       ['flux', 'gradient', 'iterations']),
    # Flat cells with the data in space, p = 1, level 1, multipole method
    # and Calderon preconditioner, as in the final campaign of the report.
    'naca_mixed_flat_p1_fmm': ('naca_mixed.prm',
                               {'Finite element order': '1'}, MIXED),
    # The exact parametric geometry, p = 2, level 1: folded pairs (order
    # 10) at the trailing edge, skewed cells (order 12) on the tips, kinked
    # pairs (order 10) along the edges of the tips and near-singular up to
    # a diameter apart, the Dirichlet datum by its L2 projection.
    'naca_mixed_exact_p2_fmm': ('naca_mixed.prm',
                                {'Exact geometry': 'true',
                                 'Exact geometry map': 'parametric',
                                 'Galerkin singular quadrature order for '
                                 'kinked pairs': '10',
                                 'Galerkin near singular distance ratio '
                                 'for kinked pairs': '1.0'}, MIXED),
    # One cycle of refinement by the curvature of the surface at 10
    # degrees on the exact geometry: the cells of the nose cut along the
    # chord, the tips made conforming along their edges.
    'naca_mixed_curvature_p2_fmm': ('naca_mixed.prm',
                                    {'Exact geometry': 'true',
                                     'Exact geometry map': 'parametric',
                                     'Curvature refinement cycles': '1',
                                     'Curvature refinement angle': '10'},
                                    MIXED),
    # The tips in the Dirichlet part as well: three pieces whose flux
    # unknowns no cell joins (the sharp edges have double nodes), each with
    # its rank-one term in the Calderon preconditioner; inexact GMRES with
    # its own expansions, and a cheap preconditioner (order 2, opening
    # 0.95, 32 points per leaf).
    'naca_mixed_tips_dirichlet_p1_fmm': ('naca_mixed.prm',
                                         {'Finite element order': '1',
                                          'Dirichlet boundary ids':
                                          '1,2,4,110,110',
                                          'Neumann boundary ids':
                                          '3,110,110',
                                          'Mixed Galerkin relaxed products '
                                          'below': '1e-5',
                                          'Galerkin preconditioner truncation '
                                          'order': '2',
                                          'Galerkin preconditioner opening '
                                          'criterion': '0.95',
                                          'Galerkin preconditioner maximum '
                                          'points per leaf': '32',
                                          'Galerkin relaxed truncation '
                                          'order': '4',
                                          'Galerkin relaxed opening '
                                          'criterion': '0.6'}, MIXED),
    # One cycle of graded refinement around the source on the exact
    # geometry, dense matrices: hanging nodes on curved edges, with the
    # constraints at the physical points and the contacts found on the
    # mapped edges.
    'naca_mixed_graded_p1_dense': ('naca_mixed.prm',
                                   {'Finite element order': '1',
                                    'Solution method': 'Direct',
                                    'Exact geometry': 'true',
                                    'Exact geometry map': 'parametric',
                                    'Graded refinement material ids': '3,4',
                                    'Graded refinement focus': '0.4,0.3,0',
                                    'Graded refinement size ratio': '0.4',
                                    'Graded refinement minimum size': '0.10',
                                    'Graded refinement growth exponent': '1',
                                    'Graded refinement maximum cycles': '1'},
                                   MIXED),
    # Adaptive refinement from level 1, two cycles with the jump indicator,
    # the multipole method: the near-field blocks of the cells that are not
    # refined and the traces of the previous mesh as the start of GMRES.
    'naca_mixed_adaptive_p1_fmm': ('naca_mixed.prm',
                                   {'Finite element order': '1',
                                    'Exact geometry': 'true',
                                    'Exact geometry map': 'parametric',
                                    'Set Global Refinement': 'false',
                                    'Initial refinement level': '1',
                                    'Number of cycles': '2',
                                    'Mixed Galerkin error indicator': 'jumps',
                                    'Mixed Galerkin refinement fraction': '0.5'},
                                   MIXED),
    # A Robin datum that is not zero with the multipole method: it enters
    # the far field through the expansions of the right-hand side.
    'robin_real_fma': ('sphere_real_robin.prm', {},
                       ['phi_nodes', 'robin_residual', 'iterations']),
    # The complex problem, real and imaginary parts as two blocks, with the
    # matrices and with the multipole method; the errors of the imaginary
    # part.
    'robin_complex_direct': ('sphere_complex_robin.prm', {},
                             ['phi_nodes', 'robin_residual', 'steps']),
    'robin_complex_fma': ('sphere_complex_robin.prm',
                          {'Solution method': 'FMA'},
                          ['phi_nodes', 'robin_residual', 'steps']),
    # Dirichlet and Robin conditions in 2D.
    'robin_2d': ('circle_robin_2d.prm', {},
                 ['phi_nodes', 'robin_residual', 'iterations'], 2),
    # The Robin part in the mixed Galerkin problem, multipole method: the
    # flux -(a/b) u of the unknowns in V and K^T, its identity term in the
    # potential rows.
    'robin_galerkin_fma': ('sphere_real_robin.prm',
                           {'Boundary integral formulation': 'galerkin',
                            'Linear solver': 'CG',
                            'Preconditioner': 'BlockJacobi'},
                           ['mixed_flux', 'mixed_potential',
                            'mixed_robin_flux', 'iterations']),
    # The screened kernel in the mixed Galerkin problem with Dirichlet and
    # Robin parts: the weak hypersingular operator with its term
    # kappa^2 (n_x . n_y) G, dense and through the screened multipole
    # method, with the Calderon preconditioner.
    'screened_mixed_robin': ('sphere_screened_robin.prm',
                             {'Boundary integral formulation': 'galerkin',
                              'Linear solver': 'CG',
                              'Preconditioner': 'BlockJacobi',
                              'Solution method': 'Direct',
                              'Mixed Galerkin preconditioner': 'calderon'},
                             ['mixed_flux', 'mixed_potential',
                              'mixed_robin_flux', 'iterations']),
    'screened_mixed_robin_fma': ('sphere_screened_robin.prm',
                                 {'Boundary integral formulation': 'galerkin',
                                  'Linear solver': 'CG',
                                  'Preconditioner': 'BlockJacobi',
                                  'Solution method': 'FMA',
                                  'Mixed Galerkin preconditioner':
                                    'calderon'},
                                 ['mixed_flux', 'mixed_potential',
                                  'mixed_robin_flux', 'iterations']),
    # Two slots, each exact on its manifold and wrong on the other: the
    # Galerkin data of every cell from the slot of its manifold, in the mixed
    # problem with the multipole method, and with Dirichlet conditions on
    # both manifolds, dense and multipole, where the load vector is that of
    # the single function x+y+z.
    'slots_mixed_fma': ('sphere_slots_mixed.prm', {},
                        ['mixed_flux', 'mixed_potential', 'iterations']),
    'slots_dirichlet_dg': ('sphere_slots_dirichlet.prm', {},
                           ['rhs_norm', 'iterations']),
    'slots_dirichlet_fma': ('sphere_slots_dirichlet.prm',
                            {'Solution method': 'FMA',
                             'Mapping Type': 'Q'},
                            ['rhs_norm', 'iterations']),
    # Two components with the same operators: the second reuses those of the
    # first, in the mixed problem and in the Dirichlet problem (whose load
    # vector it assembles alone); the values of the second.
    'components_mixed_dense': ('sphere_components_mixed.prm', {},
                               ['mixed_flux', 'mixed_potential',
                                'iterations']),
    'components_dirichlet_dg': ('sphere_components_dirichlet.prm', {},
                                ['rhs_norm', 'flux', 'iterations']),
    # The complex problem in the Galerkin formulation, with different real
    # and imaginary parts: one mixed system coupled by the complex a/b of the
    # Robin part, dense with the diagonal preconditioner and multipole with
    # the Calderon one, part by part, and relaxed products; with Dirichlet
    # conditions on the whole surface, two real problems with one single
    # layer.
    'complex_galerkin_dense': ('sphere_complex_galerkin.prm',
                               GALERKIN_COMPLEX,
                               ['mixed_flux_real', 'mixed_potential_real',
                                'mixed_flux', 'mixed_potential',
                                'mixed_robin_flux', 'iterations']),
    'complex_galerkin_fma': ('sphere_complex_galerkin.prm',
                             dict(GALERKIN_COMPLEX,
                                  **{'Solution method': 'FMA',
                                     'Mixed Galerkin preconditioner':
                                       'calderon',
                                     'Mixed Galerkin relaxed products below':
                                       '1e-3'}),
                             ['mixed_flux_real', 'mixed_potential_real',
                              'mixed_flux', 'mixed_potential',
                              'mixed_robin_flux', 'iterations']),
    # The errors against the exact solution: with two slots in the
    # collocation problem (Dirichlet and Neumann data of the slot of a cell
    # of the kind of each node), the numbers of the single function x+y+z;
    # and for the two components of a complex problem.
    'slots_collocation': ('sphere_slots_mixed.prm',
                          {'Boundary integral formulation': 'collocation',
                           'Preconditioner': 'ILU',
                           'Linear solver': 'GMRES'},
                          ['phi_nodes', 'flux', 'gradient']),
    'complex_collocation_errors': ('sphere_complex_galerkin.prm',
                                   {'Errors': 'exact solution'},
                                   ['phi_nodes', 'flux', 'gradient',
                                    'steps']),
    'complex_galerkin_dirichlet': ('sphere_complex_galerkin.prm',
                                   dict(GALERKIN_COMPLEX,
                                        **{'Dirichlet boundary ids': '1,2',
                                           'Robin boundary ids': ''}),
                                   ['rhs_norm', 'iterations']),
}


# The parameters of other sections than BEMProblem<dim>.
SECTIONS = {'Errors': 'BoundaryConditions'}


def set_value(text, key, value, dim=3):
    """Set every occurrence of a parameter; add it to BEMProblem<dim>, or to
    its section in SECTIONS, if the file does not have it."""
    pattern = re.compile(r'(\n\s*set ' + re.escape(key) + r'\s*=)[^\n]*')
    text, n = pattern.subn(lambda m: m.group(1) + ' ' + value, text)
    if n == 0:
        section = SECTIONS.get(key, 'BEMProblem')
        text += (f'\nsubsection {section}<{dim}>\n  set {key} = {value}\n'
                 f'end\n')
    return text


def run_case(name):
    base, settings, quantities = CASES[name][:3]
    dim = CASES[name][3] if len(CASES[name]) > 3 else 3
    text = open(os.path.join(HERE, 'base', base)).read()
    for key, value in settings.items():
        text = set_value(text, key, value, dim)
    with open(os.path.join(HERE, f'parameters_bem_{dim}.prm'), 'w') as fh:
        fh.write(text)
    executable = os.path.join(ROOT, 'build', f'bem_fma_{dim}d')
    log = os.path.join(HERE, f'{name}.log')
    start = time.time()
    with open(log, 'w') as fh:
        status = subprocess.run(['/usr/bin/time', '-f', 'WALL %e s RSS %M kB',
                                 'timeout', '-k', '30', '900', 'mpirun',
                                 '-np', '8', executable, '1'],
                                cwd=HERE, stdout=fh, stderr=subprocess.STDOUT)
    seconds = time.time() - start
    t = open(log, errors='ignore').read()
    values = {}
    for q in quantities:
        pattern, _ = QUANTITIES[q]
        found = re.findall(pattern, t)
        values[q] = found[-1] if found else None
    return status.returncode, seconds, values


def check_executables():
    ok = True
    for folder in ('RUN', 'RUN_EXP'):
        copy = os.path.join(ROOT, folder, 'bem_fma_3d')
        if os.path.exists(copy) and not filecmp.cmp(copy, EXECUTABLE,
                                                     shallow=False):
            print(f'WARNING {folder}/bem_fma_3d differs from build/bem_fma_3d:'
                  f' an executable older than the library corrupts its stack')
            ok = False
    return ok


def main():
    update = '--update' in sys.argv
    names = [a for a in sys.argv[1:] if not a.startswith('--')] or list(CASES)
    reference = json.load(open(REFERENCE)) if os.path.exists(REFERENCE) else {}
    failures = 0 if check_executables() else 1
    for name in names:
        code, seconds, values = run_case(name)
        line = f'{name:28s} {seconds:6.1f} s'
        if code != 0 or any(v is None for v in values.values()):
            missing = [q for q, v in values.items() if v is None]
            print(f'{line}  FAIL exit {code}, missing {missing}')
            failures += 1
            continue
        if update:
            reference[name] = values
            print(f'{line}  stored')
            continue
        if name not in reference:
            print(f'{line}  no reference')
            failures += 1
            continue
        bad = []
        for q, v in values.items():
            r = reference[name].get(q)
            if r is None:
                continue
            _, precision = QUANTITIES[q]
            if precision == 'exact':
                if int(v) != int(r):
                    bad.append(f'{q} {r} -> {v}')
                continue
            if precision == 'ceiling':
                if float(v) > CEILING * float(r):
                    bad.append(f'{q} {r} -> {v} (more than {CEILING} times)')
                continue
            d = abs(float(v) - float(r)) / max(abs(float(r)), 1e-300)
            if d > FORMAT[precision]:
                bad.append(f'{q} {r} -> {v} ({d:.1e})')
        if bad:
            print(f'{line}  FAIL ' + '; '.join(bad))
            failures += 1
        else:
            print(f'{line}  ok')
    if update:
        with open(REFERENCE, 'w') as fh:
            json.dump(reference, fh, indent=1, sort_keys=True)
    print('PASSED' if failures == 0 else f'{failures} FAILED')
    sys.exit(0 if failures == 0 else 1)


if __name__ == '__main__':
    main()
