#!/usr/bin/env python3
"""Regression test of the Galerkin and collocation solvers on small meshes.

    python3 run_regression.py            run every case and compare
    python3 run_regression.py --update   run every case and store the values
                                         as the new reference
    python3 run_regression.py CASE ...   run only the named cases

Each case writes parameters_bem_3.prm from a frozen base file in base/, runs
the solver built in ../build on 8 processes, and reads a few quantities from
the log. They are compared with reference.json: the load vector and the
Frobenius norm of V, printed with sixteen digits, to 1e-12 relative; the
errors, printed with six, to 1e-5 relative. The checks also cover what went
wrong before: an executable in a run folder older than the library, and runs
that fail or hang instead of finishing.

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

FORMAT = {'sixteen': 1e-12, 'six': 1e-5}

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
}

# name: (base file, settings in BEMProblem<3>, quantities to compare)
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
}


def set_value(text, key, value):
    """Set every occurrence of a parameter; add it to BEMProblem<3> if the
    file does not have it."""
    pattern = re.compile(r'(\n\s*set ' + re.escape(key) + r'\s*=)[^\n]*')
    text, n = pattern.subn(lambda m: m.group(1) + ' ' + value, text)
    if n == 0:
        text += (f'\nsubsection BEMProblem<3>\n  set {key} = {value}\n'
                 f'end\n')
    return text


def run_case(name):
    base, settings, _ = CASES[name]
    text = open(os.path.join(HERE, 'base', base)).read()
    for key, value in settings.items():
        text = set_value(text, key, value)
    with open(os.path.join(HERE, 'parameters_bem_3.prm'), 'w') as fh:
        fh.write(text)
    log = os.path.join(HERE, f'{name}.log')
    start = time.time()
    with open(log, 'w') as fh:
        status = subprocess.run(['timeout', '-k', '30', '900', 'mpirun',
                                 '-np', '8', EXECUTABLE, '1'],
                                cwd=HERE, stdout=fh, stderr=subprocess.STDOUT)
    seconds = time.time() - start
    t = open(log, errors='ignore').read()
    values = {}
    for q in CASES[name][2]:
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
