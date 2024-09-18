from pyscf import gto, scf, lo
from pyscf.tools import cubegen

mol = gto.Mole()
mol.build(
    atom='''H 0 0 0; H 0 0 0.74''',
    basis='sto-3g'
)

m = scf.RHF(mol).run()

# Save the density matrix as the initial guess for the next calculation
dm_init_guess = m.make_rdm1()

m = scf.RHF(mol)
res = m.kernel(dm_init_guess)

C = lo.orth_ao(m, 'lowdin')
orbs = C[:,m.mo_occ>0] # Only get occupied orbitals

for i in range(C.shape[1]):
    cubegen.orbital(mol, f'H2_mo{i+1}.cube', C[:,i])

print(res)

mol = gto.Mole()
mol.build(
    atom='''He 0 0 0''',
    basis='sto-3g'
)

m = scf.RHF(mol).run()

# Save the density matrix as the initial guess for the next calculation
dm_init_guess = m.make_rdm1()

m = scf.RHF(mol)
res = m.kernel(dm_init_guess)

print(res)

C = lo.orth_ao(m, 'nao')
orbs = C[:,m.mo_occ>0] # Only get occupied orbitals

for i in range(C.shape[1]):
    cubegen.orbital(mol, f'He_mo{i+1}.cube', C[:,i])
