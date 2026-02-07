# This file provides an exact diagonalization interface for the RISB Python library:
#     https://thenoursehorse.github.io/risb
#
# The RISB library was developed by:
#     H. L. Nourse and B. J. Powell (2016–2023),
#     R. H. McKenzie (2016–2022).
#
# This interface was written by Chenrui Wang, 2025.
# Copyright (C) 2025 Chenrui Wang
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You may obtain a copy of the License at:
#     https://www.gnu.org/licenses/gpl-3.0.txt


import numpy as np
import time

from scipy.sparse import coo_matrix, csr_matrix
from scipy.sparse.linalg import eigsh
from numpy.typing import ArrayLike
from typing import TypeAlias, TypeVar
from itertools import product
from functools import wraps
from joblib import Parallel, delayed

GfStructType: TypeAlias = list[tuple[str, int]]
MFType: TypeAlias = dict[ArrayLike]


def timing_counter(func):
    """
    A decorator that counts the number of calls to a function
    and times each individual call.
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        # Increment the call counter before executing the function.
        # The counter is an attribute of the decorator itself.
        wrapper.calls += 1
        
        # Record the start time
        start_time = time.time()
        
        # Call the original function (e.g., set_h_int)
        result = func(*args, **kwargs)
        
        # Record the end time and calculate duration
        end_time = time.time()
        duration = end_time - start_time
        
        # Print the results for this specific call
        print(f"--- [PROFILING] '{func.__name__}' call #{wrapper.calls}: took {duration:.4f} seconds ---")
        
        # Return the original function's result
        return result

    # Initialize the call counter on the wrapper function
    wrapper.calls = 0
    return wrapper

class SolveEmbeddingSparse:
    """
    Impurity solver of embedding space using user-defined ED solver.
    Parameters
    ----------
    V : numpy array
        Tensor for Interaction Hamiltonian in the embedding space.
    gf_struct : list of pairs [ (str,int), ...]
        Structure of the matrices. It must be a
        list of pairs, each containing the name of the
        matrix block as a string and the size of that block.
        For example: ``[ ('up', 3), ('down', 3) ]``.

    """
    def __init__(self,V: ArrayLike, gf_struct, Sop: bool= False):
        '''
        Sop: bool, whether generate the spin operators when initializing the embedding class
        '''
        #: dict[tuple[str,int]] : Block matrix structure of c-electrons.
        self.gf_struct = gf_struct
        self.L = self.count_orbitals() # including spin 2*number of total orbitals
        self.L_total = 2 * self.L # including impurities
        self.N_particles = int(self.L_total/2) # solve impurity for half-filling
        # preperation for the Exact Diagonalization
        self.basis = self.create_basis(self.L_total, self.N_particles)
        self.index = {state: i for i, state in enumerate(self.basis)}
        self.dim = len(self.basis)
        self.map_dict = self.construct_start_index(gf_struct)
        # Do gf_struct as a map
        self.gf_struct_dict = self._dict_gf_struct(self.gf_struct)
    
        self.V_int = V # interaction tensor
        # V_int can be intialized with hubbard U=1 (apply for Hubbard only), 
        # and further update using method: update_h_int_val by renewing the interaction strength
        self.U_val = 1

        # Ground state of the embedding problem
        self.gs_vector = None
        self.gs_energy = None
        # calculate the free energy
        # C.48 equation in the Slave-boson thesis
        self.free_energy = None

        #: scipy.sparse.csr_matrix: Embedding Hamiltonian. It is the sum of 
        #: :attr:`h0_loc`,:attr:`h_int`,:attr:`h_hybr`,:attr:`h_bath`
        self.h_emb: csr_matrix = csr_matrix((self.dim, self.dim), dtype=np.complex128)

        #: scipy.sparse.csr_matrix:Single-particle quadratic couplings of
        #: c-electron terms in ::attr:`h_emb`.
        self.h0_loc: csr_matrix = csr_matrix((self.dim, self.dim), dtype=np.complex128)

        #: scipy.sparse.csr_matrix: Interaction couplings of
        #: c-electron terms in ::attr:`h_emb`.
        self.h_int: csr_matrix = csr_matrix((self.dim, self.dim), dtype=np.complex128)
        # set the interaction before iteration to save time cost
        self.set_h_int(self.V_int)

        #: scipy.sparse.csr_matrix: Bath terms in :attr:`h_emb`.
        self.h_bath: csr_matrix = csr_matrix((self.dim, self.dim), dtype=np.complex128)

        #: scipy.sparse.csr_matrix: Hybridization terms in :attr:`h_emb`.
        self.h_hybr: csr_matrix = csr_matrix((self.dim, self.dim), dtype=np.complex128)
        self.h_hybr_half: csr_matrix = csr_matrix((self.dim, self.dim), dtype=np.complex128)


        #: dict[numpy.ndarray] : f-electron density matrix.
        self.rho_f = {}

        #: dict[numpy.ndarray] : c-electron density matrix.
        self.rho_c = {}

        #: dict[numpy.ndarray] : Density matrix of hybridization terms
        #: (c- and f-electrons).
        self.rho_cf = {}

        #: dict[coo_matrix]: effective spin operators
        # keys: orbital name str or list of name (for orbital group)
        self.S2_cache = {}
        #: total Sz operator
        self.Sz_total = None

    @staticmethod
    def _dict_gf_struct(gf_struct: GfStructType) -> dict[str, int]: # GfStructType -> dict
        return dict(gf_struct)
    
    @staticmethod
    def check_hermiticity(H, atol=1e-6) -> bool:
        """Check if H is Hermitian within tolerance atol."""
        H_dag = H.getH()  # Hermitian conjugate
        diff = (H - H_dag).tocoo()
        max_diff = np.max(np.abs(diff.data)) if diff.nnz > 0 else 0.0
        return max_diff < atol

    # count the number of orbitals for a given gf_struct
    def count_orbitals(self):
    # count the number of particles in a given state
        N = 0
        for bl_name, n_orb in self.gf_struct:
            N += n_orb
        return N


    def count_particles(self, state):
        return bin(state).count('1')
    
    # create the basis of states with N particles in L
    def create_basis(self, L, N):
        return [i for i in range(1 << L) if self.count_particles(i) == N]
    
    # given gf_struct, construct the start index mapping dict to simplify the ED hamiltonian construction
    def construct_start_index(self, gf_struct):
    # return a dict with keys as block names and values as start index j_start
    #  (2(j+j_start) for up and 2(j+j_start)+1 for dn in future calculation)
        map_dict = {}
        idx_up, idx_dn = 0, 0
        for name, n_orb in gf_struct:
            if name.startswith("up"):
                map_dict[name] = idx_up
                idx_up += n_orb
            elif name.startswith("dn"):
                map_dict[name] = idx_dn
                idx_dn += n_orb
            else:
                raise ValueError("spin channel not recognized")
        return map_dict

    def fermion_sign(self, state, i, j): #fermion sign for hopping c_i^dag c_j, return parity of fermions between [i,j-1](i<j) or [j+1,i] (i>j)
        if i == j:
            return 1
        elif i < j:
            return (-1) ** self.count_particles(state & (((1 << j) - 1) & ~((1 << i) - 1)))
        else:
            return (-1) ** self.count_particles(state & (((1 << (i+1)) - 1) & ~((1 << (j+1)) - 1)))
    
    def aux_fermion_sign(self, state, i, j): #fermion sign for hopping f_i f_j^dag, return parity of fermions between 1+[i+1,j](i<j) or [j+1,i] (i>j)
        if i == j:
            return 1
        elif i < j:
            return (-1) ** (self.count_particles(state & (((1 << (j+1)) - 1) & ~((1 << (i+1)) - 1)))+1)
        else:
            return (-1) ** self.count_particles(state & (((1 << (i+1)) - 1) & ~((1 << (j+1)) - 1)))

    def apply_c_dag_c(self, state, i, j):
        if i!= j: # not onsite term
            if not (state & (1 << j)) or (state & (1 << i)):
                return None, 0
            new_state = state ^ (1 << i) ^ (1 << j)
            sign = self.fermion_sign(state, i, j)
        elif i==j:
            if not (state & (1 << i)): # no particle
                return None, 0
            new_state = state 
            sign = 1
        return new_state, sign
    
    def apply_f_f_dag(self,state,i,j): # f_i f_j^dag
        if i!=j:
            if (state &(1<<j)) or not (state & (1<<i)):
                return None, 0
            new_state = state ^ (1<<i) ^ (1<<j)
            sign = self.aux_fermion_sign(state,i,j)
        elif i==j:
            if (state & (1 << i)): # with particle
                return None,0
            new_state = state
            sign = 1
        return new_state, sign
    
    def apply_c_dag_c_dag_c_c(self, state, i, j, k, l):

        if i==j or k==l: # not occur in our case
            return None, 0
        if not (state & (1 << l)) or not (state & (1 << k)):
            return None, 0
        
        state1 = state ^ (1 << l) ^ (1 << k)
        if k<l:# k<l: [k,l-1]-1
            sign1 = self.fermion_sign(state, k, l)*(-1) # -1 for overcounting of k
        elif k>l: #k>l:[l+1,k]
            sign1 = self.fermion_sign(state, k, l)
        
        if (state1 & (1 << j)) or (state1 & (1 << i)):
            return None, 0
        state2 = state1 ^ (1 << i) ^ (1 << j)
        if i<j: #i<j:1+[i,j-1]
            sign2 = self.fermion_sign(state1, i, j)*(-1)
        elif i>j: #i>j:[j+1,i]
            sign2 = self.fermion_sign(state1, i, j)
        
        return state2, sign1 * sign2
    
    def apply_c_dag_c_c_dag_c(self, state, i, j, k, l):
        # c^\dagger_i c_j c^\dagger_k c_l operator on state
        state1, sign1 = self.apply_c_dag_c(state,k,l)
        if state1 is None:
            return None, 0
        state2, sign2 = self.apply_c_dag_c(state1,i,j)    
        return state2, sign1*sign2
    
    def create_c_dag_c_matrix(self, i: int, j: int) -> csr_matrix:

        r'''
        Create sparse matrix representation of c c^\dagger operator 
        in the current Fock basis.
        '''
        row, col, data = [], [], []
        
        for istate, state in enumerate(self.basis):
            new_state, sign = self.apply_c_dag_c(state, i, j)
            if new_state in self.index:
                jstate = self.index[new_state]
                row.append(jstate)
                col.append(istate)
                data.append(sign)
        op = coo_matrix((data, (row, col)), shape=(self.dim, self.dim), dtype=np.complex128)
        return op.tocsr()
    
    def create_c_c_dag_matrix(self, i: int, j: int) -> csr_matrix:

        r'''
        Create sparse matrix representation of c c^\dagger operator 
        in the current Fock basis.
        '''
        row, col, data = [], [], []
        
        for istate, state in enumerate(self.basis):
            new_state, sign = self.apply_f_f_dag(state, i, j)
            if new_state in self.index:
                jstate = self.index[new_state]
                row.append(jstate)
                col.append(istate)
                data.append(sign)
        
        op = coo_matrix((data, (row, col)), shape=(self.dim, self.dim), dtype=np.complex128)
        return op.tocsr()

    def build_sparse_matrix(self, row, col, data):
        return coo_matrix((data, (row, col)), shape=(self.dim, self.dim)).tocsr()
    
    @timing_counter
    def set_h_bath(self, Lambda_c: MFType, test: bool = False, n_jobs = -2) -> None:
        '''
        Set the bath hamiltonian, the default number of threads is maximum-2
        '''
        L = self.L
        # --- 1. Build task list ---
        tasks = []
        for bl, mat in Lambda_c.items():
            idx0 = self.map_dict[bl]
            nz = np.array(np.nonzero(np.abs(mat) > 1e-5)).T

            spin_offset = 0 if bl.startswith("up") else 1
            base = 2 * idx0

            for i, j in nz:
                val = mat[i, j]
                # Correct index mapping to bath fermions
                creator = L + base + 2*i + spin_offset
                annihilator = L + base + 2*j + spin_offset
                tasks.append((creator, annihilator, val))

        # --- 2. Worker function ---
        def process_task(task):
            creator, annihilator, val = task
            row_frag, col_frag, data_frag = [], [], []

            for istate, state in enumerate(self.basis):
                new_state, sign = self.apply_f_f_dag(state, creator, annihilator)
                if new_state in self.index:
                    jstate = self.index[new_state]
                    row_frag.append(jstate)
                    col_frag.append(istate)
                    data_frag.append(sign * val)

            return row_frag, col_frag, data_frag
        # --- 3. Dispatch parallel job ---
        if n_jobs == 1:
            all_results = [process_task(t) for t in tasks]
        else:
            all_results = Parallel(n_jobs=n_jobs, backend="loky", max_nbytes="1M")(
                delayed(process_task)(t) for t in tasks
            )

        # --- 4. Collect results ---
        row = [it for res in all_results for it in res[0]]
        col = [it for res in all_results for it in res[1]]
        data = [it for res in all_results for it in res[2]]

        self.h_bath = self.build_sparse_matrix(row, col, data)
        '''
        # Old parallelization over all basis(too much cost when combine all the results)
        # precompute the non-zero indices
        precomputed_nonzero_idx = {}
        for bl, mat in Lambda_c.items():
            precomputed_nonzero_idx[bl] = np.array(np.nonzero(np.abs(mat) > 1e-5)).T

        def process_state(state_tuple):
            # an auxillary function to process with single state
            istate, state = state_tuple
            row_frag, col_frag, data_frag = [], [], []

            for bl, mat in Lambda_c.items():
                #assert np.allclose(mat, mat.conj().T), f"{bl} not Hermitian"
                idx0 = self.map_dict[bl]
                nonzero_idx = precomputed_nonzero_idx[bl] # first filter out the non-zero indices
                for i, j in nonzero_idx:
                    if bl.startswith("up"):
                        new_state, sign = self.apply_f_f_dag(state, L + 2 * (i + idx0), L + 2 * (j + idx0))
                    else:
                        new_state, sign = self.apply_f_f_dag(state, L + 2 * (i + idx0) + 1, L + 2 * (j + idx0) + 1)
                    if new_state in self.index:
                        jstate = self.index[new_state]
                        row_frag.append(jstate)
                        col_frag.append(istate)
                        data_frag.append(sign * mat[i, j])
            return row_frag, col_frag, data_frag
        # Parrallel running, keep an option with no parrallelization
        if n_jobs == 1:
            all_results = [process_state((istate, state)) 
                           for istate, state in enumerate(self.basis)]
        else:
            all_results = Parallel(n_jobs=n_jobs)(
                delayed(process_state)((istate, state)) 
                for istate, state in enumerate(self.basis)
            )
        
        row = [item for res in all_results for item in res[0]]
        col = [item for res in all_results for item in res[1]]
        data = [item for res in all_results for item in res[2]]
        self.h_bath = self.build_sparse_matrix(row, col, data)
        '''
        if test:
            print("Test: solve the h_bath part only")
            import matplotlib.pyplot as plt
            import scipy.sparse
            
            # plot the sparse matrix
            def plot_sparse_matrix(matrix, title="Sparsity Pattern"):
                plt.figure(figsize=(6, 6))
                plt.spy(matrix, markersize=1)
                plt.title(title)
                plt.xlabel("Column index")
                plt.ylabel("Row index")
                plt.grid(False)
                plt.show()

            plot_sparse_matrix(self.h_bath, title="h_bath Sparsity Pattern")

            print("h_bath:",self.h_bath)
            eigenvalue, eigenvector = eigsh(self.h_bath, k=1, which='SA')  # SA: smallest algebraic
            print("GS energy of h_bath:",eigenvalue[0])

    @timing_counter
    def set_h_hybr(self, D: MFType, test: bool= False, n_jobs = -2) -> None:
        '''
        Set the hybrid hamiltonian, the default number of threads is maximum-1
        '''
        L = self.L
        # --- 1. Build task list ---
        tasks = []
        for bl, mat in D.items():
            idx0 = self.map_dict[bl]
            nonzero_idx = np.array(np.nonzero(np.abs(mat) > 1e-5)).T
            if nonzero_idx.size == 0:
                continue
            spin_offset = 0 if bl.startswith("up") else 1
            base_c = 2 * idx0
            base_f = L + 2 * idx0

            for i, j in nonzero_idx:
                val = mat[i, j]
                # c^dag f
                c_dag_idx = base_c + 2*i + spin_offset
                f_idx = base_f + 2*j + spin_offset
                
                # f^dag c
                f_dag_idx = base_f + 2*j + spin_offset
                c_idx = base_c + 2*i + spin_offset

                # (op_type, idx1, idx2, value)
                tasks.append( ("c_dag_f", c_dag_idx, f_idx, val) )
                tasks.append( ("f_dag_c", f_dag_idx, c_idx, val.conj()) )
        # --- 2. Worker function ---
        def process_task(task):
            op_type, idx1, idx2, val = task
            # results for every task
            row_frag, col_frag, data_frag = [], [], []
            row_h_frag, col_h_frag, data_h_frag = [], [], []

            for istate, state in enumerate(self.basis):
                new_state, sign = self.apply_c_dag_c(state, idx1, idx2)
                if new_state in self.index:
                    jstate = self.index[new_state]
                    row_frag.append(jstate)
                    col_frag.append(istate)
                    data_frag.append(sign * val)
                    # h_hybr_half only contains c^dag f
                    if op_type == "c_dag_f":
                        row_h_frag.append(jstate)
                        col_h_frag.append(istate)
                        data_h_frag.append(sign * val)

            return (row_frag, col_frag, data_frag, 
                    row_h_frag, col_h_frag, data_h_frag)
        
        # --- 3. Dispatch parallel job ---
        if n_jobs == 1:
            all_results = [process_task(t) for t in tasks]
        else:
            all_results = Parallel(n_jobs=n_jobs, backend="loky", max_nbytes="1M")(
                delayed(process_task)(t) for t in tasks
            )
        # --- 4. Collect results ---
        row, col, data = [], [], []
        row_h, col_h, data_h = [], [], []
        for res in all_results:
            row.extend(res[0])
            col.extend(res[1])
            data.extend(res[2])
            row_h.extend(res[3])
            col_h.extend(res[4])
            data_h.extend(res[5])

        self.h_hybr = self.build_sparse_matrix(row, col, data)
        self.h_hybr_half = self.build_sparse_matrix(row_h, col_h, data_h)


        '''
        # Old parallelization over all basis(too much cost when combine all the results)
        # precompute the non-zero indices
        precomputed_nonzero_idx = {}
        for bl, mat in D.items():
            precomputed_nonzero_idx[bl] = np.array(np.nonzero(np.abs(mat) > 1e-5)).T
        def process_state(state_tuple):
            # an auxillary function to process with single state
            istate, state = state_tuple
            row_frag, col_frag, data_frag = [], [], []
            row_h_frag, col_h_frag, data_h_frag = [], [], [] # save the matrix for free energy calculation, only c^dag f terms

            for bl, mat in D.items():
                idx0 = self.map_dict[bl]
                nonzero_idx = precomputed_nonzero_idx[bl] # first filter out the non-zero indices
                for i, j in nonzero_idx:
                    if bl.startswith("up"):
                        new_state, sign = self.apply_c_dag_c(state, 2 * (i + idx0), L + 2 * (j + idx0)) # c^dag f
                        new_state_conj, sign_conj = self.apply_c_dag_c(state, L + 2 * (j + idx0), 2 * (i + idx0)) # f^dag c
                    else:
                        new_state, sign = self.apply_c_dag_c(state, 2 * (i + idx0) + 1, L + 2 * (j + idx0) + 1)
                        new_state_conj, sign_conj = self.apply_c_dag_c(state, L + 2 * (j + idx0) + 1, 2 * (i + idx0) + 1)
                    if new_state in self.index:
                        jstate = self.index[new_state]
                        row_frag.append(jstate)
                        col_frag.append(istate)
                        #data.append(mat[i, j])
                        data_frag.append(sign * mat[i, j])
                        row_h_frag.append(jstate)
                        col_h_frag.append(istate)
                        data_h_frag.append(sign * mat[i, j])
                    if new_state_conj in self.index:
                        jstate_conj = self.index[new_state_conj]
                        row_frag.append(jstate_conj)
                        col_frag.append(istate)
                        #data.append(mat[i, j].conj())
                        data_frag.append(sign_conj * mat[i, j].conj())
            return (row_frag, col_frag, data_frag, 
                    row_h_frag, col_h_frag, data_h_frag)
        
        # Parrallel running, keep an option with no parrallelization
        if n_jobs == 1:
            all_results = [process_state((istate, state)) 
                           for istate, state in enumerate(self.basis)]
        else:
            all_results = Parallel(n_jobs=n_jobs)(
                delayed(process_state)((istate, state)) 
                for istate, state in enumerate(self.basis)
            )
        
        row, col, data = [], [], []
        row_h, col_h, data_h = [], [], []
        for res in all_results:
            row.extend(res[0])
            col.extend(res[1])
            data.extend(res[2])
            row_h.extend(res[3])
            col_h.extend(res[4])
            data_h.extend(res[5])

        self.h_hybr = self.build_sparse_matrix(row, col, data)
        self.h_hybr_half = self.build_sparse_matrix(row_h, col_h, data_h)
        '''
        if test:
            print("Test: solve the h_hybr part only")
            print("h_hybr:",self.h_hybr)
            eigenvalue, eigenvector = eigsh(self.h_hybr, k=1, which='SA')  # SA: smallest algebraic
            print("GS energy of h_hybr:",eigenvalue[0])
    '''
    def set_h_hybr(self, D:MFType,test = True) -> None:
        row, col, data = [], [], []
        L = self.L
        for bl, mat in D.items():
            idx0 = self.map_dict[bl]
            for istate, state in enumerate(self.basis):
                for i, j in product(range(mat.shape[0]), repeat=2):
                    if abs(mat[i, j]) < 1e-5:
                        continue
                    if bl.startswith("up"):
                        new_state, sign = self.apply_c_dag_c(state, 2 * (j + idx0), L + 2 * (i + idx0)) # c^dag f
                        #new_state_conj, sign_conj = self.apply_c_dag_c(state, L + 2 * (i + idx0), 2 * (j + idx0)) # f^dag c
                    else:
                        new_state, sign = self.apply_c_dag_c(state, 2 * (j + idx0) + 1, L + 2 * (i + idx0) + 1)
                        #new_state_conj, sign_conj = self.apply_c_dag_c(state, L + 2 * (i + idx0) + 1, 2 * (j + idx0) + 1)
                    if new_state in self.index:
                        jstate = self.index[new_state]
                        row.append(jstate)
                        col.append(istate)
                        #data.append(mat[i, j])
                        # Add Hermite conjugate term
                        data.append(sign * mat[i, j])
                        row.append(istate)
                        col.append(jstate)
                        data.append(sign * mat[i, j].conj())

        self.h_hybr = self.build_sparse_matrix(row, col, data)
        if test:
            print("Test: solve the h_hybr part only")
            print("h_hybr:",self.h_hybr)
            eigenvalue, eigenvector = eigsh(self.h_hybr, k=1, which='SA')  # SA: smallest algebraic
            print("GS energy of h_hybr:",eigenvalue[0])
        '''
    @timing_counter
    def set_h0_loc(self, h0_loc_matrix:MFType, test: bool = False, n_jobs: int = -2) -> None:
        '''
        # Old parallelization over all basis(too much cost when combine all the results)
        # precompute the non-zero indices
        precomputed_nonzero_idx = {}
        for bl, mat in h0_loc_matrix.items():
            precomputed_nonzero_idx[bl] = np.array(np.nonzero(np.abs(mat) > 1e-5)).T

        # auxillary function to process with a single state
        def process_state(state_tuple):
            istate, state = state_tuple
            row_frag, col_frag, data_frag = [], [], []
            for bl, mat in h0_loc_matrix.items():
                idx0 = self.map_dict[bl]
                nonzero_idx = precomputed_nonzero_idx[bl]
                for i, j in nonzero_idx:
                    if bl.startswith("up"):
                        new_state, sign = self.apply_c_dag_c(state, 2 * (i + idx0), 2 * (j + idx0))
                    else:
                        new_state, sign = self.apply_c_dag_c(state, 2 * (i + idx0) + 1, 2 * (j + idx0) + 1)
                    if new_state in self.index:
                        jstate = self.index[new_state]
                        row_frag.append(jstate)
                        col_frag.append(istate)
                        data_frag.append(sign * mat[i, j])
            return row_frag, col_frag, data_frag
        # Parrallel running, keep an option with no parrallelization
        if n_jobs == 1:
            all_results = [process_state((istate, state)) 
                           for istate, state in enumerate(self.basis)]
        else:
            all_results = Parallel(n_jobs=n_jobs)(
                delayed(process_state)((istate, state)) 
                for istate, state in enumerate(self.basis)
            )
        '''
        tasks = [] # all tasks to be calculated
        for bl, mat in h0_loc_matrix.items():
            idx0 = self.map_dict[bl]
            if bl.startswith("up"):
                base = 2 * idx0
                spin_offset = 0
            else:
                base = 2 * idx0
                spin_offset = 1
            nz = np.array(np.nonzero(np.abs(mat) > 1e-5)).T
            for i, j in nz:
                tasks.append((bl, i, j, base, spin_offset, mat[i, j]))

        def process_ij(info): # process one task for all basis
            bl, i, j, base, spin_offset, val = info
            row_frag, col_frag, data_frag = [], [], []

            creator = 2 * i + base + spin_offset
            annihilator = 2 * j + base + spin_offset

            for istate, state in enumerate(self.basis):
                new_state, sign = self.apply_c_dag_c(state, creator, annihilator)
                if new_state in self.index:
                    jstate = self.index[new_state]
                    row_frag.append(jstate)
                    col_frag.append(istate)
                    data_frag.append(sign * val)

            return row_frag, col_frag, data_frag
        
        if n_jobs == 1:
            all_results = [process_ij(task) for task in tasks]
        else:
            all_results = Parallel(n_jobs=n_jobs, backend="loky", max_nbytes="1M")(
                delayed(process_ij)(task) for task in tasks
        )
            
        row = [item for res in all_results for item in res[0]]
        col = [item for res in all_results for item in res[1]]
        data = [item for res in all_results for item in res[2]]

        self.h0_loc = self.build_sparse_matrix(row, col, data)

        if test:
            print("Test: solve the h0_loc part only")
            print("h_loc:",self.h0_loc)
            eigenvalue, eigenvector = eigsh(self.h0_loc, k=1, which='SA')  # SA: smallest algebraic
            print("GS energy of h_loc:",eigenvalue[0])

    @timing_counter
    def set_h_int(self, V, test: bool = False, n_jobs: int = 8) -> None:
        '''
        Set the interaction hamiltonian, the default number of threads is maximum-1
        '''
        # 1) find nonzero interaction entries
        tol = 1e-6
        # 1) find nonzero interaction entries
        nonzero_idx = np.array(np.nonzero(np.abs(V) > tol)).T  # shape (nnz,4)
        if nonzero_idx.size == 0:
            # empty interaction
            self.h_int = self.build_sparse_matrix([], [], [])
            print("Empty interaction!")
            return
        '''
        # 2) an auxillary function to process with single state
        def process_state(state_tuple):
            # state_tuple: idx, state (in binary form)
            istate, state = state_tuple
            row_frag, col_frag, data_frag = [], [], []
            for i, j, k, l in nonzero_idx:
                new_state, sign = self.apply_c_dag_c_dag_c_c(state, 2 * i, 2 * j + 1, 2 * k + 1, 2 * l)
                if new_state in self.index:
                    jstate = self.index[new_state]
                    row_frag.append(jstate)
                    col_frag.append(istate)
                    data_frag.append(sign * V[i, j, k, l])
            return row_frag, col_frag, data_frag
        
        # 3) Parrallel running, keep an option with no parrallelization
        if n_jobs == 1:
            all_results = [process_state((istate, state)) 
                           for istate, state in enumerate(self.basis)]
        else:
            all_results = Parallel(n_jobs=n_jobs)(
                delayed(process_state)((istate, state)) 
                for istate, state in enumerate(self.basis)
            )
        # 4) Collect the results
        row = [item for res in all_results for item in res[0]]
        col = [item for res in all_results for item in res[1]]
        data = [item for res in all_results for item in res[2]]
        '''
        # 3) Parallel over nonzero interactions instead of basis states, rather than all basis(too much cost when combine all the results)
        def process_interaction(idx_range):
            row_frag, col_frag, data_frag = [], [], []
            for (i, j, k, l) in nonzero_idx[idx_range[0]:idx_range[1]]:
                for istate, state in enumerate(self.basis):
                    new_state, sign = self.apply_c_dag_c_dag_c_c(
                        state, 2 * i, 2 * j + 1, 2 * k + 1, 2 * l
                    )
                    if new_state in self.index:
                        jstate = self.index[new_state]
                        row_frag.append(jstate)
                        col_frag.append(istate)
                        data_frag.append(sign * V[i, j, k, l])
            return row_frag, col_frag, data_frag


        # Split interactions across workers
        nnz = len(nonzero_idx)
        chunk = (nnz + n_jobs - 1) // n_jobs
        ranges = [(i, min(i + chunk, nnz)) for i in range(0, nnz, chunk)]

        if n_jobs == 1:
            all_results = [process_interaction(rng) for rng in ranges]
        else:
            all_results = Parallel(n_jobs=n_jobs, backend="loky")(
                delayed(process_interaction)(rng) for rng in ranges
            )

        # 4) Collect results
        row = [x for r in all_results for x in r[0]]
        col = [x for r in all_results for x in r[1]]
        data = [x for r in all_results for x in r[2]]

        # 5) build the final sparse matrix
        self.h_int = self.build_sparse_matrix(row, col, data)

        if test:
            print("Test: solve the interaction part only")
            '''
            # Check hermicity
            H = self.h_int.tocsr()
            for r, c, v in zip(H.nonzero()[0], H.nonzero()[1], H.data):
                if abs(v - np.conj(H[c, r])) > 1e-6:
                    print(f"H[{r},{c}] = {v}, H[{c},{r}] = {H[c, r]}")
            '''

            eigenvalue, eigenvector = eigsh(self.h_int, k=1, which='SA')  # SA: smallest algebraic
            print("GS energy of h_int:",eigenvalue[0])

    def update_h_int_val(self, U_val_new):
        """
        if the interaction contains hubbard term only, one can initializing the interaction term
        using U=1 when set the embedding solver, and update the interaction by renew the 
        interaction strength to save time cost, especially for large system when loop over different
        interaction strength.
        Note: this method should avoid zero value of interaction strength.
        """
        self.h_int *= U_val_new / self.U_val
        self.U_val = U_val_new

    @timing_counter
    def set_h_emb(
        self,
        Lambda_c:MFType,
        D: MFType,
        h0_loc_matrix: MFType) -> None:
        """
        Set up the embedding Hamiltonian.

        Parameters
        ----------
        Lambda_c : dict[numpy.ndarray]
            Bath coupling. Each key in dictionary must follow
            :attr:`gf_struct`.
        D : dict[numpy.ndarray]
            Hybridization coupling. Each key in dictionary must follow
            :attr:`gf_struct`.
        h0_loc_matrix : dict[numpy.ndarray], optional
            Single-particle quadratic couplings of the c-electrons. Each key
            in dictionary must follow :attr:`gf_struct`.

        """
        if h0_loc_matrix is not None:
            self.set_h0_loc(h0_loc_matrix)
        self.set_h_bath(Lambda_c)
        self.set_h_hybr(D)
        #self.set_h_int(self.V_int) # remove this line because h_int already initialized at the beginning
        
        self.h_emb = self.h0_loc + self.h_hybr + self.h_int + self.h_bath
        #self.h_emb = self.h0_loc + self.h_int #+ self.h_bath #+ self.h_hybr
        #self.h_emb = self.h0_loc
        
    @timing_counter
    def solve(self, test: bool = False) -> None:
        """
        Solve for the groundstate in the half-filled number sector of the embedding problem.
        if test==True, test the time cost of the solving process
        """
        # Check hermicities of the Hamiltonian
        for name, H in [
            ("h0_loc", self.h0_loc),
            ("h_bath", self.h_bath),
            ("h_hybr", self.h_hybr),
            ("h_int", self.h_int),
            ("h_emb", self.h_emb),
        ]:
            if not self.check_hermiticity(H):
                print(f"Warning: {name} is not Hermitian!")


        print("Start solving...") if test else None
        start_time = time.time() if test else None
        eigenvalue, eigenvector = eigsh(self.h_emb, k=1, which='SA',tol=1e-8, maxiter=10000, ncv=100)  # SA: smallest algebraic
        
        end_time = time.time() if test else None
        print("Solve finished, cost {:.2f}s".format(end_time - start_time)) if test else None

        self.gs_energy = eigenvalue[0]
        print("Ground state energy: {:.8f}".format(self.gs_energy)) if test else None
        self.gs_vector = eigenvector[:,0]
        self.free_energy = self.gs_vector.conj().T @ self.h_hybr_half @ self.gs_vector + self.gs_vector.conj().T @ self.h0_loc @ self.gs_vector

    
    def get_rho_f(self, bl: str) -> np.ndarray:
        """
        Return f-electron density matrix.
        delta_ab = <f_b f_a^dag> = <psi|f_b f_a^dag|psi>
        Parameters
        ----------
        bl : str
            Which block in :attr:`gf_struct` to return.
        
        Returns
        -------
        numpy.ndarray
            The f-electron density matrix :attr:`rho_f` from impurity.
        """
        L = self.L
        bl_size = self.gf_struct_dict[bl]
        self.rho_f[bl] = np.zeros([bl_size,bl_size],dtype=complex)
        # construct operator matrix for f-electrons
        idx0 = self.map_dict[bl]
        for a, b in product(range(bl_size), repeat=2):
            if bl.startswith("up"):
                i = L + 2 * (a + idx0)
                j = L + 2 * (b + idx0)
            elif bl.startswith("dn"):
                i = L + 2 * (a + idx0) + 1
                j = L + 2 * (b + idx0) + 1
            else:
                raise ValueError(f"Unknown block label {bl}")

            op_mat = self.create_c_c_dag_matrix(j, i) 
            val = self.gs_vector.conj().T @ (op_mat @ self.gs_vector) # f_b f_a^dag
            self.rho_f[bl][a, b] = val
        return self.rho_f[bl]

    def get_rho_c(self, bl: str) -> np.ndarray:
        """
        Return c-electron density matrix.
        <c_a^dag c_b> = <psi|c_a^dag c_b|psi>
        Parameters
        ----------
        bl : str
            Which block in :attr:`gf_struct` to return.
        
        Returns
        -------
        numpy.ndarray
            The c-electron density matrix :attr:`rho_c` from impurity.
        """
        L = self.L
        bl_size = self.gf_struct_dict[bl]
        self.rho_c[bl] = np.zeros([bl_size,bl_size],dtype=complex)
        # construct operator matrix for f-electrons
        idx0 = self.map_dict[bl]
        for a, b in product(range(bl_size), repeat=2):
            if bl.startswith("up"):
                i = 2 * (a + idx0)
                j = 2 * (b + idx0)
            elif bl.startswith("dn"):
                i = 2 * (a + idx0) + 1
                j = 2 * (b + idx0) + 1
            else:
                raise ValueError(f"Unknown block label {bl}")

            op_mat = self.create_c_dag_c_matrix(i, j) 
            val = self.gs_vector.conj().T @ (op_mat @ self.gs_vector) #
            self.rho_c[bl][a, b] = val
        return self.rho_c[bl]
    
    def get_rho_cf(self, bl: str) -> np.ndarray:
        """
        Return c-electron density matrix.
        <c_alpha^dag f_a> = <psi|c_alpha^dag f_a|psi>
        Parameters
        ----------
        bl : str
            Which block in :attr:`gf_struct` to return.
        
        Returns
        -------
        numpy.ndarray
            The c,f-electron density matrix :attr:`rho_cf` from impurity.
        """
        L = self.L
        bl_size = self.gf_struct_dict[bl]
        self.rho_cf[bl] = np.zeros([bl_size,bl_size],dtype=complex)
        # construct operator matrix for f-electrons
        idx0 = self.map_dict[bl]
        for alpha, a in product(range(bl_size), repeat=2):
            if bl.startswith("up"):
                i = 2 * (alpha  + idx0)
                j = L + 2 * (a + idx0)
            elif bl.startswith("dn"):
                i = 2 * (alpha + idx0) + 1
                j = L + 2 * (a + idx0) + 1
            else:
                raise ValueError(f"Unknown block label {bl}")

            op_mat = self.create_c_dag_c_matrix(i, j) 
            val = self.gs_vector.conj().T @ (op_mat @ self.gs_vector)
            self.rho_cf[bl][alpha, a] = val
        return self.rho_cf[bl]
    
    def get_Seff(self, orbital_key: str):
        """
        compute the effective spin for given orbitals, like "T"
        """
        # If already constructed, use cached version
        if orbital_key in self.S2_cache: # S2 operator already calculated
            S_sq = self.S2_cache[orbital_key]
            S_sq_val = self.gs_vector.conj().T @ S_sq @ self.gs_vector
            # Solve for S^{\hat}^2 = S(S+1)
            S = (np.sqrt(4*S_sq_val+1)-1)/2
            return S 
        # --- Otherwise, construct the S^2 operator ---
        def Pauli_mat(): # Pauli matrices for spin-1/2 system
            s0 = np.eye(2)
            sx = np.array([[0,1],[1,0]],dtype=complex)
            sy = np.array([[0,-1j],[1j,0]],dtype=complex)
            sz = np.array([[1,0],[0,-1]],dtype=complex)
            return s0,sx,sy,sz
        # Construction of the S^2 operator: (Sx^2 + Sy^2 + Sz^2, S = \sum_a S_a for multiorbital)
        row, col, data = [], [], []
        up_block_name = f"up_{orbital_key}" 
        bl_size = self.gf_struct_dict[up_block_name] # number of orbitals in this block
        idx0 = self.map_dict[up_block_name] # index of the first orbital in this block

        s0,sx,sy,sz = Pauli_mat() # get the Pauli matrices
        for istate, state in enumerate(self.basis): # loop over all states
            for ia, ib in product(range(bl_size),repeat=2): # loop over all pairs of orbitals
                idxa = idx0 + ia
                idxb = idx0 + ib
                for s, s1, t, t1 in product(range(2),repeat=4):
                    # apply the Pauli matrices to the state vector
                    idx1 = 2*idxa + s
                    idx2 = 2*idxa + s1
                    idx3 = 2*idxb + t
                    idx4 = 2*idxb + t1
                    new_state, sign = self.apply_c_dag_c_c_dag_c(state,idx1,idx2,idx3,idx4)
                    amplitude = 1/4 * (sx[s,s1]*sx[t,t1]+sy[s,s1]*sy[t,t1]+sz[s,s1]*sz[t,t1])
                    if new_state in self.index:
                        jstate = self.index[new_state]
                        row.append(jstate)
                        col.append(istate)
                        data.append(sign * amplitude)

        S_sq = self.build_sparse_matrix(row,col,data) # operator in matrix form
        self.S2_cache[orbital_key] = S_sq # store in class variable
        S_sq_val = self.gs_vector.conj().T @ S_sq @ self.gs_vector
        # Solve for S^{\hat}^2 = S(S+1)
        S = (np.sqrt(4*S_sq_val+1)-1)/2
        return S
    
    def get_Seff_for_orbital_group(self, orbital_keys: list[str]):
        """
        Calculate effective spin for a group of orbitals, like ["E1","E2"]
        
        Parameters
        ----------
        orbital_keys : list[str]
            block name of orbitals,  ["E1", "E2"]
        """
        key = tuple(sorted(orbital_keys))
        # If already constructed, use cached version
        if key in self.S2_cache:
            S_sq = self.S2_cache[key]
            S_sq_val = self.gs_vector.conj().T @ S_sq @ self.gs_vector
            S_val = (np.sqrt(4 * S_sq_val.real + 1) - 1) / 2
            return S_val
        # --- Otherwise, construct the S^2 operator ---
        def Pauli_mat(): # Pauli matrices for spin-1/2 system
            s0 = np.eye(2)
            sx = np.array([[0,1],[1,0]],dtype=complex)
            sy = np.array([[0,-1j],[1j,0]],dtype=complex)
            sz = np.array([[1,0],[0,-1]],dtype=complex)
            return s0,sx,sy,sz

        row, col, data = [], [], []
        
        # construct orbital indices for target orbitals
        orbital_indices = [] 
        for orb_key in orbital_keys:
            up_block_name = f"up_{orb_key}"
            idx0 = self.map_dict[up_block_name]
            bl_size = self.gf_struct_dict[up_block_name]
            for i in range(bl_size):
                orbital_indices.append(idx0 + i)

        s0,sx,sy,sz = Pauli_mat()
        
        for istate, state in enumerate(self.basis):
            for ia_orb_idx, ib_orb_idx in product(orbital_indices, repeat=2):                
                for s, s1, t, t1 in product(range(2), repeat=4):
                    idx1 = 2*ia_orb_idx + s
                    idx2 = 2*ia_orb_idx + s1
                    idx3 = 2*ib_orb_idx + t
                    idx4 = 2*ib_orb_idx + t1
                    
                    new_state, sign = self.apply_c_dag_c_c_dag_c(state, idx1, idx2, idx3, idx4)
                    amplitude = 0.25 * (sx[s,s1]*sx[t,t1] + sy[s,s1]*sy[t,t1] + sz[s,s1]*sz[t,t1])
                    if new_state is not None and new_state in self.index:
                        jstate = self.index[new_state]
                        row.append(jstate)
                        col.append(istate)
                        data.append(sign * amplitude)

        S_sq = self.build_sparse_matrix(row, col, data)
        self.S2_cache[key] = S_sq # store in class variable
        S_sq_val = self.gs_vector.conj().T @ S_sq @ self.gs_vector  
        S_val = (np.sqrt(4 * S_sq_val.real + 1) - 1) / 2
        return S_val
    
    def get_Sz(self):
        if self.gs_vector is None:
            print("Warning: Ground state not solved yet. Returning 0.")
            return 0.0
        if self.Sz_total is None:
            sz_total_mat = 0
            #print(self.L/2)
            for i in range(self.L//2):
                n_up_matrix = self.create_c_dag_c_matrix(2 * i, 2 * i)
                #n_up_val = self.gs_vector.conj().T @ (n_up_matrix @ self.gs_vector)
        
                n_dn_matrix = self.create_c_dag_c_matrix(2 * i + 1, 2 * i + 1)
                #n_dn_val = self.gs_vector.conj().T @ (n_dn_matrix @ self.gs_vector)
                
                sz_total_mat += 0.5 * (n_up_matrix - n_dn_matrix)
            self.Sz_total = sz_total_mat # store in class variable
            sz_total_exp_val = self.gs_vector.conj().T @ sz_total_mat @ self.gs_vector
                
            return sz_total_exp_val
        else:
            sz_total_mat = self.Sz_total
            sz_total_exp_val = self.gs_vector.conj().T @ sz_total_mat @ self.gs_vector
            return sz_total_exp_val

        


class EmbeddingSparseDummy:
    """
    Dummy embedding solver referencing an existing SolveEmbeddingSparse object.

    This dummy does not solve the embedding problem but retrieves data
    (e.g., density matrices) from a reference embedding, possibly applying rotations.

    Parameters
    ----------
    embedding : SolveEmbeddingSparse
        The actual embedding solver to reference.
    rotations : list[callable], optional
        Rotation functions to apply to density matrices (rho_c, rho_f, rho_cf).
    """

    def __init__(self, embedding, rotations=None):
        if rotations is None:
            rotations = []
        self.embedding = embedding
        self.rotations = rotations

    def set_h_emb(self, *args, **kwargs):
        pass  # dummy does not build the Hamiltonian

    def solve(self, *args, **kwargs):
        pass  # dummy does not solve

    def get_rho_f(self, bl: str) -> np.ndarray:
        rho = self.embedding.get_rho_f(bl)
        for rot in self.rotations:
            rho = rot(rho)
        return rho

    def get_rho_c(self, bl: str) -> np.ndarray:
        rho = self.embedding.get_rho_c(bl)
        for rot in self.rotations:
            rho = rot(rho)
        return rho

    def get_rho_cf(self, bl: str) -> np.ndarray:
        rho = self.embedding.get_rho_cf(bl)
        for rot in self.rotations:
            rho = rot(rho)
        return rho

