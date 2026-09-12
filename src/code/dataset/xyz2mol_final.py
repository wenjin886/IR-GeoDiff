from collections import defaultdict
import itertools
import networkx as nx
import copy

from rdkit import Chem

import numpy as np
import torch

__ATOM_LIST__ = \
    ['h',  'he',
     'li', 'be', 'b',  'c',  'n',  'o',  'f',  'ne',
     'na', 'mg', 'al', 'si', 'p',  's',  'cl', 'ar',
     'k',  'ca', 'sc', 'ti', 'v ', 'cr', 'mn', 'fe', 'co', 'ni', 'cu',
     'zn', 'ga', 'ge', 'as', 'se', 'br', 'kr',
     'rb', 'sr', 'y',  'zr', 'nb', 'mo', 'tc', 'ru', 'rh', 'pd', 'ag',
     'cd', 'in', 'sn', 'sb', 'te', 'i',  'xe',
     'cs', 'ba', 'la', 'ce', 'pr', 'nd', 'pm', 'sm', 'eu', 'gd', 'tb', 'dy',
     'ho', 'er', 'tm', 'yb', 'lu', 'hf', 'ta', 'w',  're', 'os', 'ir', 'pt',
     'au', 'hg', 'tl', 'pb', 'bi', 'po', 'at', 'rn',
     'fr', 'ra', 'ac', 'th', 'pa', 'u',  'np', 'pu']

INDEX2TYPE = {0: "H", 
              1: "B", 2:"C", 3:"N", 4:"O", 5:"F",
              6: "Al", 7: "Si", 8: "P", 9: "S", 10: "Cl", 
              11: "As", 12: "Se", 13:"Br"}


atomic_valence = defaultdict(list)
atomic_valence[1] = [1]
atomic_valence[5] = [3] # B
atomic_valence[6] = [4] # C
atomic_valence[7] = [3] # N
atomic_valence[8] = [2] # O
atomic_valence[9] = [1] # F
atomic_valence[13] = [3] # Al
atomic_valence[14] = [4] # Si
atomic_valence[15] = [3, 5] # P
atomic_valence[16] = [2, 4, 6] # S
atomic_valence[17] = [1] # Cl
atomic_valence[33] = [3, 5] # As
atomic_valence[34] = [2, 4, 6] # Se
atomic_valence[35] = [1] # Br


atomic_valence_electrons = {}
atomic_valence_electrons[1] = 1
atomic_valence_electrons[5] = 3
atomic_valence_electrons[6] = 4
atomic_valence_electrons[7] = 5
atomic_valence_electrons[8] = 6
atomic_valence_electrons[9] = 7
atomic_valence_electrons[13] = 3
atomic_valence_electrons[14] = 4
atomic_valence_electrons[15] = 5
atomic_valence_electrons[16] = 6
atomic_valence_electrons[17] = 7
atomic_valence_electrons[33] = 5
atomic_valence_electrons[34] = 6
atomic_valence_electrons[35] = 7



# http://www.wiredchemist.com/chemistry/data/bond_energies_lengths.html
bonds1 = {1: { 1:  74,  
               5: 119,  6: 109,  7: 101,  8: 96,   9: 92,
              13: 150, 14: 148, 15: 144, 16: 134, 17: 127,
                                33: 140, 34: 146, 35: 141},
        
          5: { 1: 119, # B
               5: 170,  6: 156,  7: 150,  8: 145,  9: 135,
              13: 210, 14: 195, 15: 185, 16: 185, 17: 185,
                                33: 200, 34: 200, 35: 175},
          
          6: { 1: 109,
               5: 156,  6: 154,  7: 147,  8: 143,  9: 135,
              13: 195, 14: 185, 15: 184, 16: 182, 17: 177,
                                33: 185, 34: 185, 35: 194},
          
          7: { 1: 101, 
               5: 150,  6: 147,  7: 145,  8: 140,  9: 136,
              13: 190, 14: 175, 15: 177, 16: 175, 17: 175,  
                                33: 180, 34: 180, 35: 214},
          
          8: { 1: 96,
               5: 145,  6: 143,  7: 140,  8: 148,  9: 142,
              13: 185, 14: 163, 15: 163, 16: 163, 17: 164,
                                33: 178, 34: 175, 35: 172},

          9:  {1: 92,
               5: 135,  6: 135,  7: 136,  8: 142,  9: 142,
              13: 175, 14: 160, 15: 154, 16: 158, 17: 166,
                                33: 171, 34: 165, 35: 178},

         13: { 1: 150,
               5: 210,  6: 195,  7: 190,  8: 185,  9: 175,
              13: 250, 14: 235, 15: 225, 16: 225, 17: 225,
                                33: 240, 34: 240, 35: 240},

         14: { 1: 148, 
               5: 195,  6: 185,  7: 175,  8: 163,  9: 160,
              13: 235, 14: 233, 15: 221, 16: 200, 17: 202,
                                33: 225, 34: 225, 35: 215},
        
         15: { 1: 144, 
               5: 185,  6: 184,  7: 177,  8: 163,  9: 154,
              13: 225, 14: 221, 15: 210, 16: 203, 17: 203,
                                33: 215, 34: 215, 35: 222},

         16: { 1: 134,
               5: 185,  6: 182,  7: 175 , 8: 163,  9: 158,
              13: 225, 14: 200, 15: 203, 16: 203, 17: 207,
                                33: 215, 34: 215, 35: 225},

         17: { 1: 127, 
               5: 185,  6: 177,  7: 175,  8: 164,  9: 166,
              13: 225, 14: 202, 15: 203, 16: 207, 17: 199,
                                33: 215, 34: 215, 35: 214},
        
         33: { 1: 140, # As
               5: 200,  6: 185,  7: 180,  8: 178,  9: 171,
              13: 240, 14: 225, 15: 215, 16: 215, 17: 215,
                                33: 243, 34: 230, 35: 233},

         34: { 1: 146,
               5: 200,  6: 185,  7: 180,  8: 175,  9: 165,
              13: 240, 14: 225, 15:215, 16: 215, 17: 215,
                                33: 230, 34: 230, 35: 230},

         35: { 1: 141,
               5: 175,  6: 194,  7: 214,  8: 172,  9: 178,
              13: 240, 14: 215, 15: 222, 16: 225, 17: 214,
                                33: 233, 34: 230, 35: 228},
 }
# BOND_THRESHOLD = 0.4 # qm9s
# BOND_THRESHOLD = 0.3 # qme14s

class GeometriesError(ValueError):
    def __init__(self, last_smi):
        self.last_smi = last_smi
    def __str__(self):
        return f"Please check the 3D geometries. ({self.last_smi})"

class NotCompleteMoleculeError(ValueError):
    def __init__(self, last_smi):
        self.last_smi = last_smi
    def __str__(self):
        return f"Not a complete molecule! ({self.last_smi})"
    
class XYZ2MOL:
    def __init__(self, atom_charge, pos, num_atoms=None, output_log=True, bond_threshold=0.4):
        """
        attribute:
            atoms: list, idx of atom is the charge or the idx in periodic table
            pos:  np.array or list
            num_atoms: int
            bond_threshold: 0.4 qm9s, 0.3 qme14s

        """
        self.output_log = output_log
        self.BOND_THRESHOLD = bond_threshold
        if num_atoms is not None: atom_charge = atom_charge[:num_atoms]
        self.atoms = [int(atom) for atom in atom_charge]
        
        if num_atoms is None:
            self.num_atoms = len(self.atoms)
        else:
            self.num_atoms = num_atoms
        
        
        if type(pos) == list:
            self.xyz = pos[:self.num_atoms]
        elif type(pos) == np.ndarray:
            pos = pos.astype(np.float64)
            self.xyz = pos[:self.num_atoms].tolist()
        elif type(pos) == torch.Tensor:
            self.xyz = pos[:self.num_atoms].tolist()

        self.single_valence_atoms = []
        for key, val in atomic_valence.items():
            if len(val) == 1 and val[0] == 1:
                self.single_valence_atoms.append(key)
    
    def get_atomic_charge(self, atom, atomic_valence_electrons, BO_valence):
        """
        """

        if atom == 1:
            charge = 1 - BO_valence
        elif atom == 5:
            charge = 3 - BO_valence
        elif atom == 15 and BO_valence == 5:
            charge = 0
        elif atom == 16 and BO_valence in [4, 6]:
            charge = 0
        else:
            charge = atomic_valence_electrons - 8 + BO_valence

        return charge
    
    def set_atomic_charges(self, mol, atoms, atomic_valence_electrons,
                           BO_valences, BO_matrix, mol_charge):
     
        q = 0
        for i, atom in enumerate(atoms):
            a = mol.GetAtomWithIdx(i)
           
            charge = self.get_atomic_charge(atom, atomic_valence_electrons[atom], BO_valences[i])
            q += charge
            if atom == 6:
                number_of_single_bonds_to_C = list(BO_matrix[i, :]).count(1)
                if number_of_single_bonds_to_C == 2 and BO_valences[i] == 2:
                    q += 1
                    charge = 0
                if number_of_single_bonds_to_C == 3 and q + 1 < mol_charge:
                    q += 2
                    charge = 1

            if (abs(charge) > 0):
                a.SetFormalCharge(int(charge))

        return mol

    def set_atomic_radicals(self, mol, atoms, atomic_valence_electrons, BO_valences):
        """
        The number of radical electrons = absolute atomic charge
        """
        for i, atom in enumerate(atoms):
            a = mol.GetAtomWithIdx(i)
          
            charge = self.get_atomic_charge(
                atom,
                atomic_valence_electrons[atom],
                BO_valences[i])

            if (abs(charge) > 0):
                a.SetNumRadicalElectrons(abs(int(charge)))

        return mol
        
    def get_mol_skeleton(self):
        mol_pos = Chem.RWMol()
        for atom in self.atoms:
            mol_pos.AddAtom(Chem.Atom(atom))
       
        # Set coordinates
        conf = Chem.Conformer(mol_pos.GetNumAtoms())
        for i in range(self.num_atoms):
            conf.SetAtomPosition(i, 
                                (self.xyz[i][0], self.xyz[i][1], self.xyz[i][2]))
        mol_pos.AddConformer(conf)

        AC = np.zeros((self.num_atoms, self.num_atoms), dtype=int)
        D = np.zeros((self.num_atoms, self.num_atoms))
        
        mol_AC = Chem.RWMol(mol_pos)
        for i in range(self.num_atoms):
            for j in range(i+1, self.num_atoms):
                p1 = np.array(self.xyz[i])
                p2 = np.array(self.xyz[j])
                dist = np.sqrt(np.sum((p1 - p2) ** 2))
                assert self.atoms[i] in bonds1 and self.atoms[j] in bonds1[self.atoms[i]], \
                    f"Please check single bond len setting of {self.atoms[i]} and {self.atoms[j]}."
                assert bonds1[self.atoms[i]][self.atoms[j]]  == bonds1[self.atoms[j]][self.atoms[i]], \
                    f"Please check single bond len setting of {self.atoms[i]} and {self.atoms[j]} ({bonds1[self.atoms[i]][self.atoms[j]]} and {bonds1[self.atoms[j]][self.atoms[i]]})."

                if dist < (bonds1[self.atoms[i]][self.atoms[j]]/100 + self.BOND_THRESHOLD):
                    AC[i, j] = 1
                    AC[j, i] = 1
                    D[i, j] = dist
                    D[j, i] = dist
                    mol_AC.AddBond(i, j, Chem.BondType.SINGLE)
        
        AC_valences = list(AC.sum(axis=1))
        mol_AC = self.set_atomic_radicals(mol_AC, self.atoms, atomic_valence_electrons, AC_valences)
        
        mol_AC = mol_AC.GetMol()
        smi_AC = Chem.MolToSmiles(mol_AC)
        
    
        
        return D, mol_pos, AC, mol_AC, smi_AC
    
    def check_BO(self, atoms, AC, modify=False, D=None):

        valences_list_of_lists = []
        
        # Obtain num of neighbourgs
        AC_valence = list(AC.sum(axis=1))
        
        if modify is True:
            assert D is not None
        AC_modified = AC.copy()
        D_modified = D.copy()

        for i,(atomicNum,valence) in enumerate(zip(atoms, AC_valence)):
            # valence can't be smaller than number of neighbourgs
            possible_valence = [x for x in atomic_valence[atomicNum] if x >= valence]
            if not possible_valence:
                if self.output_log:
                    print('check_BO | Valence of atom',i,'is',valence,'which bigger than allowed max',max(atomic_valence[atomicNum]))
                
                if modify: 
                    bond_diff = D[i, :].copy()
                    for j, at_j in enumerate(atoms):
                        if D[i, j] > 0:
                            bond_diff[j] = D[i, j] - np.array(bonds1[atomicNum][at_j]/100)
                    j = np.argmax(bond_diff)
                    assert D[i, j] != 0, f"Distance between {i}({atomicNum}) and {j}({atoms[j]}) is {'%.4f' % D[i, j]}. No bond can be removed."
                    if self.output_log: 
                        print(f"check_BO | Distance between {i}({atomicNum}) and {j}({atoms[j]}), {'%.4f' % D[i, j]}, " \
                              f"differs the standard one ({'%.4f' % (bonds1[atomicNum][atoms[j]]/100)}) most. Remove the bond.")
                    AC_modified[i, j] = 0
                    AC_modified[j, i] = 0
                    D_modified[i, j] = 0
                    D_modified[j, i] = 0
                    
                    return self.check_BO(atoms, AC_modified, modify, D_modified)

                
            valences_list_of_lists.append(possible_valence)
        return AC_valence, valences_list_of_lists, AC_modified, D_modified
    
    def get_UA(self, maxValence_list, valence_list):
        """
        Unsaturated atoms
        """
        UA = []
        DU = []
        for i, (maxValence, valence) in enumerate(zip(maxValence_list, valence_list)):
            if not maxValence - valence > 0:
                continue
            UA.append(i)
            DU.append(maxValence - valence)

        return UA, DU

    def find_independent_rings(self, mol):
     
        Chem.GetSymmSSSR(mol)

        rings = mol.GetRingInfo().AtomRings() # 最小环
        n = len(rings)
        
        adj = [[] for _ in range(n)]
        for i in range(n):
            for j in range(i + 1, n):
                if set(rings[i]) & set(rings[j]):
                    adj[i].append(j) 
                    adj[j].append(i)

        visited = [False] * n
        groups = []
        def dfs(u, group):
            visited[u] = True
            group.append(u)
            for v in adj[u]:
                if not visited[v]:
                    dfs(v, group)

        for i in range(n):
            if not visited[i]:
                group = []
                dfs(i, group)
                groups.append(group)

        independent_rings = []
        for group in groups:
            atom_set = set()
            for ring_idx in group:
                atom_set.update(rings[ring_idx])
            independent_rings.append(tuple(atom_set))

        return independent_rings
    
    def get_bonds(self, UA, AC):
        """
        Obtain bonds between two unsaturated atoms which can attribute more valence to unsaturated atoms.
        """
        bonds = []
        for k, i in enumerate(UA):
            for j in UA[k + 1:]:
                if AC[i, j] != 0:
                    bonds.append(tuple(sorted([i, j])))

        return bonds

    def get_UA_pairs(self, UA, AC, use_graph=True):
        """
        Return the most simplified bonds between two unsaturated atoms, which can describe the matching but contain the least num of edges.
        """

        bonds = self.get_bonds(UA, AC)
        if len(bonds) == 0:
            return [()]

        if use_graph:
            G = nx.Graph()
            G.add_edges_from(bonds)
            UA_pairs = [list(nx.max_weight_matching(G))]
            return UA_pairs

        max_atoms_in_combo = 0
        UA_pairs = [()]
        for combo in list(itertools.combinations(bonds, int(len(UA) / 2))):
            flat_list = [item for sublist in combo for item in sublist]
            atoms_in_combo = len(set(flat_list))
            if atoms_in_combo > max_atoms_in_combo:
                max_atoms_in_combo = atoms_in_combo
                UA_pairs = [combo]

            elif atoms_in_combo == max_atoms_in_combo:
                UA_pairs.append(combo)

        return UA_pairs
    
    def get_BO(self, AC, UA, DU, valences, UA_pairs, use_graph=True):
        """
        For each iteration, add bond order on all bonds between UA.
        Itering until no more UA.
        """
        BO = AC.copy()
        DU_save = []

        while DU_save != DU:
            for i, j in UA_pairs:
                if BO[i, j] < 3:
                    BO[i, j] += 1
                    BO[j, i] += 1
            BO_valence = list(BO.sum(axis=1))
            DU_save = copy.copy(DU)
            UA, DU = self.get_UA(valences, BO_valence)
            UA_pairs = self.get_UA_pairs(UA, AC, use_graph=use_graph)[0]

        return BO

    def get_terminal_atoms(self, AC):
        
        AC_valence = list(AC.sum(axis=1))
        saturated_atoms = [] 
        terminal_atoms = [] 

        for i, atom in enumerate(self.atoms):
            # if atom in [1, 9, 17] and AC_valence[i] == 1: saturated_atoms.append(i)
            if atom in self.single_valence_atoms and AC_valence[i] == 1: saturated_atoms.append(i)

        changed = True
        while changed:
            changed = False
            for i, atom in enumerate(self.atoms):

                if i in saturated_atoms or i in terminal_atoms: continue

                AC_i = AC[i, :]
                AC_i_sum = AC_i.sum()
                neighbor_atom_idx = np.where(AC_i > 0)[0] 
                neighbo_sa = [j for j in neighbor_atom_idx if j in saturated_atoms]

                if atom in [6, 7, 8]:
                    if len(neighbo_sa) == AC_i_sum - 1: 
                        if atomic_valence[atom][0] == AC_i_sum: 
                            saturated_atoms.append(i)
                            changed = True 
                        elif atomic_valence[atom][0] > AC_i_sum: 
                            terminal_atoms.append((i, atomic_valence[atom][0] - AC_i_sum)) 
                elif atom in [16, 34]: # S, Se
                    if AC_i_sum == 1: terminal_atoms.append((i, 1))
        return terminal_atoms, saturated_atoms

    def update_terminal_atoms_BO(self, AC):
     
        terminal_atoms, saturated_atoms = self.get_terminal_atoms(AC=AC)
        terminal_atoms = list(set(terminal_atoms))

        if terminal_atoms == []:
            return AC
        else:
            BO = AC.copy()
            for i, num_bonds_to_add in terminal_atoms:
                if i in saturated_atoms: continue # saturated_atoms keep updating.
                pos = np.squeeze(np.argwhere(BO[i, :] != 0)) # neighbor atoms
                
                if pos.size > 1:
                    ua = [int(j) for j in pos if (j not in saturated_atoms)]  
                else: 
                    ua = [int(pos)]
                if len(ua) == 1: 
                    ua0_valence = BO[ua[0], :].sum() 
                    if ua0_valence >= max(atomic_valence[self.atoms[ua[0]]]):
                        pass
                    elif ua0_valence + num_bonds_to_add <= min(atomic_valence[self.atoms[ua[0]]]):
                        if BO[i, ua[0]] + num_bonds_to_add < 4: 
                            BO[i, ua[0]] += num_bonds_to_add
                            BO[ua[0], i] += num_bonds_to_add 

                    else:
                        if BO[i, ua[0]] + 1 < 4: 
                            BO[i, ua[0]] += 1
                            BO[ua[0], i] += 1
                    
                    saturated_atoms.append(i)
                    if ua[0] in terminal_atoms: saturated_atoms.append(ua[0])
                       
                else:
                    continue
                    
            return BO

    def is_aromatic_ring(self, ring_atoms, ring_valance):
        if len(ring_atoms) < 5: return False
        pi_electrons = 0

        for atom, valance in zip(ring_atoms, ring_valance):
            if atom == 6:
                if valance == 3: 
                    pi_electrons += 1
                else:
                    return False

            elif atom in [7, 15]:
                if (valance == 2):
                    pi_electrons += 1
                elif (valance == 3):
                    pi_electrons += 2
                else:
                    return False

            elif atom in [8, 16]:
                pi_electrons += 2

            else: return False

        return (pi_electrons >= 2) and (pi_electrons - 2) % 4 == 0

    def update_aromatic_rings_BO(self, BO, D, independent_rings):
        
        BO_new = BO.copy()
        
        BO = np.array(BO)
        BO_new = np.array(BO_new)
        D = np.array(D)
        
        all_BO_sum = BO.sum(axis=1)
        
        for i, ring in enumerate(independent_rings): 
            ring_atoms = np.array([self.atoms[i] for i in ring])
            ring_valance = [all_BO_sum[i] for i in ring]
            if not self.is_aromatic_ring(ring_atoms, ring_valance): continue

            
            sub_BO = BO[np.ix_(ring, ring)]
            sub_BO_new = BO_new[np.ix_(ring, ring)]
            sub_D = D[np.ix_(ring, ring)]
            

            BO_valence, valences_list_of_lists, _, _ = self.check_BO(ring_atoms, sub_BO, modify=False, D=sub_D)

            valences = [min(i) for i in valences_list_of_lists] 

            best_sub_BO_new = sub_BO_new.copy()
            valences = np.array(valences)
            for i, a_i in enumerate(ring_atoms):
                if a_i == 6:
                    idx = ring[i]
                    val_diff = all_BO_sum[idx] - BO_valence[i]
                    valences[i] = 4 - val_diff
                elif a_i in [7, 15]:
                    idx = ring[i]
                    if all_BO_sum[idx] == 3: valences[i] = 2 

            UA, DU_from_BO = self.get_UA(valences, BO_valence)

            
            UA_pairs_list = self.get_UA_pairs(UA, sub_BO)
            for UA_pairs in UA_pairs_list:
                sub_BO_new = self.get_BO(sub_BO_new, UA, DU_from_BO, valences, UA_pairs, use_graph=True)
                best_sub_BO_new = sub_BO_new
                     
            BO_new[np.ix_(ring, ring)] = best_sub_BO_new

        return BO_new
    
    def get_bonds_from_BO(self, UA, BO):
        """
        Obtain bonds between two unsaturated atoms which can attribute more valence to unsaturated atoms.
        """
        bonds = []
        for k, i in enumerate(UA):
            for j in UA[k + 1:]:
                if BO[i, j] > 0:
                    atoms_bo = [sorted([i, j]), BO[i,j]]
                    bonds.append(tuple(atoms_bo))

        return bonds

    def choose_equiv_bonds_old(self, UA, BO, D):
        """
        Based on the distance as the smaller distance, the higher bond order
        """
        bonds = self.get_bonds_from_BO(UA, BO)
        bonds_type_dist = {}
        for bond in bonds:
            bond_atoms, bo = bond
            a_i, a_j = bond_atoms
            dist = D[a_i, a_j]
            if bo in bonds_type_dist:
                dist_, _ = bonds_type_dist[bo]
                if dist < dist_:
                    bonds_type_dist[bo] = [dist, [a_i, a_j]]    
            else:
                bonds_type_dist[bo] = [dist, [a_i, a_j]]
        new_UA = []
        for key, value in bonds_type_dist.items():
            new_UA += value[1]
        new_UA = list(set(new_UA))
        return new_UA
    
    def _get_max_rel_dist_diff_single_neighbor_type(self, rel_dist_diff_list, diff_limit):
        """
        diff_limit: same bond type: 0.007, different bond type: 0.06
        """
        max_rel_diff_i, is_update = -1000, False

        for rel_diff_i in sorted(rel_dist_diff_list):
            if (rel_diff_i > max_rel_diff_i) and abs(rel_diff_i - max_rel_diff_i) > diff_limit:
                if max_rel_diff_i != -1000: is_update = True
                max_rel_diff_i = rel_diff_i
            else:
                if self.output_log: print(f"No update: rel_diff_i: {rel_diff_i} | max_rel_diff_i: {max_rel_diff_i} | diff: {abs(rel_diff_i - max_rel_diff_i)} | diff_limit: {diff_limit}")
                else: pass

        return max_rel_diff_i, is_update
    
    def choose_equiv_bonds(self, UA, BO, D):
        
        if len(UA) == 1: return None
        if len(UA) == 2: return UA


        core_atom, core_atom_type = UA[0], self.atoms[UA[0]]
        bonds_of_core = {"neighbor_atom":[], "neighbor_type":[], "bo":[], "rel_dist_diff":[]}
        for i in range(1, len(UA)):
            neighbo_atom, neighbo_atom_type = UA[i], self.atoms[UA[i]]
            bonds_of_core["neighbor_atom"].append(neighbo_atom)
            bonds_of_core["neighbor_type"].append(neighbo_atom_type)
            bonds_of_core["bo"].append(BO[core_atom, neighbo_atom])
            bonds_of_core["rel_dist_diff"].append(bonds1[core_atom_type][neighbo_atom_type]/100 - D[core_atom, neighbo_atom])
        
        if self.output_log: 
            print(f"core_atom: {core_atom}, core_atom_type: {core_atom_type}")
            print(f"bonds_of_core: {bonds_of_core}")
        
        if len(set(bonds_of_core["bo"])) != 1:
            return None 
           
        if len(set(bonds_of_core["neighbor_type"])) == 1:
            max_rel_diff_i, is_update = self._get_max_rel_dist_diff_single_neighbor_type(bonds_of_core["rel_dist_diff"], diff_limit=0.007)
            if max_rel_diff_i < 0: return None 
            if is_update: 
                idx_bond = bonds_of_core["rel_dist_diff"].index(max_rel_diff_i)
                return list(sorted([core_atom, bonds_of_core["neighbor_atom"][idx_bond]]))
            else: return None
        
        else:
            neighbor_type_list = list(set(bonds_of_core["neighbor_type"]))

            max_rel_dist_diff_each_neighbor_type = []
            for neighbor_type in neighbor_type_list:
                idx_bond_i = [i for i, x in enumerate(bonds_of_core["neighbor_type"]) if x == neighbor_type]
                if len(idx_bond_i) == 1: 
                    max_rel_dist_diff_each_neighbor_type.append(bonds_of_core["rel_dist_diff"][idx_bond_i[0]])
                else:
                    rel_dist_diff_list_i = [bonds_of_core["rel_dist_diff"][i] for i in idx_bond_i]
                    max_rel_diff_i, is_update = self._get_max_rel_dist_diff_single_neighbor_type(rel_dist_diff_list_i, diff_limit=0.007)
                    if is_update: max_rel_dist_diff_each_neighbor_type.append(max_rel_diff_i)
            
            max_rel_diff, is_update = self._get_max_rel_dist_diff_single_neighbor_type(max_rel_dist_diff_each_neighbor_type, diff_limit=0.06)
            if max_rel_diff < 0: return None
            if is_update:
                idx_bond = bonds_of_core["rel_dist_diff"].index(max_rel_diff)
                return list(sorted([core_atom, bonds_of_core["neighbor_atom"][idx_bond]]))
            else: return None 
            
            
    
    def update_N_C_BO(self, BO, D):
        """
        args:
            atom_idx: 7, 6 (7: N, 6: C) 
        """

        step = 0
        skip_UA = []
        while True:
            BO_valence, valences_list_of_lists, BO, D = self.check_BO(self.atoms, BO, modify=True, D=D)
            
            if [] in valences_list_of_lists: return None
            valences = [min(i) for i in valences_list_of_lists]
            UA, DU_from_BO = self.get_UA(valences, BO_valence)
            DU_from_BO = np.array(DU_from_BO)
            UA = np.array(UA)
            sort_idx = np.argsort(DU_from_BO)[::-1]
            UA = UA[sort_idx] 

            ua_list = [atom for atom in UA if self.atoms[atom] in [6, 7]] 
            if ua_list == []:
                return BO
            else:
                UA_new = []
                for i in ua_list:
                    UA_new_i = [i]
                    pos = np.squeeze(np.argwhere(BO[i, :] != 0), axis=-1)

                    for j in pos: 
                        if (j in UA) or ((self.atoms[j] in [15, 16, 33, 34]) and (BO_valence[j] < max(atomic_valence[self.atoms[j]]))): 
                            UA_new_i.append(j)
                    
                    if UA_new_i in skip_UA: 
                        continue

                    UA_new.append(UA_new_i)
                    
        
                
                if UA_new == []: return BO
                num_ua_atoms = [len(i) for i in UA_new]
                min_idx = num_ua_atoms.index(min(num_ua_atoms))
                UA_new = UA_new[min_idx]

                if self.output_log: print(f"{step}: before choose_equiv_bonds | UA_new: {UA_new}")
                _UA_new = self.choose_equiv_bonds(UA_new, BO, D)
                if self.output_log: 
                    print(f"{step}: after choose_equiv_bonds | UA_new: {_UA_new}\n")
                    
                
                if _UA_new is None: 
                    skip_UA.append(UA_new)
                    step += 1
                    continue
                else: 
                    UA_new = _UA_new
                    
                UA_pairs_list = self.get_UA_pairs(UA_new, BO)
                if len(UA_pairs_list) == 1 and UA_pairs_list[0] == (): return BO
                for UA_pairs in UA_pairs_list:
                    for i, j in UA_pairs:
                        if BO[i, j] < 3:
                            BO[i, j] += 1
                            BO[j, i] += 1
            
            if step > 9:
                print(f"Warning: Reach the step maximum {step}. (func: update_N_C_BO)")
                return BO 
            step += 1
    
    def charge_is_OK(self, BO, charge, atomic_valence_electrons, atoms, allow_charged_fragments=True):
        # total charge
        Q = 0

        # charge fragment list
        q_list = []

        BO_valences = list(BO.sum(axis=1))
        for i, atom in enumerate(atoms):
            q = self.get_atomic_charge(atom, atomic_valence_electrons[atom], BO_valences[i])
            if (q != 0):
                if (atom in [13, 33, 34]) and (BO_valences[i] in atomic_valence[atom]):
                    q = 0
                elif  (atom in [1, 6, 7, 8, 9]): 
                    if (not allow_charged_fragments): return False
                    
                
            Q += q
            if atom == 6:
                number_of_single_bonds_to_C = list(BO[i, :]).count(1)
                if number_of_single_bonds_to_C == 2 and BO_valences[i] == 2:
                    Q += 1
                    q = 2
                if number_of_single_bonds_to_C == 3 and Q + 1 < charge:
                    Q += 2
                    q = 1

            if q != 0:
                q_list.append(q)
        
        return (charge == Q)


    def update_bonds_iteratively(self, BO, D):
        
        max_iter = 20 # set a maximum number of iterations to prevent dead loop
        for _ in range(max_iter):
            
            current_valences = BO.sum(axis=1)
            
            
            urgent_atoms = [
                i for i, v in enumerate(current_valences) 
                if v not in atomic_valence.get(self.atoms[i], [v])
            ]
            if self.output_log: print(f"urgent_atoms: {urgent_atoms}")
            if not urgent_atoms:
                break

            min_num_ua_neighbors, i, neighbors = 1000, -1, []
            for urgent_atom in urgent_atoms:
                neighbors_i = [j for j, bond_order in enumerate(BO[urgent_atom]) if bond_order > 0]
                num_ua_neighbors_i = 0
                for idx in neighbors_i:
                    if idx in urgent_atoms:
                        num_ua_neighbors_i += 1
                if num_ua_neighbors_i < min_num_ua_neighbors:
                    min_num_ua_neighbors = num_ua_neighbors_i
                    i = urgent_atom
                    neighbors = neighbors_i
            if self.output_log: print(f"i: {i} | neighbors: {neighbors}")
        
            possible_partners_ua, possible_partners_others = [], []
            for j in neighbors:
                # check if the partner can be upgraded
                can_upgrade = current_valences[j] < max(atomic_valence.get(self.atoms[j], [current_valences[j]]))
                # check if the bond order is less than 3
                bond_not_full = BO[i, j] < 3
                if can_upgrade and bond_not_full:
                    # possible_partners.append(j)
                    if j in urgent_atoms: possible_partners_ua.append(j)
                    else: possible_partners_others.append(j)

          
            if self.output_log: print(f"possible_partners_ua: {possible_partners_ua} | possible_partners_others: {possible_partners_others}")
            if len(possible_partners_ua) == 0 and len(possible_partners_others) == 0: break

            if len(possible_partners_ua) > 0: 
                forward_partners = [j for j in possible_partners_ua if j > i]
                if forward_partners: best_partner = min(forward_partners) # if there is a forward partner, choose the smallest index
                else: best_partner = max(possible_partners_ua) # if there is no forward partner (like at the end of the chain), choose the largest index, often the only choice
            else:
                forward_partners = [j for j in possible_partners_others if j > i]
                if forward_partners: best_partner = min(forward_partners) # if there is a forward partner, choose the smallest index
                else: best_partner = max(possible_partners_others) # if there is no forward partner (like at the end of the chain), choose the largest index, often the only choice
            
      
            BO[i, best_partner] += 1
            BO[best_partner, i] += 1
            updated_in_iter = True
           
            if updated_in_iter:
                continue 

        return BO


    def get_mol_smi(self, mol_pos, BO):
        bondTypeDict = {
            1: Chem.BondType.SINGLE,
            2: Chem.BondType.DOUBLE,
            3: Chem.BondType.TRIPLE
        }

        mol_BO = Chem.RWMol(mol_pos)
        for i in range(self.num_atoms):
            for j in range(i+1, self.num_atoms):
                assert BO[i,j] < 4, f"Please check bond order between atom {i} and {j}: {BO[i,j]}."
                if BO[i,j] > 0:
                    mol_BO.AddBond(i, j, bondTypeDict[BO[i,j]])
        
        BO_valences = list(BO.sum(axis=1))
        mol_BO = self.set_atomic_radicals(mol_BO, self.atoms, atomic_valence_electrons, BO_valences)
        mol_BO = mol_BO.GetMol()
        return mol_BO, Chem.MolToSmiles(mol_BO)
    
    def global_remove_bond(self, AC, D, mol_pos):
    
        D = np.array(D)
        new_D = D.copy()
        positions = np.argwhere(D != 0)

        D_diff = D.copy()
        for i, j in positions:
            bond_len_ref = bonds1[self.atoms[i]][self.atoms[j]]
            D_diff[i,j] = D[i, j] - bond_len_ref/100
        
        max_dist_diff = np.max(D_diff)
        if max_dist_diff < 0.15:
            if self.output_log: print(f"global_remove_bond | No bond can been removed as the maiximum of dist difference is only {'%.4f' % max_dist_diff}.")
            return None, None, None

        removed_bond = np.argwhere(D_diff == np.max(D_diff))
        new_AC = AC.copy()
        for i, j in removed_bond:
            new_AC[i, j] = AC[i, j] - 1
            new_D[i, j] = 0
            bond_len_ref = bonds1[self.atoms[i]][self.atoms[j]]
            if self.output_log:
                print(f"global_remove_bond | Bond between {self.atoms[i]}({i}) and {self.atoms[j]}({j}) has been removed. (Dist: {'%.4f' % D[i,j]}, Ref: {'%.4f' % (bond_len_ref/100)})")
        mol_AC, _ = self.get_mol_smi(mol_pos, new_AC)
        return new_AC, new_D, mol_AC
        # return new_AC


    def BO2mol(self, mol, BO_matrix, atoms, atomic_valence_electrons):
        """
        based on code written by Paolo Toscani

        From bond order, atoms, valence structure and total charge, generate an
        rdkit molecule.

        args:
            mol - rdkit molecule
            BO_matrix - bond order matrix of molecule
            atoms - list of integer atomic symbols
            atomic_valence_electrons -
            mol_charge - 0, total charge of molecule

        optional:
            allow_charged_fragments - bool - allow charged fragments

        returns
            mol - updated rdkit molecule with bond connectivity

        """

        l = len(BO_matrix)
        l2 = len(atoms)
        BO_valences = list(BO_matrix.sum(axis=1))

        if (l != l2):
            raise RuntimeError('sizes of adjMat ({0:d}) and Atoms {1:d} differ'.format(l, l2))

        rwMol = Chem.RWMol(mol)

        bondTypeDict = {
            1: Chem.BondType.SINGLE,
            2: Chem.BondType.DOUBLE,
            3: Chem.BondType.TRIPLE
        }

        for i in range(l):
            for j in range(i + 1, l):
                bo = int(round(BO_matrix[i, j]))
                if (bo == 0):
                    continue
                bt = bondTypeDict.get(bo, Chem.BondType.SINGLE)
                rwMol.AddBond(i, j, bt)

        mol = rwMol.GetMol()
        mol = self.set_atomic_radicals(mol, atoms, atomic_valence_electrons, BO_valences)

        return mol  
       
    def xyz2mol(self, process_fragments=False):
        D, mol_pos, AC, mol_AC, smi = self.get_mol_skeleton()
        # if AC is None:
        #     return mol_AC, smi

        frags = Chem.GetMolFrags(mol_AC)
        if len(frags) > 1 and not process_fragments:
            print(f"Not a complete molecule!({smi})")
            return mol_AC, smi

        if self.output_log: print(f"skeleton | {smi}")
        # print(AC)
        step = 0 
        while True:
            
            if step >= 10:
                print(f"Warning | Step reached the maximum ({step}).")
                break
            step += 1

            BO = self.update_terminal_atoms_BO(AC)
            if BO is None:
                AC, D, mol_AC = self.global_remove_bond(AC, D, mol_pos)
                if AC is None: return mol, smi
                continue
            else:
                mol, smi = self.get_mol_smi(mol_pos, BO)
                _, valences_list_of_lists, _, _ = self.check_BO(self.atoms, BO, modify=False, D=D)
                if [] in valences_list_of_lists: 
                    AC, D, mol_AC = self.global_remove_bond(AC, D, mol_pos)
                    if AC is None: return mol, smi
                    continue
            if self.output_log: print(f"step {step} update_terminal_atoms_BO | {smi}")

            independent_rings = self.find_independent_rings(mol_AC)
            BO = self.update_aromatic_rings_BO(BO, D, independent_rings)
            if BO is None:
                AC, D, mol_AC = self.global_remove_bond(AC, D, mol_pos)
                if AC is None: return mol, smi
                continue
            else:
                mol, smi = self.get_mol_smi(mol_pos, BO)
            if self.output_log: print(f"step {step} update_aromatic_rings_BO | {smi}")

            
            BO = self.update_N_C_BO(BO, D) 
            if BO is None:
                AC, D, mol_AC = self.global_remove_bond(AC, D, mol_pos)
                if AC is None: return mol, smi
                continue
            else:
                mol, smi = self.get_mol_smi(mol_pos, BO)
            if self.output_log: print(f"step {step} update_N_C_BO | {smi}")

            BO = self.update_bonds_iteratively(BO, D)
            if BO is None:
                AC, D, mol_AC = self.global_remove_bond(AC, D, mol_pos)
                if AC is None: return mol, smi
                continue
            else:
                mol, smi = self.get_mol_smi(mol_pos, BO)
            if self.output_log: print(f"step {step} update_other_bonds | {smi}")
            if self.output_log: print(f"step {step} charge_is_OK | {smi}")
            charge_OK = self.charge_is_OK(BO, 0, atomic_valence_electrons, self.atoms, allow_charged_fragments=False)
            if not charge_OK:
                if self.output_log: print("CHARGE IS NOT 0!")
                AC, D, mol_AC = self.global_remove_bond(AC, D, mol_pos)
                if AC is None: return mol, smi
                continue

            break
            
        if BO is None: 
            return mol, smi
        else:
            mol = self.BO2mol(
                    mol_pos,
                    BO,
                    self.atoms,
                    atomic_valence_electrons,
                    )
        
        return mol, Chem.MolToSmiles(mol)
 

if __name__ == "__main__":
    pass

