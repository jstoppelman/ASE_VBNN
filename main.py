#!/ff_files/usr/ff_files/bin/ff_files/env python
import numpy as np
from copy import deepcopy
from ase.io import read, write
from SAPTFF_OpenMM import SAPT_ForceField
from schnet_nn import *
import torch

def main():
    #Reference electronic energies, the shift variable is used for the H22 element
    ref_A_acetate = -228.3671507582417632
    ref_B_emim = -344.2359084163039711
    ref_A_acetic = -228.9282132443254056
    ref_B_nhc = -343.8068210638700180
    shift = ref_A_acetic + ref_B_nhc - ref_A_acetate - ref_B_emim
    H_kJmol = 2625.5
    shift *= H_kJmol

    #PDB, forcefield and residue file for constructing SAPT_ForceField classes
    pdbtemplate_d1 = "./ff_files/emim.pdb"
    pdbtemplate_d2 = "./ff_files/nhc.pdb"
    xml_ff_file = "./ff_files/pb.xml"
    xml_res_file = "./ff_files/pb_residues.xml"

    saptff_d1 = SAPT_ForceField(pdbtemplate_d1, xml_res_file, xml_ff_file, Drude_hyper_force=True, exclude_monomer_intra=True)
    saptff_d2 = SAPT_ForceField(pdbtemplate_d2, xml_res_file, xml_ff_file, Drude_hyper_force=True, exclude_monomer_intra=True)

    #Constuct residue list in order to partition total dimer positions into separate monomer
    #positions for the intramolecular neural networks included in the Diabat_NN classes
    res_list1 = saptff_d1.res_list()
    nn_d1 = Diabat_NN("./ff_files/emim_model", "./ff_files/acetate_model", "./ff_files/d1_inter", res_list1)

    res_list2 = saptff_d2.res_list()
    nn_d2 = Diabat_NN("./ff_files/nhc_model", "./ff_files/acetic_model", "./ff_files/d2_inter", res_list2)
    
    #Model for predicting the H12 energy and force
    off_diag = torch.load("./ff_files/best_model")

    #Main class for MD simulation inside of ASE
    ase_md = ASE_MD("input.xyz", './ff_files/md_tmp', saptff_d1, saptff_d2, nn_d1, nn_d2, off_diag, shift=shift)
    ase_md.create_system("traj")
    ase_md.run_md(1000)

if __name__ == "__main__":
    main()

