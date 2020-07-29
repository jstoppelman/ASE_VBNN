import schnetpack as sch
from schnetpack import AtomsData, Properties
from schnetpack.data import AtomsConverter
import schnetpack.train as trn
from schnetpack.md import MDUnits
import numpy as np
import torch
from torch.optim import Adam
import sys, os, shutil
from copy import deepcopy
from ase import units, Atoms
from ase.io import read, write
from ase.io import Trajectory
from ase.calculators.calculator import Calculator, all_changes
from ase.md import VelocityVerlet, Langevin, MDLogger
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution

def pos_diabat2(positions):
    """
    Reorder diabat 1 positions to be in the same order as diabat 2.
    Hard-coded in for now to work for EMIM+/acetate 

    Parameters
    -----------
    positions : np.ndarray
        array containing 3*N positions for the simulated dimer

    Returns
    -----------
    positions : np.ndarray
        array containing the reordered 3*N positions
    inds : list
        list containing the indices at which the new positions
        have been modified in comparison to the old positions.
    """
    inds = [i for i in range(len(positions))]
    p_o1 = positions[19]
    p_o2 = positions[20]
    h = positions[3]
    dist1 = p_o1 - h
    dist2 = p_o2 - h
    dist1 = np.power(np.dot(dist1, dist1), 1/2)
    dist2 = np.power(np.dot(dist2, dist2), 1/2)
    if dist1 < dist2:
        inds.append(inds.pop(3))
        h_atom = positions[3]
        positions = np.delete(positions, 3, axis=0)
        positions = np.append(positions, [h_atom], axis=0)
    else:
        inds.insert(19, inds.pop(20))
        inds.append(inds.pop(3))
        new_pos = np.empty_like(positions)
        new_pos[:] = positions
        o1_atom = positions[19]
        o2_atom = positions[20]
        new_pos[19] = o2_atom
        new_pos[20] = o1_atom
        h_atom = new_pos[3]
        new_pos = np.delete(new_pos, 3, axis=0)
        new_pos = np.append(new_pos, [h_atom], axis=0)
        positions = new_pos
    return positions, inds

def reorder(coords, inds):
    """
    Reorder the input (positions or forces from diabat 2)
    to align with the positions and forces of diabat 1

    Parameters
    -----------
    coords : np.ndarray
        array containing 3*N positions or forces for the simulated dimer
    inds : list
        list containing the indices that the diabat 1 positions
        were reordered to in pos_diabat2

    Returns
    -----------
    coords : np.ndarray
        array containing the reorderd posiitons or forces
    """
    reord_list = [inds.index(i) for i in range(len(inds))]
    return coords[reord_list]

class Diabat_NN:
    """
    Class for obtaining the energies and forces from SchNetPack
    neural networks. This computes energies and forces from the
    intra- and intermolecular neural networks for the diabatic states.
    """
    def __init__(self, nn_monA, nn_monB, nn_dimer, res_list):
        """
        Parameters
        -----------
        nn_monA : str
            location of the neural network for monomer A in the diabat
        nn_monB : str
            location of the neural network for monomer B in the diabat
        nn_dimer : str
            location of the neural network for the dimer
        res_list : dictionary
            dictionary containing the indices of the monomers in the diabat
        """
        self.nn_monA = torch.load(nn_monA)
        self.nn_monB = torch.load(nn_monB)
        self.nn_dimer = torch.load(nn_dimer)
        self.res_list = res_list
        self.converter = AtomsConverter(device='cuda')

    def compute_energy_intra(self, xyz):
        """
        Compute the energy for the intramolecular components of the dimer

        Parameters
        -----------
        xyz : ASE Atoms Object
            ASE Atoms Object used as the input for the neural networks.
        
        Returns
        -----------
        energy : np.ndarray
            Intramoleculer energy in kJ/mol
        forces : np.ndarray
            Intramolecular forces in kJ/mol/A
        """ 
        monA = self.res_list[0]
        monB = self.res_list[1]
        
        xyzA = xyz[monA]
        xyzB = xyz[monB]

        inputA = self.converter(xyzA)
        inputB = self.converter(xyzB)
        resultA = self.nn_monA(inputA)
        resultB = self.nn_monB(inputB)

        energyA = resultA['energy'].detach().cpu().numpy()[0][0]
        forcesA = resultA['forces'].detach().cpu().numpy()[0]
        energyB = resultB['energy'].detach().cpu().numpy()[0][0]
        forcesB = resultB['forces'].detach().cpu().numpy()[0]

        intra_eng = energyA + energyB
        intra_forces = np.append(forcesA, forcesB, axis=0)
        return np.asarray(intra_eng), np.asarray(intra_forces)

    def compute_energy_inter(self, xyz):
        """
        Compute the energy for the intermolecular components of the dimer.

        Parameters
        -----------
        xyz : ASE Atoms Object
            ASE Atoms Object used as the input for the neural network.

        Returns
        -----------
        energy : np.ndarray
            Intermoleculer energy 
        forces : np.ndarray
            Intermolecular forces
        """
        inp = self.converter(xyz)
        result = self.nn_dimer(inp)

        energy = result['energy'].detach().cpu().numpy()[0][0]
        forces = result['forces'].detach().cpu().numpy()[0]
        return np.asarray(energy), np.asarray(forces)

class EVB_Hamiltonian(Calculator):
    """ 
    ASE Calculator for running EVB simulations using OpenMM forcefields 
    and SchNetPack neural networks. Modeled after SchNetPack calculator.
    """
    energy = "energy"
    forces = "forces"
    implemented_properties = [energy, forces]

    def __init__(self, saptff_d1, saptff_d2, nn_d1, nn_d2, off_diag, shift=0, **kwargs):
        """
        Parameters
        -----------
        saptff_d1 : Object
            Contains OpenMM force field for diabat 1.
        saptff_d2 : Object
            Contains OpenMM force field for diabat 2.
        nn_d1 : Object
            Diabat_NN instance for diabat 1.
        nn_d2 : Object
            Diabat NN instance for diabat 2.
        off_diag : PyTorch model
            Model for predicting H12 energy and forces.
        shift : float
            Diabat 2 energy shift to diabat 1 reference.
        **kwargs : dict
            additional args for ASE base calculator.
        """
        Calculator.__init__(self, **kwargs)
        self.saptff_d1 = saptff_d1
        self.saptff_d2 = saptff_d2
        self.nn_d1 = nn_d1
        self.nn_d2 = nn_d2
        self.off_diag = off_diag
        self.shift = shift

        self.converter = AtomsConverter(device='cuda')
        self.energy_units = units.kJ / units.mol
        self.forces_units = units.kJ / units.mol / units.Angstrom

    def saptff_energy_force_d1(self, xyz):
        """
        Compute OpenMM energy and forces

        Parameters
        -----------
        xyz : ASE Atoms object
            Used to supply positions to the SAPTFF_Forcefield class

        Returns
        -----------
        energy : np.ndarray
            Energy in kJ/mol
        forces : np.ndarray
            Forces in kJ/mol/A
        """
        pos_d1 = xyz.get_positions()
        omm_pos = self.saptff_d1.set_xyz(pos_d1)
        energy, forces = self.saptff_d1.compute_energy(omm_pos)
        return energy, forces

    def saptff_energy_force_d2(self, xyz):
        """
        Compute OpenMM energy and forces

        Parameters
        -----------
        xyz : ASE Atoms object
            Used to supply positions to the SAPTFF_Forcefield class

        Returns
        -----------
        energy : np.ndarray
            Energy in kJ/mol
        forces : np.ndarray
            Forces in kJ/mol/A
        """

        xyz, reord_inds = pos_diabat2(xyz.get_positions())
        omm_pos = self.saptff_d2.set_xyz(xyz)
        energy, forces = self.saptff_d2.compute_energy(omm_pos)
        forces = reorder(forces, reord_inds)
        return energy, forces

    def nn_energy_force_d1(self, xyz):
        """
        Compute Diabat neural network energy and forces

        Parameters
        -----------
        xyz : ASE Atoms object
            Used to supply positions to the Diabat_NN class

        Returns
        -----------
        energy : np.ndarray
            Energy in kJ/mol
        forces : np.ndarray
            Forces in kJ/mol/A
        """

        intra_eng, intra_forces = self.nn_d1.compute_energy_intra(xyz)
        inter_eng, inter_forces = self.nn_d1.compute_energy_inter(xyz)
        total_forces = intra_forces + inter_forces
        return intra_eng + inter_eng, total_forces

    def nn_energy_force_d2(self, xyz):
        """
        Compute Diabat neural network energy and forces

        Parameters
        -----------
        xyz : ASE Atoms object
            Used to supply positions to the Diabat_NN class

        Returns
        -----------
        energy : np.ndarray
            Energy in kJ/mol
        forces : np.ndarray
            Forces in kJ/mol/A
        """

        symb = xyz.get_chemical_symbols()
        symb.append(symb.pop(3))
        xyz, reord_inds = pos_diabat2(xyz.get_positions())
        tmp_Atoms = Atoms(symb, positions=xyz)
        intra_eng, intra_forces = self.nn_d2.compute_energy_intra(tmp_Atoms)
        inter_eng, inter_forces = self.nn_d2.compute_energy_inter(tmp_Atoms)
        total_forces = intra_forces + inter_forces
        total_forces = reorder(total_forces, reord_inds)
        return intra_eng + inter_eng, total_forces

    def diagonalize(self, d1_energy, d2_energy, h12_energy):
        """
        Forms matrix and diagonalizes using np to obtain ground-state
        eigenvalue and eigenvector.

        Parameters
        -----------
        d1_energy : np.ndarray
            Total diabat 1 energy
        d2_energy : np.ndarray
            Total diabat 2 energy
        h12_energy : np.ndarray
            Off-diagonal energy

        Returns
        -----------
        eig[l_eig] : np.ndarray
            Ground-state eigenvalue
        eigv[:, l_eig] : np.ndarray
            Ground-state eigenvector
        """
        hamiltonian = [[d1_energy, h12_energy], [h12_energy, d2_energy]]
        eig, eigv = np.linalg.eig(hamiltonian)
        l_eig = np.argmin(eig)
        return eig[l_eig], eigv[:, l_eig]

    def calculate_forces(self, d1_forces, d2_forces, h12_forces, ci):
        """
        Uses Hellmann-Feynman theorem to calculate forces on each atom.

        Parameters
        -----------
        d1_forces : np.ndarray
            forces for diabat 1
        d2_forces : np.ndarray
            forces for diabat 2
        h12_forces : np.ndarray
            forces from off-diagonal elements
        ci : np.ndarray
            ground-state eigenvector

        Returns
        -----------
        np.ndarray
            Forces calculated from Hellman-Feynman theorem
        """
        return ci[0]**2 * d1_forces + 2 * ci[0] * ci[1] * h12_forces + ci[1]**2 * d2_forces

    def calculate(self, atoms=None, properties=['energy'], system_changes=all_changes):
        """
        Obtains the total energy and force using the above methods.

        Parameters
        -----------
        atoms : ASE Atoms object
            atoms object containing coordinates
        properties : list
            Not used, follows SchNetPack format
        system_changes : list
            List of changes for ASE
        """

        Calculator.calculate(self, atoms)
        result = {}

        ff_energy_d1, ff_forces_d1 = self.saptff_energy_force_d1(atoms)
        ff_energy_d2, ff_forces_d2 = self.saptff_energy_force_d2(atoms)
        nn_energy_d1, nn_forces_d1 = self.nn_energy_force_d1(atoms)
        nn_energy_d2, nn_forces_d2 = self.nn_energy_force_d2(atoms)
        
        diabat1_energy = ff_energy_d1 + nn_energy_d1
        diabat2_energy = ff_energy_d2 + nn_energy_d2 + self.shift
        diabat1_forces = ff_forces_d1 + nn_forces_d1
        diabat2_forces = ff_forces_d2 + nn_forces_d2

        inp = self.converter(atoms)
        off_diag = self.off_diag(inp)

        h12_energy = off_diag['energy'].detach().cpu().numpy()[0][0]
        h12_forces = off_diag['forces'].detach().cpu().numpy()[0]

        energy, ci = self.diagonalize(diabat1_energy, diabat2_energy, h12_energy)
        forces = self.calculate_forces(diabat1_forces, diabat2_forces, h12_forces, ci)
        
        result["energy"] = energy.reshape(-1) * self.energy_units
        result["forces"] = forces.reshape((len(atoms), 3)) * self.forces_units

        self.results = result

class ASE_MD:
    """
    Setups and runs the MD simulation. Serves as an interface to the EVB Hamiltonian class and ASE.
    """
    def __init__(self, ase_atoms, tmp, calc_omm_d1, calc_omm_d2, calc_nn_d1, calc_nn_d2, off_diag, shift=0):
        """
        Parameters
        -----------
        ase_atoms : str
            Location of input structure, gets created to ASE Atoms object.
        tmp : str
            Location for tmp directory.
        calc_omm_d1 : Object
            Contains OpenMM force field for diabat 1.
        calc_omm_d2 : Object
            Contains OpenMM force field for diabat 2.
        calc_nn_d1 : Object
            Diabat_NN instance for diabat 1.
        calc_nn_d2 : Object
            Diabat NN instance for diabat 2.
        off_diag : PyTorch model
            Model for predicting H12 energy and forces.
        shift : float
            Diabat 2 energy shift to diabat 1 reference.
        """

        self.tmp = tmp
        if not os.path.isdir(self.tmp):
            os.makedirs(self.tmp)
        
        self.mol = read(ase_atoms)
        calculator = EVB_Hamiltonian(calc_omm_d1, calc_omm_d2, calc_nn_d1, calc_nn_d2, off_diag, shift)
        self.mol.set_calculator(calculator)
        self.md = None

    def create_system(self, name, time_step=1.0, temp=300, temp_init=None, restart=False, store=1, nvt=False, friction=0.001):
        """
        Parameters
        -----------
        name : str
            Name for output files.
        time_step : float, optional
            Time step in fs for simulation.
        temp : float, optional
            Temperature in K for NVT simulation.
        temp_init : float, optional
            Optional different temperature for initialization than thermostate set at.
        restart : bool, optional
            Determines whether simulation is restarted or not, 
            determines whether new velocities are initialized.
        store : int, optional
            Frequency at which output is written to log files.
        nvt : bool, optional
            Determines whether to run NVT simulation, default is False.
        friction : float, optional
            friction coefficient in fs^-1 for Langevin integrator
        """
        if temp_init is None: temp_init = temp
        if not self.md or restart:
            MaxwellBoltzmannDistribution(self.mol, temp_init * units.kB)
        
        if not nvt:
            self.md = VelocityVerlet(self.mol, time_step * units.fs)
        else:
            self.md = Langevin(self.mol, time_step * units.fs, temp * units.kB, friction/units.fs)

        logfile = os.path.join(self.tmp, "{}.log".format(name))
        trajfile = os.path.join(self.tmp, "{}.traj".format(name))

        logger = MDLogger(self.md, self.mol, logfile, stress=False, peratom=False, header=True, mode="a")
        trajectory = Trajectory(trajfile, "w", self.mol)
        self.md.attach(logger, interval=store)
        self.md.attach(trajectory.write, interval=store)

    def write_mol(self, name, ftype="xyz", append=False):
        """
        Write out current molecule structure.
        Parameters
        -----------
        name : str
            Name of the output file.
        ftype : str, optional
            Determines output file format, default xyz.
        append : bool, optional
            Append to existing output file or not.
        """
        path = os.path.join(self.tmp, "{}.{}".format(name, ftype))
        write(path, self.mol, format=ftype, append=append)

    def run_md(self, steps):
        """
        Run MD simulation.
        Parameters
        -----------
        steps : int
            Number of MD steps
        """
        self.md.run(steps)

