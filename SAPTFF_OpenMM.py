from __future__ import print_function
from simtk.openmm.app import *
from simtk.openmm import *
from simtk.unit import *
from sys import stdout
import numpy as np

#**********************************************
# this routine uses OpenMM to evaluate energies of SAPT-FF force field
# we use the DrudeSCFIntegrator to solve for equilibrium Drude positions.
# positions of atoms are frozen by setting mass = 0 in .xml file
#**********************************************


class SAPT_ForceField:
    # Setup OpenMM simulation object with template pdb
    # and xml force field files
    def __init__(self, pdbtemplate, residuexml, saptxml , platformtype='CPU', Drude_hyper_force=True, exclude_monomer_intra=False):

        # load bond definitions before creating pdb object (which calls createStandardBonds() internally upon __init__).  Note that loadBondDefinitions is a static method
        # of Topology, so even though PDBFile creates its own topology object, these bond definitions will be applied...
        Topology().loadBondDefinitions(residuexml)
        self.pdb = PDBFile( pdbtemplate )  # this is used for topology, coordinates are not used...
        self.integrator = DrudeSCFIntegrator(0.00001*picoseconds) # Use the SCF integrator to optimize Drude positions    
        self.modeller = Modeller(self.pdb.topology, self.pdb.positions)
        self.forcefield = ForceField( saptxml )
        self.modeller.addExtraParticles(self.forcefield)  # add Drude oscillators

        # by default, no cutoff is used, so all interactions are computed.  This is what we want for gas phase PES...no Ewald!!
        self.system = self.forcefield.createSystem(self.modeller.topology, constraints=None, rigidWater=True)

        # Obtain a list of real atoms excluding Drude particles for obtaining forces later
        # Also set particle mass to 0 in order to optimize Drude positions without affecting atom positions
        self.realAtoms = []
        for i in range(self.system.getNumParticles()):
            if self.system.getParticleMass(i)/dalton > 1.0:
                self.realAtoms.append(i)
            self.system.setParticleMass(i,0)

        # add "hard wall" hyper force to Drude/parent atoms to prevent divergence with SCF integrator...
        if Drude_hyper_force == True:
            self.add_Drude_hyper_force()

        for i in range(self.system.getNumForces()):
            f = self.system.getForce(i)
            f.setForceGroup(i)

        self.platform = Platform.getPlatformByName(platformtype)
        self.simulation = Simulation(self.modeller.topology, self.system, self.integrator, self.platform)

        # now exclude intra-molecular Non-bonded interactions if requested....
        if exclude_monomer_intra == True:
            self.add_exclusions_monomer_intra()

    #***********************************
    # this method excludes all intra-molecular non-bonded interactions in the system
    # for Parent/Drude interactions, exclusion is replaced with a damped Thole interaction...
    #***********************************
    def add_exclusions_monomer_intra( self ):    

        drudeForce = [f for f in [self.system.getForce(i) for i in range(self.system.getNumForces())] if type(f) == DrudeForce][0]
        nbondedForce = [f for f in [self.system.getForce(i) for i in range(self.system.getNumForces())] if type(f) == NonbondedForce][0]
        customNonbondedForce = [f for f in [self.system.getForce(i) for i in range(self.system.getNumForces())] if type(f) == CustomNonbondedForce][0]
        
        # map from global particle index to drudeforce object index
        particleMap = {}
        for i in range(drudeForce.getNumParticles()):
            particleMap[drudeForce.getParticleParameters(i)[0]] = i

        # can't add duplicate ScreenedPairs, so store what we already have
        flagexceptions = {}
        for i in range(nbondedForce.getNumExceptions()):
            (particle1, particle2, charge, sigma, epsilon) = nbondedForce.getExceptionParameters(i)
            string1=str(particle1)+"_"+str(particle2)
            string2=str(particle2)+"_"+str(particle1)
            flagexceptions[string1]=1
            flagexceptions[string2]=1

        # can't add duplicate customNonbonded exclusions, so store what we already have
        flagexclusions = {}
        for i in range(customNonbondedForce.getNumExclusions()):
            (particle1, particle2) = customNonbondedForce.getExclusionParticles(i)
            string1=str(particle1)+"_"+str(particle2)
            string2=str(particle2)+"_"+str(particle1)
            flagexclusions[string1]=1
            flagexclusions[string2]=1

        print(' adding exclusions ...')

        # add all intra-molecular exclusions, and when a drude pair is
        # excluded add a corresponding screened thole interaction in its place
        for res in self.simulation.topology.residues():
            for i in range(len(res._atoms)-1):
                for j in range(i+1,len(res._atoms)):
                    (indi,indj) = (res._atoms[i].index, res._atoms[j].index)
                    # here it doesn't matter if we already have this, since we pass the "True" flag
                    nbondedForce.addException(indi,indj,0,1,0,True)
                    # make sure we don't already exclude this customnonbond
                    string1=str(indi)+"_"+str(indj)
                    string2=str(indj)+"_"+str(indi)
                    if string1 in flagexclusions or string2 in flagexclusions:
                        continue
                    else:
                        customNonbondedForce.addExclusion(indi,indj)
                    # add thole if we're excluding two drudes
                    if indi in particleMap and indj in particleMap:
                        # make sure we don't already have this screened pair
                        if string1 in flagexceptions or string2 in flagexceptions:
                            continue
                        else:
                            drudei = particleMap[indi]
                            drudej = particleMap[indj]
                            drudeForce.addScreenedPair(drudei, drudej, 2.0)
    
        # now reinitialize to make sure changes are stored in context
        state = self.simulation.context.getState(getEnergy=False,getForces=False,getVelocities=False,getPositions=True)
        positions = state.getPositions()
        self.simulation.context.reinitialize()
        self.simulation.context.setPositions(positions)

    # this method adds a "hard wall" hyper bond force to
    # parent/drude atoms to prevent divergence using the 
    # Drude SCFIntegrator ...
    def add_Drude_hyper_force( self ):
        drudeForce = [f for f in [self.system.getForce(i) for i in range(self.system.getNumForces())] if type(f) == DrudeForce][0]

        hyper = CustomBondForce('step(r-rhyper)*((r-rhyper)*khyper)^powh')
        hyper.addGlobalParameter('khyper', 100.0)
        hyper.addGlobalParameter('rhyper', 0.02)
        hyper.addGlobalParameter('powh', 6)
        self.system.addForce(hyper)

        for i in range(drudeForce.getNumParticles()):
            param = drudeForce.getParticleParameters(i)
            drude = param[0]
            parent = param[1]
            hyper.addBond(drude, parent)

    def res_list(self):
        #Return dictionary of indices for each monomer in the dimer. 
        #Used for diababt intramolecular neural networks
        res_dict = {}
        k = 0
        for res in self.pdb.topology.residues():
            res_dict[k] = []
            for i in range(len(res._atoms)):
                res_dict[k].append(res._atoms[i].index)
            k+=1
        return res_dict

    def set_xyz(self, xyz):
        self.xyz_pos = xyz
        for i in range(len(self.xyz_pos)):
            # update pdb positions
            self.pdb.positions[i] = Vec3(self.xyz_pos[i][0]/10 , self.xyz_pos[i][1]/10, self.xyz_pos[i][2]/10)*nanometer
        # now update positions in modeller object
        self.modeller = Modeller(self.pdb.topology, self.pdb.positions)
        # add dummy site and shell initial positions
        self.modeller.addExtraParticles(self.forcefield)
        return self.modeller.positions 
    
    # Compute the energy for a particular configuration
    # the input xyz array should follow the same structure
    # as the template pdb file
    def compute_energy( self , xyz ): 

        # set positions in simulation context
        self.simulation.context.setPositions(xyz)

        # integrate one step to optimize Drude positions.  Note that atoms won't move if masses are set to zero
        self.simulation.step(1)

        # get energy
        #state = self.simulation.context.getState(getEnergy=True,getForces=True,getPositions=True)
        #eSAPTFF = state.getPotentialEnergy()

        # if you want energy decomposition, uncomment these lines...
        #for j in range(self.system.getNumForces()):
        #    f = self.system.getForce(j)
        #    print(type(f), str(self.simulation.context.getState(getEnergy=True, groups=2**j).getPotentialEnergy()))
        eSAPTFF = 0
        SAPTFF_forces = np.zeros_like(self.xyz_pos)
        for j in range(self.system.getNumForces()):
            f = self.system.getForce(j)
            if f.__class__.__name__ in ["NonbondedForce", "CustomNonbondedForce", "DrudeForce"]:
                energy = self.simulation.context.getState(getEnergy=True, groups=2**j).getPotentialEnergy()
                eSAPTFF += energy/kilojoule_per_mole
                forces = self.simulation.context.getState(getForces=True, groups=2**j).getForces(asNumpy=True)[self.realAtoms]
                SAPTFF_forces += forces/kilojoule_per_mole*nanometers
        return eSAPTFF, SAPTFF_forces/10.0


