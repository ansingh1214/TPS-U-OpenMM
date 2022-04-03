import numpy as np
from netCDF4 import Dataset
from netCDF4 import stringtochar
from simtk.unit import *
from simtk.openmm.app import *
from simtk.openmm import *
from sys import stdout
from tqdm import tqdm
import time
import matplotlib.pyplot as plt
import MDAnalysis
from MDAnalysis.lib.distances import calc_dihedrals
from openmmtools.integrators import VVVRIntegrator
from parmed.openmm.reporters import NetCDFReporter as Reporter
import sys
from tps_openmm import *
from functions import Functions
import os.path
from os import path
def engine(topol,dims,conf,vels,nsteps,dt,random_vel=False):
    top = GromacsTopFile(topol, periodicBoxVectors=dims)
    system = top.createSystem(nonbondedMethod=PME, nonbondedCutoff=1.4*nanometer,rigidWater=True,ewaldErrorTolerance=0.0005)
    platform = Platform.getPlatformByName('CUDA')
    integrator = VVVRIntegrator(300*kelvin, 1/picosecond, 0.002*picoseconds)
    integrator.setConstraintTolerance(0.00001)
    simulation = Simulation(top.topology, system, integrator,platform)
    simulation.context.setPositions(conf)
    simulation.context.setVelocitiesToTemperature(300)

    if random_vel:
        temp_vel = simulation.context.getState(getVelocities=True).getVelocities(asNumpy=True) * picosecond / nanometer
        temp_vel[22:] = vels[22:]
        simulation.context.setVelocities(temp_vel)
    else:
        simulation.context.setVelocities(vels)

    trj = 'tmp/' + '{:.2f}'.format(float(sys.argv[1])) + '.nc'
    reporter = Reporter(trj, dt,vels=True)
    simulation.reporters.append(reporter)
    simulation.step(nsteps)
    #scaling from AMBER to OpenMM units (nm, nm/ps) 
    confs = reporter._out._ncfile.variables['coordinates'][:]*.1
    velocities  = reporter._out._ncfile.variables['velocities'][:]*.1*20.455

    return [confs, velocities]

def tps_engine(root, funcs, tps):
    N_trajs = int(root.variables['ntraj'][0])
    pbar1 = tqdm(desc="Accepted: ",total=N_trajs)
    pbar2 = tqdm(desc="Attempts: ",total=float('inf'))
    pbar1.update(root.variables['current_step'][0])
    choice = [tps.forward_shooting, tps.backward_shooting, tps.forward_shifting, tps.backward_shifting]
    while(root.variables['current_step'][0] < N_trajs):
        rand1 = np.random.randint(6)
        if rand1 < 6:
            traj, frame, N_prev = tps.C_point_selector()
            confs, vels = tps.two_sided_shooting(traj,frame)
            if confs is not None:
                y_prev = funcs.get_y(root.variables['coordinates'][traj],-1)
                y      = funcs.get_y(confs, -1)
                N      = funcs.C_states(confs).shape[0]

                curr = root.variables['current_step'][0]
                moves= root.variables['attempts'][0]
                if tps.C_U_acceptor(y, y_prev, N, N_prev):
                    root.variables['coordinates'][curr] = confs
                    root.variables['velocities'][:]  = vels

                    root.variables['move'][moves] = np.array([1,traj,frame])
                    root.variables['y'][moves] = y
                    root.variables['current_step'][0] += 1
                    if curr % 20 == 0:
                        root.sync()
                    pbar1.update(1)
                
                else:
                    root.variables['move'][moves] = np.array([0,traj,frame])
                    root.variables['y'][moves] = y_prev
            else:
                moves= root.variables['attempts'][0]
                root.variables['move'][moves] = np.array([0,traj,frame])
                    

        else:
            rand = np.random.randint(2)
            if rand == 0:
                traj, frame = tps.uniform_point_selector(shift=True)
            else:
                traj, frame = tps.uniform_point_selector(shift=True,shiftBack=True)
            if traj is not None and frame is not None:
                confs, vels = choice[2+rand](traj,frame)
                if confs is not None:
                    y_prev = funcs.get_y(root.variables['coordinates'][traj],-1)
                    y      = funcs.get_y(confs, -1)
                    if tps.C_U_acceptor(y, y_prev):
                        if funcs.hA(confs[0]) and funcs.hB(confs[-1]):
                            curr = root.variables['current_step'][0]
                            root.variables['coordinates'][curr] = confs
                            root.variables['velocities'][curr]  = vels
                            root.variables['move'][curr] = np.array([2+rand,traj,frame])
                            root.variables['y'][curr] = y
                            root.variables['current_step'][0] += 1
                            if curr % 20 == 0:
                                root.sync()
                        pbar1.update(1)
        root.variables['attempts'][0] += 1
        pbar2.update(1)


if __name__ == "__main__":
    N_trajs  = 2300
    N_frames = 400
    N_atoms  = 9022
    tps_dt   = 5
    y0       = float(sys.argv[1])
    dim      = np.array([3.0,3.0,18.0])
    dimension= np.array([[dim[0],0,0],[0,dim[1],0],[0,0,dim[2]]])
    filename = 'tps_trajs/' + '{:.2f}'.format(y0) + '.nc'
    init     = 'init_trajs/' + '{:.2f}'.format(y0) + '.nc'
    topology = '../alanine/topol.top'
    if path.exists(filename):
        netcdf   = NC('a',filename,N_trajs)
    else:
        netcdf   = NC('w',filename,N_trajs,N_frames,N_atoms,tps_dt,dimension,topology,init)
    
    root     = netcdf.root
    funcs    = Functions(dim,y0)
    print(int(root.variables['current_step'][0]), ' | ', funcs.bias)
    tps      = TPS(root, funcs, engine)
    tps_engine(root, funcs, tps)
    root.close()

