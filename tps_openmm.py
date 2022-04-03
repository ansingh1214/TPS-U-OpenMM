import numpy as np
from netCDF4 import Dataset
from netCDF4 import stringtochar

class NC:
    def __init__(self,mode,name,trajs=None,frames=None,natoms=None,dt=None,dim=None,topol=None,init_traj=None):
        if mode == 'w':
            self.root = Dataset(name,'w',format='NETCDF4')
            self.root.createDimension('moves',)
            self.root.createDimension('trajectories',)
            self.root.createDimension('frames',frames)
            self.root.createDimension('atoms',natoms)
            self.root.createDimension('spatial',3)
            self.root.createDimension('nchars',len(topol))
            self.root.createDimension('moveinfo',3)
            self.root.createVariable('ntraj','i4')
            self.root.createVariable('coordinates','f4',("trajectories","frames","atoms","spatial",))
            self.root.createVariable('velocities','f4',("frames","atoms","spatial",))
            self.root.createVariable('dt','i',)
            self.root.createVariable('dimensions','f4',('spatial','spatial',))
            self.root.createVariable('current_step','i')
            self.root.createVariable('attempts','i')
            self.root.createVariable('topol_file','S1',('nchars',))
            self.root.createVariable('move','i',('moves','moveinfo',))
            self.root.createVariable('y','f4',('moves',))
            self.root.variables['ntraj'][0]      = trajs
            self.root.variables['dimensions'][:] = dim
            self.root.variables['topol_file'][:] = stringtochar(np.array([topol],dtype='S'))
            self.root.variables['topol_file']._Encoding = 'ascii'
            self.root.variables['dt'][0] = dt
            self.root.variables['attempts'][0] = 0
            self.root.variables['current_step'][0] = 0
            if init_traj is not None:   
                init = Dataset(init_traj,'r')
                self.root.variables['coordinates'][0,:] = init.variables['coordinates'][:]
                self.root.variables['velocities'][:] = init.variables['velocities'][:]
                self.root.variables['current_step'][0] = 1
                self.root.variables['attempts'][0] = 1
                self.root.variables['move'][0] = np.array([-1,-1,-1])
                init.close()
        if mode == 'r':
            self.root = Dataset(name,'r')
        if mode == 'a':
            self.root = Dataset(name,'a')
            if trajs is not None:
                self.root.variables['ntraj'][0] = trajs


class TPS:
    def __init__(self,root,funcs,engine):
        self.root   = root
        self.funcs  = funcs
        self.engine = engine

    def uniform_point_selector(self,shift=False,shiftBack=False):
        N_frames = self.root.dimensions['frames'].size
        curr     = self.root.variables['current_step']
        rand_traj= np.random.randint(curr)
        if shift:
            coords   = self.root.variables['coordinates'][rand_traj]
            if shiftBack:
                indicator = self.funcs.hB
            else:
                indicator = self.funcs.hA
            rand_f   = np.random.randint(N_frames)
            if not indicator(coords[rand_f]):
                return None, None

        else:
            rand_f   = np.random.randint(N_frames)
        return rand_traj, rand_f


    def C_point_selector(self):
        N_frames = self.root.dimensions['frames'].size
        curr     = self.root.variables['current_step'][0]
        coords   = self.root.variables['coordinates'][curr-1]
        C_ind    = self.funcs.C_states(coords)
        rand_ind = np.random.randint(C_ind.shape[0])
        rand_f   = C_ind[rand_ind]
        return curr-1, rand_f, C_ind.shape[0] 
    
    def C_acceptor(self,traj,N_prev):
        N    = self.funcs.C_states(traj).shape[0]
        rand = np.random.random()
        if(N_prev/N > 1):
            return True
        else:
            if rand < N_prev/N:
                return True
        return False

    def C_U_acceptor(self, y, y_prev, N=None, N_prev=None):
        py    = self.funcs.tps_u_bias(y,y_prev)
        if N_prev is None or N is None:
            pN    = 1
        else:
            pN    = N_prev/N

        p    = pN * py
        rand = np.random.random()
        flag = False
        if(p > 1):
            flag = True
        else:
            if rand < p:
                flag = True
        print('N = {} | N_prev = {} | y = {:1.3f} | y_prev = {:1.3f} | py = {:1.3f} | p = {:.3f} | {}'.format(N,N_prev,y,y_prev,py,p,flag))
        return flag
        
    def forward_shooting(self,traj,frame,random_vel):
        root  = self.root
        shape = root.variables['coordinates'][0].shape
        dt    = int(root.variables['dt'][0])
        topol = str(root.variables['topol_file'][:])
        dim   = root.variables['dimensions'][:]
        nsteps= int((shape[0] - frame - 1)*dt)
        if nsteps == 0:
            return None, None
        temp_conf   = np.zeros(shape)
        temp_vels   = np.zeros(shape)
        temp_conf[:frame+1,:] = root.variables['coordinates'][traj][:frame+1]
        temp_vels[:frame+1,:] = root.variables['velocities'][traj][:frame+1]
        run_data = self.engine(topol,dim,temp_conf[frame],temp_vels[frame],nsteps,dt,random_vel)
        temp_conf[frame+1:,:] = run_data[0]
        temp_vels[frame+1:,:] = run_data[1]
        return temp_conf, temp_vels 

    def backward_shooting(self, traj, frame, random_vel):
        root  = self.root
        shape = root.variables['coordinates'][0].shape
        dt    = int(root.variables['dt'][0])
        topol = str(root.variables['topol_file'][:])
        dim   = root.variables['dimensions'][:]
        #only for reverse frames for consistency
        frame = shape[0] - frame - 1
        nsteps= int((shape[0] - frame - 1)*dt)
        if nsteps == 0:
            return None, None
        temp_conf   = np.zeros(shape)
        temp_vels   = np.zeros(shape)
        temp_conf[:frame+1,:] = np.flip(root.variables['coordinates'][traj],axis=0)[:frame+1]
        temp_vels[:frame+1,:] = -np.flip(root.variables['velocities'][traj],axis=0)[:frame+1]
        run_data = self.engine(topol,dim,temp_conf[frame],temp_vels[frame],nsteps,dt,random_vel)
        temp_conf[frame+1:,:] = run_data[0]
        temp_vels[frame+1:,:] = run_data[1]
        return np.flip(temp_conf,axis=0),-np.flip(temp_vels,axis=0)

    def two_sided_shooting(self, traj, frame):
        root  = self.root
        shape = root.variables['coordinates'][0].shape
        dt    = int(root.variables['dt'][0])
        topol = str(root.variables['topol_file'][:])
        dim   = root.variables['dimensions'][:]
        nsteps_f= int((shape[0] - frame - 1)*dt)
        nsteps_b= shape[0]*dt - nsteps_f
        if nsteps_f == 0 or nsteps_b == 0:
            return None, None
 
        temp_conf   = np.zeros(shape)
        temp_vels   = np.zeros(shape)
        conf        = root.variables['coordinates'][traj][frame]
        vels        = root.variables['velocities'][frame]
        forward     = self.engine(topol,dim,conf,vels,nsteps_f,dt,True)
        flip = False
        if self.funcs.hA(forward[0][-1]):
            indicator = self.funcs.hB
            flip = True
        elif self.funcs.hB(forward[0][-1]):
            indicator = self.funcs.hA
        else:
            return None, None
        
        backward    = self.engine(topol,dim,forward[0][0],-forward[1][0],nsteps_b,dt,False)
        if not indicator(backward[0][-1]):
            return None, None

        temp_conf[frame+1:,:] = forward[0]
        temp_vels[frame+1:,:] = forward[1]
        temp_conf[:frame+1,:] = np.flip(backward[0],axis=0)
        temp_vels[:frame+1,:] = -np.flip(backward[1],axis=0)
        if flip:
            temp_conf =  np.flip(temp_conf,axis=0)
            temp_vels = -np.flip(temp_vels,axis=0)
        return temp_conf, temp_vels 
                
    def forward_shifting(self, traj, frame):
        root  = self.root
        shape = root.variables['coordinates'][0].shape
        dt    = int(root.variables['dt'][0])
        topol = str(root.variables['topol_file'][:])
        dim   = root.variables['dimensions'][:]
        nsteps= int(frame*dt)
        if nsteps == 0:
            return None, None
        temp_conf   = np.zeros(shape)
        temp_vels   = np.zeros(shape)
        temp_conf[:shape[0]-frame] = root.variables['coordinates'][traj][frame:]
        temp_vels[:shape[0]-frame] = root.variables['velocities'][traj][frame:]
        run_data = self.engine(topol,dim,temp_conf[shape[0]-frame-1],temp_vels[shape[0]-frame-1],nsteps,dt)
        temp_conf[shape[0]-frame:,:] = run_data[0]
        temp_vels[shape[0]-frame:,:] = run_data[1]
        return temp_conf,temp_vels

    def backward_shifting(self, traj, frame):
        root  = self.root
        shape = root.variables['coordinates'][0].shape
        dt    = int(root.variables['dt'][0])
        topol = str(root.variables['topol_file'][:])
        dim   = root.variables['dimensions'][:]
        #only for reverse frames for consistency
        frame = shape[0] - frame - 1
        nsteps= int(frame*dt)
        if nsteps == 0:
            return None, None
        temp_conf   = np.zeros(shape)
        temp_vels   = np.zeros(shape)
        temp_conf[:shape[0]-frame] = np.flip(root.variables['coordinates'][traj],axis=0)[frame:]
        temp_vels[:shape[0]-frame] = -np.flip(root.variables['velocities'][traj],axis=0)[frame:]
        run_data = self.engine(topol,dim,temp_conf[shape[0]-frame-1],temp_vels[shape[0]-frame-1],nsteps,dt)
        temp_conf[shape[0]-frame:,:] = run_data[0]
        temp_vels[shape[0]-frame:,:] = run_data[1]
        return np.flip(temp_conf,axis=0),-np.flip(temp_vels,axis=0)


