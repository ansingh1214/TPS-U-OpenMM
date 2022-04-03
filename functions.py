#Set of Functions unique to the system that define the states and CV
class Functions:
    def __init__(self, dims, y0):
        self.dims  = np.array([dims[0],dims[1],dims[2],90,90,90])
        self.phi_indices = [4,6,8,14]
        self.psi_indices = [6,8,14,16]
        self.indicatorA = np.array([[35,140],[-100,95]]) * np.pi/180
        self.indicatorB = np.array([[-180, -35],[95,180]]) * np.pi/180
#        self.indicatorC = np.array([[-30,30],[-180,180]])* np.pi/180
#        self.indicatorC = np.array([[-20,20],[-180,180]])* np.pi/180
        self.indicatorC = np.array([[-25,12],[-180,180]])* np.pi/180
        self.water_idx  = np.arange(22,9022,3)
        self.adp_idx    = np.array([1,5,7,9,11,15,17,19]) - 1
        self.adp_mass   = np.array([12,12,14,12,12,12,14,12])
#        self.bias       = 200
        self.bias       = 100
        self.y0         = y0

    def ramachandran(self,conf):
        if np.ndim(conf) == 2:
            conf = np.expand_dims(conf,0)

        phi_atoms = np.transpose(conf[:,self.phi_indices,:],(1,0,2))
        psi_atoms = np.transpose(conf[:,self.psi_indices,:],(1,0,2))
        phi = calc_dihedrals(phi_atoms[0],phi_atoms[1],phi_atoms[2],
                             phi_atoms[3],self.dims)
        psi = calc_dihedrals(psi_atoms[0],psi_atoms[1],psi_atoms[2],
                             psi_atoms[3],self.dims)
        rama = np.array([phi,psi])
        return rama
    
    def hA(self, conf):
        rama = self.ramachandran(conf)
        phi = rama[0,0]
        psi = rama[1,0]
        flag = True
        if phi < self.indicatorA[0][0] or phi > self.indicatorA[0][1]:
            flag = False
        if psi < self.indicatorA[1][0] or psi > self.indicatorA[1][1]:
            flag = False
        return flag
        
    def hB(self, conf):
        rama = self.ramachandran(conf)
        phi = rama[0,0]
        psi = rama[1,0]
        flag = True
        if phi < self.indicatorB[0][0] or phi > self.indicatorB[0][1]:
            flag = False
        if psi < self.indicatorB[1][0] or psi > self.indicatorB[1][1]:
            flag = False
        return flag
    
    def hC(self, conf):
        rama = self.ramachandran(conf)
        phi = rama[0,0]
        psi = rama[1,0]
        flag = True
        if phi < self.indicatorC[0][0] or phi > self.indicatorC[0][1]:
            flag = False
        if psi < self.indicatorC[1][0] or psi > self.indicatorC[1][1]:
            flag = False
        return flag
    
    def C_states(self,conf):
        rama = self.ramachandran(conf)
        temp = []
        for i in range(conf.shape[0]):
            if self.hC(conf[i]):
                temp.append(i)
        return np.array(temp)

    def get_y(self, conf, y_frame):
        adp_com   = np.average(conf[y_frame, self.adp_idx, 2], weights=self.adp_mass) 
        water_com = np.mean(conf[y_frame, self.water_idx, 2])
        return np.abs(adp_com - water_com)

    def tps_u_bias(self, y, y_prev):
        return np.exp(-self.bias*(y-self.y0)**2 + self.bias*(y_prev - self.y0)**2)


