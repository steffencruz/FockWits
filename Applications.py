def BHM(self,tmax,k,J=1,U=0.1):
    #Implement simple two-mode Bose-Hubbard simulation
    j = np.complex(0,1)
    BKR = {}
    for t in np.linspace(0,tmax,10):
        for layer in range(k):
            phi = -j*J*t/k
            r = -U*t/(2*k)
            BS = self.BS(phi)
            K = np.kron(self.mode.K(r),self.mode.K(r))
            R = np.kron(self.mode.R(-r),self.mode.R(-r))
            BK = np.matmul(BS,K)
            BKR[t] = np.matmul(BK,R)
    return BKR