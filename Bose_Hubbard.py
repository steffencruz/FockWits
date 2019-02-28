
from FockWit import *


if __name__=='__main__':

    tmax = 4
    k=100
    n=30
    J=100.
    U=100.
    m = TwoQMode(2)

    v0 = np.zeros(m.n_dim)
    # encoding for state |0> |2>
    v0[12]=1

    print('Preparing Matrix')
    BHM = m.BHM(tmax=tmax,k=k,n=n,J=J,U=U)
    print('Complete')
    fig,ax = plt.subplots(1,1,figsize=(10,7))
    labels = ['|{:04b}>'.format(i) for i in range(m.n_dim)]

    s = range(m.n_dim)
    p0 = plt.plot(s,v0,'ko-',ms=3,lw=1,label='Ref [T=0]')
    pt = None
    plt.legend()

    plt.xticks(s,labels,rotation=80,fontsize=7)
    plt.xlabel('Configuration')
    plt.ylabel('Probability')
    plt.title('Results for Bose–Hubbard with {} Qubits representing each mode'.format(m.mode.n_qubits_per_mode))
    for t,mat in sorted(BHM.items()):

        vt = np.matmul(mat,v0)
        vt = vt*vt.conj()
        if pt is None:
            pt = plt.plot(s,vt,'o-',ms=3,lw=1,alpha=0.4,label='T={}'.format(round(t,3)))
        else:
            pt[0].set_data(s,vt)
            pt[0].set_label('T={}'.format(round(t,3)))

        plt.pause(0.1)
