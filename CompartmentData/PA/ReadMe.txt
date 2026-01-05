*eventTrace.npy
Compartment "dose rate" arrays. Every energy event is binned into 0.002 s bins
size is [2,n]
[1,:]  is the number of events/total number of events
[2,:] is the LEFT bin edge (recall bin width = 2ms) 

*DVH.npy
DVH for each compartment [2,n]
[0,:] is the Volume in percentage [100,0] w/ 0.0001 prescision
[1,:] is the dose value in Gy, w/ 0.0025 step size
