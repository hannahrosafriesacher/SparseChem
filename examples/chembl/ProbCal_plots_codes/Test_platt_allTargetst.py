import numpy as np

a=np.array([[0,-1],[0,0],[0,1],[0,0],[-1,-1], [0,-1], [-1,-1],[0,0], [-1,0],[-1,1],[1,0]])
a_Target0=a[:,0]
index0=np.nonzero(a_Target0)
a_target0_nonzero=a_Target0[np.nonzero(a_Target0)]
a_Target1=a[:,1]
index1=np.nonzero(a_Target1)
a_target1_nonzero=a_Target1[np.nonzero(a_Target1)]

list_index=[index0, index1]
index_numpy=np.array(list_index)
print(index_numpy)

#print(a.shape)
#print(a_Target1)
#print(a_target1_nonzero.flatten())
#print(np.nonzero(a_Target1)[0].flatten())

#b=np.zeros_like(a)
#print(b)

#c=np.insert(arr=b[:0],obj=np.nonzero(a_Target0)[0].flatten(), values= a_target0_nonzero.flatten(), axis=1)
#print(c)