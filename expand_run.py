import os
import numpy as np
from shutil import copyfile

def build_dataset_dict(param):
    flds = {}

#<HDF5 dataset "BFLD": shape (205, 105, 3), type "<f4">
#<HDF5 dataset "EFLD": shape (205, 105, 3), type "<f4">
#<HDF5 dataset "SP01": shape (80109, 6), type "<f4">
#<HDF5 dataset "SP01INDEX": shape (80109, 2), type "<i4">
#<HDF5 dataset "SP01INJECTOR": shape (1,), type "<f4">
    nn = np.array(param['node_number'])
    ncells = np.array(param['ncells'])
    nprocs = nn[1]*nn[0]
    nx,ny = ncells/nn
# Note we are assuming that the number of grid point ons a processor
# is constant for the expansion

    nsh = (ncells[1]/nn[1] + 5, ncells[0]/nn[0] + 5, 1)

    flds['BFLD'] = np.concatenate((np.ones(nsh), np.zeros(nsh), 
                                   np.zeros(nsh)), axis=2).astype('<f4')

    vdrift = param['vdrift']
    vth = param['vth']
    flds['EFLD'] = np.cross(flds['BFLD'], vdrift).astype('<f4')

    npart = nx*ny*param['num_par'][0]*param['num_par'][1]

#SP01 -> Npart: X, Y, Vx, Vy, Vz, Z=.25?
    flds['SP01'] = np.concatenate((np.random.rand(npart, 1) - .5,
                                   np.random.rand(npart, 1) - .5,
                                   vth*np.random.randn(npart, 1) + vdrift[0],
                                   vth*np.random.randn(npart, 1) + vdrift[1],
                                   vth*np.random.randn(npart, 1) + vdrift[2],
                                   np.zeros((npart, 1)) + .25), 
                                   axis=1).astype('<f4')

    flds['SP01INDEX'] = np.concatenate((np.random.randint(4,4+nx, size=(npart,1)),
                                        np.random.randint(4,4+ny, size=(npart,1))),
                                        axis=1).astype('<i4')

    flds['SP01INJECTOR'] = array([param['planepos']]).astype('<f4')

    return flds

#======================================================================

def build_attrs(path):
    attrs = {}
    with h5py.File(path,'r') as f:
        for k,v in f.attrs.iteritems():
            attrs[k] = v
    return attrs

#======================================================================

def build_maps(old_param, new_param):
    nn = np.array(old_param['node_number'])
    new_nn = np.array(new_param['node_number'])

    nprocs = new_nn[0]*new_nn[1]
    ctrng = np.arange(nprocs)

    #new_proc_map = {c:None for c in np.arange(nprocs)}
    new_proc_map = []
    inv_proc_map = {}

    for c,(x,y) in enumerate(zip(ctrng%new_nn[0], ctrng/new_nn[0])):
        if x < nn[0]:
            new_proc_map.append(x + y*nn[0])
            inv_proc_map[x + y*nn[0]] = c
        else: 
            new_proc_map.append(-1) #generate new distro


    wny = lambda n: n/new_nn[0]
    neighbor_map = (lambda n: -1 if n%new_nn[0] == 0 else (n - 1),
                    lambda n: (wny(n) - 1)%nprocs + n%new_nn[0],
                    lambda n: (n + 1) if (n + 1)%new_nn[0] != 0 else -1,
                    lambda n: (wny(n) + 1)%nprocs + n%new_nn[0])

    neighbors = {n:[nm(n) for nm in neighbor_map] for n in range(nprocs)}

    return new_proc_map, inv_proc_map, neighbors

#======================================================================

def is_new_edge(old_param, inv_proc_map, n):
    nn = np.array(old_param['node_number'])

    right_most_procs = [(1+c)*nn[0] - 1 for c in range(nn[1])]
    edge_procs = [inv_proc_map[rp]+1 for rp in right_most_procs]

    return n in edge_procs 

#======================================================================

old_input = '../orig/input/input'
new_input = './input/input'

cur_dir = os.getcwd()
old_dir = '../orig/Restart/'
new_dir = './Restart/'

fname = 'Rest_proc{:05d}.h5'

old_param = read_input(old_input)
new_param = read_input(new_input)

new_proc, inv_proc_map, neighbors = build_maps(old_param, new_param)

_path = os.path.join(old_dir, fname.format(0))
attrs = build_attrs(_path)

print 'Moving into new restart dir',new_dir
os.chdir(new_dir)

for new_id,old_id in enumerate(new_proc):
    new_fname = fname.format(new_id)
    if old_id == -1:
        print "Creating", new_fname
        flds = build_dataset_dict(new_param)
        with h5py.File(new_fname, 'w') as f:
            for k,v in flds.iteritems():
                dset = f.create_dataset(k, v.shape, dtype=v.dtype) 
                f[k][:] = v
            for k,v in attrs.iteritems():
                f.attrs[k] = v
            f.attrs['NPART'] = np.array(f['SP01'].shape[0]).astype('int32')
    else:
        old_fname = os.path.join(cur_dir, old_dir, fname.format(old_id))
# Symbolicly Link 
#   Fast but doesn't seem to work
        #print "linking", old_fname, "to", new_fname
        #os.symlink(old_fname, new_fname)
# Copy
#   Slow but works well?
        print "coppying", old_fname, "to", new_fname
        copyfile(old_fname, new_fname)
# Move!
#   Fast but dangerous!
        #print "movinb", old_fname, "to", new_fname
#        os.rename(old_fname, new_fname)


    with h5py.File(new_fname, 'r+') as f:
        new_inj = np.array([new_param['planepos']]).astype('<f4')
        inj = f['SP01INJECTOR']
        inj[:] = new_inj

# Might not need to do this
        # Do we need to update boundry conditions?
        if is_new_edge(old_param, inv_proc_map, new_id):
            print 'Proc number', new_id,'is a new edge'
            right_BFLD = f['BFLD']
            right_EFLD = f['EFLD']
            with h5py.File(fname.format(new_id-1), 'r') as g:
                left_BFLD = g['BFLD']
                left_EFLD = g['EFLD']
                # [Y, X, comp]
                right_BFLD[:,:5,:] = left_BFLD[:,-5:,:]
                right_EFLD[:,:5,:] = left_EFLD[:,-5:,:]

os.chdir(cur_dir)

# just checking
#os.chdir(new_dir)
#for new_id,old_id in enumerate(new_proc):
#    with h5py.File(new_fname.format(c), 'r') as f:
#       print f['SP01INJECTOR'][:]
#os.chdir(cur_dir)

sys.exit()

##Match boundrys
#right_most_procs = [(1+c)*nn[0] - 1 for c in range(nn[1])]
#for pc in right_most_procs:
#    inv_proc_map[pc]



#for c in ctrng:
#    with h5py.File(new_fname.format(c), 'r') as f:
#        inj = f['SP01INJECTOR']
#        print inj[:]

#print new_proc_map

#                                          ny,  nx, xyz
# f['BFLD'] = <HDF5 dataset "BFLD": shape (105, 205, 3), type "<f4">
# f['EFLD'] = <HDF5 dataset "EFLD": shape (105, 205, 3), type "<f4">

#                                         nparts, x,y,vx,vy,vz,z 
# f['SP01'] = <HDF5 dataset "SP01": shape (271041, 6), type "<f4">

# f['SP01INDEX'] = <HDF5 dataset "SP01INDEX": shape (271041, 2), type "<i4">
# f['SP01INJECTOR'] = <HDF5 dataset "SP01INJECTOR": shape (1,), type "<f4">

def viz_proc_layout(nnp):
    vizstr = '[{:=4d}, {:=4d}, ..., {:=4d}]'

    _tr =  arange(nnp[0]*nnp[1]).reshape(nnp[::-1])[::-1,:]

    for _t in _tr:
        print vizstr.format(*_t[[0,1,-1]])

