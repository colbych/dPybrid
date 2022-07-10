#import future
import os
import h5py
import numpy as np
import matplotlib as mpl 
import matplotlib.pyplot as plt 
from matplotlib.colors import LogNorm
from scipy.ndimage import gaussian_filter as gf
import pdb

phase_vars = 'p1x1 p2x1 p3x1 ptx1 etx1'.split()

#======================================================================

def qloader(num=None, path='./'):
    import glob

    if path[-1] != '/': path = path + '/'
    
    bpath = path+"Output/Fields/Magnetic/Total/{}/Bfld_{}.h5"
    choices = glob.glob(bpath.format('x', '*'))
    choices = [int(c[-11:-3]) for c in choices]
    choices.sort()

    dpath = path+"Output/Phase/*"
    dens_vars = [c[len(dpath)-1:] for c in glob.glob(dpath)]

    bpath = bpath.format('{}', '{:08d}')
    epath = path+"Output/Fields/Electric/Total/{}/Efld_{:08d}.h5"
    dpath = path+"Output/Phase/{}/Sp01/dens_sp01_{:08d}.h5"


    while num not in choices:
        _ =  'Select from the following possible movie numbers: '\
             '\n{0} '.format(choices)
        num = int(input(_))

    d = {}
    for k in 'xyz':
        print(bpath.format(k, num))
        print(epath.format(k, num))
        with h5py.File(bpath.format(k,num),'r') as f:
            d['b'+k] = f['DATA'][:]

            if 'xx' not in d:
                _N2,_N1 = f['DATA'][:].shape #python is fliped
                x1,x2 = f['AXIS']['X1 AXIS'][:],f['AXIS']['X2 AXIS'][:]
                dx1 = (x1[1]-x1[0])/_N1
                dx2 = (x2[1]-x2[0])/_N2

                d['xx'] = dx1*np.arange(_N1) + dx1/2. + x1[0]
                d['yy'] = dx2*np.arange(_N2) + dx2/2. + x2[0]

        try:
            with h5py.File(epath.format(k,num),'r') as f:
                d['e'+k] = f['DATA'][:]
        except:
            pass

        d['b'+k+'_xx'] = d['xx']
        d['e'+k+'_yy'] = d['yy']
     
    for k in dens_vars:
        print(dpath.format(k,num))
        with h5py.File(dpath.format(k,num),'r') as f:
            d[k] = f['DATA'][:]

            _N2,_N1 = f['DATA'][:].shape #python is fliped
            x1,x2 = f['AXIS']['X1 AXIS'][:],f['AXIS']['X2 AXIS'][:]
            dx1 = (x1[1]-x1[0])/_N1
            dx2 = (x2[1]-x2[0])/_N2
            d[k+'_xx'] = dx1*np.arange(_N1) + dx1/2. + x1[0]
            d[k+'_yy'] = dx2*np.arange(_N2) + dx2/2. + x2[0]

    return d

#======================================================================

def get_output_times(path='./', sp=1, output_type='Phase'):
    import glob
    phase_vars = 'p1x1 p2x1 p3x1 ptx1 etx1'.split()
    
    if output_type.lower() == 'phase':
        _fn = "Output/Phase/{var}/Sp{sp:02d}/dens_sp{sp:02d}_*.h5"
    elif output_type.lower() == 'raw':
        _fn = "Output/Raw/Sp{sp:02d}/raw_sp{sp:02d}_*.h5"
    elif output_type.lower() == 'field':
        _fn = "Output/Fields/Magnetic/Total/x/Bfld_*.h5"
    elif output_type.lower() == 'flow':
        _fn = "Output/Phase/FluidVel/Sp{sp:02d}/x/Vfld_*.h5"
    elif output_type.lower() == 'pres':
        _fn = "Output/Phase/PressureTen/Sp{sp:02d}/x/Pfld_*.h5"
    else:
        raise TypeError

    for _pv in phase_vars:
        fname =  _fn.format(var=_pv, sp=sp)
        dpath = os.path.join(path, fname)
        choices = glob.glob(dpath)
        choices = [int(c[-11:-3]) for c in choices]
        choices.sort()

        if len(choices) > 0:
            return np.array(choices)

    print("No files found in path: {}".format(_fn.format(var=_pv, sp=sp)))
    raise FileNotFoundError

#======================================================================

def dens_loader(dens_vars=None, num=None, path='./', sp=1, verbose=False):
    import glob

    if path[-1] != '/': path = path + '/'

    choices = get_output_times(path=path, sp=sp)
    
    dpath = path+"Output/Phase/*"
    if dens_vars is None:
        dens_vars = [c[len(dpath)-1:] for c in glob.glob(dpath)]
    else:
        if not type(dens_vars) in (list, tuple):
            dens_vars = [dens_vars]

    for _k in 'FluidVel PressureTen'.split():
        if _k in dens_vars:
            dens_vars.pop(dens_vars.index(_k ))

    print(dens_vars)
    dens_vars.sort()

    dpath = path+"Output/Phase/{dv}/Sp{sp:02d}/dens_sp{sp:02d}_{tm}.h5"
    
    if verbose: print(dpath.format(dv=dens_vars[0], sp=sp, tm='*'))


    dpath = path+"Output/Phase/{dv}/Sp{sp:02d}/dens_sp{sp:02d}_{tm:08}.h5"

    d = {}
    while num not in choices:
        _ =  'Select from the following possible movie numbers: '\
             '\n{0} '.format(choices)
        num = int(input(_))

    for k in dens_vars:
        if verbose: print(dpath.format(dv=k, sp=sp, tm=num))
        with h5py.File(dpath.format(dv=k,sp=sp,tm=num),'r') as f:
            d[k] = f['DATA'][:]

            _ = f['DATA'].shape #python is fliped
            if len(_) < 3:
                _N2,_N1 = _
                x1,x2 = f['AXIS']['X1 AXIS'][:], f['AXIS']['X2 AXIS'][:]
                dx1 = (x1[1]-x1[0])/_N1
                dx2 = (x2[1]-x2[0])/_N2
                d[k+'_xx'] = dx1*np.arange(_N1) + dx1/2. + x1[0]
                d[k+'_yy'] = dx2*np.arange(_N2) + dx2/2. + x2[0]

            else:
                _N3,_N2,_N1 = _
                x1 = f['AXIS']['X1 AXIS'][:]
                x2 = f['AXIS']['X2 AXIS'][:]
                x3 = f['AXIS']['X3 AXIS'][:]
                dx1 = (x1[1]-x1[0])/_N1
                dx2 = (x2[1]-x2[0])/_N2
                dx3 = (x3[1]-x3[0])/_N3
                d[k+'_xx'] = dx1*np.arange(_N1) + dx1/2. + x1[0]
                d[k+'_yy'] = dx2*np.arange(_N2) + dx2/2. + x2[0]
                d[k+'_zz'] = dx3*np.arange(_N3) + dx3/2. + x3[0]

            if k == 'etx1':
                d['etx1_yy'] = np.exp(d['etx1_yy'])

    _id = "{}:{}:{}".format(os.path.abspath(path), num, "".join(dens_vars))
    d['id'] = _id
    return d

#======================================================================

def raw_loader(dens_vars=None, num=None, path='./', sp=1):
    import glob

    if path[-1] != '/': path = path + '/'

    choices = get_output_times(path=path, sp=sp, output_type='Raw')
    dpath = path+"Output/Raw/Sp{sp:02d}/raw_sp{sp:02d}_{tm:08}.h5"

    d = {}
    while num not in choices:
        _ =  'Select from the following possible movie numbers: '\
             '\n{0} '.format(choices)
        num = int(input(_))

    if type(dens_vars) is str:
        dens_vars = dens_vars.split()
    elif dens_vars is None:
        dens_vars = 'p1 p2 p3 q tag x1 x2'.split()
    print(dpath.format(sp=sp, tm=num))
    with h5py.File(dpath.format(sp=sp,tm=num),'r') as f:
        for k in dens_vars:
            d[k] = f[k][:]

    return d

#======================================================================

def pres_loader(pres_vars=None, num=None, path='./', sp=1, verbose=False):
    import glob

    if path[-1] != '/': path = path + '/'

    choices = get_output_times(path=path, sp=sp, output_type='pres')
    dpath = path+"Output/Phase/PressureTen/Sp{sp:02d}/{dv}/Pfld_{tm:08}.h5"

    d = {}
    while num not in choices:
        _ =  'Select from the following possible movie numbers: '\
             '\n{0} '.format(choices)
        num = int(input(_))

    if type(pres_vars) is str:
        pres_vars = pres_vars.split()
    elif pres_vars is None:
        pres_vars = 'xx yy zz xy yz zx x y z'.split()
    #print(dpath.format(sp=sp, tm=num))

    for k in pres_vars:
        if verbose: print(dpath.format(sp=sp, dv=k, tm=num))

        with h5py.File(dpath.format(sp=sp, dv=k, tm=num),'r') as f:
            kc = 'p'+k
            _ = f['DATA'].shape #python is fliped
            dim = len(_)
            print(kc,_)
            d[kc] = f['DATA'][:]
            if dim < 3:
                _N2,_N1 = _
                x1,x2 = f['AXIS']['X1 AXIS'][:], f['AXIS']['X2 AXIS'][:]
                dx1 = (x1[1]-x1[0])/_N1
                dx2 = (x2[1]-x2[0])/_N2
                d[kc+'_xx'] = dx1*np.arange(_N1) + dx1/2. + x1[0]
                d[kc+'_yy'] = dx2*np.arange(_N2) + dx2/2. + x2[0]

            else:
                _N3,_N2,_N1 = _
                x1 = f['AXIS']['X1 AXIS'][:]
                x2 = f['AXIS']['X2 AXIS'][:]
                x3 = f['AXIS']['X3 AXIS'][:]
                dx1 = (x1[1]-x1[0])/_N1
                dx2 = (x2[1]-x2[0])/_N2
                dx3 = (x3[1]-x3[0])/_N3
                d[kc+'_xx'] = dx1*np.arange(_N1) + dx1/2. + x1[0]
                d[kc+'_yy'] = dx2*np.arange(_N2) + dx2/2. + x2[0]
                d[kc+'_zz'] = dx3*np.arange(_N3) + dx3/2. + x3[0]

    _id = "{}:{}:{}".format(os.path.abspath(path), num, "".join(pres_vars))
    d['id'] = _id

    return d

#======================================================================

def flow_loader(flow_vars=None, num=None, path='./', sp=1, verbose=False):
    import glob

    if path[-1] != '/': path = path + '/'

    choices = get_output_times(path=path, sp=sp, output_type='flow')
    dpath = path+"Output/Phase/FluidVel/Sp{sp:02d}/{dv}/Vfld_{tm:08}.h5"

    d = {}
    while num not in choices:
        _ =  'Select from the following possible movie numbers: '\
             '\n{0} '.format(choices)
        num = int(input(_))

    if type(flow_vars) is str:
        flow_vars = flow_vars.split()
    elif flow_vars is None:
        flow_vars = 'x y z'.split()
    #print(dpath.format(sp=sp, tm=num))

    for k in flow_vars:
        if verbose: print(dpath.format(sp=sp, dv=k, tm=num))

        with h5py.File(dpath.format(sp=sp, dv=k, tm=num),'r') as f:
            kc = 'u'+k
            _ = f['DATA'].shape #python is fliped
            dim = len(_)
            print(kc,_)
            d[kc] = f['DATA'][:]
            if dim < 3:
                _N2,_N1 = _
                x1,x2 = f['AXIS']['X1 AXIS'][:], f['AXIS']['X2 AXIS'][:]
                dx1 = (x1[1]-x1[0])/_N1
                dx2 = (x2[1]-x2[0])/_N2
                d[kc+'_xx'] = dx1*np.arange(_N1) + dx1/2. + x1[0]
                d[kc+'_yy'] = dx2*np.arange(_N2) + dx2/2. + x2[0]

            else:
                _N3,_N2,_N1 = _
                x1 = f['AXIS']['X1 AXIS'][:]
                x2 = f['AXIS']['X2 AXIS'][:]
                x3 = f['AXIS']['X3 AXIS'][:]
                dx1 = (x1[1]-x1[0])/_N1
                dx2 = (x2[1]-x2[0])/_N2
                dx3 = (x3[1]-x3[0])/_N3
                d[kc+'_xx'] = dx1*np.arange(_N1) + dx1/2. + x1[0]
                d[kc+'_yy'] = dx2*np.arange(_N2) + dx2/2. + x2[0]
                d[kc+'_zz'] = dx3*np.arange(_N3) + dx3/2. + x3[0]

    _id = "{}:{}:{}".format(os.path.abspath(path), num, "".join(flow_vars))
    d['id'] = _id

    return d

#======================================================================

def track_loader(dens_vars=None, num=None, path='./', sp=1):
    import glob

    if path[-1] != '/': path = path + '/'

    choices = get_output_times(path=path, sp=sp, output_type='Raw')
    dpath = path+"Output/Raw/Sp{sp:02d}/raw_sp{sp:02d}_{tm:08}.h5"

    d = {}
    while num not in choices:
        _ =  'Select from the following possible movie numbers: '\
             '\n{0} '.format(choices)
        num = int(input(_))

    if type(dens_vars) is str:
        dens_vars = dens_vars.split()
    elif dens_vars is None:
        dens_vars = 'p1 p2 p3 q tag x1 x2'.split()
    print(dpath.format(sp=sp, tm=num))
    with h5py.File(dpath.format(sp=sp,tm=num),'r') as f:
        for k in dens_vars:
            d[k] = f[k][:]

    return d

#======================================================================

def field_loader(field_vars='all', components='all', num=None, 
                 path='./', slc=None, verbose=False):
    import glob
    _field_choices_ = {'B':'Magnetic',
                       'E':'Electric',
                       'J':'CurrentDens'}
    _ivc_ = {v: k for k, v in _field_choices_.items()}

    if components == 'all':
        components = 'xyz'

    if path[-1] != '/': path = path + '/'
    
    p = read_input(path=path)
    dim = len(p['ncells'])

    fpath = path+"Output/Fields/*"

    if field_vars == 'all':
        field_vars = [c[len(fpath)-1:] for c in glob.glob(fpath)]
        field_vars = [_ivc_[k] for k in field_vars]
    else:
        if isinstance(field_vars, str):
            field_vars = field_vars.upper().split()
        elif not type(field_vars) in (list, tuple):
            field_vars = [field_vars]

    if slc is None:
        if dim == 1:
            slc = np.s_[:]
        elif dim == 2:
            slc = np.s_[:,:]
        elif dim == 3:
            slc = np.s_[:,:,:]

    fpath = path+"Output/Fields/{f}/{T}{c}/{v}fld_{t}.h5"

    T = '' if field_vars[0] == 'J' else 'Total/'
    test_path = fpath.format(f = _field_choices_[field_vars[0]],
                             T = T,
                             c = 'x',
                             v = field_vars[0],
                             t = '*')
    
    if verbose: print(test_path)
    choices = glob.glob(test_path)
    #num_of_zeros = len()
    choices = [int(c[-11:-3]) for c in choices]
    choices.sort()

    fpath = fpath.format(f='{f}', T='{T}', c='{c}', v='{v}', t='{t:08d}')

    d = {}
    while num not in choices:
        _ =  'Select from the following possible movie numbers: '\
             '\n{0} '.format(choices)
        num = int(input(_))

    for k in field_vars:
        T = '' if k == 'J' else 'Total/'

        for c in components:
            ffn = fpath.format(f = _field_choices_[k],
                               T = T,
                               c = c,
                               v = k,
                               t = num)

            kc = k.lower()+c
            if verbose: print(ffn)
            with h5py.File(ffn, 'r') as f:
                d[kc] = f['DATA'][slc]

                _ = f['DATA'].shape #python is fliped
                if dim < 3:
                    _N2,_N1 = _
                    x1,x2 = f['AXIS']['X1 AXIS'][:], f['AXIS']['X2 AXIS'][:]
                    dx1 = (x1[1]-x1[0])/_N1
                    dx2 = (x2[1]-x2[0])/_N2
                    d[kc+'_xx'] = dx1*np.arange(_N1) + dx1/2. + x1[0]
                    d[kc+'_yy'] = dx2*np.arange(_N2) + dx2/2. + x2[0]

                    d[kc+'_xx'] = d[kc+'_xx'][slc[1]]
                    d[kc+'_yy'] = d[kc+'_yy'][slc[0]]
                else:
                    _N3,_N2,_N1 = _
                    x1 = f['AXIS']['X1 AXIS'][:]
                    x2 = f['AXIS']['X2 AXIS'][:]
                    x3 = f['AXIS']['X3 AXIS'][:]
                    dx1 = (x1[1]-x1[0])/_N1
                    dx2 = (x2[1]-x2[0])/_N2
                    dx3 = (x3[1]-x3[0])/_N3
                    d[kc+'_xx'] = dx1*np.arange(_N1) + dx1/2. + x1[0]
                    d[kc+'_yy'] = dx2*np.arange(_N2) + dx2/2. + x2[0]
                    d[kc+'_zz'] = dx3*np.arange(_N3) + dx3/2. + x3[0]

                    d[kc+'_xx'] = d[kc+'_xx'][slc[2]]
                    d[kc+'_yy'] = d[kc+'_yy'][slc[1]]
                    d[kc+'_zz'] = d[kc+'_zz'][slc[0]]

    return d


#======================================================================

#def load_all(field_vars='all', components='all', num=None, 
#                 path='./', slc=None, verbose=False):
#    for sp in range(5):
#        try:
#            d = dens_loader(num=num, path=path, sp=sp):

#======================================================================

def slice_from_window(w, p):
    bs = p['boxsize']
    nc = p['ncells']
    
    if w == 'all':
        w = [0., bs[0], 0., bs[1]]
    ip0 = max(np.int(np.round(w[0]/1./bs[0]*nc[0])), 0)
    ip1 = min(np.int(np.round(w[1]/1./bs[0]*nc[0])), nc[0])
    jp0 = max(np.int(np.round(w[2]/1./bs[1]*nc[1])), 0)
    jp1 = min(np.int(np.round(w[3]/1./bs[1]*nc[1])), nc[1])

    return np.s_[jp0:jp1, ip0:ip1]

#======================================================================

def _add_ExB(d):
    bm2 = d['bx']**2 + d['by']**2 + d['bz']**2
    d['exbx'] = (d['ey']*d['bz'] + d['ez']*d['by'])/bm2
    d['exby'] = (d['ez']*d['bx'] + d['ex']*d['bz'])/bm2
    d['exbz'] = (d['ex']*d['by'] + d['ey']*d['bx'])/bm2

    d['bm'] = np.sqrt(bm2)

#======================================================================

def pcm(d, k, ax=None, corse_res=(1,1), **kwargs):
    if ax is None:
        ax = plt.gca()
    
    rax = np.s_[::corse_res[0]]
    ray = np.s_[::corse_res[1]]

    pvar = k
    if type(k) is str:
        pvar = d[k]

    pc = ax.pcolormesh(d[k+'_xx'][rax], d[k+'_yy'][ray], d[k][ray,rax], **kwargs)

    return pc

#======================================================================

def ims(d, k, ax=None, corse_res=(1,1), **kwargs):
    if ax is None:
        ax = plt.gca()
    
    ax.set_aspect('auto')
    rax = np.s_[::corse_res[0]]
    ray = np.s_[::corse_res[1]]

    pvar = k
    if type(k) is str:
        pvar = d[k]

    ext = [d[k+'_'+2*_v][rax][_c] for _c,_v in zip([0,-1,0,-1],'xxyy')]
    im = ax.imshow(d[k][ray,rax], extent=ext, origin='lower', **kwargs)

    return im

#======================================================================

def calc_gamma(d, c, overwrite=False):
    if type(d) is dict:
        if 'gamma' in d:
            print('gamma already defined! Use overwite for overwrite.')
            if not overwrite:
                return None

        k = 'p1x1_yy'
        if 'ptx1_yy' in d:
            k = 'ptx1_yy'

        p1 = d[k]
        d['gamma'] = 1./np.sqrt(1. - (p1/c)**2)
        return d['gamma']

    else: # d is a numpy array of velocities 
        return 1./np.sqrt(1. - (d/c)**2)

#======================================================================

def calc_energy(d, c):
    if 'gamma' not in d:
        calc_gamma(d, c)
    
    gam = d['gamma']
    eng = (gam-1)*c**2
    
    d['eng'] = eng

#======================================================================

def prefix_fname_with_date(fname=''):
    """ Appends the current date to the begining of a file name """
    import datetime
    return datetime.date.today().strftime('%Y.%m.%d.') + fname

#======================================================================

def ask_to_save_fig(fig, fname=None, path=''):
    from os.path import join
    if input('Save Fig?\n> ') == 'y':
        if fname is None:
            fname = input('Save As:')

        fname = join(path, prefix_fname_with_date(fname))
        print('Saving {}...'.format(fname))
        fig.savefig(fname)

#======================================================================

def run_mean_fields(fname=None):
    """ Grabs the energy values from a p3d.stdout file and returns them
        as a numpy array.
    
    Args:
        fname (str, optional): Name of the p3d.stdout file to grab.
            If None it will ask.
    """
    if fname is None:
        fname = input('Enter dHybrid out file: ')

    flds = {k:[] for k in 'xyz'}
    with open(fname, 'r') as f:
        for line in f:
            if line[1:6] == 'Field':
                line = line.strip().split()
                comp = line[-2][0]
                if comp in 'xyz': 
                    flds[comp].append(line[-1])

    flds = np.array([flds[k] for k in 'xyz' ]).astype('float')
    return flds

#======================================================================

def read_input(path='./'):
    """Parse dHybrid input file for simulation information

    Args:
        path (str): path of input file
    """
    import os

    path = os.path.join(path, "input/input")
    inputs = {}
    repeated_sections = {}
    # Load in all of the input stuff
    with open(path) as f:
        in_bracs = False
        for line in f:
            # Clean up string
            line = line.strip()

            # Remove comment '!'
            trim_bang = line.find('!')
            if trim_bang > -1:
                line = line[:trim_bang].strip()

            # Is the line not empty?
            if line:
                if not in_bracs:
                    in_bracs = True
                    current_key = line

# The input has repeated section and keys for differnt species
# This section tries to deal with that
                    sp_counter = 1
                    while current_key in inputs:
                        inputs[current_key+"_01"] = inputs[current_key]
                        sp_counter += 1
                        current_key = "{}_{:02d}".format(line, sp_counter)
                        repeated_sections[current_key] = sp_counter

                    inputs[current_key] = []

                else:
                    if line == '{':
                        continue
                    elif line == '}':
                        in_bracs = False
                    else:
                        inputs[current_key].append(line)

    # Parse the input and cast it into usefull types
    param = {}
    repeated_keys = {}
    for key,inp in inputs.items():
        for sp in inp:
            k = sp.split('=')
            k,v = [v.strip(' , ') for v in k]

            _fk = k.find('(') 
            if _fk > 0:
                k = k[:_fk]

            if k in param:
                param["{}_{}".format(k, key)] = param[k]
                k = "{}_{}".format(k, key)

            param[k] = [_auto_cast(c.strip()) for c in v.split(',')]

            if len(param[k]) == 1:
                param[k] = param[k][0]
    
    return param

#======================================================================

def _auto_cast(k):
    """Takes an input string and tries to cast it to a real type

    Args:
        k (str): A string that might be a int, float or bool
    """

    k = k.replace('"','').replace("'",'')

    for try_type in [int, float]:
        try:
            return try_type(k)
        except:
            continue

    if k == '.true.':
        return True
    if k == '.false.':
        return False

    return str(k)

#======================================================

def calc_psi(f):
    """ Calculated the magnetic scaler potential for a 2D simulation
    Args:
        d (dict): Dictionary containing the fields of the simulation
            d must contain bx, by, xx and yy
    Retruns:
        psi (numpy.array(len(d['xx'], len(d['yy']))) ): Magnetic scaler
            potential
    """

    bx = f['bx']
    by = f['by']
    dy = f['bx_yy'][1] - f['bx_yy'][0]
    dx = f['bx_xx'][1] - f['bx_xx'][0]

    psi = 0.0*bx
    psi[1:,0] = np.cumsum(bx[1:,0])*dy
    psi[:,1:] = (psi[:,0] - np.cumsum(by[:,1:], axis=1).T*dx).T
    #psi[:,1:] = psi[:,0] - np.cumsum(by[:,1:], axis=1)*dx
    return psi

#======================================================

def div_B(f):
    """Have you come looking to figure out which axis is X?

    I have tried so many times to figure this out so you are in luck!
    So long as this thing is zero you should know that 
    axis0 is y
    axis1 is x
    
    regards,
    A smarter hopefully fatter version of you
    """

    bx = f['bx']
    by = f['by']
    dy = f['bx_yy'][1] - f['bx_yy'][0]
    dx = f['bx_xx'][1] - f['bx_xx'][0]

    return ((np.roll(bx, -1, axis=1) - np.roll(bx, 1, axis=1))/dx + 
            (np.roll(by, -1, axis=0) - np.roll(bx, 1, axis=0))/dx)/2.

#======================================================

def spt(d, k, q=0, ax=None, rng='all', sigma=0., yscale=1., **kwargs):
    if ax is None:
        ax = plt.gca()

    yy = d[k+'_yy']/yscale
    xx = d[k+'_xx']

    if rng == 'all':
        rng = np.s_[:,:]
    else:
        lb,up = [np.abs(xx - r).argmin() for r in rng]
        rng = np.s_[:,lb:up]
  
    pvar = yy**q*np.mean(d[k][rng], axis=1)
    if sigma > 0:
        pvar = gf(pvar, sigma=sigma, mode='constant')

    ax.plot(yy, pvar, **kwargs)
    ax.set_yscale('log')
    if np.min(yy) > 0.:
        ax.set_xscale('log')
    return ax,yy,pvar

#======================================================

def eff(E0=2000., path='./', num=None):
    d = dens_loader('etx1', path=path, num=num)

    yy = d['etx1_yy']
    xx = d['etx1_xx']
    dE = np.log(yy[1]) - np.log(yy[0])

    ip = np.abs(yy - E0).argmin()
  
    EfE = np.sum(dE*(yy*d['etx1'][ip:, :]), axis=0)

    return EfE, xx

#======================================================

def fft_ksp(d, f, axis=1):
    if type(f) == str:
        x = d[f+'_xx']
        f = d[f]
    else:
        x = d['bx_xx']

    mf = np.mean(f, axis=[1,0][axis])

    nn = len(mf)
    Ff = np.fft.fft(mf)/(1.0*nn)

    k = np.arange(nn)/(x[-1] - x[0])*2.*np.pi

    return k[:nn//2], Ff[:nn//2]

#======================================================================

def fft_ksp_dict(d, smooth=None):
    global_kk = np.logspace(-3,-.5, 10000)
    x = d['xx']
    fd = {'tt':d['tt']}
    for v in "bx by bz".split():
        nn = d[v].shape[1]
        #Ff = np.fft.fft(d[v], axis=1)/(1.0*nn)
        _f = d[v]
        if smooth:
            _f = gf(_f, sigma=smooth, mode='wrap')

        Ff = np.fft.fft(_f, axis=1)/np.sqrt(nn*2.*np.pi)
        #Ff = np.fft.fft(d[v], axis=1)*np.sqrt((x[-1] - x[0])/2./np.pi/nn)
        k = np.arange(nn)/(x[-1] - x[0])*2.*np.pi
    
        fd[v] = Ff[:, :nn//2]
    
    fd['kk'] = k[:nn//2]

    return fd

#======================================================

def fft2D(d, f):
    if type(f) == str:
        f = d[f]
        x = d[f+'_xx']
        y = d[f+'_yy']
    else:
        x = d['bx_xx']
        y = d['bx_yy']


    ny,nx = f.shape
    kx = np.arange(nx)/(x[-1] - x[0])*2.*np.pi
    ky = np.arange(ny)/(y[-1] - y[0])*2.*np.pi

    F = np.fft.fft2(f)/(1.0*nx*ny)

    return kx[:nx/2], ky[:ny/2], F[:ny/2, :nx/2]

#======================================================

#def fft2Dmag(d, f):
#    kx, ky, F = fft2D(d, f)
#    kk = np.sqrt(kx**2 + ky**2).flat


#======================================================

def calc_flow(d):
    ff = [d[k+'x1'] for k in 'p1 p2 p3'.split()]
    pp = [d[k+'x1_yy'] for k in 'p1 p2 p3'.split()]
    dps = [p[1] - p[0] for p in pp]
    n = np.sum(ff[0]*dps[0], axis=0)
    return [np.sum(p*f.T*dp, axis=1)/n for p,f,dp in zip(pp,ff,dps)]

#======================================================

def build_gam(d, C=None):
    if C is None:
        print('!!!Warning!!! speed of light not given, using 50')
        C = 50.
    d['gtx1_xx'] = d['ptx1_xx']
    pp = d['ptx1_yy']
    gam = np.sqrt(pp**2/C**2 + 1.)
    d['gtx1_yy'] = (gam - 1.)
    d['gtx1'] = (gam/pp*d['ptx1'].T).T

    return None

#======================================================

def build_vel(d, C=None):
    if C is None:
        print('!!!Warning!!! speed of light not given, using 50')
        C = 50.
    d['vtx1_xx'] = d['ptx1_xx']
    pp = d['ptx1_yy']
    gam = np.sqrt(pp**2/C**2 + 1.)
    d['vtx1_yy'] = pp/gam
    d['vtx1'] = (gam**3*d['ptx1'].T).T

    return None

#======================================================

def time_cbar(tms, ax, cmap='jet', title='Time ($\Omega_{ci}^{-1}$)'):
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="2.5%", pad=0.05)

    cmap = mpl.cm.get_cmap(cmap)
    norm = mpl.colors.Normalize(vmin=tms[0], vmax=tms[-1])
    cb1 = mpl.colorbar.ColorbarBase(cax, cmap=cmap, norm=norm)

    cax.text(.5, 1.05, title,
             transform=cax.transAxes, ha='center')

#======================================================
# Restart Part Mapper
class PartMapper(object):
    def __init__(self, path):
        self.path=path
        
        self.p = read_input(path)
        
        self.px,self.py = self.p['node_number']
        self.nx,self.ny = self.p['ncells']
        self.rx,self.ry = self.p['boxsize']
        
        self.dx = self.rx/1./self.nx
        self.dy = self.ry/1./self.ny
        
    def _box_center(self, ip, jp):
        dx = self.dx
        dy = self.dy
        
        npx = self.nx//self.px
        Mx = (self.nx/1./self.px - npx)*self.px
        
        npy = self.ny//self.py
        My = (self.ny/1./self.py - npy)*self.py
        
        if ip < Mx:
            xr = dx*(npx + 1)*ip + dx/2.
        else:
            xr = dx*(Mx + npx*ip) + dx/2.
            
        if jp < My:
            yr = dy*(npy + 1)*jp + dy/2.
        else:
            yr = dy*(My + npy*jp) + dy/2.

        return xr,yr
    
    def xrange_to_nums(self, x0, x1):
        i0 = np.int(np.floor(x0/self.rx*self.px))
        i1 = np.int(np.min([np.ceil(x1/self.rx*self.px), self.px - 1]))
        
        nums = range(i0, i1)
        for _ny in range(1, self.py):
            nums += range(i0 + _ny*self.px, i1 + _ny*self.px)
        
        return nums
        
    def _num_to_index(self, num):
        ip = num%self.px
        jp = num//self.px
        return ip,jp

    def _index_to_num(self, ip, jp):
        num = self.px*jp + ip
        return num
    
    def parts_from_index(self, ip, jp, sp='SP01'):
        fname = self.path+'/Restart/Rest_proc{:05d}.h5'
        num = self._index_to_num(ip, jp)
        bcx,bcy = self._box_center(ip, jp)
        dx,dy = self.dx,self.dy
        
        with h5py.File(fname.format(num),'r') as f:

            pts = f[sp][:]
            ind = f[sp+'INDEX'][:]
            pts[:, 0] = pts[:,0] + bcx + dx*(ind[:,0] - 4)
            pts[:, 1] = pts[:,1] + bcy + dy*(ind[:,1] - 4)
        
        return pts

    def parts_from_num(self, num, sp='SP01'):
        ip, jp = self._num_to_index(num)
        return self.parts_from_index(ip, jp, sp=sp)

#======================================================

#def calc_dens(d, k='p1x1'):
def moments(d, k='p1x1'):
    y = d[k+'_yy']
    x = d[k+'_xx']
    fp = d[k]
    dp = y[1] - y[0]

    n = dp*np.sum(fp, axis=0)
    u = dp*np.sum((fp.T*y).T, axis=0)/n
    T = dp*np.sum((fp.T*y**2).T + fp*u**2 - 2.*u*(fp.T*y).T, axis=0)/n

    return x,n,u,T

#======================================================

def dens_movie(path='./',
               ax=None,
               cmap='jet', 
               rng=np.s_[1:],
               mvar='p1x1',
               avg_r=0):

    if ax is None:
        plt.figure(77).clf()
        fig,ax = plt.subplots(1, 1, num=77)
        fig.set_size_inches(8.5, 11./4.)

    tms = get_output_times(path=path)[rng]
    shock_loc = []
    
    #d = dens_loader(mvar, path=path, num=tms[-1])
    #dens,xx = quick_dens(d, mvar)
    #ymax = dens.max()

    r_in_time = [[] for _ in range(avg_r + 1)]

    print("We will be working on {} lines...".format(len(tms)))
    for _c,tm in enumerate(tms):
        print("{},".format(_c), end="")
        cid = plt.cm.get_cmap(cmap)(_c/1./len(tms))
        d = dens_loader(mvar, path=path, num=tm)
        dens,xx = calc_dens(d, mvar)

        ax.plot(xx, dens, linewidth=.5, color=cid)

        # Find the shock location
        #ddens = .0001 + 0.*dens
        ddens = np.abs(dens[5:] - dens[3:-2]) + .00001
        sl = (np.abs(dens[4:-1] - 2.0)/ddens).argmin()
        sl = sl + 4

        # Try a different way
        #sl = np.abs(dens[6:] - dens[1:-5] 
        #          + dens[5:-1] - dens[2:-4] 
        #          + dens[4:-2] - dens[3:-3]).argmax() + 3

        shock_loc.append(xx[sl]) 

        r_in_time[0].append(np.mean(dens[:sl]))
        for _d,_r in enumerate(r_in_time[1:]):
            ip0 = int(np.round(_d*1./avg_r*sl))
            ip1 = int(np.round((_d+1.)/avg_r*sl))
            _r.append(np.mean(dens[ip0:ip1]))
    
    ax.set_xlim(0, 1.5*shock_loc[-1])
    ax.minorticks_on()
    p = read_input(path=path)

    return fig, ax, shock_loc, p['dt']*tms, r_in_time

#======================================================

def rotate_ten(f, p, overwrite=False, full_rotate=False):

    if 'ppar' in p and not overwrite:
        print('Warning: {} was found in the'.format('ppar') + 
              'restored data: nothing will be rotated!!!!')
        pass

    if full_rotate:
        print("Warning: Full rotation not implemented! Exiting!")
        pass

    bmag = np.sqrt(f['bx']**2 + f['by']**2 + f['bz']**2)
    bbx,bby,bbz = (f[k]/bmag for k in 'bx by bz'.split())

    p['ppar'] = (bbx*(bbx*p['pxx'] +
                      bby*p['pxy'] +
                      bbz*p['pzx'])+
                 bby*(bbx*p['pxy'] +
                      bby*p['pyy'] +
                      bbz*p['pyz'])+
                 bbz*(bbx*p['pzx'] +
                      bby*p['pyz'] +
                      bbz*p['pzz']))

    p['pperp1'] = (p['pxx'] +
                      p['pyy'] +
                      p['pzz'] -
                      p['ppar'])/2.

    p['pperp2'] = p['pperp1']

#======================================================
# Making jet3 a permanent colormap
def get_jet3():
    N = 1024
    jt = plt.cm.jet(np.linspace(0, 1, N))

    ln = N/8
    shp = np.concatenate([np.arange(0, 3*ln, 1), np.arange(3*ln, 5*ln, N/8), np.arange(5*ln, 8*ln, 1)])

    rr = np.arange(len(shp))/(len(shp) - 1.0)
    cdict = {}
    cdict['red'] = np.vstack((rr, jt[shp,0], jt[shp,0])).T
    cdict['green'] = np.vstack((rr, jt[shp,1], jt[shp,1])).T
    cdict['blue'] = np.vstack((rr, jt[shp,2], jt[shp,2])).T
    j3 = mpl.colors.LinearSegmentedColormap('jet3', segmentdata=cdict, N=256)

    return j3

#======================================================

def find_shock(d, k):
    #This assumes that the value of k is nearly constant
    #upstream and increases at the shock
    n = np.sum(np.abs(d[k]), axis=0)
    n = n - 3.0*np.mean(n[3*(len(n)//4):])
    ip = np.cumsum(n).argmax()
    xp = d[k+"_xx"][ip]

    return ip,xp

#======================================================
from matplotlib.colors import LinearSegmentedColormap      
      
cm_data = [[0.0051932, 0.098238, 0.34984],      
           [0.0090652, 0.10449, 0.35093],      
           [0.012963, 0.11078, 0.35199],      
           [0.01653, 0.11691, 0.35307],      
           [0.019936, 0.12298, 0.35412],      
           [0.023189, 0.12904, 0.35518],      
           [0.026291, 0.13504, 0.35621],      
           [0.029245, 0.14096, 0.35724],      
           [0.032053, 0.14677, 0.35824],      
           [0.034853, 0.15256, 0.35923],      
           [0.037449, 0.15831, 0.36022],      
           [0.039845, 0.16398, 0.36119],      
           [0.042104, 0.16956, 0.36215],      
           [0.044069, 0.17505, 0.36308],      
           [0.045905, 0.18046, 0.36401],      
           [0.047665, 0.18584, 0.36491],      
           [0.049378, 0.19108, 0.36581],      
           [0.050795, 0.19627, 0.36668],      
           [0.052164, 0.20132, 0.36752],      
           [0.053471, 0.20636, 0.36837],      
           [0.054721, 0.21123, 0.36918],      
           [0.055928, 0.21605, 0.36997],      
           [0.057033, 0.22075, 0.37075],      
           [0.058032, 0.22534, 0.37151],      
           [0.059164, 0.22984, 0.37225],      
           [0.060167, 0.2343, 0.37298],      
           [0.061052, 0.23862, 0.37369],      
           [0.06206, 0.24289, 0.37439],      
           [0.063071, 0.24709, 0.37505],      
           [0.063982, 0.25121, 0.37571],      
           [0.064936, 0.25526, 0.37636],      
           [0.065903, 0.25926, 0.37699],      
           [0.066899, 0.26319, 0.37759],      
           [0.067921, 0.26706, 0.37819],      
           [0.069002, 0.27092, 0.37877],      
           [0.070001, 0.27471, 0.37934],      
           [0.071115, 0.2785, 0.37989],      
           [0.072192, 0.28225, 0.38043],      
           [0.07344, 0.28594, 0.38096],      
           [0.074595, 0.28965, 0.38145],      
           [0.075833, 0.29332, 0.38192],      
           [0.077136, 0.297, 0.38238],      
           [0.078517, 0.30062, 0.38281],      
           [0.079984, 0.30425, 0.38322],      
           [0.081553, 0.30786, 0.3836],      
           [0.083082, 0.31146, 0.38394],      
           [0.084778, 0.31504, 0.38424],      
           [0.086503, 0.31862, 0.38451],      
           [0.088353, 0.32217, 0.38473],      
           [0.090281, 0.32569, 0.38491],      
           [0.092304, 0.32922, 0.38504],      
           [0.094462, 0.33271, 0.38512],      
           [0.096618, 0.33616, 0.38513],      
           [0.099015, 0.33962, 0.38509],      
           [0.10148, 0.34304, 0.38498],      
           [0.10408, 0.34641, 0.3848],      
           [0.10684, 0.34977, 0.38455],      
           [0.1097, 0.3531, 0.38422],      
           [0.11265, 0.35639, 0.38381],      
           [0.11575, 0.35964, 0.38331],      
           [0.11899, 0.36285, 0.38271],      
           [0.12232, 0.36603, 0.38203],      
           [0.12589, 0.36916, 0.38126],      
           [0.12952, 0.37224, 0.38038],      
           [0.1333, 0.37528, 0.3794],      
           [0.13721, 0.37828, 0.37831],      
           [0.14126, 0.38124, 0.37713],      
           [0.14543, 0.38413, 0.37584],      
           [0.14971, 0.38698, 0.37445],      
           [0.15407, 0.38978, 0.37293],      
           [0.15862, 0.39253, 0.37132],      
           [0.16325, 0.39524, 0.36961],      
           [0.16795, 0.39789, 0.36778],      
           [0.17279, 0.4005, 0.36587],      
           [0.17775, 0.40304, 0.36383],      
           [0.18273, 0.40555, 0.36171],      
           [0.18789, 0.408, 0.35948],      
           [0.19305, 0.41043, 0.35718],      
           [0.19831, 0.4128, 0.35477],      
           [0.20368, 0.41512, 0.35225],      
           [0.20908, 0.41741, 0.34968],      
           [0.21455, 0.41966, 0.34702],      
           [0.22011, 0.42186, 0.34426],      
           [0.22571, 0.42405, 0.34146],      
           [0.23136, 0.4262, 0.33857],      
           [0.23707, 0.42832, 0.33563],      
           [0.24279, 0.43042, 0.33263],      
           [0.24862, 0.43249, 0.32957],      
           [0.25445, 0.43453, 0.32643],      
           [0.26032, 0.43656, 0.32329],      
           [0.26624, 0.43856, 0.32009],      
           [0.27217, 0.44054, 0.31683],      
           [0.27817, 0.44252, 0.31355],      
           [0.28417, 0.44448, 0.31024],      
           [0.29021, 0.44642, 0.30689],      
           [0.29629, 0.44836, 0.30351],      
           [0.30238, 0.45028, 0.30012],      
           [0.30852, 0.4522, 0.29672],      
           [0.31465, 0.45411, 0.29328],      
           [0.32083, 0.45601, 0.28984],      
           [0.32701, 0.4579, 0.28638],      
           [0.33323, 0.45979, 0.28294],      
           [0.33947, 0.46168, 0.27947],      
           [0.3457, 0.46356, 0.276],      
           [0.35198, 0.46544, 0.27249],      
           [0.35828, 0.46733, 0.26904],      
           [0.36459, 0.46921, 0.26554],      
           [0.37092, 0.47109, 0.26206],      
           [0.37729, 0.47295, 0.25859],      
           [0.38368, 0.47484, 0.25513],      
           [0.39007, 0.47671, 0.25166],      
           [0.3965, 0.47859, 0.24821],      
           [0.40297, 0.48047, 0.24473],      
           [0.40945, 0.48235, 0.24131],      
           [0.41597, 0.48423, 0.23789],      
           [0.42251, 0.48611, 0.23449],      
           [0.42909, 0.48801, 0.2311],      
           [0.43571, 0.48989, 0.22773],      
           [0.44237, 0.4918, 0.22435],      
           [0.44905, 0.49368, 0.22107],      
           [0.45577, 0.49558, 0.21777],      
           [0.46254, 0.4975, 0.21452],      
           [0.46937, 0.49939, 0.21132],      
           [0.47622, 0.50131, 0.20815],      
           [0.48312, 0.50322, 0.20504],      
           [0.49008, 0.50514, 0.20198],      
           [0.49709, 0.50706, 0.19899],      
           [0.50415, 0.50898, 0.19612],      
           [0.51125, 0.5109, 0.1933],      
           [0.51842, 0.51282, 0.19057],      
           [0.52564, 0.51475, 0.18799],      
           [0.53291, 0.51666, 0.1855],      
           [0.54023, 0.51858, 0.1831],      
           [0.5476, 0.52049, 0.18088],      
           [0.55502, 0.52239, 0.17885],      
           [0.56251, 0.52429, 0.17696],      
           [0.57002, 0.52619, 0.17527],      
           [0.57758, 0.52806, 0.17377],      
           [0.5852, 0.52993, 0.17249],      
           [0.59285, 0.53178, 0.17145],      
           [0.60052, 0.5336, 0.17065],      
           [0.60824, 0.53542, 0.1701],      
           [0.61597, 0.53723, 0.16983],      
           [0.62374, 0.539, 0.16981],      
           [0.63151, 0.54075, 0.17007],      
           [0.6393, 0.54248, 0.17062],      
           [0.6471, 0.54418, 0.17146],      
           [0.65489, 0.54586, 0.1726],      
           [0.66269, 0.5475, 0.17404],      
           [0.67048, 0.54913, 0.17575],      
           [0.67824, 0.55071, 0.1778],      
           [0.686, 0.55227, 0.18006],      
           [0.69372, 0.5538, 0.18261],      
           [0.70142, 0.55529, 0.18548],      
           [0.7091, 0.55677, 0.18855],      
           [0.71673, 0.5582, 0.19185],      
           [0.72432, 0.55963, 0.19541],      
           [0.73188, 0.56101, 0.19917],      
           [0.73939, 0.56239, 0.20318],      
           [0.74685, 0.56373, 0.20737],      
           [0.75427, 0.56503, 0.21176],      
           [0.76163, 0.56634, 0.21632],      
           [0.76894, 0.56763, 0.22105],      
           [0.77621, 0.5689, 0.22593],      
           [0.78342, 0.57016, 0.23096],      
           [0.79057, 0.57142, 0.23616],      
           [0.79767, 0.57268, 0.24149],      
           [0.80471, 0.57393, 0.24696],      
           [0.81169, 0.57519, 0.25257],      
           [0.81861, 0.57646, 0.2583],      
           [0.82547, 0.57773, 0.2642],      
           [0.83227, 0.57903, 0.27021],      
           [0.839, 0.58034, 0.27635],      
           [0.84566, 0.58167, 0.28263],      
           [0.85225, 0.58304, 0.28904],      
           [0.85875, 0.58444, 0.29557],      
           [0.86517, 0.58588, 0.30225],      
           [0.87151, 0.58735, 0.30911],      
           [0.87774, 0.58887, 0.31608],      
           [0.88388, 0.59045, 0.32319],      
           [0.8899, 0.59209, 0.33045],      
           [0.89581, 0.59377, 0.33787],      
           [0.90159, 0.59551, 0.34543],      
           [0.90724, 0.59732, 0.35314],      
           [0.91275, 0.59919, 0.36099],      
           [0.9181, 0.60113, 0.369],      
           [0.9233, 0.60314, 0.37714],      
           [0.92832, 0.60521, 0.3854],      
           [0.93318, 0.60737, 0.39382],      
           [0.93785, 0.60958, 0.40235],      
           [0.94233, 0.61187, 0.41101],      
           [0.94661, 0.61422, 0.41977],      
           [0.9507, 0.61665, 0.42862],      
           [0.95457, 0.61914, 0.43758],      
           [0.95824, 0.62167, 0.4466],      
           [0.9617, 0.62428, 0.4557],      
           [0.96494, 0.62693, 0.46486],      
           [0.96798, 0.62964, 0.47406],      
           [0.9708, 0.63239, 0.48329],      
           [0.97342, 0.63518, 0.49255],      
           [0.97584, 0.63801, 0.50183],      
           [0.97805, 0.64087, 0.51109],      
           [0.98008, 0.64375, 0.52035],      
           [0.98192, 0.64666, 0.5296],      
           [0.98357, 0.64959, 0.53882],      
           [0.98507, 0.65252, 0.548],      
           [0.98639, 0.65547, 0.55714],      
           [0.98757, 0.65842, 0.56623],      
           [0.9886, 0.66138, 0.57526],      
           [0.9895, 0.66433, 0.58425],      
           [0.99027, 0.66728, 0.59317],      
           [0.99093, 0.67023, 0.60203],      
           [0.99148, 0.67316, 0.61084],      
           [0.99194, 0.67609, 0.61958],      
           [0.9923, 0.67901, 0.62825],      
           [0.99259, 0.68191, 0.63687],      
           [0.99281, 0.68482, 0.64542],      
           [0.99297, 0.68771, 0.65393],      
           [0.99306, 0.69058, 0.6624],      
           [0.99311, 0.69345, 0.67081],      
           [0.99311, 0.69631, 0.67918],      
           [0.99307, 0.69916, 0.68752],      
           [0.993, 0.70201, 0.69583],      
           [0.9929, 0.70485, 0.70411],      
           [0.99277, 0.70769, 0.71238],      
           [0.99262, 0.71053, 0.72064],      
           [0.99245, 0.71337, 0.72889],      
           [0.99226, 0.71621, 0.73715],      
           [0.99205, 0.71905, 0.7454],      
           [0.99184, 0.72189, 0.75367],      
           [0.99161, 0.72475, 0.76196],      
           [0.99137, 0.72761, 0.77027],      
           [0.99112, 0.73049, 0.77861],      
           [0.99086, 0.73337, 0.78698],      
           [0.99059, 0.73626, 0.79537],      
           [0.99031, 0.73918, 0.80381],      
           [0.99002, 0.7421, 0.81229],      
           [0.98972, 0.74504, 0.8208],      
           [0.98941, 0.748, 0.82937],      
           [0.98909, 0.75097, 0.83798],      
           [0.98875, 0.75395, 0.84663],      
           [0.98841, 0.75695, 0.85533],      
           [0.98805, 0.75996, 0.86408],      
           [0.98767, 0.763, 0.87286],      
           [0.98728, 0.76605, 0.8817],      
           [0.98687, 0.7691, 0.89057],      
           [0.98643, 0.77218, 0.89949],      
           [0.98598, 0.77527, 0.90845],      
           [0.9855, 0.77838, 0.91744],      
           [0.985, 0.7815, 0.92647],      
           [0.98447, 0.78462, 0.93553],      
           [0.98391, 0.78776, 0.94463],      
           [0.98332, 0.79091, 0.95375],      
           [0.9827, 0.79407, 0.9629],      
           [0.98205, 0.79723, 0.97207],      
           [0.98135, 0.80041, 0.98127]]      
      
batlow = LinearSegmentedColormap.from_list('batlow', cm_data)      
#def perp_spec(ar, sumax=2, lens=3*(2*np.pi)):
#    """
#      PerpSpectrum(ar,sumax=2,lenx=2*pi,leny=2*pi,lenz=2*pi)
#      ar -> Array to compute the spectrum of
#      sumax -> Axis of magnetic field direction. Right now only x,y,z = 0,1,2
#      lenx,leny,lenz -> System size in x,y,z directions to take into 
#                        account the anisotropy of system if any
#      RETURNS:
#      kk -> Wavenumber array
#      fekp -> Spectrum of the array
#    """
#
#    nf - np.fft
#    mar = ar - np.mean(ar)
#    nn  = np.shape(mar)
#    kk = nf.fftshift(nf.fftfreq(n))*n*(2*pi/l) for n,l in zip(nn,lens)
#
#  
#    far = nf.fftshift(nf.fftn(mar))/(np.prod(nn))
#
#    fftea = 0.5*np.abs(far)**2
#
#    ffteb = np.sum(fftea,axis=sumax)
#
#    fekp = np.zeros(min(nn[:2]))
#
#    kp = np.sqrt(np.sum(np.meshgrid(kx**2, ky**2), axis=0))
#
#    dk = np.abs(kp[1,0] - kp[0,0])
#    kk = kp[nn[0]/2, nn[1]/2]
#
##   for i in range(len(fekp)):
##      fekp[i]= np.sum(np.ma.MaskedArray(ffteb, ~((kp[nx/2,i+ny/2]-dk < kp) & (kp < kp[nx/2,i+ny/2]+dk))))
##
##        fekp[i]= np.sum(np.ma.MaskedArray(ffteb, ~((kp[nx/2,i+ny/2]-dk < kp) & (kp < kp[nx/2,i+ny/2]+dk))))
##    
#
#   return kk,fekp/dk
