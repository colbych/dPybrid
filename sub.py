import os
import h5py
import numpy as np
import matplotlib as mpl 
import matplotlib.pyplot as plt 
from matplotlib.colors import LogNorm
from scipy.ndimage import gaussian_filter as gf

phase_vars = 'p1x1 p2x1 p3x1 ptx1 etx1'.split()

#======================================================================

def qloader(num=None, path='./'):
    import glob

    if path[-1] is not '/': path = path + '/'
    
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
        num = int(raw_input(_))

    d = {}
    for k in 'xyz':
        print bpath.format(k,num)
        print epath.format(k,num)
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
        print dpath.format(k,num)
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

    print "No files found in path: {}".format(_fn.format(var=_pv, sp=sp))
    raise FileNotFoundError

#======================================================================

def dens_loader(dens_vars=None, num=None, path='./', sp=1, verbose=False):
    import glob

    if path[-1] is not '/': path = path + '/'

    choices = get_output_times(path=path, sp=sp)
    
    dpath = path+"Output/Phase/*"
    if dens_vars is None:
        dens_vars = [c[len(dpath)-1:] for c in glob.glob(dpath)]
    else:
        if not type(dens_vars) in (list, tuple):
            dens_vars = [dens_vars]

    if 'FluidVel' in dens_vars:
        dens_vars.pop(dens_vars.index('FluidVel'))

    print dens_vars
    dens_vars.sort()

    dpath = path+"Output/Phase/{dv}/Sp{sp:02d}/dens_sp{sp:02d}_{tm}.h5"
    
    if verbose: print dpath.format(dv=dens_vars[0], sp=sp, tm='*')


    dpath = path+"Output/Phase/{dv}/Sp{sp:02d}/dens_sp{sp:02d}_{tm:08}.h5"

    d = {}
    while num not in choices:
        _ =  'Select from the following possible movie numbers: '\
             '\n{0} '.format(choices)
        num = int(raw_input(_))

    for k in dens_vars:

        if verbose: print dpath.format(dv=k,sp=sp,tm=num)
        with h5py.File(dpath.format(dv=k,sp=sp,tm=num),'r') as f:
            d[k] = f['DATA'][:]

            _N2,_N1 = f['DATA'][:].shape #python is fliped
            x1,x2 = f['AXIS']['X1 AXIS'][:],f['AXIS']['X2 AXIS'][:]
            dx1 = (x1[1]-x1[0])/_N1
            dx2 = (x2[1]-x2[0])/_N2
            d[k+'_xx'] = dx1*np.arange(_N1) + dx1/2. + x1[0]
            d[k+'_yy'] = dx2*np.arange(_N2) + dx2/2. + x2[0]

            if k == 'etx1':
                d['etx1_yy'] = np.exp(d['etx1_yy'])
    _id = "{}:{}:{}".format(os.path.abspath(path), num, "".join(dens_vars))
    d['id'] = _id
    return d

#======================================================================

def raw_loader(dens_vars=None, num=None, path='./', sp=1):
    import glob

    if path[-1] is not '/': path = path + '/'

    choices = get_output_times(path=path, sp=sp, output_type='Raw')
    dpath = path+"Output/Raw/Sp{sp:02d}/raw_sp{sp:02d}_{tm:08}.h5"

    d = {}
    while num not in choices:
        _ =  'Select from the following possible movie numbers: '\
             '\n{0} '.format(choices)
        num = int(raw_input(_))

    if type(dens_vars) is str:
        dens_vars = dens_vars.split()
    elif dens_vars is None:
        dens_vars = 'p1 p2 p3 q tag x1 x2'.split()
    print dpath.format(sp=sp,tm=num)
    with h5py.File(dpath.format(sp=sp,tm=num),'r') as f:
        for k in dens_vars:
            d[k] = f[k][:]

    return d

#======================================================================

def flow_loader(flow_vars=None, num=None, path='./', sp=1, verbose=False):
    import glob

    if path[-1] is not '/': path = path + '/'

    choices = get_output_times(path=path, sp=sp, output_type='flow')
    dpath = path+"Output/Phase/FluidVel/Sp{sp:02d}/{dv}/Vfld_{tm:08}.h5"

    d = {}
    while num not in choices:
        _ =  'Select from the following possible movie numbers: '\
             '\n{0} '.format(choices)
        num = int(raw_input(_))

    if type(flow_vars) is str:
        flow_vars = flow_vars.split()
    elif flow_vars is None:
        flow_vars = 'x y z'.split()
    #print dpath.format(sp=sp, tm=num)

    for k in flow_vars:
        if verbose: print dpath.format(sp=sp, dv=k, tm=num)

        with h5py.File(dpath.format(sp=sp, dv=k, tm=num),'r') as f:
            d[k] = f['DATA'][:]

            _N2,_N1 = f['DATA'][:].shape #python is fliped
            x1,x2 = f['AXIS']['X1 AXIS'][:],f['AXIS']['X2 AXIS'][:]
            dx1 = (x1[1]-x1[0])/_N1
            dx2 = (x2[1]-x2[0])/_N2
            d[k+'_xx'] = dx1*np.arange(_N1) + dx1/2. + x1[0]
            d[k+'_yy'] = dx2*np.arange(_N2) + dx2/2. + x2[0]

    _id = "{}:{}:{}".format(os.path.abspath(path), num, "".join(flow_vars))
    d['id'] = _id

    return d

#======================================================================

def track_loader(dens_vars=None, num=None, path='./', sp=1):
    import glob

    if path[-1] is not '/': path = path + '/'

    choices = get_output_times(path=path, sp=sp, output_type='Raw')
    dpath = path+"Output/Raw/Sp{sp:02d}/raw_sp{sp:02d}_{tm:08}.h5"

    d = {}
    while num not in choices:
        _ =  'Select from the following possible movie numbers: '\
             '\n{0} '.format(choices)
        num = int(raw_input(_))

    if type(dens_vars) is str:
        dens_vars = dens_vars.split()
    elif dens_vars is None:
        dens_vars = 'p1 p2 p3 q tag x1 x2'.split()
    print dpath.format(sp=sp,tm=num)
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
    _ivc_ = {v: k for k, v in _field_choices_.iteritems()}

    if components == 'all':
        components = 'xyz'


    if path[-1] is not '/': path = path + '/'
    
    fpath = path+"Output/Fields/*"

    if field_vars == 'all':
        field_vars = [c[len(fpath)-1:] for c in glob.glob(fpath)]
        field_vars = [_ivc_[k] for k in field_vars]
    else:
        if isinstance(field_vars, basestring):
            field_vars = field_vars.upper().split()
        elif not type(field_vars) in (list, tuple):
            field_vars = [field_vars]

    if slc is None:
        slc = np.s_[:,:]

    fpath = path+"Output/Fields/{f}/{T}{c}/{v}fld_{t}.h5"

    T = '' if field_vars[0] == 'J' else 'Total/'
    test_path = fpath.format(f = _field_choices_[field_vars[0]],
                             T = T,
                             c = 'x',
                             v = field_vars[0],
                             t = '*')
    
    if verbose: print test_path
    choices = glob.glob(test_path)
    #num_of_zeros = len()
    choices = [int(c[-11:-3]) for c in choices]
    choices.sort()

    fpath = fpath.format(f='{f}', T='{T}', c='{c}', v='{v}', t='{t:08d}')

    d = {}
    while num not in choices:
        _ =  'Select from the following possible movie numbers: '\
             '\n{0} '.format(choices)
        num = int(raw_input(_))

    for k in field_vars:
        T = '' if k == 'J' else 'Total/'

        for c in components:
            ffn = fpath.format(f = _field_choices_[k],
                               T = T,
                               c = c,
                               v = k,
                               t = num)

            kc = k.lower()+c
            if verbose: print ffn
            with h5py.File(ffn,'r') as f:
                d[kc] = f['DATA'][slc]

                _N2,_N1 = f['DATA'].shape #python is fliped
                x1,x2 = f['AXIS']['X1 AXIS'][:], f['AXIS']['X2 AXIS'][:]
                dx1 = (x1[1]-x1[0])/_N1
                dx2 = (x2[1]-x2[0])/_N2
                d[kc+'_xx'] = dx1*np.arange(_N1) + dx1/2. + x1[0]
                d[kc+'_yy'] = dx2*np.arange(_N2) + dx2/2. + x2[0]

                d[kc+'_xx'] = d[kc+'_xx'][slc[1]]
                d[kc+'_yy'] = d[kc+'_yy'][slc[0]]

    return d


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
    im = ax.imshow(d[k][ray,rax], extent=ext, origin='low', **kwargs)

    return im

#======================================================================

def calc_gamma(d, c, overwrite=False):
    if type(d) is dict:
        if 'gamma' in d:
            print 'gamma already defined! Use overwite for overwrite.'
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
    if raw_input('Save Fig?\n> ') == 'y':
        if fname is None:
            fname = raw_input('Save As:')

        fname = join(path, prefix_fname_with_date(fname))
        print 'Saving {}...'.format(fname)
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
        fname = raw_input('Enter dHybrid out file: ')

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
    for inp in inputs.itervalues():
        for sp in inp:
            k = sp.split('=')
            k,v = [v.strip(' , ') for v in k]

            _fk = k.find('(') 
            if _fk > 0:
                k = k[:_fk]

            param[k] = [_auto_cast(c.strip()) for c in v.split(',')]

            if len(param[k]) == 1:
                param[k] = param[k][0]
    
    return param

#======================================================================

def _auto_cast(k):
    """Takes an imput string and trys to cast it to a real type

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

    return k[:nn/2], Ff[:nn/2]

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
        print '!!!Warning!!! speed of light not given, using 50'
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
        print '!!!Warning!!! speed of light not given, using 50'
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

    print "We will be working on {} lines...".format(len(tms))
    for _c,tm in enumerate(tms):
        print "{},".format(_c),
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
