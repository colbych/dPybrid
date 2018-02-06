import h5py
import numpy as np

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

def dens_loader(dens_vars=None, num=None, path='./', sp=1):
    import glob

    if path[-1] is not '/': path = path + '/'
    
    dpath = path+"Output/Phase/*"
    if dens_vars is None:
        dens_vars = [c[len(dpath)-1:] for c in glob.glob(dpath)]
    else:
        if not type(dens_vars) in (list, tuple):
            dens_vars = [dens_vars]

    dpath = path+"Output/Phase/{dv}/Sp{sp:02d}/dens_sp{sp:02d}_{tm}.h5"
    
    print dpath.format(dv=dens_vars[0], sp=sp, tm='*')

    choices = glob.glob(dpath.format(dv=dens_vars[0], sp=sp, tm='*'))
    choices = [int(c[-11:-3]) for c in choices]
    choices.sort()

    dpath = path+"Output/Phase/{dv}/Sp{sp:02d}/dens_sp{sp:02d}_{tm:08}.h5"

    d = {}
    while num not in choices:
        _ =  'Select from the following possible movie numbers: '\
             '\n{0} '.format(choices)
        num = int(raw_input(_))

    for k in dens_vars:
        print dpath.format(dv=k,sp=sp,tm=num)
        with h5py.File(dpath.format(dv=k,sp=sp,tm=num),'r') as f:
            d[k] = f['DATA'][:]

            _N2,_N1 = f['DATA'][:].shape #python is fliped
            x1,x2 = f['AXIS']['X1 AXIS'][:],f['AXIS']['X2 AXIS'][:]
            dx1 = (x1[1]-x1[0])/_N1
            dx2 = (x2[1]-x2[0])/_N2
            d[k+'_xx'] = dx1*np.arange(_N1) + dx1/2. + x1[0]
            d[k+'_yy'] = dx2*np.arange(_N2) + dx2/2. + x2[0]

    return d

#======================================================================

def field_loader(field_vars='all', components='all', num=None, path='./'):
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

    fpath = path+"Output/Fields/{f}/{T}{c}/{v}fld_{t}.h5"

    T = '' if field_vars[0] == 'J' else 'Total/'
    test_path = fpath.format(f = _field_choices_[field_vars[0]],
                             T = T,
                             c = 'x',
                             v = field_vars[0],
                             t = '*')
    
    print test_path
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
            print ffn
            with h5py.File(ffn,'r') as f:
                d[kc] = f['DATA'][:]

                _N2,_N1 = f['DATA'][:].shape #python is fliped
                x1,x2 = f['AXIS']['X1 AXIS'][:],f['AXIS']['X2 AXIS'][:]
                dx1 = (x1[1]-x1[0])/_N1
                dx2 = (x2[1]-x2[0])/_N2
                d[kc+'_xx'] = dx1*np.arange(_N1) + dx1/2. + x1[0]
                d[kc+'_yy'] = dx2*np.arange(_N2) + dx2/2. + x2[0]

    return d


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
        import matplotlib.pyplot as plt
        ax = plt.gca()
    
    rax = np.s_[::corse_res[0]]
    ray = np.s_[::corse_res[1]]

    pvar = k
    if type(k) is str:
        pvar = d[k]

    pc = ax.pcolormesh(d[k+'_xx'][rax], d[k+'_yy'][ray], d[k][ray,rax], **kwargs)

    return pc

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

def read_input(path='./input/input'):
    """Parse dHybrid input file for simulation information

    Args:
        path (str): path of input file
    """

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
