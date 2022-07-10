import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
import sub
plt.rcParams.update({'font.size': 8})

# Fields that we want to load

class MultiColor(object):
    def __init__(self, path, time, side='lower'):
        self.path = path
        self.time = time
        self.side = side

        self.load_data()
        self.init_figure()
        self.calc_values()
        self.build_plot_info()
        self.plot_data()
        self.clean_up_fig()
        self.save_figure()

#=====================================================================#

    def load_data(self):
        param = sub.read_input(path=self.path)
        self.param = param
        f = sub.field_loader(path=self.path, num=self.time)
        u = sub.flow_loader(path=self.path, num=self.time)
        n = sub.dens_loader('x3x2x1', path=self.path, num=self.time)
        p = sub.pres_loader(path=self.path, num=self.time)

        d = {}
        for k in 'bx by bz ex ey ez'.split():
            d[k] = f[k]

        for k in 'ux uy uz'.split():
            d[k] = u[k]

        d['n'] = n['x3x2x1']

        for k in 'pxx pyy pzz pxy pyz pzx px py pz'.split():
            d[k] = p[k]

        d['bx_xx'] = f['bx_xx']
        d['bx_yy'] = f['bx_yy']
        d['xx'] = f['bx_xx']
        d['yy'] = f['bx_yy']
        d['time'] = self.param['dt']*self.time
        d['dx'] = d['xx'][1] - d['xx'][0]

        self.d = d

#=====================================================================#

    def init_figure(self):
        fig = plt.figure(1)
        fig.clf()
        fig.set_size_inches(11, 10.)

        self.fig = fig
        self.ax = [fig.add_subplot(9,3,1+_) for _ in range(9*3)]

#=====================================================================#

    def calc_values(self):
        d = self.d

        sub.rotate_ten(d, d)
        dx = self.d['dx']

        ddx = lambda f : (np.roll(f, -1, 1) - np.roll(f, 1, 1))/(2*dx)
        ddy = lambda f : (np.roll(f, -1, 0) - np.roll(f, 1, 0))/(2*dx)
        ddz = lambda f : 0.*f

        d['bm'] = np.sqrt(d['bx']**2 + d['by']**2 + d['bz']**2)
        d['em'] = np.sqrt(d['ex']**2 + d['ey']**2 + d['ez']**2)

        d['jx'] = ddy(d['bz']) - ddz(d['by'])
        d['jy'] = ddz(d['bx']) - ddx(d['bz'])
        d['jz'] = ddx(d['by']) - ddy(d['bx'])

        d['uxbx'] = -d['uy']*d['bz'] + d['uz']*d['by']
        d['uxby'] = -d['uz']*d['bx'] + d['ux']*d['bz']
        d['uxbz'] = -d['ux']*d['by'] + d['uy']*d['bx']

        d['jxbx'] = (d['jy']*d['bz'] - d['jz']*d['by'])/d['n'] 
        d['jxby'] = (d['jz']*d['bx'] - d['jx']*d['bz'])/d['n']
        d['jxbz'] = (d['jx']*d['by'] - d['jy']*d['bx'])/d['n']

        d['tpar'] = d['ppar']/d['n']
        d['tperp'] = d['pperp1']/d['n']

        d['firehose'] = 1. - (d['tpar'] - d['tperp'])/d['bm']**2
        d['mirror'] = 1. - 2.*(d['tperp'] - d['tpar'])/d['bm']**2

        d['psi'] = sub.calc_psi(d)
    
#=====================================================================#

    class PI(object):
        def __init__(self, key, title, cmap, sig=None, posdef=False):
            self.key = key
            self.title = title
            self.cmap = cmap
            self.sig = sig
            self.posdef = posdef

    def build_plot_info(self):
        PI = self.PI
        info = [PI('bx', '$B_x$', 'bwr'),
                PI('by', '$B_y$', 'bwr'),
                PI('bz', '$B_z$', sub.batlow),
                PI('bm', '$|B|$', sub.batlow),
                PI('ex', '$E_x$', 'bwr'),
                PI('ey', '$E_y$', 'bwr'),
                PI('ez', '$E_z$', 'bwr'),
                #PI('em', '$|E|$', sub.batlow),
                PI('ux', '$U_x$', 'bwr'),
                PI('uy', '$U_y$', 'bwr'),
                PI('uz', '$U_z$', 'bwr'),
                PI('jx', '$J_x$', 'bwr'),
                PI('jy', '$J_y$', 'bwr'),
                PI('jz', '$J_z$', 'bwr'),
                PI('uxbx', '$-u\\times B_x$', 'bwr'),
                PI('uxby', '$-u\\times B_y$', 'bwr'),
                PI('uxbz', '$-u\\times B_z$', 'bwr'),
                PI('jxbx', '$J/n\\times B_x$', 'bwr'),
                PI('jxby', '$J/n\\times B_y$', 'bwr'),
                PI('jxbz', '$J/n\\times B_z$', 'bwr'),
                PI('pxx', '$P_{xx}$', sub.batlow),
                PI('pyy', '$P_{yy}$', sub.batlow),
                PI('pzz', '$P_{zz}$', sub.batlow),
                PI('n', '$n$', sub.batlow),
                PI('tpar', '$T_{\parallel}$', sub.batlow),
                PI('tperp', '$T_{\perp}$', sub.batlow),
                PI('firehose', 'Firehose ($1 - \\beta_\parallel/2'
                              ' + \\beta_\perp/2$)', sub.batlow),
                PI('mirror', 'Mirror ($1 - \\beta_\perp'
                              ' + \\beta_\parallel/2$)', sub.batlow)]

        self.plot_info = info

#=====================================================================#

    def plot_data(self):
        d = self.d
        for ctr in range(len(self.plot_info)):
            a = self.ax[ctr]
            pi = self.plot_info[ctr]
            k = pi.key
            cmap = pi.cmap

            #self.sig = sig
            #self.posdef = posdef

            vmean = np.mean(d[k])
            vstd =  np.std(d[k])
            vmax = np.max(d[k])
            vmin = np.min(d[k])
            if np.abs(vmean) < vstd/2.:
                vmax = np.max([np.abs(vmin), vmax])
                vmin = -vmax

            ctargs = {'levels':20, 'linewidths':.5}
            self.ims(d, k, ax=a, cmap=cmap, vmin=vmin, vmax=vmax, 
                    ctargs=ctargs, cbar=1)

#=====================================================================#

    def ims(self, d, k, ax, cbar=None, cont=None, ctargs=None, **kwargs):

        if type(k) is str: 
            plt_val = d[k].T
        else: 
            plt_val = k.T

        # Use the dict values of xx and yy to set extent
        ext = [d['xx'][0],
               d['xx'][-1],
               d['yy'][0],
               d['yy'][-1]]

        if ctargs is None: ctargs = {}

        im = ax.imshow(plt_val.T, origin='lower', extent=ext, **kwargs)

        # Code to implement for a cbar
        if cbar:
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", "3%", pad="1.5%")
            plt.colorbar(im, cax=cax)

            cax.xaxis.set_tick_params(which='both',labelsize=6)
            cax.yaxis.set_tick_params(which='both',labelsize=6)
            cax.minorticks_on()

        if ctargs:
            if 'colors' not in ctargs: ctargs['colors'] = 'k'
            if 'linestyles' not in ctargs: ctargs['linestyles'] = 'solid'
           
            cts = ax.contour(d['xx'], d['yy'], d['psi'], **ctargs)

#======================================================

    def clean_up_fig(self):
        ly = len(self.d['yy'])
        if self.side == 'lower':
            ylim = self.d['yy'][0], self.d['yy'][ly//2] 
        else:
            ylim = self.d['yy'][ly//2], self.d['yy'][ly-1] 

        for _c,a in enumerate(self.ax):
            a.minorticks_on()
            ttl = self.plot_info[_c].title
            if _c == 1:
                ttl = ttl + ", Time = {:.1f} $\Omega_c^{{-1}}$".format(self.d['time'])
            a.set_title(ttl, fontsize=6)
            a.set_ylim(ylim)
            a.tick_params(axis='x', labelsize=6)
            a.tick_params(axis='y', labelsize=6)

        plt.tight_layout()

#======================================================

    def save_figure(self):
        sname = self.path.split("/")[-1]
        side = self.side
        time = self.time
        fname = "./multicolor_figs/{}_{}_{}.png"
        fname = fname.format(sname, side, time)
        self.fig.savefig(fname, dpi=400)

#======================================================

if __name__ == "__main__":
    simpath = "/scratch/08570/tg878691/thickgfld1"
    tms = sub.get_output_times(simpath)
    for tm in tms:
        MC = MultiColor(simpath, tm)
