import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm 

def plot_density(
        Wan, Wan_idx,
        title=None, save_name=None, mark_home_cell=False,
        mark_center=False, show_lattice=False, omit_sites=None,
        show=False, interpolate=False,
        scatter_size=40, lat_size=2, fig=None, ax=None, cbar=True, return_fig=False
        ):

    center = Wan.centers[Wan_idx]

    if not hasattr(Wan, "positions"):
        Wan.get_supercell(Wan_idx, omit_sites=omit_sites)

    positions = Wan.positions

    # Extract arrays for plotting or further processing
    xs = positions['all']['xs']
    ys = positions['all']['ys']
    w0i_wt = positions['all']['wt']

    xs_ev_home = positions['home even']['xs']
    ys_ev_home = positions['home even']['ys']
    xs_odd_home = positions['home odd']['xs']
    ys_odd_home = positions['home odd']['ys']

    xs_omit = positions['omit']['xs']
    ys_omit = positions['omit']['ys']
    w0omit_wt = positions['omit']['wt']

    xs_ev = positions['even']['xs']
    ys_ev = positions['even']['ys']
    w0ev_wt = positions['even']['wt']

    xs_odd = positions['odd']['xs']
    ys_odd = positions['odd']['ys']
    w0odd_wt = positions['odd']['wt']
        
    if fig is None:
        fig, ax = plt.subplots()

    # Weight plot
    if interpolate:
        from scipy.interpolate import griddata
        grid_x, grid_y = np.mgrid[min(xs):max(xs):2000j, min(ys):max(ys):2000j]
        grid_z = griddata((xs, ys), w0i_wt, (grid_x, grid_y), method='linear')
        dens_plot = plt.pcolormesh(grid_x, grid_y, grid_z, cmap='plasma', norm=LogNorm(vmin=2e-16, vmax=1))
    else:
        dens_plot = ax.scatter(xs, ys, c=w0i_wt, s=scatter_size, cmap='plasma', norm=LogNorm(vmin=2e-16, vmax=1), marker='h', zorder=0)

    if show_lattice:
        scat = ax.scatter(xs_ev, ys_ev, marker='o', c='k', s=lat_size, zorder=2)
        scat = ax.scatter(xs_odd, ys_odd, marker='o', s=lat_size, zorder=2, facecolors='none', edgecolors='k')

    if omit_sites is not None :
        ax.scatter(xs_omit, ys_omit, s=2, marker='x', c='g')

    if mark_home_cell:
        scat = ax.scatter(xs_ev_home, ys_ev_home, marker='o', s=lat_size, zorder=2, facecolors='none', edgecolors='b')
        scat = ax.scatter(xs_odd_home, ys_odd_home, marker='o', s=lat_size, zorder=2, facecolors='none', edgecolors='r')

    if cbar:
        cbar = plt.colorbar(dens_plot, ax=ax)
        # cbar.set_label(rf"$|\langle \phi_{{\vec{{R}}, j}}| w_{{0, {Wan_idx}}}\rangle|^2$", rotation=270)
        cbar.set_label(rf"$|w_{Wan_idx}(\mathbf{{r}} )|^2$", rotation=270)
        cbar.ax.get_yaxis().labelpad = 20

    ax.set_title(title)

    if mark_center:
        ax.scatter(center[0], center[1],
            marker='x', 
            label=fr"Center $\mathbf{{r}}_c = ({center[0]: .3f}, {center[1]: .3f})$", c='g', zorder=1)
        ax.legend(loc='upper right')

    # Saving
    if save_name is not None:
        plt.savefig(f'Wan_wt_{save_name}.png')

    if show:
        plt.show()

    if return_fig:
        return fig, ax


def plot_phase():
    # Phase plot
    #     fig2, ax2 = plst.subplots()
    #     figs.append(fig2)
    #     axs.append(ax2)

    #     scat = ax2.scatter(xs, ys, c=w0i_phase, cmap='hsv')

    #     cbar = plt.colorbar(scat, ax=ax2)
    #     cbar.set_label(
    #         rf"$\phi = \tan^{{-1}}(\mathrm{{Im}}[w_{{0, {Wan_idx}}}(r)]\  / \ \mathrm{{Re}}[w_{{0, {Wan_idx}}}(r)])$", 
    #         rotation=270)
    #     cbar.ax.get_yaxis().labelpad = 20
    #     ax2.set_title(title)

    #     # Saving
    #     if save_name is not None:
    #         plt.savefig(f'Wan_wt_{save_name}.png')
        
    #     if show:
    #         plt.show()
    return


def plot_decay(
        Wan, Wan_idx, fit_deg=None, fit_rng=None, ylim=None, 
        omit_sites=None, fig=None, ax=None, title=None, show=False, 
        return_fig=True
        ):
    
    if fig is None:
        fig, ax = plt.subplots()

    if not hasattr(Wan, "positions"):
        Wan.get_supercell(Wan_idx, omit_sites=omit_sites)

    # Extract arrays for plotting or further processing
    positions = Wan.positions
    r = positions['all']['r']
    r_omit = positions['omit']['r']
    r_ev = positions['even']['r']
    r_odd = positions['odd']['r']

    w0i_wt = positions['all']['wt']
    w0omit_wt = positions['omit']['wt']
    w0ev_wt = positions['even']['wt']
    w0odd_wt = positions['odd']['wt']

    # binning data
    max_r = np.amax(r)
    num_bins = int(np.ceil(max_r))
    r_bins = [[i, i + 1] for i in range(num_bins)]
    r_ledge = [i for i in range(num_bins)]
    r_cntr = [0.5 + i for i in range(num_bins)]
    w0i_wt_bins = [[] for i in range(num_bins)]

    # bins of weights
    for i in range(r.shape[0]):
        for j, r_intvl in enumerate(r_bins):
            if r_intvl[0] <= r[i] < r_intvl[1]:
                w0i_wt_bins[j].append(w0i_wt[i])
                break

    # average value of bins
    avg_w0i_wt_bins = []
    for i in range(num_bins):
        if len(w0i_wt_bins[i]) != 0:
            avg_w0i_wt_bins.append(sum(w0i_wt_bins[i])/len(w0i_wt_bins[i]))

    # numpify
    avg_w0i_wt_bins = np.array(avg_w0i_wt_bins)
    r_ledge = np.array(r_ledge)
    r_cntr = np.array(r_cntr)

    if fit_rng is None:
        cutoff = int(0.7*max_r)
        init_r = int(0.2*max_r)
        fit_rng = [init_r, cutoff]
    else:
        cutoff = fit_rng[-1]

    # scatter plot
    # plt.scatter(r, w0i_wt, zorder=1, s=10)
    if omit_sites is not None:
        ax.scatter(r_omit[r_omit<cutoff], w0omit_wt[r_omit<cutoff], zorder=1, s=10, c='g', label='omitted site')

    ax.scatter(r[r<cutoff], w0i_wt[r<cutoff], zorder=1, s=10, c='b')

    # ax.scatter(r_ev[r_ev<cutoff], w0ev_wt[r_ev<cutoff], zorder=1, s=10, c='b')
    # ax.scatter(r_odd[r_odd<cutoff], w0odd_wt[r_odd<cutoff], zorder=1, s=10, c='b')

    # bar of avgs
    ax.bar(r_ledge[r_ledge<cutoff], avg_w0i_wt_bins[r_ledge<cutoff], width=1, align='edge', ec='k', zorder=0, ls='-', alpha=0.3)

    # fit line
    if fit_deg is None:
        deg = 1 # polynomial fit degree
    r_fit = r_cntr[np.logical_and(r_cntr > fit_rng[0], r_cntr < fit_rng[1])]
    w0i_wt_fit = avg_w0i_wt_bins[np.logical_and(r_cntr > fit_rng[0], r_cntr < fit_rng[1])]
    fit = np.polyfit(r_fit, np.log(w0i_wt_fit), deg)
    fit_line = np.sum(np.array([r_fit**(deg-i) * fit[i] for i in range(deg+1)]), axis=0)
    fit_label = rf"$Ce^{{{fit[-2]: 0.2f} r  {'+'.join([fr'{c: .2f} r^{deg-j}' for j, c in enumerate(fit[:-3])])}}}$"
    ax.plot(r_fit, np.exp(fit_line), c='lime', ls='--', lw=2.5, label=fit_label)

    ax.legend()
    ax.set_xlabel(r'$|\mathbf{r}- \mathbf{{r}}_c|$', size=12)
    ax.set_ylabel(rf"$|w_{Wan_idx}(\mathbf{{r}}- \mathbf{{r}}_c)|^2$", size=12)
    # ax.set_xlabel(r'$|\vec{R}+\vec{\tau}_j|$')
    # ax.set_xlim(-4e-1, cutoff)
    if ylim is None:
        ax.set_ylim(0.8*min(w0i_wt[r<cutoff]), 1.5)
    else:
        ax.set_ylim(ylim)
    ax.set_yscale('log')

    ax.set_title(title)

    if show:
        plt.show()

    if return_fig:
        return fig, ax


def plot_centers(Wan, omit_sites=None, center_scale=200,
        section_home_cell=True, color_home_cell=True, translate_centers=False,
        title=None, save_name=None, show=False, legend=False, pmx=4, pmy=4,
        kwargs_centers={'s': 80, 'marker': '*', 'c': 'g'},
        kwargs_omit={'s': 50, 'marker': 'x', 'c':'k'},
        kwargs_lat_ev={'s':10, 'marker': 'o', 'c':'k'}, 
        kwargs_lat_odd={'s':10, 'marker': 'o', 'facecolors':'none', 'edgecolors':'k'},
        fig=None, ax=None
        ):
    lat_vecs = Wan.model.get_lat_vecs()
    orbs = Wan.model.get_orb_vecs(Cartesian=False)
    centers = Wan.centers

    # Initialize arrays to store positions and weights
    positions = {
        'all': {'xs': [], 'ys': []},
        'centers': {'xs': [[] for i in range(centers.shape[0])], 'ys':[[] for i in range(centers.shape[0])]},
        'home even': {'xs': [], 'ys': []},
        'home odd': {'xs': [], 'ys': []},
        'omit': {'xs': [], 'ys': []},
        'even': {'xs': [], 'ys': []},
        'odd': {'xs': [], 'ys': []},
    }
    for tx, ty in Wan.supercell:
        if translate_centers:
            for j in range(centers.shape[0]):
                center = centers[j] + tx * lat_vecs[0] + ty * lat_vecs[1]
                positions['centers']['xs'][j].append(center[0])
                positions['centers']['ys'][j].append(center[1])
        for i, orb in enumerate(orbs):
            # Extract relevant parameters
            pos = orb[0] * lat_vecs[0] + tx * lat_vecs[0] + orb[1] * lat_vecs[1] + ty * lat_vecs[1]
            x, y = pos[0], pos[1]

            # Store values in 'all'
            positions['all']['xs'].append(x)
            positions['all']['ys'].append(y)

            # Handle omit site if applicable
            if omit_sites is not None and i in omit_sites:
                positions['omit']['xs'].append(x)
                positions['omit']['ys'].append(y)
            # Separate even and odd index sites
            if i % 2 == 0:
                positions['even']['xs'].append(x)
                positions['even']['ys'].append(y)
                if tx == ty == 0:
                    positions['home even']['xs'].append(x)
                    positions['home even']['ys'].append(y)
            else:
                positions['odd']['xs'].append(x)
                positions['odd']['ys'].append(y)
                if tx == ty == 0:
                    positions['home odd']['xs'].append(x)
                    positions['home odd']['ys'].append(y)


    # Convert lists to numpy arrays (batch processing for cleanliness)
    for key, data in positions.items():
        for sub_key in data:
            positions[key][sub_key] = np.array(data[sub_key])

    # All positions
    xs = positions['all']['xs']
    ys = positions['all']['ys']

    # home cell site positions
    xs_ev_home = positions['home even']['xs']
    ys_ev_home = positions['home even']['ys']
    xs_odd_home = positions['home odd']['xs']
    ys_odd_home = positions['home odd']['ys']

    # omitted site positions
    xs_omit = positions['omit']['xs']
    ys_omit = positions['omit']['ys']

    # sublattice positions
    xs_ev = positions['even']['xs']
    ys_ev = positions['even']['ys']
    xs_odd = positions['odd']['xs']
    ys_odd = positions['odd']['ys']

        
    if fig is None:
        fig, ax = plt.subplots()

    # Weight plot

    if omit_sites is not None :
        ax.scatter(xs_omit, ys_omit, **kwargs_omit)

    if color_home_cell:
        # Zip the home cell coordinates into tuples
        home_ev_coords = set(zip(xs_ev_home, ys_ev_home))

        # Filter even sites: Keep (x, y) pairs that are not in home_coordinates
        out_even = [(x, y) for x, y in zip(xs_ev, ys_ev) if (x, y) not in home_ev_coords]
        if out_even:
            xs_ev_out, ys_ev_out = zip(*out_even)
        else:
            xs_ev_out, ys_ev_out = [], []  # In case no points are left

        # Zip the home cell coordinates into tuples
        home_odd_coords = set(zip(xs_odd_home, ys_odd_home))

        # Filter even sites: Keep (x, y) pairs that are not in home_coordinates
        out_odd = [(x, y) for x, y in zip(xs_odd, ys_odd) if (x, y) not in home_odd_coords]
        if out_even:
            xs_odd_out, ys_odd_out = zip(*out_odd)
        else:
            xs_odd_out, ys_odd_out = [], []  # In case no points are left

        ax.scatter(xs_ev_home, ys_ev_home, zorder=2, **kwargs_lat_ev)
        ax.scatter(xs_odd_home, ys_odd_home, zorder=2, **kwargs_lat_odd)
        
        if 'c' in kwargs_lat_ev.keys():
            kwargs_lat_ev.pop('c')
        if 'c' in kwargs_lat_odd.keys():
            kwargs_lat_odd.pop('c')

        ax.scatter(xs_ev_out, ys_ev_out, zorder=2, **kwargs_lat_ev)
        ax.scatter(xs_odd_out, ys_odd_out, zorder=2, **kwargs_lat_odd)
    
    else:
        ax.scatter(xs_ev, ys_ev, zorder=2, **kwargs_lat_ev)
        ax.scatter(xs_odd, ys_odd, zorder=2, **kwargs_lat_odd)

    # draw lines sectioning out home supercell
    if section_home_cell:
        c1 = np.array([0,0])
        c2 = c1 + lat_vecs[0]
        c3 = c1 + lat_vecs[1]
        c4 = c1 + lat_vecs[0] + lat_vecs[1]

        ax.plot([c1[0], c2[0]], [c1[1], c2[1]], c='k', ls='--', lw=1)
        ax.plot([c1[0], c3[0]], [c1[1], c3[1]], c='k', ls='--', lw=1)
        ax.plot([c3[0], c4[0]], [c3[1], c4[1]], c='k', ls='--', lw=1)
        ax.plot([c2[0], c4[0]], [c2[1], c4[1]], c='k', ls='--', lw=1)

    # scatter centers
    for i in range(centers.shape[0]):
        if translate_centers:
            x = positions['centers']['xs'][i]
            y = positions['centers']['ys'][i]
            if i==0:
                label = "Wannier centers"
            else:
                label=None
            ax.scatter(
                x, y, zorder=1, label=label, s=np.exp(11*Wan.spread[i])*center_scale, **kwargs_centers)
        else:
            center = centers[i]
            label = "Wannier centers"
            ax.scatter(
                center[0], center[1], zorder=1, label=label, **kwargs_centers)

    if legend:
        ax.legend(loc='upper right')
    
    center_sc = (1/2) * (lat_vecs[0] + lat_vecs[1])
    ax.set_xlim(center_sc[0] - pmx, center_sc[0] + pmx)
    ax.set_ylim(center_sc[1] - pmy, center_sc[1] + pmy)

    ax.set_title(title)

    # Saving
    if save_name is not None:
        plt.savefig(f'{save_name}.png', dpi=700)

    if show:
        plt.show()
    
    return fig, ax
    

def plot_interp_bands(Wan, k_path, nk=101, k_label=None, red_lat_idx=None, 
    fig=None, ax=None, title=None, scat_size=3, 
    lw=2, lc='b', ls='solid', cmap="plasma", show=False, cbar=True):

    return Wan.tilde_states.plot_interp_bands(
        k_path, nk=nk, k_label=k_label, red_lat_idx=red_lat_idx, fig=fig, ax=ax,
        title=title, scat_size=scat_size, lw=lw, lc=lc, ls=ls, cmap=cmap, show=show, cbar=cbar
        )