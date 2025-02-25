import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

def plot_bands(
        model, k_path, 
        nk=101, evals=None, evecs=None, k_vec=None,
        k_label=None, title=None, save_name=None, sub_lat=False,
        red_lat_idx=None, blue_lat_idx=None, show=False
        ):
    """

    Args:
        model (pythtb.model):
        k_path (list): list of high symmetry points to interpolate
        evals (np.array, optional): If specifying, indices must correspond to interpolated path.
        evecs (np.array, optional): If specifying, indices must correspond to interpolated path.
        k_vec (np.array, optional): k_vec corresponding to evals and evecs 
        k_label (_type_, optional): _description_. Defaults to None.
        title (_type_, optional): _description_. Defaults to None.
        save_name (_type_, optional): _description_. Defaults to None.
        sub_lat (bool, optional): _description_. Defaults to False.
        red_lat_idx (_type_, optional): _description_. Defaults to None.
        blue_lat_idx (_type_, optional): _description_. Defaults to None.
        show (bool, optional): _description_. Defaults to False.

    Returns:
        fig, ax: matplotlib fig and ax
    """
    
    fig, ax = plt.subplots()

    (k_vec, k_dist, k_node) = model.k_path(k_path, nk, report=False)

    ax.set_xlim(0, k_node[-1])
    ax.set_xticks(k_node)
    if k_label is not None:
        ax.set_xticklabels(k_label)
    for n in range(len(k_node)):
        ax.axvline(x=k_node[n], linewidth=0.5, color='k')

    if evecs is None and evals is None:
        # must diagonalize model on interpolated path
        # generate k-path and labels
    
        # diagonalize
        evals, evecs = model.solve_all(k_vec, eig_vectors=True)
        evecs = np.transpose(evecs, axes=(1, 0, 2)) #[k, n, orb]
        evals = np.transpose(evals, axes=(1, 0)) #[k, n]

    n_eigs = evecs.shape[1]

    if sub_lat:
        # scattered bands with sublattice color
        for k in range(nk):
            for n in range(n_eigs): # band idx
                # color is weight on all high energy (odd) sites in unit cell
                col = sum([ abs(evecs[k, n, i])**2 for i in red_lat_idx ])
                scat = ax.scatter(k_dist[k], evals[k, n], c=col, cmap='bwr', marker='o', s=3, vmin=0, vmax=1)

        cbar = fig.colorbar(scat, ticks=[1,0])
        cbar.ax.set_yticklabels([r'$\psi_1$', r'$\psi_2$'])
        cbar.ax.get_yaxis().labelpad = 20

    else:
        # continuous bands
        for n in range(n_eigs):
          ax.plot(k_dist, evals[:, n], c='blue')

    ax.set_title(title)
    ax.set_ylabel(r"Energy $E(\mathbf{{k}})$ ")

    if save_name is not None:
        plt.savefig(save_name)

    if show:
        plt.show()

    return fig, ax
    

def plot_Wan(
        w0, Wan_idx, orbs, lat_vecs, plot_phase=False, plot_decay=False,
        title=None, save_name=None, omit_site=None,
        fit_deg=None, fit_rng=None, ylim=None, show=False):
    
    nx, ny = w0.shape[0], w0.shape[1]

    supercell = [(i,j) for i in range(-int((nx-nx%2)/2), int((nx-nx%2)/2)) 
                for j in range(-int((ny-ny%2)/2), int((ny-ny%2)/2))]

    xs = []
    ys = []
    r = []
    w0i_wt = []
    w0i_phase = []

    xs_omit = []
    ys_omit = []
    r_omit = []
    w0omit_wt = []

    r_ev = []
    r_odd = []
    w0ev_wt = []
    w0odd_wt = []

    for tx, ty in supercell:
        for i, orb in enumerate(orbs):
            phase = np.arctan(w0[tx, ty, Wan_idx, i].imag/w0[tx, ty, Wan_idx, i].real) 
            wt = np.abs(w0[tx, ty, Wan_idx, i])**2
            pos = orb[0]*lat_vecs[0] + tx*lat_vecs[0] + orb[1]*lat_vecs[1]+ ty*lat_vecs[1]

            x, y, rad = pos[0], pos[1], np.sqrt(pos[0]**2 + pos[1]**2)

            xs.append(x)
            ys.append(y)
            r.append(rad)
            w0i_wt.append(wt)
            w0i_phase.append(phase)

            if omit_site is not None and i == omit_site:
                xs_omit.append(x)
                ys_omit.append(y)
                r_omit.append(rad)
                w0omit_wt.append(wt)
            elif i%2 ==0:
                r_ev.append(rad)
                w0ev_wt.append(wt)
            else:
                r_odd.append(rad)
                w0odd_wt.append(wt)

    # numpify
    xs = np.array(xs)
    ys = np.array(ys)
    r = np.array(r)
    w0i_wt = np.array(w0i_wt)
    xs_omit = np.array(xs_omit)
    ys_omit = np.array(ys_omit)
    r_omit = np.array(r_omit)
    w0omit_wt = np.array(w0omit_wt)
    r_ev = np.array(r_ev)
    w0ev_wt = np.array(w0ev_wt)
    r_odd = np.array(r_odd)
    w0odd_wt = np.array(w0odd_wt)

    figs = []
    axs = []

    fig, ax = plt.subplots()
    figs.append(fig)
    axs.append(ax)

    # Weight plot
    scat = ax.scatter(xs, ys, c=w0i_wt, cmap='plasma', norm=LogNorm(vmin=2e-16, vmax=1))

    if omit_site is not None :
        ax.scatter(xs_omit, ys_omit, s=2, marker='x', c='g')

    cbar = plt.colorbar(scat, ax=ax)
    # cbar.set_label(rf"$|\langle \phi_{{\vec{{R}}, j}}| w_{{0, {Wan_idx}}}\rangle|^2$", rotation=270)
    cbar.set_label(rf"$|w_{Wan_idx}(\mathbf{{r}})|^2$", rotation=270)
    cbar.ax.get_yaxis().labelpad = 20
    ax.set_title(title)

    # Saving
    if save_name is not None:
        plt.savefig(f'Wan_wt_{save_name}.png')

    if show:
        plt.show()

    if plot_phase:
        # Phase plot
        fig2, ax2 = plt.subplots()
        figs.append(fig2)
        axs.append(ax2)

        scat = ax2.scatter(xs, ys, c=w0i_phase, cmap='hsv')

        cbar = plt.colorbar(scat, ax=ax2)
        cbar.set_label(
            rf"$\phi = \tan^{{-1}}(\mathrm{{Im}}[w_{{0, {Wan_idx}}}(r)]\  / \ \mathrm{{Re}}[w_{{0, {Wan_idx}}}(r)])$", 
            rotation=270)
        cbar.ax.get_yaxis().labelpad = 20
        ax2.set_title(title)

        # Saving
        if save_name is not None:
            plt.savefig(f'Wan_wt_{save_name}.png')
        
        if show:
            plt.show()

    if plot_decay:
        fig3, ax3 = plt.subplots()
        figs.append(fig3)
        axs.append(ax3)

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
        if omit_site is not None:
            ax3.scatter(r_omit[r_omit<cutoff], w0omit_wt[r_omit<cutoff], zorder=1, s=10, c='g', label='omitted site')

        ax3.scatter(r_ev[r_ev<cutoff], w0ev_wt[r_ev<cutoff], zorder=1, s=10, c='b', label='low energy sites')
        ax3.scatter(r_odd[r_odd<cutoff], w0odd_wt[r_odd<cutoff], zorder=1, s=10, c='r', label='high energy sites')

        # bar of avgs
        ax3.bar(r_ledge[r_ledge<cutoff], avg_w0i_wt_bins[r_ledge<cutoff], width=1, align='edge', ec='k', zorder=0, ls='-', alpha=0.3)

        # fit line
        if fit_deg is None:
            deg = 1 # polynomial fit degree
        
        r_fit = r_cntr[np.logical_and(r_cntr > fit_rng[0], r_cntr < fit_rng[1])]
        w0i_wt_fit = avg_w0i_wt_bins[np.logical_and(r_cntr > fit_rng[0], r_cntr < fit_rng[1])]
        fit = np.polyfit(r_fit, np.log(w0i_wt_fit), deg)
        fit_line = np.sum(np.array([r_fit**(deg-i) * fit[i] for i in range(deg+1)]), axis=0)
        fit_label = rf"$Ce^{{{fit[-2]: 0.2f} r  {'+'.join([fr'{c: .2f} r^{deg-j}' for j, c in enumerate(fit[:-3])])}}}$"
        ax3.plot(r_fit, np.exp(fit_line), c='lime', ls='--', lw=2.5, label=fit_label)

        ax3.legend()
        ax3.set_xlabel(r'$|\mathbf{r}|$')
        ax3.set_ylabel(rf"$|w_{Wan_idx}(\mathbf{{r}})|^2$")
        # ax.set_xlabel(r'$|\vec{R}+\vec{\tau}_j|$')
        # ax.set_xlim(-4e-1, cutoff)
        if ylim is None:
            ax3.set_ylim(0.8*min(w0i_wt[r<cutoff]), 1.5)
        else:
            ax3.set_ylim(ylim)
        ax3.set_yscale('log')

        ax3.set_title(title)

        if save_name is not None:
            plt.savefig(f'Wan_decay_{save_name}.png')

        if show:
            plt.show()
    
    return figs, axs