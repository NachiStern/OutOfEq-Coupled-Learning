import os
from IPython.display import display, HTML

import numpy as np
import scipy as sp

import matplotlib.pyplot as plt; plt.rcdefaults()
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
from matplotlib import collections as mc
import matplotlib as mpl
import seaborn as sns
import scipy.spatial as spatial

import sys
#sys.path.insert(0, '../')
#sys.path.insert(0, '../python_src/')

#import network_solver as ns




    

def createGIF(iname, oname, show=True, delay=5):
        
    os.system("convert -delay {0} -loop 0 {1} {2}".format(delay, iname, oname))

    if show:
        
        IMG_TAG = """<img src="{0}" alt="some_text">""".format(oname)

        display(HTML(IMG_TAG))
        
        
        
        
gray = "#969696"
blue = "#377eb8"
red = "#e41a1c"


def lighten_color(color, amount=0.5):
    """
    Lightens the given color by multiplying (1-luminosity) by the given amount.
    Input can be matplotlib color string, hex string, or RGB tuple.

    Examples:
    >> lighten_color('g', 0.3)
    >> lighten_color('#F034A3', 0.6)
    >> lighten_color((.3,.55,.1), 0.5)
    """
    import matplotlib.colors as mc
    import colorsys
    try:
        c = mc.cnames[color]
    except:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], 1 - amount * (1 - c[1]), c[2])



def is_in_bounds(pos_list, left, right, lower, upper):
        
    for pos in pos_list:
        if pos[0] > left and pos[0] < right and pos[1] > lower and pos[1] < upper:
            return True
        
    return False


def show_network(ax, net, disp=None, strain=None, styles={}, box_mult=1.0, periodic=True, oriented=False, alpha=0.75):
    # use this function
      
    boxsize=0.5
    padding=0.0
    
    edgei = np.copy(net.edgei)
    edgej = np.copy(net.edgej)
    node_pos = np.copy(net.node_pos)
    DIM = net.DIM
    NE = net.NE
    NN = net.NN
    L = np.copy(net.L)
    center = 0.5 * np.ones(DIM, float)
    
     
    if disp is None:
        disp = np.zeros(DIM*NN, float)

    if strain is None:
        strain = np.zeros([DIM, DIM], float)
    else:
        boxL = 0.5 + padding
        corners = np.array([[-boxL, -boxL], 
                            [boxL, -boxL], 
                            [boxL, boxL], 
                            [-boxL, boxL]])
        corners[0] = def_tensor[0:2, 0:2].dot(corners[0])
        corners[1] = def_tensor[0:2, 0:2].dot(corners[1])
        corners[2] = def_tensor[0:2, 0:2].dot(corners[2])
        corners[3] = def_tensor[0:2, 0:2].dot(corners[3])

        ax.add_patch(mpatches.Polygon(corners, True, fill=False, lw=4.0))


    def_tensor = np.identity(DIM) + strain
    patches = []



    for i in range(NN):
        node_pos[DIM*i:DIM*i+DIM] += padding * L
    
    L += 2.0 * padding * L
    
    left = -box_mult*boxsize*1.05
    right = box_mult*boxsize*1.05
    lower = -box_mult*boxsize*1.05
    upper = box_mult*boxsize*1.05
    
    edges = []
    edge_index = []
    
    for i in range(NE):
        posi = node_pos[DIM*edgei[i]:DIM*edgei[i]+DIM]+disp[DIM*edgei[i]:DIM*edgei[i]+DIM]
        posj = node_pos[DIM*edgej[i]:DIM*edgej[i]+DIM]+disp[DIM*edgej[i]:DIM*edgej[i]+DIM]

        posi /= L
        posj /= L

        bvec = posj - posi
        bvec -= np.rint(bvec)

        posi -= np.floor(posi)
        posj -= np.floor(posj)

        posi -= center
        posj -= center
        
        pos_list = [tuple(posi),tuple(posi+bvec)]
        
        if is_in_bounds(pos_list, left, right, lower, upper):
            edges.append(pos_list)    
            edge_index.append(i)
        
        if periodic:
            
            pos_list = [tuple(posi+np.array([1.0, 0.0])),tuple(posi+bvec+np.array([1.0, 0.0]))]
            if is_in_bounds(pos_list, left, right, lower, upper):
                edges.append(pos_list)    
                edge_index.append(i)
                
            pos_list = [tuple(posi+np.array([1.0, -1.0])),tuple(posi+bvec+np.array([1.0, -1.0]))]
            if is_in_bounds(pos_list, left, right, lower, upper):
                edges.append(pos_list)    
                edge_index.append(i)

            pos_list = [tuple(posi+np.array([0.0, -1.0])),tuple(posi+bvec+np.array([0.0, -1.0]))]
            if is_in_bounds(pos_list, left, right, lower, upper):
                edges.append(pos_list)    
                edge_index.append(i)

            pos_list = [tuple(posi+np.array([-1.0, -1.0])),tuple(posi+bvec+np.array([-1.0, -1.0]))]
            if is_in_bounds(pos_list, left, right, lower, upper):
                edges.append(pos_list)    
                edge_index.append(i)

            pos_list = [tuple(posi+np.array([-1.0, 0.0])),tuple(posi+bvec+np.array([-1.0, 0.0]))]
            if is_in_bounds(pos_list, left, right, lower, upper):
                edges.append(pos_list)    
                edge_index.append(i)
                
            pos_list = [tuple(posi+np.array([-1.0, 1.0])),tuple(posi+bvec+np.array([-1.0, 1.0]))]
            if is_in_bounds(pos_list, left, right, lower, upper):
                edges.append(pos_list)    
                edge_index.append(i)
                
            pos_list = [tuple(posi+np.array([0.0, 1.0])),tuple(posi+bvec+np.array([0.0, 1.0]))]
            if is_in_bounds(pos_list, left, right, lower, upper):
                edges.append(pos_list)    
                edge_index.append(i)
                
            pos_list = [tuple(posi+np.array([1.0, 1.0])),tuple(posi+bvec+np.array([1.0, 1.0]))]
            if is_in_bounds(pos_list, left, right, lower, upper):
                edges.append(pos_list)    
                edge_index.append(i)


    for i, b in enumerate(edges):
        edges[i] = [def_tensor.dot(b[0])[0:2], def_tensor.dot(b[1])[0:2]]
    
    ls = []
    colors = []
    lw = []
    
    for i, b in enumerate(edge_index):
        
        if b in styles and 'color' in styles[b]:
            colors.append(styles[b]['color'])
        else:
            colors.append(gray)
            
            
        if b in styles and 'ls' in styles[b]:
            ls.append(styles[b]['ls'])
        else:
            ls.append('solid')
            
        if b in styles and 'lw' in styles[b]:
            lw.append(styles[b]['lw'])
        else:
            lw.append(2.0)
                    
    if not oriented:
        lc = mc.LineCollection(edges, zorder=-1, linestyle=ls, lw=lw, alpha=alpha, color=colors)  
        ax.add_collection(lc)
    else:
        
        X = []
        Y = []
        U = []
        V = []
        
        for i, b in enumerate(edge_index):
            posi = edges[i][0]
            posj = edges[i][1]
            bvec = (posj - posi) * 0.9
            
            
            if b in styles and 'orient' in styles[b]:
                if styles[b]['orient'] == 1:
                    
                    X.append(posi[0])
                    Y.append(posi[1])
                    U.append(bvec[0])
                    V.append(bvec[1])
                else:
                    X.append(posj[0])
                    Y.append(posj[1])
                    U.append(-bvec[0])
                    V.append(-bvec[1])
            else:
                X.append(posi[0])
                Y.append(posi[1])
                U.append(bvec[0])
                V.append(bvec[1])
                    
                    
        ax.quiver(X, Y, U, V, edgecolors=colors, facecolors=colors, units='xy', scale=1.0, linewidths=np.array(lw)/2, alpha=alpha)
    
    
#     if box_mult > 1.0:
#         boxL = 0.5 + padding
#         corners = np.array([[-boxL, -boxL], 
#                             [boxL, -boxL], 
#                             [boxL, boxL], 
#                             [-boxL, boxL]])
#         corners[0] = def_tensor[0:2, 0:2].dot(corners[0])
#         corners[1] = def_tensor[0:2, 0:2].dot(corners[1])
#         corners[2] = def_tensor[0:2, 0:2].dot(corners[2])
#         corners[3] = def_tensor[0:2, 0:2].dot(corners[3])

#         ax.add_patch(mpatches.Polygon(corners, True, fill=False, lw=4.0, ls='dashed'))
        
    
    ax.set_xlim(-box_mult*boxsize, box_mult*boxsize)
    ax.set_ylim(-box_mult*boxsize, box_mult*boxsize)
    
    
def show_oriented_nodes(ax, net, nodes, orientation, styles={}, shadow=False):
    
    edgei = np.copy(net.edgei)
    edgej = np.copy(net.edgej)
    node_pos = np.copy(net.node_pos)
    DIM = net.dim
    L = np.copy(net.L)
    center = 0.5 * np.ones(DIM, float)
    
    for b in nodes:
        
        if b in styles and 'color' in styles[b]:
            color = styles[b]['color']
        else:
            color = 'k'
            
        if b in styles and 'size' in styles[b]:
            size = styles[b]['size']
        else:
            size = 200
            
        if b in styles and 'sign' in styles[b]:
            sign = styles[b]['sign']
        else:
            sign = 1
        
        posi = node_pos[DIM*b:DIM*b+DIM] / L
        posj = node_pos[DIM*orientation[b]:DIM*orientation[b]+DIM] / L

        bvec = posj - posi
        bvec -= np.rint(bvec)
        
        bvec *= sign
        
        angle = np.degrees(np.arctan2(bvec[1], bvec[0]))+30.0
                
        posi -= np.floor(posi)
        posi -= center
        
        x = [posi[0]]
        y = [posi[1]]
                    
        if shadow:
            ax.scatter(x, y, marker=(3, 0, angle), s=1.25*size, facecolor='#636363', alpha=0.5)
            
            
        ax.scatter(x, y, marker=(3, 0, angle), s=size, facecolor=color, alpha=1.0)
    
    
def show_nodes(ax, net, nodes, disp=None, strain=None, styles={}, marker='o', shadow=False, shadow_offset=[0.0, 0.0]):
    
    
    edgei = np.copy(net.edgei)
    edgej = np.copy(net.edgej)
    node_pos = np.copy(net.node_pos)
    DIM = net.DIM
    NE = net.NE
    NN = net.NN
    L = np.copy(net.L)
    center = 0.5 * np.ones(DIM, float)
    
    if disp is None:
        disp = np.zeros(DIM*NN, float)
    
    if strain is None:
        strain = np.zeros([DIM, DIM], float)

    def_tensor = np.identity(DIM) + strain 
        
    x1 = []
    y1 = []

    for i in range(len(nodes)):
        
        posi = node_pos[DIM*nodes[i]:DIM*nodes[i]+DIM]+disp[DIM*nodes[i]:DIM*nodes[i]+DIM]

        posi /= L

        posi -= np.floor(posi)

        posi -= center

        posi = def_tensor.dot(posi)

        x1.append(posi[0])
        y1.append(posi[1])

    colors = []
    sizes = []

    for b in nodes:
        
        if b in styles and 'color' in styles[b]:
            colors.append(styles[b]['color'])
        else:
            colors.append('k')
            
        if b in styles and 'size' in styles[b]:
            sizes.append(styles[b]['size'])
        else:
            sizes.append(200)
    
    if shadow:
        ax.scatter(np.array(x1)+np.full_like(x1, shadow_offset[0]), np.array(y1)+np.full_like(y1, shadow_offset[1]), marker=marker , s=1.25*np.array(sizes), facecolor='#636363', alpha=0.5)
    
    ax.scatter(x1, y1, marker=marker , s=sizes, facecolor=colors, alpha=1.0, linewidths=0.0)

def show_vecs(ax, net, u, strain=None, stream=False):
    
    edgei = np.copy(net.edgei)
    edgej = np.copy(net.edgej)
    node_pos = np.copy(net.node_pos)
    DIM = net.dim
    NE = net.NE
    NN = net.NN
    L = np.copy(net.L)
    center = 0.5 * np.ones(DIM, float)
    
    disp = np.copy(u)
        
    if strain is None:
        strain = np.zeros([DIM, DIM], float)

    def_tensor = np.identity(DIM) + strain 
    
    X = np.zeros(NN, float)
    Y = np.zeros(NN, float)
    U = np.zeros(NN, float)
    V = np.zeros(NN, float)
    
    for i in range(NN):
        
        pos = node_pos[DIM*i:DIM*i+DIM]

        pos /= L

        pos -= np.floor(pos)

        pos -= center
        
        pos = def_tensor.dot(pos)
        
        u = disp[DIM*i:DIM*i+DIM]
        u /= L
        
        X[i] = pos[0]
        Y[i] = pos[1]
        U[i] = u[0]
        V[i] = u[1]
     
    if not stream:
        ax.quiver(X, Y, U, V, units='xy', scale=1.0, width=0.005)
    else:
        
        mag = np.sqrt(U*U + V*V) 
        
        x = np.linspace(X.min(), X.max(), 1000)
        y = np.linspace(Y.min(), Y.max(), 1000)

        xi, yi = np.meshgrid(x,y)

        #then, interpolate your data onto this grid:

        gu = sp.interpolate.griddata(zip(X,Y), U, (xi,yi))
        gv = sp.interpolate.griddata(zip(X,Y), V, (xi,yi))
        gmag = sp.interpolate.griddata(zip(X,Y), mag, (xi,yi))

        lw = 6*gmag/np.nanmax(gmag)
         #now, you can use x, y, gu, gv and gspeed in streamplot:

        ax.streamplot(xi, yi, gu, gv, density=1.0, color='k', linewidth=lw)
        
        
        
def scalar_field(ax, x, y, z, sigma, NGRID, L, cmap=mpl.cm.viridis):
        
    x_grid = np.linspace(0, L[0], NGRID)
    y_grid = np.linspace(0, L[1], NGRID)
    
    weights = np.exp(-((x_grid[:,None,None]-x[None, None, :])**2 
                  + (y_grid[None,:,None]-y[None, None, :])**2) / (2.0 * sigma**2))
    
    z_est = np.sum(weights * z[None, None, :], axis=2) / np.sum(weights, axis=2)
        
#     z_est /= np.sum(np.exp(-((x_grid[:,None,None]-x[None, None, :])**2 
#                   + (y_grid[None,:,None]-y[None, None, :])**2) / (2.0 * sigma)), axis=2)
            
    im = ax.imshow(z_est.T, cmap=cmap, origin='lower',
                vmin=np.min(z_est), vmax=np.max(z_est), extent=[0, L[0], 0, L[1]])
    im.set_interpolation('bilinear')
    
    return im

#     cf = ax1.contourf(z_est.T, cmap=mpl.cm.Blues, levels=np.linspace(0, 1.0, 11))
#     return cf
        
         
def show_corners(ax, net, corners, styles={}, radius=0.5, periodic=False):
    
    DIM = net.dim
    
    center = 0.5 * np.ones(DIM, float)
    
    patches = []
    patches_to_corners = []
    
    for ci, corner in enumerate(corners):
        
        patch_corners = np.zeros([len(corner), 2], float)

        vi = corner[0]
        posi = net.node_pos[DIM*vi:DIM*vi+DIM] / net.L
        patch_corners[0] = posi - np.floor(posi) - center


        for vj in range(1, len(corner)):
            posj = net.node_pos[DIM*corner[vj]:DIM*corner[vj]+DIM] / net.L

            bvec = posj - posi
            bvec -= np.rint(bvec)

            patch_corners[vj] = posi - np.floor(posi) - center + radius * bvec


        patches.append(mpatches.Polygon(patch_corners))

        patches_to_corners.append(ci)

        if periodic:
            patch_cornersN = np.copy(patch_corners)
            patch_cornersN[:, 1] += 1

            patches.append(mpatches.Polygon(patch_cornersN))
            patches_to_corners.append(ci)

            patch_cornersNE = np.copy(patch_corners)
            patch_cornersNE[:, 0] += 1
            patch_cornersNE[:, 1] += 1

            patches.append(mpatches.Polygon(patch_cornersNE))
            patches_to_corners.append(ci)

            patch_cornersE = np.copy(patch_corners)
            patch_cornersE[:, 0] += 1

            patches.append(mpatches.Polygon(patch_cornersE))
            patches_to_corners.append(ci)

            patch_cornersSE = np.copy(patch_corners)
            patch_cornersSE[:, 0] += 1
            patch_cornersSE[:, 1] -= 1

            patches.append(mpatches.Polygon(patch_cornersSE))
            patches_to_corners.append(ci)

            patch_cornersS = np.copy(patch_corners)
            patch_cornersS[:, 1] -= 1

            patches.append(mpatches.Polygon(patch_cornersS))
            patches_to_corners.append(ci)

            patch_cornersSW = np.copy(patch_corners)
            patch_cornersSW[:, 0] -= 1
            patch_cornersSW[:, 1] -= 1

            patches.append(mpatches.Polygon(patch_cornersSW))
            patches_to_corners.append(ci)


            patch_cornersW = np.copy(patch_corners)
            patch_cornersW[:, 0] -= 1

            patches.append(mpatches.Polygon(patch_cornersW))
            patches_to_corners.append(ci)

            patch_cornersNW = np.copy(patch_corners)
            patch_cornersNW[:, 0] -= 1
            patch_cornersNW[:, 1] += 1

            patches.append(mpatches.Polygon(patch_cornersNW))
            patches_to_corners.append(ci)
                            
        
    colors = []
    for ci in patches_to_corners:
        
        if ci in styles and 'color' in styles[ci]:
            colors.append(styles[ci]['color'])
        else:
            colors.append('b')
            
            
    pc = mc.PatchCollection(patches, color=colors, zorder=-2)
    ax.add_collection(pc)
   
        
        
        
def show_facets(ax, net, facets, styles={}, periodic=False, alpha=1.0):
    
    DIM = net.dim
    
    center = 0.5 * np.ones(DIM, float)
    
    patches_to_facets = []
    patches = []
    
    left = -0.5*1.05
    right = 0.5*1.05
    lower = -0.5*1.05
    upper = 0.5*1.05
    
    patch_index = 0
    for fi, facet in enumerate(facets):
        corners = np.zeros([len(facet['nodes']), 2], float)

        posi = net.node_pos[DIM*facet['nodes'][0]:DIM*facet['nodes'][0]+DIM] / net.L
        corners[0] = posi - np.floor(posi) - center


        for j in range(1,len(facet['nodes'])):
            posj = net.node_pos[DIM*facet['nodes'][j]:DIM*facet['nodes'][j]+DIM] / net.L

            bvec = posj - posi
            bvec -= np.rint(bvec)

            corners[j] = posi - np.floor(posi) - center + bvec

            posj = posi
            
            
        if is_in_bounds(corners.tolist(), left, right, lower, upper):
            patches.append(mpatches.Polygon(corners))
            patches_to_facets.append(fi)
        
        if periodic:
            cornersN = np.copy(corners)
            cornersN[:, 1] += 1
            
            if is_in_bounds(cornersN.tolist(), left, right, lower, upper):
                patches.append(mpatches.Polygon(cornersN))
                patches_to_facets.append(fi)
            
            cornersNE = np.copy(corners)
            cornersNE[:, 0] += 1
            cornersNE[:, 1] += 1
            
            if is_in_bounds(cornersNE.tolist(), left, right, lower, upper):
                patches.append(mpatches.Polygon(cornersNE))
                patches_to_facets.append(fi)
            
            cornersE = np.copy(corners)
            cornersE[:, 0] += 1
            
            if is_in_bounds(cornersE.tolist(), left, right, lower, upper):
                patches.append(mpatches.Polygon(cornersE))
                patches_to_facets.append(fi)
            
            cornersSE = np.copy(corners)
            cornersSE[:, 0] += 1
            cornersSE[:, 1] -= 1
            
            if is_in_bounds(cornersSE.tolist(), left, right, lower, upper):
                patches.append(mpatches.Polygon(cornersSE))
                patches_to_facets.append(fi)
            
            cornersS = np.copy(corners)
            cornersS[:, 1] -= 1
            
            if is_in_bounds(cornersS.tolist(), left, right, lower, upper):
                patches.append(mpatches.Polygon(cornersS))
                patches_to_facets.append(fi)
            
            cornersSW = np.copy(corners)
            cornersSW[:, 0] -= 1
            cornersSW[:, 1] -= 1
            
            if is_in_bounds(cornersSW.tolist(), left, right, lower, upper):
                patches.append(mpatches.Polygon(cornersSW))
                patches_to_facets.append(fi)
            
            
            cornersW = np.copy(corners)
            cornersW[:, 0] -= 1
            
            if is_in_bounds(cornersW.tolist(), left, right, lower, upper):
                patches.append(mpatches.Polygon(cornersW))
                patches_to_facets.append(fi)
            
            cornersNW = np.copy(corners)
            cornersNW[:, 0] -= 1
            cornersNW[:, 1] += 1
            
            if is_in_bounds(cornersNW.tolist(), left, right, lower, upper):
                patches.append(mpatches.Polygon(cornersNW))
                patches_to_facets.append(fi)
            
        
    colors = []
    for fi in patches_to_facets:
        
        if fi in styles and 'color' in styles[fi]:
            colors.append(styles[fi]['color'])
        else:
            colors.append('w')
            
        
    pc = mc.PatchCollection(patches, color=colors, zorder=-2, alpha=alpha)
    ax.add_collection(pc)
        
        
        
def show_voronoi(ax, net, verts, styles={}, periodic=False):
    
    
    L = net.L    
    vert_pos = np.vstack([net.node_pos.reshape((net.NN, 2))]*9)
    
    vert_pos[net.NN:2*net.NN, :] += np.array([L[0], 0.0])
    vert_pos[2*net.NN:3*net.NN, :] += L
    vert_pos[3*net.NN:4*net.NN, :] += np.array([0.0, L[1]])
    vert_pos[4*net.NN:5*net.NN, :] += np.array([-L[0], L[1]])
    vert_pos[5*net.NN:6*net.NN, :] += np.array([-L[0], 0.0])
    vert_pos[6*net.NN:7*net.NN, :] += np.array([-L[0], -L[1]])
    vert_pos[7*net.NN:8*net.NN, :] += np.array([0.0, -L[1]])
    vert_pos[8*net.NN:9*net.NN, :] += np.array([L[0], -L[1]])
        
    vor = spatial.Voronoi(vert_pos)
    
      
#     fig = spatial.voronoi_plot_2d(vor, ax=ax, point_size=3, show_vertices=False)
#     plt.show()
    
    vor_net = ns.Network2D(vor.vertices.shape[0], vor.vertices.flatten(), 0, [], [], net.L)
    
    facets = []
    facet_styles = {}
    fi = 0
    for vi in verts:
        region = vor.regions[vor.point_region[vi]]

        facet = {}
        facet['nodes'] = region

        facets.append(facet)
        if vi in styles:
            facet_styles[fi] = styles[vi]
        
        fi += 1    
    
    show_facets(ax, vor_net, facets, styles=facet_styles, periodic=periodic, alpha=1.0)
    
    

def frame(network, disp, Gamma, K, pert, meas, label, boxsize=0.5, padding=0.0, save=False, show_removed=False, ostrain=[]):

    sns.set(color_codes=True)
    sns.set_context('poster', font_scale=1.75)
    # sns.set_palette("hls", 6)
    sns.set_palette("bright")
    sns.set_style('ticks', {'xtick.direction': 'in','ytick.direction': 'in', 'axes.linewidth': 2.0})
    gray = "#969696"
    blue = "#377eb8"
    red = "#e41a1c"
    
    edgei = network.edgei
    edgej = network.edgej
    node_pos = np.copy(network.node_pos)
    DIM = network.dim
    NE = network.NE
    NN = network.NN
    L = np.copy(network.L)
    center = 0.5 * np.ones(DIM, float)
    
    fig = plt.figure(1, (8, 8))

    ax = fig.add_axes([0,0,1,1], frameon=True, aspect=1.0)

    patches = []


    def_tensor = np.identity(DIM) + Gamma 

    boxL = 0.5 + padding
    corners = np.array([[-boxL, -boxL], 
                        [boxL, -boxL], 
                        [boxL, boxL], 
                        [-boxL, boxL]])
    corners[0] = def_tensor[0:2, 0:2].dot(corners[0])
    corners[1] = def_tensor[0:2, 0:2].dot(corners[1])
    corners[2] = def_tensor[0:2, 0:2].dot(corners[2])
    corners[3] = def_tensor[0:2, 0:2].dot(corners[3])

    ax.add_patch(mpatches.Polygon(corners, True, fill=False, lw=2.0))


    for i in range(NN):
        node_pos[DIM*i:DIM*i+DIM] += padding * L
    
    L += 2.0 * padding * L
    # center -= padding
    
#     center = np.array([0.0, 0.0])
#     for i in range(NN):
#         center += node_pos[DIM*i:DIM*i+DIM] / L
#     center /= NN
    
    edges = []
    edge_index = []

    for i in range(NE):
        posi = node_pos[DIM*edgei[i]:DIM*edgei[i]+DIM]+disp[DIM*edgei[i]:DIM*edgei[i]+DIM]
        posj = node_pos[DIM*edgej[i]:DIM*edgej[i]+DIM]+disp[DIM*edgej[i]:DIM*edgej[i]+DIM]

        posi /= L
        posj /= L

        bvec = posj - posi
        bvec -= np.rint(bvec)

        posi -= np.floor(posi)
        posj -= np.floor(posj)

        posi -= center
        posj -= center


        if posi[0]+bvec[0] > 0.5 and posi[1]+bvec[1] > 0.5:
            c1  = (0.5 - posi[0]) / bvec[0]
            c2  = (0.5 - posi[1]) / bvec[1]
            c  = np.min([c1, c2])
            edges.append([tuple(posi),tuple(posi+c*bvec)])

            if c1 < c2:
                edges.append([tuple(posi+np.array([-1.0 ,0.0])+c1*bvec),tuple(posi+np.array([-1.0 ,0.0])+c2*bvec)])  
            else:
                edges.append([tuple(posi+np.array([0.0 ,-1.0])+c1*bvec),tuple(posi+np.array([0.0 ,-1.0])+c2*bvec)]) 

            c  = np.min([-(-0.5 - posj[0]) / bvec[0],-(-0.5 - posj[1]) / bvec[1]])
            edges.append([tuple(posj),tuple(posj-c*bvec)])  
            
            edge_index.append(i)
            edge_index.append(i)
            edge_index.append(i)

        elif posi[0]+bvec[0] > 0.5 and posi[1]+bvec[1] < -0.5:
            c1  = (0.5 - posi[0]) / bvec[0]
            c2  = (-0.5 - posi[1]) / bvec[1]
            c  = np.min([c1, c2])
            edges.append([tuple(posi),tuple(posi+c*bvec)])

            if c1 < c2:
                edges.append([tuple(posi+np.array([-1.0 ,0.0])+c1*bvec),tuple(posi+np.array([-1.0 ,0.0])+c2*bvec)])  
            else:
                edges.append([tuple(posi+np.array([0.0 ,1.0])+c1*bvec),tuple(posi+np.array([0.0 ,1.0])+c2*bvec)]) 
            c  = np.min([-(-0.5 - posj[0]) / bvec[0],-(0.5 - posj[1]) / bvec[1]])
            edges.append([tuple(posj),tuple(posj-c*bvec)]) 
            
            edge_index.append(i)
            edge_index.append(i)
            edge_index.append(i)

        elif posi[0]+bvec[0] < -0.5 and posi[1]+bvec[1] > 0.5:
            c1 = (-0.5 - posi[0]) / bvec[0]
            c2 = (0.5 - posi[1]) / bvec[1]
            c  = np.min([c1, c2])
            edges.append([tuple(posi),tuple(posi+c*bvec)])

            if c1 < c2:
                edges.append([tuple(posi+np.array([1.0 ,0.0])+c1*bvec),tuple(posi+np.array([1.0 ,0.0])+c2*bvec)])  
            else:
                edges.append([tuple(posi+np.array([0.0 ,-1.0])+c1*bvec),tuple(posi+np.array([0.0 ,-1.0])+c2*bvec)])            

            c  = np.min([-(-0.5 - posj[1]) / bvec[1],-(0.5 - posj[0]) / bvec[0]])
            edges.append([tuple(posj),tuple(posj-c*bvec)])
            
            edge_index.append(i)
            edge_index.append(i)
            edge_index.append(i)

        elif posi[0]+bvec[0] < -0.5 and posi[1]+bvec[1] < -0.5:
            c1  = (-0.5 - posi[0]) / bvec[0]
            c2  = (-0.5 - posi[1]) / bvec[1]
            c  = np.min([c1, c2])
            edges.append([tuple(posi),tuple(posi+c*bvec)])

            if c1 < c2:
                edges.append([tuple(posi+np.array([1.0 ,0.0])+c1*bvec),tuple(posi+np.array([1.0 ,0.0])+c2*bvec)])  
            else:
                edges.append([tuple(posi+np.array([0.0 ,1.0])+c1*bvec),tuple(posi+np.array([0.0 ,1.0])+c2*bvec)])

            c  = np.min([-(0.5 - posj[0]) / bvec[0],-(0.5 - posj[1]) / bvec[1]])
            edges.append([tuple(posj),tuple(posj-c*bvec)])  
            
            edge_index.append(i)
            edge_index.append(i)
            edge_index.append(i)

        elif posi[0]+bvec[0] > 0.5:
            c  = (0.5 - posi[0]) / bvec[0]
            edges.append([tuple(posi),tuple(posi+c*bvec)])
            c  = (-0.5 - posj[0]) / bvec[0]
            edges.append([tuple(posj),tuple(posj+c*bvec)])
            
            edge_index.append(i)
            edge_index.append(i)

        elif posi[1]+bvec[1] > 0.5:
            c  = (0.5 - posi[1]) / bvec[1]
            edges.append([tuple(posi),tuple(posi+c*bvec)])
            c  = (-0.5 - posj[1]) / bvec[1]
            edges.append([tuple(posj),tuple(posj+c*bvec)])
            
            edge_index.append(i)
            edge_index.append(i)

        elif posi[0]+bvec[0] < -0.5:
            c  = (-0.5 - posi[0]) / bvec[0]
            edges.append([tuple(posi),tuple(posi+c*bvec)])
            c  = (0.5 - posj[0]) / bvec[0]
            edges.append([tuple(posj),tuple(posj+c*bvec)])
            
            edge_index.append(i)
            edge_index.append(i)

        elif posi[1]+bvec[1] < -0.5:
            c  = (-0.5 - posi[1]) / bvec[1]
            edges.append([tuple(posi),tuple(posi+c*bvec)])
            c  = (0.5 - posj[1]) / bvec[1]
            edges.append([tuple(posj),tuple(posj+c*bvec)])
            
            edge_index.append(i)
            edge_index.append(i)

        else:
            edges.append([tuple(posi),tuple(posi+bvec)])
            
            edge_index.append(i)

    for i, b in enumerate(edges):
        edges[i] = [def_tensor.dot(b[0])[0:2], def_tensor.dot(b[1])[0:2]]

    #show input nodes
    inodesi = pert.istrain_nodesi
    inodesj = pert.istrain_nodesj
    
    x1 = []
    y1 = []

    for i in range(pert.NIstrain):
        
        posi = node_pos[DIM*inodesi[i]:DIM*inodesi[i]+DIM]+disp[DIM*inodesi[i]:DIM*inodesi[i]+DIM]
        posj = node_pos[DIM*inodesj[i]:DIM*inodesj[i]+DIM]+disp[DIM*inodesj[i]:DIM*inodesj[i]+DIM]
        
        posi /= L
        posj /= L

        bvec = posj - posi
        bvec -= np.rint(bvec)

        posi -= np.floor(posi)
        posj -= np.floor(posj)

        posi -= center
        posj -= center
        
        posi = def_tensor.dot(posi)
        posj = def_tensor.dot(posj)
        
        x1.append(posi[0])
        x1.append(posj[0])
        
        y1.append(posi[1])
        y1.append(posj[1])
        
        
    ax.scatter(x1, y1, marker='o' , edgecolor='', s=200, linewidth=0.0, facecolor=blue, alpha=1.0)

    #show output nodes
    onodesi = meas.ostrain_nodesi
    onodesj = meas.ostrain_nodesj
    
    x1 = []
    y1 = []

    for i in range(meas.NOstrain):
        if ostrain[i] < 0:
            continue
        
        posi = node_pos[DIM*onodesi[i]:DIM*onodesi[i]+DIM]+disp[DIM*onodesi[i]:DIM*onodesi[i]+DIM]
        posj = node_pos[DIM*onodesj[i]:DIM*onodesj[i]+DIM]+disp[DIM*onodesj[i]:DIM*onodesj[i]+DIM]
        
        posi /= L
        posj /= L

        bvec = posj - posi
        bvec -= np.rint(bvec)

        posi -= np.floor(posi)
        posj -= np.floor(posj)

        posi -= center
        posj -= center
        
        posi = def_tensor.dot(posi)
        posj = def_tensor.dot(posj)
        
        x1.append(posi[0])
        x1.append(posj[0])
        
        y1.append(posi[1])
        y1.append(posj[1])

    ax.scatter(x1, y1, marker='s' , edgecolor='', s=200, linewidth=0.0, facecolor='r', alpha=1.0)
    
    x1 = []
    y1 = []

    for i in range(meas.NOstrain):
        if ostrain[i] > 0:
            continue
        
        posi = node_pos[DIM*onodesi[i]:DIM*onodesi[i]+DIM]+disp[DIM*onodesi[i]:DIM*onodesi[i]+DIM]
        posj = node_pos[DIM*onodesj[i]:DIM*onodesj[i]+DIM]+disp[DIM*onodesj[i]:DIM*onodesj[i]+DIM]
        
        posi /= L
        posj /= L

        bvec = posj - posi
        bvec -= np.rint(bvec)

        posi -= np.floor(posi)
        posj -= np.floor(posj)

        posi -= center
        posj -= center
        
        posi = def_tensor.dot(posi)
        posj = def_tensor.dot(posj)
        
        x1.append(posi[0])
        x1.append(posj[0])
        
        y1.append(posi[1])
        y1.append(posj[1])

    ax.scatter(x1, y1, marker='o' , edgecolor='', s=200, linewidth=0.0, facecolor='k', alpha=1.0)
        
    carray = []
    for b in edge_index:
        carray.append(K[b])

    #add lines to plot
    if not show_removed:
        ls = ['solid' for b in range(len(edges))]
        lw = (4*np.array(carray)+1.0)
        lc = mc.LineCollection(edges, zorder=-1, linestyle=ls, lw=lw, alpha=0.8,color=gray)
    else:
        ls = [(0.0, (8.0,4.0)) if K[b] < 0.1 else 'solid' for b in edge_index]
        lw = [3.0 if K[b] < 0.1 else 5.0 for b in edge_index]
        colors = ["r" if K[b] < 0.1 else gray for b in edge_index]
        lc = mc.LineCollection(edges, zorder=-1, linestyle=ls, lw=lw, alpha=0.8,color=colors)
    
    

    ax.add_collection(lc)

#     cb = plt.colorbar(lc)
#     cb.set_clim(0, 1.0)

    ax.get_xaxis().set_ticks([])
    ax.get_yaxis().set_ticks([])
    ax.set_xlim(-boxsize, boxsize)
    ax.set_ylim(-boxsize, boxsize)
    
    if save:
        plt.savefig(label, bbox_inches='tight')

    plt.show()