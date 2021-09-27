import os, sys

#sys.path.insert(0, '../')
#sys.path.insert(0, '../python_src/')

import numpy as np
import numpy.linalg as la
import numpy.random as rand
import scipy.sparse as sparse
import scipy.sparse.linalg as sla
from netCDF4 import Dataset, chartostring, stringtoarr
import itertools as it
import shelve
import copy

#import network_solver as ns

class Network:
    def __init__(self, net):
        self.NN = net['NN']
        self.NE = net['NE']
        self.DIM = net['DIM']
        self.node_pos = np.array(net['node_pos'])
        self.edgei = net['edgei']
        self.edgej = net['edgej']
        #self.L = net['L']
        
        box_mat = np.eye(2)      
        L = box_mat.diagonal()
        self.L = L
        
        bvecij = np.zeros(self.DIM*self.NE, float)
        eq_length = np.zeros(self.NE, float)
        for i in range(self.NE):
            bvec =  self.node_pos[self.DIM*self.edgej[i]:self.DIM*self.edgej[i]+self.DIM]-self.node_pos[self.DIM*self.edgei[i]:self.DIM*self.edgei[i]+self.DIM]
            bvec -= np.rint(bvec / L) * L
            bvecij[self.DIM*i:self.DIM*i+self.DIM] = bvec
            eq_length[i] = la.norm(bvec)
        self.bvecij = bvecij
        self.eq_length = eq_length
        


def convert_jammed_state_to_network(label, index, DIM=2):

#     directory="/data1/home/rocks/data/network_states"
    directory="."
    
    fn = "{1}/{0}.nc".format(label, directory)

    data = Dataset(fn, 'r')

    # print data

    NN = len(data.dimensions['NP'])

    node_pos = data.variables['pos'][index]
        
    rad = data.variables['rad'][index]
    box_mat = data.variables['BoxMatrix'][index]

    L = np.zeros(DIM, float)
    for d in range(DIM):
        L[d] = box_mat[d *(DIM+1)]
        
    for i in range(NN):
        node_pos[DIM*i:DIM*i+DIM] *= L

    edgei = []
    edgej = []

    NE = 0
    edgespernode = np.zeros(NN, int)
        
    gridL = np.max([int(round(NN**(1.0/DIM))/4.0), 1])
    NBINS = gridL**DIM
    
    print("Grid Length:", gridL, "Number Bins:", NBINS, "Nodes per Bin:", 1.0 * NN / NBINS)
    
    tmp = []
    for d in range(DIM):
        tmp.append(np.arange(gridL))
    
    bin_to_grid = list(it.product(*tmp))
    
    grid_to_bin = {x: i for i, x in enumerate(bin_to_grid)}
    
    grid_links = set()
                
    for i in range(NBINS):
        bini = bin_to_grid[i]
        for j in range(i+1, NBINS):
            binj = bin_to_grid[j]
            
            link = True
                        
            for d in range(DIM):
                dist = bini[d] - binj[d]
                dist -= np.rint(1.0*dist/gridL) * gridL
                                
                if np.abs(dist) > 1:
                    link = False
                    
            if link:
                grid_links.add(tuple(sorted([i,j])))
      
        
    bin_nodes = [[] for b in range(NBINS)]
        
    for n in range(NN):
        pos = node_pos[DIM*n:DIM*n+DIM]
        ipos = tuple(np.floor(pos / L * gridL).astype(int))
        
        bin_nodes[grid_to_bin[ipos]].append(n)
                
    # add edges within each bin
    for ibin in range(NBINS):
        for i in range(len(bin_nodes[ibin])):
            for j in range(i+1,len(bin_nodes[ibin])):
                
                ni = bin_nodes[ibin][i]
                nj = bin_nodes[ibin][j]
                
                posi = node_pos[DIM*ni:DIM*ni+DIM]
                posj = node_pos[DIM*nj:DIM*nj+DIM]
                bvec = posj - posi
                bvec -= np.rint(bvec / L) * L
                l0 = la.norm(bvec)
                
                if l0 < rad[ni] + rad[nj]:
                    NE += 1
                    edgei.append(ni)
                    edgej.append(nj)
                    edgespernode[ni] += 1
                    edgespernode[nj] += 1
     
    # add edge between bins
    for (bini, binj) in grid_links:
        for ni in bin_nodes[bini]:
            for nj in bin_nodes[binj]:
                
                posi = node_pos[DIM*ni:DIM*ni+DIM]
                posj = node_pos[DIM*nj:DIM*nj+DIM]
                bvec = posj - posi
                bvec -= np.rint(bvec / L) * L
                l0 = la.norm(bvec)
                
                if l0 < rad[ni] + rad[nj]:
                    NE += 1
                    edgei.append(ni)
                    edgej.append(nj)
                    edgespernode[ni] += 1
                    edgespernode[nj] += 1
                    
                    
    node_pos_tmp = np.copy(node_pos)
    edgei_tmp = np.copy(edgei)
    edgej_tmp = np.copy(edgej)
    rad_tmp = np.copy(rad)

    index_map = list(range(NN))
    rattlers = set()
    for i in range(NN):
        if edgespernode[i] < DIM+1:
            #print("Removing", i, edgespernode[i])
            index_map.remove(i)
            rattlers.add(i)        

    rev_index_map = -1 * np.ones(NN, int)
    for i in range(len(index_map)):
        rev_index_map[index_map[i]] = i

    NN = len(index_map)
    node_pos = np.zeros(DIM*NN, float)
    rad = np.zeros(NN)

    for i in range(NN):
        node_pos[DIM*i:DIM*i+DIM] = node_pos_tmp[DIM*index_map[i]:DIM*index_map[i]+DIM]
        rad[i] = rad_tmp[index_map[i]]

    edgei = []
    edgej = []
    for i in range(NE):
        if edgei_tmp[i] not in rattlers and edgej_tmp[i] not in rattlers:
            edgei.append(rev_index_map[edgei_tmp[i]])
            edgej.append(rev_index_map[edgej_tmp[i]])

    NE = len(edgei)
    
    print("Number Rattlers:", len(rattlers))
    
    print("NN", NN)
    print("NE", NE) 
        
    net = {}
    net['source'] = fn
    
    net['DIM'] = DIM
    net['box_L'] = L
    
    net['NN'] = NN
    net['node_pos'] = node_pos
    net['rad'] = rad
    
    net['NE'] = NE
    net['edgei'] = np.array(edgei)
    net['edgej'] = np.array(edgej)
    
    net['box_mat'] = box_mat.reshape([DIM,DIM])
    net['L'] = L
    
    return net


def load_network(db_fn, index):
    
    with shelve.open(db_fn) as db:
        net = db["{}".format(index)]
        
    return net

def convert_to_network_object(net, periodic=True):
    net2 = Network(net)
    return net2

# def convert_to_network_object(net, periodic=True):
    
#     DIM = net['DIM']
   
#     NN = net['NN']
#     node_pos = np.array(net['node_pos'])
#     #box_mat = net['box_mat']
#     box_mat = np.eye(2)
    
#     NE = net['NE']
#     edgei = net['edgei']
#     edgej = net['edgej']
    
    
#     for i in range(NN):
#         node_pos[DIM*i:DIM*i+DIM] = box_mat.dot(node_pos[DIM*i:DIM*i+DIM])
        
    
#     L = box_mat.diagonal()
    
#     bvecij = np.zeros(DIM*NE, float)
#     eq_length = np.zeros(NE, float)
#     for i in range(NE):
#         bvec =  node_pos[DIM*edgej[i]:DIM*edgej[i]+DIM]-node_pos[DIM*edgei[i]:DIM*edgei[i]+DIM]
#         bvec -= np.rint(bvec / L) * L
        
#         bvecij[DIM*i:DIM*i+DIM] = bvec
#         eq_length[i] = la.norm(bvec)
    
        
#     if DIM == 2:
#         cnet = ns.Network2D(NN, node_pos, NE, edgei, edgej, L)
#     elif DIM == 3:
#         cnet = ns.Network3D(NN, node_pos, NE, edgei, edgej, L)
        
#     cnet.setInteractions(bvecij, eq_length, np.ones(NE, float) / eq_length)
    
# #     print("convert", cnet.NE)
    
#     return cnet





def prune_network(net, rem_nodes, rem_edges):
    
    DIM = net['DIM']
   
    NN = net['NN']
    node_pos = net['node_pos']
    rad = net['rad']
    
    NE = net['NE']
    edgei = net['edgei']
    edgej = net['edgej']
    
    if 'boundary' in net:
        boundary = net['boundary']
    else:
        boundary = set()
    
    # map from original periodic network to current network
    if 'node_map' in net:
        node_map = net['node_map']
    else:
        node_map = {i:i for i in range(NN)}
    
    rem_nodes = set(rem_nodes)
    rem_edges = set(rem_edges)
    
#     print("Removing", len(rem_nodes), "/", NN, "nodes and", len(rem_edges), "/", NE, "edges...")
    
    
    local_node_map = {}
    
    NN_tmp = 0
    node_pos_tmp = []
    boundary_tmp = set()
    rad_tmp = []
    for v in range(NN):
        
        if v not in rem_nodes:
        
            node_pos_tmp.extend(node_pos[DIM*v:DIM*v+DIM])
            rad_tmp.append(rad[v])
            
            if v in boundary:
                boundary_tmp.add(NN_tmp)
                
            local_node_map[v] = NN_tmp

            NN_tmp += 1
            
    NE_tmp = 0
    edgei_tmp = []
    edgej_tmp = []
    for e in range(NE):
        
        if edgei[e] not in local_node_map or edgej[e] not in local_node_map:
            rem_edges.add(e)
        
        if e in rem_edges:
            if edgei[e] in local_node_map:
                boundary_tmp.add(local_node_map[edgei[e]])
            if edgej[e] in local_node_map:
                boundary_tmp.add(local_node_map[edgej[e]])
        
        else :
            edgei_tmp.append(local_node_map[edgei[e]])
            edgej_tmp.append(local_node_map[edgej[e]])
            NE_tmp += 1
      
    node_map_tmp = {}
    
    for v in node_map:
        if node_map[v] in local_node_map:
            node_map_tmp[v] = local_node_map[node_map[v]]
            
    
    
    new_net = copy.deepcopy(net)
    
    new_net['NN'] = NN_tmp
    new_net['node_pos'] = np.array(node_pos_tmp)
    new_net['rad'] = np.array(rad_tmp)
    
    new_net['NE'] = NE_tmp
    new_net['edgei'] = np.array(edgei_tmp)
    new_net['edgej'] = np.array(edgej_tmp)
    
    new_net['boundary'] = boundary_tmp
    new_net['node_map'] = node_map_tmp
    
    
#     print("Removed", NN-NN_tmp, "/", NN, "nodes and", NE-NE_tmp, "/", NE, "edges.")
    
    return new_net


def make_finite(net):
    
    
    DIM = net['DIM']
   
    NN = net['NN']
    node_pos = net['node_pos']
    box_mat = net['box_mat']
        
    NE = net['NE']
    edgei = net['edgei']
    edgej = net['edgej']
    
    rem_edges = set()
    
    for b in range(NE):
        posi = node_pos[DIM*edgei[b]:DIM*edgei[b]+DIM]
        posj = node_pos[DIM*edgej[b]:DIM*edgej[b]+DIM]
        
        bvec = posj-posi
        bvec -= np.rint(bvec)
        
        if not ((posi+bvec <= 1.0).all() and (posi+bvec >= 0.0).all()):
            rem_edges.add(b)
    
    return prune_network(net, set(), rem_edges)
            

def make_ball(net, radius, center=None):
    
    DIM = net['DIM']
   
    NN = net['NN']
    node_pos = net['node_pos']
    box_mat = net['box_mat']
    
    NE = net['NE']
    edgei = net['edgei']
    edgej = net['edgej']
    
    if 'boundary' in net:
        boundary = net['boundary']
    else:
        boundary = set()
    
    if center is None:
        center = 0.5
        
    rem_nodes = set()      
        
    for v in range(NN):
        
        posi = node_pos[DIM*v:DIM*v+DIM]
        
        bvec = posi - center
        bvec -= np.rint(bvec)
        
        if la.norm(bvec) > radius / 2.0:
            rem_nodes.add(v)
        
    
    
    return prune_network(net, rem_nodes, set())
    
    
def choose_boundary_edge(net, theta, phi=0):
    
    DIM = net['DIM']
   
    NN = net['NN']
    node_pos = net['node_pos']
    L = net['box_mat'].diagonal()
    
    NE = net['NE']
    edgei = net['edgei']
    edgej = net['edgej']
    
    boundary = net['boundary']
    
    
    Z = np.zeros(NN, float)
    
    for b in range(NE):
        Z[edgei[b]] += 1
        Z[edgej[b]] += 1
    
    
    
    vec = np.zeros(DIM, float)
    
    if DIM == 2:
        theta = theta*np.pi/180
        vec[0] = np.cos(theta)
        vec[1] = np.sin(theta)
    if DIM == 3:
        theta = theta*np.pi/180
        phi = phi * np.pi / 180
        
        vec[0] = np.sin(theta)*np.cos(phi)
        vec[1] = np.sin(theta)*np.sin(phi)
        vec[2] = np.cos(theta)
    
    
    angles = np.zeros(len(boundary), float)
    center = 0.5
    
    boundary_edges = []
    angles = []
    for b in range(NE):
        
        if edgei[b] in boundary and edgej[b] in boundary:
            boundary_edges.append(b)
            
            posi = node_pos[DIM*edgei[b]:DIM*edgei[b]+DIM]
            posj = node_pos[DIM*edgej[b]:DIM*edgej[b]+DIM]
            
            bvec = posj - posi
            bvec -= np.rint(bvec)
            
            pos = posi + 0.5*bvec - center
            pos /= la.norm(pos)
            angles.append(np.arccos(np.dot(pos, vec)))
           
    asort = np.argsort(angles)
        
    for i in asort:
        
        b = boundary_edges[i]
                
        if Z[edgei[b]] >= DIM + 1 and Z[edgej[b]] >= DIM + 1:
            bmin = boundary_edges[i]
            break
            
        print("skipping", b, "Z:", Z[edgei[b]], Z[edgej[b]])
    
    
    return (bmin, edgei[bmin], edgej[bmin])


def choose_boundary_nodes(net, theta, edge_map, phi=0):
    
    DIM = net['DIM']
   
    NN = net['NN']
    node_pos = net['node_pos']
    L = net['box_mat'].diagonal()
    
    NE = net['NE']
    edgei = net['edgei']
    edgej = net['edgej']
    
    boundary = list(net['boundary'])
    
    
    vec = np.zeros(DIM, float)
    
    if DIM == 2:
        theta = theta*np.pi/180
        vec[0] = np.cos(theta)
        vec[1] = np.sin(theta)
    if DIM == 3:
        theta = theta*np.pi/180
        phi = phi * np.pi / 180
        
        vec[0] = np.sin(theta)*np.cos(phi)
        vec[1] = np.sin(theta)*np.sin(phi)
        vec[2] = np.cos(theta)
    
    
    angles = np.zeros(len(boundary), float)
    center = 0.5
    
    angles = []
    for b in boundary:
            
            pos = node_pos[DIM*b:DIM*b+DIM] - center
            pos -= np.rint(pos)
            
            pos /= la.norm(pos)
            angles.append(np.arccos(np.dot(pos, vec)))
           
    asort = np.argsort(angles)
    
    bi = boundary[asort[0]]
    
    for j in asort[1:]:
        
        bj = boundary[j]
        
    
        if tuple(sorted([bi, bj])) not in edge_map:
            break
        
    
    return (bi, bj)






# def bondExists(nw, nodei, nodej):
# 	NE = nw.NE
# 	edgei = nw.edgei
# 	edgej = nw.edgej
    
# 	for b in range(NE):
# 		if (edgei[b] == nodei and edgej[b] == nodej) or (edgei[b] == nodej and edgej[b] == nodei):
# 			return True
		
# 	return False


    
#old

def load_jammed_network(db_fn, index):
    
    with shelve.open(db_fn) as db:
        net = db["{}".format(index)]
            
            
    
            
    DIM = net['DIM']
   
    NN = net['NN']
    node_pos = net['node_pos']
#     L = net['box_mat'].diagonal()
    L = net['box_L']
    
#     for i in range(NN):
#         node_pos[DIM*i:DIM*i+DIM] = L * node_pos[DIM*i:DIM*i+DIM]
    
    
    NE = net['NE']
    edgei = net['edgei']
    edgej = net['edgej']
    
    bvecij = np.zeros(DIM*NE, float)
    eq_length = np.zeros(NE, float)
    for i in range(NE):
        bvec =  node_pos[DIM*edgej[i]:DIM*edgej[i]+DIM]-node_pos[DIM*edgei[i]:DIM*edgei[i]+DIM]
        bvec -= np.rint(bvec / L) * L
        
        bvecij[DIM*i:DIM*i+DIM] = bvec
        eq_length[i] = la.norm(bvec)
    
    
    if DIM == 2:
        cnet = ns.Network2D(NN, node_pos, NE, edgei, edgej, L)
    elif DIM == 3:
        cnet = ns.Network3D(NN, node_pos, NE, edgei, edgej, L)
        
    cnet.setInteractions(bvecij, eq_length, np.ones(NE, float) / eq_length)
#     cnet.fix_trans = True
#     cnet.fix_rot = False
    
    return cnet

    

def convertToFlowNetwork(net):
    DIM = 1
    NN = net.NN
    NE = net.NE
    node_pos = np.arange(0, 1, 1.0/NN)
    edgei = net.edgei
    edgej = net.edgej
    
    L = np.array([1.0])
    
    bvecij = np.zeros(NE, float)
    eq_length = np.zeros(NE, float)
    for b in range(NE):
        bvec =  node_pos[DIM*edgej[b]:DIM*edgej[b]+DIM]-node_pos[DIM*edgei[b]:DIM*edgei[b]+DIM]
        bvec -= np.rint(bvec / L) * L
        bvec /= la.norm(bvec)
        
        eq_length[b] = 1.0
        
        bvecij[DIM*b:DIM*b+DIM] = bvec
        
        
    fnet = ns.Network1D(NN, node_pos, NE, edgei, edgej, L)
    fnet.setInteractions(bvecij, eq_length, np.ones(NE, float))
#     fnet.fix_trans = True
#     fnet.fix_rot = False
        
    return fnet
