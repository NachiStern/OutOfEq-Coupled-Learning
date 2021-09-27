import numpy as np
import numpy.linalg as la
import networkx as nx

    
def closest_edge(net, pos): # return array of sorted indices of edges, acording to distance from location "pos". Location of edge is decided by its mid-point
        
    NE = net.NE
    L = net.L
    dim = net.DIM
    
    dist = np.zeros(NE, float)
    
    for i in range(NE):
                
        bi = net.edgei[i]
        bj = net.edgej[i]
        
        posi = net.node_pos[dim*bi:dim*bi+dim]
        posj = net.node_pos[dim*bj:dim*bj+dim]
        
        bvec = posj - posi
        bvec -= np.round(bvec/net.L)*net.L
        
        bpos = posi + bvec/2.0
        
        vec = pos - bpos
        vec -= np.rint(vec / L) * L
        
        dist[i] = la.norm(vec)
                        
    return np.argsort(dist)

def closest_node(net, pos):
    
    NN = net.NN
    L = net.L
    dim = net.DIM
    
    dist = np.zeros(NN, float)
    for i in range(NN):
        posi = net.node_pos[dim*i:dim*i+dim]
        
        vec = pos - posi
        vec -= np.rint(vec / L) * L
        
        dist[i] = la.norm(vec)
        
    return np.argsort(dist)
        
    
def get_facets_2D(net):
    
    DIM = net.DIM
    
    G = nx.Graph()
    G.add_nodes_from(range(net.NN))
    G.add_edges_from(zip(net.edgei, net.edgej))

    def angle(ni, nj):
        posi = net.node_pos[DIM*ni:DIM*ni+DIM]
        posj = net.node_pos[DIM*nj:DIM*nj+DIM]
        bvec = posj - posi
        bvec -= np.rint(bvec / net.L) * net.L
        bvec /= la.norm(bvec)

        theta = np.arctan2(bvec[1], bvec[0])

        return theta

    DG = G.to_directed()
    unvisited = set(DG.edges())

    facets_to_nodes = []

    cedge = unvisited.pop()
    facet = [cedge[0], cedge[1]]

    # for i in range(10):
    while len(unvisited) > 0:

        ci = cedge[0]
        cj = cedge[1]
        ctheta = angle(ci, cj) + np.pi 

    #     print "Current Facet:", facet
    #     print "Current Edge:", cedge


        neighbors = []
        nangles = []
        for neigh in DG.neighbors(cj):
            if (cj, neigh) not in unvisited:
                continue

            ntheta = angle(cj, neigh) 

            dtheta = ctheta - ntheta
            dtheta -= np.floor(dtheta / (2*np.pi)) * 2*np.pi

            if neigh == ci:
                dtheta += 2*np.pi

            neighbors.append(neigh)
            nangles.append(dtheta)

        cedge = (cj, neighbors[np.argmin(nangles)])

    #     print "Next Edge", cedge

        unvisited.discard(cedge)

        if facet[0] == cedge[1]:

    #         print "Found new facet:", facet

            facets_to_nodes.append(np.copy(facet))

            if len(unvisited) > 0:
                cedge = unvisited.pop()
                facet = [cedge[0], cedge[1]]
        else:
            facet.append(cedge[1])                

    # print("Number Facets:", len(facets_to_nodes))

    nodes_to_edges = {}
    for i in range(net.NE):
        nodes_to_edges[tuple(sorted((net.edgei[i], net.edgej[i])))] = i



    # edges_to_facets = {tuple(sorted(edge)): [] for edge in G.edges()}
    facets_to_edges = [[] for fi in range(len(facets_to_nodes))]

    for fi, facet in enumerate(facets_to_nodes):
        for i in range(len(facet)):
            ni = facet[i]
            nj = facet[(i+1) % len(facet)]

            edge = tuple(sorted((ni, nj)))

            # edges_to_facets[edge].append(fi)

            facets_to_edges[fi].append(nodes_to_edges[edge])
            
            
    facets = []
    for fi in range(len(facets_to_nodes)):
        facets.append({"nodes": facets_to_nodes[fi], 'edges': facets_to_edges[fi]})

    # return (facets_to_nodes, facets_to_edges)
    
    return facets

def calc_facet_strain_2D(net, facets, disp):
    
    DIM = net.DIM

    shear = np.zeros(len(facets), float)
    bulk = np.zeros(len(facets), float)
    for fi in range(len(facets)):
        X = np.zeros([DIM, DIM], float)
        Y = np.zeros([DIM, DIM], float)

        for i in facets[fi]['edges']:
            posi = net.node_pos[DIM*net.edgei[i]:DIM*net.edgei[i]+DIM] 
            posj = net.node_pos[DIM*net.edgej[i]:DIM*net.edgej[i]+DIM]

            bvec = posj - posi
            bvec -= np.rint(bvec / net.L) * net.L

            dispi = disp[DIM*net.edgei[i]:DIM*net.edgei[i]+DIM] 
            dispj = disp[DIM*net.edgej[i]:DIM*net.edgej[i]+DIM]

            du = dispj - dispi

            X += np.outer(bvec, bvec)
            Y += np.outer(du, bvec)


        Gamma = la.inv(X).dot(Y)

        R = 0.5 * (Gamma - Gamma.transpose())
        eps = 0.5 * (Gamma + Gamma.transpose())

        gamma = eps - 1.0/DIM * np.trace(eps) * np.eye(2)

        shear[fi] = 0.5 * la.norm(gamma)**2
        bulk[fi] = 0.5 * np.trace(eps)**2
        
    return (bulk, shear)


def calc_max_ext(net, facets, disp):
    
    DIM = net.DIM

    max_ext = np.zeros(len(facets), float)
    for fi in range(len(facets)):
        
        nodes = facets[fi]['nodes']
        NN = len(nodes)
        
        ext = []

        for ni in range(NN):
            for nj in range(ni+1, NN):
                posi = net.node_pos[DIM*nodes[ni]:DIM*nodes[ni]+DIM] 
                posj = net.node_pos[DIM*nodes[nj]:DIM*nodes[nj]+DIM]

                bvec = posj - posi
                bvec -= np.rint(bvec / net.L) * net.L
                bvec /= la.norm(bvec)
                
                dispi = disp[DIM*nodes[ni]:DIM*nodes[ni]+DIM] 
                dispj = disp[DIM*nodes[nj]:DIM*nodes[nj]+DIM]                
                
                du = dispj - dispi
                
                ext.append(np.abs(bvec.dot(du)))
                                
        max_ext[fi] = np.max(ext)
        
    return max_ext