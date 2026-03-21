"""
DeepACSON skeleton extraction and cross-section sampling.

This is a derivative of the DeepACSON CSD (Cross-Section Decomposition) code
from Abdollahzadeh et al. (2019, 2021).

Original implementation: https://github.com/aAbdz/DeepACSON
Source commit: https://github.com/aAbdz/DeepACSON/commit/ad894cda498139b13b8b4e9e8944d88b3e23afe4
Citation:
    Abdollahzadeh A, Belevich I, Jokitalo E, Tohka J, Sierra A.
    "DeepACSON automated segmentation of white matter in 3D electron microscopy."
    Commun Biol. 2021 Feb 18;4(1):179.

The code below is organized by original source file. Sections marked "verbatim"
are copied without modification. All changes are marked with:
    # --- MODIFIED ---
    # Reason: ...
    # Original: ...
    # ---

MIT License

Copyright (c) 2020 Ali Abdollahzadeh

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import sys

import numpy as np
import skfmm
from scipy.interpolate import RegularGridInterpolator as rgi
from skimage.measure import label, regionprops

# ===================================================================
# DeepACSON/CSD/plane_rotation.py (verbatim)
# ===================================================================

def rotate_vector(vector, rot_mat):

    "rotating a vector by a rotation matrix"
    rotated_vec = np.dot(vector,rot_mat)
    return rotated_vec


def rotation_matrix_3D(vector, theta):

    """counterclockwise rotation about a unit vector by theta radians using
    Euler-Rodrigues formula: https://en.wikipedia.org/wiki/Euler-Rodrigues_formula"""

    a=np.cos(theta/2.0)
    b,c,d=-vector*np.sin(theta/2.0)
    aa,bb,cc,dd=a**2, b**2, c**2, d**2
    bc,ad,ac,ab,bd,cd=b*c, a*d, a*c, a*b, b*d, c*d

    rot_mat=np.array([[aa+bb-cc-dd, 2*(bc+ad), 2*(bd-ac)],
                         [2*(bc-ad), aa+cc-bb-dd, 2*(cd+ab)],
                         [2*(bd+ac), 2*(cd-ab), aa+dd-bb-cc]])
    return rot_mat

# --- BUGFIX ---
# Original code used exact zero check and clamped normalization (1e-5),
# which fails for near-antiparallel vectors: cross product magnitude ~1e-6
# gets divided by 1e-5, producing a non-unit axis (|n|≈0.18). This causes
# rotation_matrix_3D (Euler-Rodrigues) to produce a scaled-down matrix
# (factor ≈0.03), collapsing the cross-section grid and inflating measured
# radii to the full grid size (~11 μm).
# Fix: detect near-zero cross product and use an arbitrary perpendicular axis.
# Original:
#   n = np.cross(vec1, vec2)
#   if np.array_equal(n, np.array([0, 0, 0])):
#       n = vec1
#   s = max(np.sqrt(np.dot(n,n)), 1e-5)
#   n = n/s
#   return n
# ---
def unit_normal_vector(vec1, vec2):

    n = np.cross(vec1, vec2)
    s = np.sqrt(np.dot(n, n))
    if s < 1e-8:
        # Parallel or antiparallel — pick an arbitrary perpendicular axis
        n = np.cross(vec1, np.array([1, 0, 0]))
        s = np.sqrt(np.dot(n, n))
        if s < 1e-8:
            n = np.cross(vec1, np.array([0, 1, 0]))
            s = np.sqrt(np.dot(n, n))
    n = n / s
    return n

def angle(vec1, vec2):

    theta=np.arccos(np.dot(vec1,vec2) / (np.sqrt(np.dot(vec1,vec1)) * np.sqrt(np.dot(vec2, vec2))))
    return theta


# ===================================================================
# DeepACSON/CSD/unit_tangent_vector.py (verbatim)
# ===================================================================

def unit_tangent_vector(curve):

    d_curve = np.gradient(curve, axis=0)
    ds = np.expand_dims((np.sum(d_curve**2, axis=1))**0.5, axis=1)
    ds[ds==0] = 1e-5
    u_tang_vec = d_curve/np.repeat(ds, curve.shape[1], axis=1)
    return u_tang_vec


# ===================================================================
# DeepACSON/CSD/skeleton3D.py (verbatim)
# ===================================================================

def pointmin(D):

    sz = D.shape
    max_D = np.max(D)
    Fx = np.zeros(sz)
    Fy = np.zeros(sz)
    Fz = np.zeros(sz)

    J = max_D * np.ones(np.array(sz)+2)
    J[1:-1,1:-1,1:-1] = D

    x = [0, 1,-1, 0, 0, 1, 1,-1,-1, 0, 1,-1, 0, 0, 1, 1,-1,-1, 1,-1, 0, 0, 1, 1,-1,-1]
    y = [0, 0, 0, 1,-1, 1,-1, 1,-1, 0, 0, 0, 1,-1, 1,-1, 1,-1, 0, 0, 1,-1, 1,-1, 1,-1]
    z = [1, 1, 1, 1, 1, 1, 1, 1, 1,-1,-1,-1,-1,-1,-1,-1,-1,-1, 0, 0, 0, 0, 0, 0, 0, 0]

    #x = [1,-1, 0, 0, 0, 0]
    #y = [0, 0, 1,-1, 0, 0]
    #z = [0, 0, 0, 0, 1,-1]

    for i in range(26):
        In = J[1+x[i]:1+sz[0]+x[i], 1+y[i]:1+sz[1]+y[i], 1+z[i]:1+sz[2]+z[i]]
        check = In<D
        D[check] = In[check]

        den = (x[i]**2 + y[i]**2 + z[i]**2)**0.5

        Fx[check]= x[i]/den
        Fy[check]= y[i]/den
        Fz[check]= z[i]/den
    return Fx, Fy, Fz


def Euler_path(Fx,Fy,Fz,start_point,step_size):

    f_start_point = np.floor(start_point).astype(int)
    sz = Fx.shape

    x = [0, 0, 0, 0, 1, 1, 1, 1]
    y = [0, 0, 1, 1, 0, 0, 1, 1]
    z = [0, 1, 0, 1, 0, 1, 0, 1]

    neighbor_inx = np.array((x,y,z)).T

    base = f_start_point + neighbor_inx
    base[base<0] = 0
    xbase=base[:,0]; xbase[xbase>=sz[0]]=sz[0]-1
    ybase=base[:,1]; ybase[ybase>=sz[1]]=sz[1]-1
    zbase=base[:,2]; zbase[zbase>=sz[2]]=sz[2]-1
    base=np.array((xbase,ybase,zbase)).T

    dist2f=np.squeeze(start_point-f_start_point)
    dist2c=1-dist2f

    perc = np.array((   dist2c[0]*dist2c[1]*dist2c[2],
                        dist2c[0]*dist2c[1]*dist2f[2],
                        dist2c[0]*dist2f[1]*dist2c[2],
                        dist2c[0]*dist2f[1]*dist2f[2],
                        dist2f[0]*dist2c[1]*dist2c[2],
                        dist2f[0]*dist2c[1]*dist2f[2],
                        dist2f[0]*dist2f[1]*dist2c[2],
                        dist2f[0]*dist2f[1]*dist2f[2]  ))



    gradient_valueX=[Fx[tuple(i)] for i in base]*perc
    gradient_valueY=[Fy[tuple(i)] for i in base]*perc
    gradient_valueZ=[Fz[tuple(i)] for i in base]*perc

    gradient_value=np.array((gradient_valueX,gradient_valueY,gradient_valueZ))

    sum_g=np.sum(gradient_value,axis=1)

    gradient=sum_g/((np.sum(sum_g**2)+0.000001)**0.5)

    end_point = start_point - step_size*gradient

    if (np.any(end_point<0) or end_point[0,0]>sz[0] or end_point[0,1]>sz[1] or end_point[0,2]>sz[2]):
        end_point=np.zeros((1,3))
    return end_point


def euler_shortest_path(D,source_point,start_point,step_size):

    Fx, Fy, Fz = pointmin(D)
    Fx = -Fx
    Fy = -Fy
    Fz = -Fz

    itr = 0
    path = start_point
    while True:

        end_point = Euler_path(Fx,Fy,Fz,start_point,step_size)

        dist_endpoint_to_all = np.sum((source_point-end_point)**2,axis=1)**0.5
        distance_to_endpoint = min(dist_endpoint_to_all)

        if(itr>=10):
            Movement = np.sum((end_point-path[itr-10])**2)**0.5
        else:
            Movement = step_size+1

        if(np.all(end_point==0) or Movement<step_size):
            break

        itr = itr+1

        path = np.append(path,end_point,axis=0)

        if(distance_to_endpoint<10*step_size):
            source_inx = source_point[np.argmin(dist_endpoint_to_all)]
            path = np.append(path,np.array(source_inx,ndmin=2),axis=0)
            break

        start_point = end_point
    return path


def get_line_length(L):

    dist = np.sum(np.sum((L[1:] - L[:-1])**2,axis=1)**0.5)
    return dist


def organize_skeleton(skel_seg,length_th):

    final_skeleton = []

    n = len(skel_seg)
    end_points = np.zeros((n*2,3))

    l = 0
    for i in range(n):
        ss = skel_seg[i]
        l = max(l,len(ss))
        end_points[i*2] = ss[0]
        end_points[i*2+1] = ss[-1]

    connecting_distance = 2

    for i in range(n):

        ss = np.asarray(skel_seg[i])

        ex = np.reshape(end_points[:,0],(-1,1)); ex = np.repeat(ex,len(ss),axis=1)
        sx = np.reshape(ss[:,0],(1,-1)); sx = np.repeat(sx,len(end_points),axis=0)

        ey = np.reshape(end_points[:,1],(-1,1)); ey = np.repeat(ey,len(ss),axis=1)
        sy = np.reshape(ss[:,1],(1,-1)); sy = np.repeat(sy,len(end_points),axis=0)

        ez = np.reshape(end_points[:,2],(-1,1)); ez = np.repeat(ez,len(ss),axis=1)
        sz = np.reshape(ss[:,2],(1,-1)); sz = np.repeat(sz,len(end_points),axis=0)

        D = (ex-sx)**2 + (ey-sy)**2 + (ez-sz)**2

        check = np.amin(D, axis=1) < connecting_distance
        check[i*2] = False
        check[i*2+1] = False

        cut_skel = [0,len(ss)]
        if(any(check)):
            for ii in range(len(check)):
                if(check[ii]):
                    line = D[ii]
                    min_ind = np.ma.argmin(line)
                    if((min_ind>2) and (min_ind<(len(line)-2))):
                        cut_skel.append(min_ind)

        cut_skel = sorted(cut_skel)
        for j in range(len(cut_skel)-1):
            skel_breaked_seg = ss[cut_skel[j]:cut_skel[j+1]]
            length_skel_seg = get_line_length(skel_breaked_seg)
            if(length_skel_seg >= length_th):
               final_skeleton.append(skel_breaked_seg)

    return final_skeleton


def skeleton(Ax):

    boundary_dist=skfmm.distance(Ax)

    source_point=np.unravel_index(np.argmax(boundary_dist), boundary_dist.shape)
    maxD=boundary_dist[source_point]

    speed_im=(boundary_dist/maxD)**1.5

    Ax=np.ones(Ax.shape)
    Ax[source_point]=0

    flag=True
    skeleton_segments=[]
    source_point = np.array(source_point,ndmin=2)
    while True:

        D=skfmm.travel_time(Ax,speed_im)
        end_point=np.unravel_index(np.ma.argmax(D), D.shape)
        max_dist=D[end_point]
        D=np.ma.MaskedArray.filled(D,max_dist)

        end_point = np.array(end_point,ndmin=2)
        shortest_line=euler_shortest_path(D,source_point,end_point,step_size=0.1)
        #shortest_line = discrete_shortest_path(D,end_point)

        line_length=get_line_length(shortest_line)
        print(line_length)

        if flag:
            length_threshold=min(40*maxD, 0.18*line_length)
            flag=False

        if(line_length<=length_threshold):
            break


        source_point=np.append(source_point,shortest_line,axis=0)

        skeleton_segments.append(shortest_line)

        shortest_line=np.floor(shortest_line).astype(int)

        for i in shortest_line:
            Ax[tuple(i)]=0

    if len(skeleton_segments)!=0:
        final_skeleton=organize_skeleton(skeleton_segments,length_threshold)
    else:
        final_skeleton=[]

    return final_skeleton

if __name__ == "__main__":
    skeleton(sys.argv[1])


# ===================================================================
# Cross-section sampling (from DeepACSON/CSD/shape_decomposition.py, restructured)
#
# The original tangent_planes_to_zone_of_interest() combines cross-section
# sampling with zone-of-interest tracking (Hausdorff distance, mean curve,
# shift imposition). We extract only the cross-section sampling logic, and
# additionally compute the cross-section area.
# ===================================================================

def sample_cross_section(binary_vol, point, tangent_vec, g_radius, g_res):
    """
    Sample a perpendicular cross-section and measure its area.

    Uses the DeepACSON approach: rotate a planar grid to be perpendicular to
    the tangent vector, trilinear-interpolate the binary volume, then find the
    connected component at the center.

    Args:
        binary_vol: 3D binary volume (float64, 1=inside axon)
        point: (3,) skeleton point coordinates in volume space
        tangent_vec: (3,) unit tangent vector at this point
        g_radius: grid radius in voxels
        g_res: grid resolution in voxels (e.g. 0.25)

    Returns:
        area_pixels: number of pixels in the central connected component,
                     or 0 if invalid (no center hit, multiple labels, too small)
    """
    sz = binary_vol.shape

    # Build sampling plane grid — verbatim from shape_decomposition.py
    x, y = np.mgrid[-g_radius:g_radius:g_res, -g_radius:g_radius:g_res]
    z = np.zeros_like(x)
    xyz = np.array([np.ravel(x), np.ravel(y), np.ravel(z)]).T

    cent_ball = (x**2+y**2)<g_res*1

    # Rotate plane to be perpendicular to tangent — verbatim
    if np.array_equal(tangent_vec, np.array([0, 0, 0])):
        return 0

    rot_axis = unit_normal_vector(tangent_vec, np.array([0,0,1]))
    theta = angle(tangent_vec, np.array([0,0,1]))
    rot_mat = rotation_matrix_3D(rot_axis, theta)
    rotated_plane = np.squeeze(rotate_vector(xyz, rot_mat))
    cross_section_plane = rotated_plane+point

    # Trilinear interpolation — verbatim
    interpolating_func = rgi((range(sz[0]),range(sz[1]),range(sz[2])),
                                binary_vol,bounds_error=False,fill_value=0)
    cross_section = interpolating_func(cross_section_plane)
    bw_cross_section = cross_section>=0.5
    bw_cross_section = np.reshape(bw_cross_section, x.shape)

    # Connected component at center — verbatim
    label_cross_section, nn = label(bw_cross_section, return_num=True)
    main_lbl = np.unique(label_cross_section[cent_ball])
    main_lbl = main_lbl[np.nonzero(main_lbl)]

    if len(main_lbl)!=1:
        return 0

    bw_cross_section = label_cross_section==main_lbl[0]

    # Size filter — verbatim
    nz_X = np.count_nonzero(np.sum(bw_cross_section, axis=0))
    nz_Y = np.count_nonzero(np.sum(bw_cross_section, axis=1))
    if (nz_X<4) or (nz_Y<4):
        return 0

    # Area via regionprops — verbatim
    props = regionprops(bw_cross_section.astype(np.int32))
    if len(props) == 0:
        return 0

    return props[0].area
