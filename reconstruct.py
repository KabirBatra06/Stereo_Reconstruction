import numpy as np
import scipy
import scipy.linalg
from scipy.optimize import least_squares
import cv2
import math
import matplotlib.pyplot as plt
import random

#  GETTING AN INITAL ESTIMATE OF F USING ANNOTATED POINTS
def F_estimate(img0_pts, img1_pts):

    A = np.empty((len(img0_pts), 9))
    i=0
    for pt1, pt2 in zip(img0_pts, img1_pts):
        A[i] = np.array([pt1[0]*pt2[0], pt1[1]*pt2[0], pt2[0], pt1[0]*pt2[1], pt1[1]*pt2[1], pt2[1], pt1[0], pt1[1], 1])
        i+=1

    _, _, V = np.linalg.svd(A)
    F = np.transpose(V)[:,-1]
    F /= F[-1]
    F = F.reshape(3,3)

    U, D, V = np.linalg.svd(F)
    D[-1] = 0
    F = U @ np.diag(D) @ V

    return F

#  GETTING EPIPOLES FROM A FUNDAMENTAL MATRIX
def get_epipoles(F):
    e0 = scipy.linalg.null_space(F).flatten()
    e1 = scipy.linalg.null_space(F.T).flatten()

    return e0/e0[-1], e1/e1[-1]

#  GETTING CAMERA PROJECTION MATRICES FOR CAMERAS IN CANNONICAL FORM
def get_canonical(F, e1):

    s = np.array([[0, -e1[2], e1[1]], 
                  [e1[2], 0, -e1[0]], 
                  [-e1[1], e1[0], 0]])

    P0 = np.column_stack((np.eye(3), [0,0,0]))
    P1 = np.column_stack((np.matmul(s,F), e1))

    return P0, P1

#  ESTIMATING WORLD POINTS GIVEN TWO IMAGE POINTS
def get_world_pt(P0, P1, pt0, pt1):

    A = np.array([pt0[0]*P0[2] - P0[0],
                  pt0[1]*P0[2] - P0[1],
                  pt1[0]*P1[2] - P1[0],
                  pt1[1]*P1[2] - P1[1]])
    
    _, _, V = np.linalg.svd(A)
    X = np.transpose(V)[:,-1]
    X /= X[-1]
    X = X.reshape(-1,)
    return X

# COST FOR LM WHEN OPTOMIZING P_PRIME AND WORLD POINTS
def cost(params, P0, img0_points, img1_points):

    retval = []
    P1 = params[:12].reshape((3,4))
    world_points = params[12:]

    i=0
    for pt0, pt1 in zip(img0_points, img1_points):

        w_p = world_points[i:i+3]

        pt1_pred = np.matmul(P1, np.append(w_p,1))
        retval.append(pt1_pred[0] - pt1[0])
        retval.append(pt1_pred[1] - pt1[1])

        pt0_pred = np.matmul(P0, np.append(w_p,1))
        retval.append(pt0_pred[0] - pt0[0])
        retval.append(pt0_pred[1] - pt0[1])
        i+=3


    return np.array(retval)

#  OPTIMIZING P_PRIME AND WORLD POINTS USING LM
def LM_estimation(P0, P1, img0_points, img1_points):

    params = []

    for p in P1.flatten():
        params.append(p)

    for pt0, pt1 in zip(img0_points, img1_points):
        w_p = get_world_pt(P0, P1, pt0, pt1)
        params.append(w_p[0])
        params.append(w_p[1])
        params.append(w_p[2])

    params = np.array(params)
    
    retval = least_squares(cost, params, method='lm', args=(P0, img0_points, img1_points))
    
    return retval.x

# COST FOR LM WHEN OPTOMIZING ONLY WORLD POINTS
def cost_reconstruct(params, P0, P1, img0_points, img1_points):

    retval = []
    world_points = params

    i=0
    for pt0, pt1 in zip(img0_points, img1_points):

        w_p = world_points[i:i+3]

        pt1_pred = np.matmul(P1, np.append(w_p,1))
        retval.append(pt1_pred[0] - pt1[0])
        retval.append(pt1_pred[1] - pt1[1])

        pt0_pred = np.matmul(P0, np.append(w_p,1))
        retval.append(pt0_pred[0] - pt0[0])
        retval.append(pt0_pred[1] - pt0[1])
        i+=3


    return np.array(retval)

#  OPTIMIZING ONLY WORLD POINTS USING LM
def LM_estimation_reconstruct(P0, P1, img0_points, img1_points):

    params = []

    for pt0, pt1 in zip(img0_points, img1_points):
        w_p = get_world_pt(P0, P1, pt0, pt1)
        params.append(w_p[0])
        params.append(w_p[1])
        params.append(w_p[2])

    params = np.array(params)
    
    retval = least_squares(cost_reconstruct, params, method='lm', args=(P0, P1, img0_points, img1_points))
    
    return retval.x

def get_F(P0, P1):
    e1 = P1[:,-1]
    s = np.array([[0, -e1[2], e1[1]], 
                  [e1[2], 0, -e1[0]], 
                  [-e1[1], e1[0], 0]])
    
    P_inv = np.linalg.pinv(P0)

    F = s @ P1 @ P_inv

    return F

#  GETTING RECTIFICATION MATRIX FOR RIGHT IMAGE
def get_H_prime(img, e1):
    h, w, _ = img.shape

    T = np.array([
        [1, 0, -w/2],
        [0, 1, -h/2],
        [0, 0, 1]
    ])

    T_back = np.array([
        [1, 0, w/2],
        [0, 1, h/2],
        [0, 0, 1]
    ])

    angle = np.arctan(-(e1[1] - h / 2)/ (e1[0] - w / 2))

    R = np.array([
        [np.cos(angle), -np.sin(angle), 0],
        [np.sin(angle), np.cos(angle), 0],
        [0, 0, 1]
    ])

    f = ((e1[0] - w/2) * math.cos(angle) - (e1[1] - h/2) * math.sin(angle))

    G = np.eye(3)
    G[2,0] = -1/f

    return T_back @ G @ R @ T

#  GETTING RECTIFICATION MATRIX FOR LEFT IMAGE
def get_H(P, P_prime, H_prime, x_points, x_prime_points):
    PP = np.linalg.pinv(P)

    M = P_prime @ PP
    H0 = H_prime @ M

    x_hat = np.zeros((len(x_points), 3))
    for i, pt in enumerate(x_points):
        val = H0 @ np.array([pt[0], pt[1], 1])
        x_hat[i] = val / val[-1]

    x_prime_hat = np.zeros((len(x_prime_points), 3))
    for i, pt in enumerate(x_prime_points):
        val = H_prime @ np.array([pt[0], pt[1], 1])
        x_prime_hat[i] = val / val[-1]

    b = x_prime_hat[:, 0]
    a = (np.linalg.inv(x_hat.T @ x_hat) @ x_hat.T) @ b
    Ha = np.eye(3)
    Ha[0, :] = a

    H1 = Ha @ H0
    H1 = H1 / H1[2, 2]

    return H1

# USING SSD TO FIND CORRESPONDENCES
def SSD_correspondence(image_1, image_2, ip_1, ip_2):
    img1 = cv2.cvtColor(image_1, cv2.COLOR_BGR2GRAY) / 255
    img2 = cv2.cvtColor(image_2, cv2.COLOR_BGR2GRAY) / 255

    window = 4 # Window for checking similarity in surrounding pixel values
    corres_ip2 = np.zeros_like(ip_1)

    # Finding point in second image with lowest SSD for interest point in first image 
    for i, p1 in enumerate(ip_1):
        if p1[0]>window and p1[1]>window and p1[1]+window < 500:
            area_1 = img1[p1[0]-window:p1[0]+window, p1[1]-window:p1[1]+window]
            ssd_min = 999999
            for p2 in ip_2:
                if p2[0]>window and p2[1]>window and p2[1]+window < 500:

                    area_2 = img2[p2[0]-window:p2[0]+window, p2[1]-window:p2[1]+window]
                    ssd = np.sum((area_1 - area_2) ** 2)
                    if ssd < ssd_min:
                        ssd_min = ssd
                        corres_ip2[i] = p2
    
    corres_ip2[:, 1] += img1.shape[1] # offsetting interest points in second image for plotting purpose
    return corres_ip2
    
# FINDING AND LABELING ALL THE CORRESPONDENCES
def correspondances(img0, img1):

    edges0 = cv2.Canny(img0, 200, 200)
    edges1 = cv2.Canny(img1, 200, 200)

    plt.imsave("task1_out/edges0.jpg", edges0, cmap = 'gray')
    plt.imsave("task1_out/edges1.jpg", edges1, cmap = 'gray')

    pt_idx0 = np.where(edges0 > 1)
    pt_idx1 = np.where(edges1 > 1)

    point_idx0 = np.hstack((pt_idx0[0].reshape(-1,1), pt_idx0[1].reshape(-1,1)))
    point_idx1 = np.hstack((pt_idx1[0].reshape(-1,1), pt_idx1[1].reshape(-1,1)))

    point_idx0 = np.split(point_idx0, np.argwhere(np.diff(point_idx0[:,0]) != 0)[:,0] + 1)
    point_idx1 = np.split(point_idx1, np.argwhere(np.diff(point_idx1[:,0]) != 0)[:,0] + 1)

    all_corres = []
    combo_image = np.concatenate((img0, img1), axis=1)
    final_corres_img0 = []
    final_corres_img1 = []
    for i in range(200, len(point_idx0)-150):
        candidates = np.concatenate(point_idx1[i-10:i+10])

        corres_ip2 = SSD_correspondence(img0, img1, point_idx0[i], candidates)
        all_corres.append(corres_ip2)

        if i % 10 == 0 and corres_ip2[0][1]>500:
            idx = len(point_idx0[i])//2
            idx2 = len(point_idx0[i])//2
            combo_image = cv2.circle(combo_image, (point_idx0[i][idx][1], point_idx0[i][idx][0]), radius=3, color=(255, 0, 255), thickness=-1)
            combo_image = cv2.circle(combo_image, (corres_ip2[idx][1], corres_ip2[idx][0]), radius=3, color=(255 ,0, 0), thickness = -1)
            combo_image = cv2.line(combo_image, (point_idx0[i][idx][1], point_idx0[i][idx][0]), (corres_ip2[idx][1], corres_ip2[idx][0]), (random.randrange(255) ,random.randrange(255) ,random.randrange(255)), 2)
            final_corres_img0.append(list(point_idx0[i][idx2]))
            final_corres_img1.append(list(corres_ip2[idx2]))

   
    cv2.imwrite("task1_out/correspondences.jpg", combo_image)

    return final_corres_img0, final_corres_img1

# CONVERTING CORRESPONDING POINTS FROM RECTIFIIED IMAGE COORDINATES TO NON RECTIFIED IMAGE COORDINATES
def get_unrectified_points(H, H_prime, img0_pts, img1_pts):

    H_inv = np.linalg.inv(H)
    H_prime_inv = np.linalg.inv(H_prime)

    retval1 = []
    retval2 = []

    for i in range(len(img0_pts)):
        val1 = H_inv @ np.array([img0_pts[i][0], img0_pts[i][0], 1])
        val1 = val1 / val1[-1]
        retval1.append([int(val1[0]), int(val1[1])])

        val2 = H_prime_inv @ np.array([img1_pts[i][0], img1_pts[i][0], 1])
        val2 = val2 / val2[-1]
        retval2.append([int(val2[0]), int(val2[1])])

    return retval1, retval2

# PLOTTING 3D REPRESENTATION
def plot_3d_triplets(points, img0):

    coords = [points[i:i+3] for i in range(0, len(points), 3) if len(points[i:i+3]) == 3]

    fig = plt.figure()
    ax1 = fig.add_subplot(1,2,1, projection='3d')
    ax2 = fig.add_subplot(1,2,2)
    ax2.imshow(img0)

    for pt in coords[:-14]:
        ax1.scatter(pt[0], pt[1], pt[2], c='blue', marker='o')

    marked = coords[-14:]
    for i in range(len(marked)):
        ax1.scatter(marked[i][0], marked[i][1], marked[i][2], c='red', marker='o')
        if i < 4:
            ax1.plot([marked[i][0], marked[i+1][0]], [marked[i][1], marked[i+1][1]], 'ro', linestyle="-")

    ax2.axis('off')
    plt.show()

def main():

    # HAND LABELED CORRESPONDENCES
    img0_points = [[129, 144], [287, 127], [427, 245], [220, 289], [114, 231], [435, 342], [208, 420], [213, 348], [120, 191], [424, 289], [252,156], [325,268], [263,322], [150,153]]
    img1_points = [[139, 129], [310, 126], [412, 265], [161, 282], [127, 213], [424, 371], [149, 417], [157, 343], [129, 176], [412, 312], [263,151], [286,273], [213,323], [155,137]]

    # GETTING INITAL ESTIMATES
    F = F_estimate(img0_points, img1_points)

    e0, e1 = get_epipoles(F)

    P0, P1 = get_canonical(F, e1)

    # OPTIMIZING ESTIMATES WITH LM
    params = LM_estimation(P0, P1, img0_points, img1_points)

    P1_refined = params[:12].reshape((3,4))

    F_refined = get_F(P0, P1_refined)

    e0_refined, e1_refined = get_epipoles(F_refined)

    # FINDING RECTIFICATION HOMOGRAPHIES
    img0 = cv2.imread("img6.jpg")
    img1 = cv2.imread("img7.jpg")

    H_prime = get_H_prime(img1, e1_refined)
    H_prime /= H_prime[-1][-1]

    H = get_H(P0, P1_refined, H_prime, img0_points, img1_points)

    H /= H[-1][-1]

    T = np.array([
        [1, 0, 0],
        [0, 1, 100],
        [0, 0, 1]
    ])

    # RECTIFYING IMAGES
    img0_out = cv2.warpPerspective(img0, H@T, (img0.shape[1], img0.shape[0]))
    img1_out = cv2.warpPerspective(img1, H_prime@T, (img1.shape[1], img1.shape[0]))

    cv2.imwrite("task1_out/img0_out.jpg", img0_out)
    cv2.imwrite("task1_out/img1_out.jpg", img1_out)

    # FINDING CORRESPONDENCES
    final_corres_img0, final_corres_img1 = correspondances(img0_out, img1_out)

    unrec_corres_img0, unrec_corres_img1 = get_unrectified_points(H, H_prime, final_corres_img0, final_corres_img1)

    for pt0, pt1 in zip(img0_points,img1_points):
        unrec_corres_img0.append(pt0)
        unrec_corres_img1.append(pt1)

    # OPTIMIZING WORLD POINTS
    world_points = []
    world_points = LM_estimation_reconstruct(P0, P1_refined, unrec_corres_img0, unrec_corres_img1)

    # 3D PLOT OF WORLD POINTS
    plot_3d_triplets(world_points, img0)

if __name__ == "__main__":
    main()