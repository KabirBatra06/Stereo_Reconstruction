import numpy as np
import cv2
from tqdm import tqdm

def gen_disp_map(image1, image2, window_x, window_y, dmax):

    # FINDING LIMITS FOR WINDOW AREA
    retval = np.zeros_like(image1)
    left = window_x // 2
    right = (window_x // 2) + 1
    up = window_y // 2
    down = (window_y // 2) + 1

    rows = image1.shape[0]
    cols = image1.shape[1]

    #PADDING IMAGE TO ACCOMODATE WINDOW
    padded_img1 = np.pad(image1, ((left, left), (up, up)))
    padded_img2 = np.pad(image2, ((left, left), (up, up)))

    for i in tqdm(range(left, rows+left)):
        for j in range(up, cols+up):
            p = padded_img1[i][j]

            cost = window_x * window_y # MAX COST IS IF ALL 1'S AND THAT IS WINDOW SIZE
            d_min = -10
            for d in range(dmax):
                if (j - d - up) >=0: 
                    q = padded_img2[i][j-d]
                    bv1 = padded_img1[i-left:i+right, j-up:j+down] > p # BIT REPRESENTATION OF WINDOW 1
                    bv2 = padded_img2[i-left:i+right, j-d-up:j-d+down] > q # BIT REPRESENTATION OF WINDOW 2
                    bv_xor = (bv1.flatten() ^ bv2.flatten()).astype(int)
                    count = np.sum(bv_xor == 1) # COUNTING 1'S TO GET COST
                    if count < cost:
                        cost = count
                        d_min = d # BEST DISPARITY VALUE

            retval[i-left][j-up] = d_min # CREATING DISPARITY MASK
    
    return retval
            


def main():
    disp_map = cv2.imread("Task3Images/disp2.png", 0)
    image1 = cv2.imread("Task3Images/im2.png", 0)
    image2 = cv2.imread("Task3Images/im6.png", 0)

    # GETTING DMAX
    disp_f = disp_map.astype(float)
    disp_f /= 4
    disp_f = disp_f.astype(int)
    dmax = np.max(disp_f)

    # DISPARITY FOR 10X10 WINDOW
    disparity1 = gen_disp_map(image1, image2, 10, 10, dmax)

    # CALCULATING ERROR WITH GROUND TRUTH
    error = abs(disparity1 - disp_f)
    within_delta = np.sum(error[disp_f>0] <= 2) 
    total_valid = np.sum(disp_f>0)

    mask1 = (error <=2) * 255 # MAKING MASK

    print("Accuracy for image1 = ", within_delta/total_valid * 100)
    disparity1 = (disparity1 / dmax) * 255
    cv2.imwrite("task3_out/out_disparity1.jpg", disparity1)
    cv2.imwrite("task3_out/out_mask1.jpg", mask1)

    # DISPARITY FOR 20X20 WINDOW
    disparity2 = gen_disp_map(image1, image2, 20, 20, dmax)

    # CALCULATING ERROR WITH GROUND TRUTH
    error = abs(disparity2 - disp_f)
    within_delta = np.sum(error[disp_f>0] <= 2) 
    total_valid = np.sum(disp_f>0)

    mask2 = (error <=2) * 255 # MAKING MASK

    print("Accuracy for image 2 = ", within_delta/total_valid * 100)
    disparity2 = (disparity2 / dmax) * 255
    cv2.imwrite("task3_out/out_disp2.jpg", disparity2)
    cv2.imwrite("task3_out/out_mask2.jpg", mask2)

if __name__ == "__main__":
    main()