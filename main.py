import cv2 as cv
import numpy as np
import os

# *****two class otsu*****
# --- for calculating the overlap between classes
# the threshold should minimize overlap
def o2_calc_variability(image, thres):
    # creating a threshold image from the image
    var_imag = np.zeros(image.shape)
    var_imag[image>thres] = 1

    # dividing the image into two classes
    pixels_above = image[var_imag == 1]
    pixels_below = image[var_imag == 0]

    # number of pixels for probability calculation
    num_total = image.size
    num_above = np.count_nonzero(var_imag == 1)

    # probability of pixel being in a class = number of pixels in class / total number of pixels
    p_above = num_above / num_total
    p_below = 1 - p_above

    # variance calculated with np.var()
    # if there are no pixels above or below the threshold, set the variance to 0 instead
    var_above = np.var(pixels_above) if len(pixels_above) > 0 else 0
    var_below = np.var(pixels_below) if len(pixels_below) > 0 else 0

    # returning the overlap between classes
    overlap = var_above*p_above + var_below*p_below
    return overlap

# ---for finding the best threshold
def o2_threshold_finding(image):
    # the range is 0 to the greatest value in the image
    t_range = range(np.max(image))
    criterias = []

    for value in t_range:
        criterias.append(o2_calc_variability(image, value))

    best_threshold = t_range[np.argmin(criterias)]

    return best_threshold

# ---main otsu with two classes method
def o2(image):
    gray_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    best_threshold = o2_threshold_finding(gray_image)

    # create the threshold image
    var_imag = np.zeros(gray_image.shape)
    var_imag[gray_image>best_threshold] = 1

    return var_imag


# *****multi class otsu*****
# for three classes
# ---calculate the variance for one threshold
# three classes, three things summed
def omulti_calc_variance(image, thres1, thres2):
    # thres1 is the smaller threshold
    if thres1 > thres2:
        return -1
    
    # thresholding image
    var_imag = np.zeros(image.shape)
    var_imag[image>thres1] = 1
    var_imag[image>thres2] = 2

    # dividing image into three classes
    pixels_1 = image[var_imag == 0]
    pixels_2 = image[var_imag == 1]
    pixels_3 = image[var_imag == 2]

    # pixel counts
    num_total = image.size
    num_class2 = np.count_nonzero(var_imag == 1)
    num_class3 = np.count_nonzero(var_imag == 2)
    num_class1 = num_total - num_class2 - num_class3

    # calculating probabilities
    p_1 = num_class1 / num_total
    p_2 = num_class2 / num_total
    p_3 = num_class3 / num_total

    # variability for the classes
    v_1 = np.var(pixels_1) if len(pixels_1) > 0 else 0
    v_2 = np.var(pixels_2) if len(pixels_2) > 0 else 0
    v_3 = np.var(pixels_3) if len(pixels_3) > 0 else 0
    v_img = np.var(image) if len(image) > 0 else 0

    # between class variance (sum of probability(vari - varimg)^2 for all classes k)
    variance = p_1*(v_1 - v_img)*(v_1 - v_img)  + p_2*(v_2 - v_img)*(v_2 - v_img) + p_3*(v_3 - v_img)*(v_3 - v_img)
   
    return variance 


def omulti_threshold_finding(image):
    t_limit = range(np.max(image))
    maxVar = 0
    op_thres1 = 0
    op_thres2 = 0

    # for every combination of thresholds
    for t in t_limit:
        for t2 in t_limit:
            curVar = omulti_calc_variance(image, t, t2)
            if curVar > maxVar:
                maxVar = curVar
                op_thres1 = t
                op_thres2 = t2

    return op_thres1, op_thres2


def omulti(image):
    gray_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    thres1, thres2 = omulti_threshold_finding(gray_image)

    # thresholding image
    var_imag = np.zeros(image.shape)
    var_imag[image>thres1] = 1
    var_imag[image>thres2] = 2

    return var_imag

# *****mean shift*****
def mean_shift(image):
    # mean shift with OpenCV's default method
    mean_sh = cv.pyrMeanShiftFiltering(image, 6, 6, maxLevel=1)
    # Convert to HSV to make the segments more noticable
    mean_sh_col = cv.cvtColor(mean_sh, cv.COLOR_BGR2HSV)

    # # saving the image
    # if not os.path.isfile("Output Data\mean_shit.png"):
    #     cv.imwrite(filename="Output Data\mean_shit.png", img=mean_sh_col)

    return mean_sh_col


def main():
    # reading images
    two_otsu = cv.imread("Input Data\OTSU2class-edge_L-150x150.png")
    multi_otsu = cv.imread("Input Data\OTSU Multiple Class-S01-150x150.png")
    mean_shit = cv.imread("Input Data\meanshit S00-150x150.png")

    # performing the segmentation on the appropriate image
    new_mean = mean_shift(mean_shit)
    new_two_otsu = o2(two_otsu)
    new_two_otsu_detailed = o2(multi_otsu)
    new_multi_otsu = omulti(multi_otsu)

    # image display
    print("*****Image Options:*****\n 1.) Otsu with two classes\n 2.) Otsu with multiple classes")
    print("3.) Mean Shift method")

    choice = input("Enter the number associated with the image you wish to display: ")
    
    if choice == "1":
        cv.namedWindow('Otsu with 2 Classes', cv.WINDOW_AUTOSIZE)
        cv.imshow('Otsu with 2 Classes', new_two_otsu)
        cv.namedWindow('Otsu with 2 Classes Detailed', cv.WINDOW_AUTOSIZE)
        cv.imshow('Otsu with 2 Classes Detailed', new_two_otsu_detailed)
    elif choice == "2":
        cv.namedWindow('Otsu with 3 Classes', cv.WINDOW_AUTOSIZE)
        cv.imshow('Otsu with 3 Classes', new_multi_otsu)
    elif choice == "3":
        cv.namedWindow('Mean Shift', cv.WINDOW_AUTOSIZE)
        cv.imshow('Mean Shift', new_mean)
    else:
        print("Invalid selection. Select a number between 1 and 3 to display an image.")
    
    cv.waitKey(0)
    cv.destroyAllWindows()


if __name__ == "__main__":
    main()