import cv2 as cv
import numpy as np
import os

# two class otsu
# --- for finding the best threshold
# the threshold should minimize overlap
def o2_calc_variability(image, thres):
    # creating a threshold image from the image
    var_imag = np.zeros(image.shape)
    var_imag[image>thres] = 1

    # dividing the image into two classes
    pixels_above = image[var_imag == 1]
    pixels_below = image[var_imag == 0]

    # number of pixels for weight calculation
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


def o2_threshold_finding(image):
    # the range is 0 to the greatest value in the image
    t_range = range(np.max(image) + 1) if np.max(image) < 255 else range(255)
    criterias = []

    for value in t_range:
        criterias.append(o2_calc_variability(image, value))

    best_threshold = t_range[np.argmin(criterias)]

    return best_threshold

def o2(image):
    gray_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    best_threshold = o2_threshold_finding(gray_image)

    # create the threshold image
    var_imag = np.zeros(gray_image.shape)
    var_imag[gray_image>best_threshold] = 1

    if not os.path.isfile("Output Data\o2_image.png"):
        cv.imwrite(filename="Output Data\o2_image.png", img=var_imag)
    elif not os.path.isfile("Output Data\o2_image2.png"):
        cv.imwrite(filename="Output Data\o2_image2.png", img=var_imag)

    return var_imag


# multi class otsu

# mean shift
def mean_shift(image):
    mean_sh = cv.pyrMeanShiftFiltering(image, 6, 6, maxLevel=1)
    mean_sh_col = cv.cvtColor(mean_sh, cv.COLOR_BGR2HSV)

    if not os.path.isfile("Output Data\mean_shit.png"):
        cv.imwrite(filename="Output Data\mean_shit.png", img=mean_sh_col)

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

    # image display
    print("*****Image Options:*****\n 1.) Otsu with two classes\n 2.) Otsu with multiple classes")
    print("3.) Mean Shift method")

    choice = input("Enter the number associated with the image you wish to display: ")
    

    if choice == "1":
        print("Otsu with two classes")
        cv.namedWindow('Otsu with 2 Classes', cv.WINDOW_AUTOSIZE)
        cv.imshow('Otsu with 2 Classes', new_two_otsu)
        cv.namedWindow('Otsu with 2 Classes Detailed', cv.WINDOW_AUTOSIZE)
        cv.imshow('Otsu with 2 Classes Detailed', new_two_otsu_detailed)
    elif choice == "2":
        print("Otsu with multiple classes")
    elif choice == "3":
        cv.namedWindow('Mean Shift', cv.WINDOW_AUTOSIZE)
        cv.imshow('Mean Shift', new_mean)
    else:
        print("Invalid selection. Select a number between 1 and 3 to display an image.")
    
    cv.waitKey(0)
    cv.destroyAllWindows()


if __name__ == "__main__":
    main()