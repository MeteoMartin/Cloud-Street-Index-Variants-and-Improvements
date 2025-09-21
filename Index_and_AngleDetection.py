# [0] Imports
from PIL import Image as pimg  # Like mpimg for img processing
import numpy as np
import matplotlib
import matplotlib.pyplot as plt  # For Data Visualisation
import matplotlib.patheffects as pe
from PIL.ImageColor import colormap
from scipy.signal import find_peaks, peak_prominences  # For finding Extrema
from datetime import datetime, timedelta
import matplotlib.dates as mdates
from osgeo import gdal
import cartopy.crs as ccrs
import os
import pandas as pd
import xarray as xr
from scipy.ndimage import gaussian_filter1d
from mpl_toolkits.axes_grid1 import make_axes_locatable


plt.style.use('tableau-colorblind10') # Makes use of a colour blind friendly colour palette
os.environ['GTIFF_SRS_SOURCE'] = 'EPSG'
os.environ['PROJ_LIB'] = '/home/XXX/miniconda3/envs/envname/share/proj'
gdal.DontUseExceptions()


# [1] Define Functions ================================================================================================
# Function to calculate the running average; 
# [Returns np.convolve with running average]
def running_average(data, window_size):
    return np.convolve(data, np.ones(window_size) / window_size, mode='valid')
#======================================================================================================================
# Function to read in the image and convert to greyscale; [Returns image as 2d-array]
# Random Winkel: dreh=np.random.uniform(low=-90, high=90, size=1).round()[0]
def image_reader(imglocation, dreh=0, mute=False):
    """:return: image_array"""
    if not mute: print(f"< Einlesewinkel: {dreh}° >")
    image_data = np.array(pimg.open(imglocation).convert('L').rotate(dreh))
    return image_data

# Function that limits the size of an selected area by a/sqrt(2) if a is the shorter array dimension
# [Returns: masked array]
def array_size_lim(array, set_radius=False, set_diameter=False, opt_lim=False, max_lim=False):
    """
    :param array: 2-dimensional numpy array
    :param set_radius: radius of the selected area
    :param set_diameter: diameter of the selected area, accepts odd numbers
    :param opt_lim: limits size by min_dim/sqrt(2), if given alone: size = opt_diameter
    :param max_lim: limits size by min_dim, if given alone: size = max_diameter
    :return: resized array
    """
    # Neue Aufstellung, 1 xor 2 wenn 3 xor 4 begrenzt durch 3 xor 4, sonst kein limit
    min_dim = min(array.shape)
    max_diameter = min_dim
    opt_diameter = int(min_dim // np.sqrt(2))
    max_radius, opt_radius, odd = max_diameter // 2, opt_diameter // 2, 0

    # Herausfiltern von 0011, 0111, 1011, 1100, 1101, 1110, 1111
    if set_radius and set_diameter or opt_lim and max_lim: exit("Conflicting parameters!")
    # Handhabung von 0000 → 0010
    elif not set_radius and not set_diameter and not opt_lim and not max_lim: opt_lim = True
    # Handhabung von xx10
    if opt_lim or max_lim:
        if not set_radius and not set_diameter:
            if opt_lim: set_radius, odd = opt_radius, opt_diameter % 2
            if max_lim: set_radius, odd = max_radius, max_diameter % 2
        else:
            if opt_lim:
                if set_radius:
                    if set_radius > opt_radius: set_radius, odd = opt_radius, opt_diameter % 2
                if set_diameter:
                    if set_diameter > opt_diameter: set_diameter = opt_diameter
                    odd = set_diameter % 2
                    set_radius = set_diameter // 2
            if max_lim:
                if set_radius:
                    if set_radius > max_radius: set_radius, odd = max_radius, max_diameter % 2
                if set_diameter:
                    if set_diameter > max_diameter: set_diameter = max_diameter
                    odd = set_diameter % 2
                    set_radius = set_diameter // 2
    else:
        if set_diameter: set_radius, odd = set_diameter // 2, set_diameter % 2

    # slice array according to given or max value
    x_min = int(array.shape[1] // 2 - set_radius)
    x_max = int(array.shape[1] // 2 + set_radius + odd)
    y_min = int(array.shape[0] // 2 - set_radius)
    y_max = int(array.shape[0] // 2 + set_radius + odd)

    # Square section of image center with a diameter of pxrange
    sq_array = array[y_min:y_max, x_min:x_max]
    #print(np.shape(sq_array))
    return sq_array

# [*] FUNCTIONS FOR INDEX CALCULATIONS --------------------------------------------------------------------------------
# Function for calculating I_CS from Klingebiel et al. 2025
# [Returns: ics0]
def ics0_function(imglocation, radius=200, dreh=0):
    """:return: Index value (float)"""
    # [Ics0] Calculate Ics without the image being rotated
    img_ics0 = np.array(pimg.open(imglocation).convert('L').rotate(dreh))
    cval_ics0 = array_size_lim(img_ics0, set_radius=radius,max_lim=False)

    # [Ics0] get values of central line (hval) and central row (vval) in cval
    hval0 = cval_ics0[pxrange // 2, :]  # takes center row/line of selected area
    vval0 = cval_ics0[:, pxrange // 2]

    # [Ics0] find the (number of) peaks
    peaks_row_avg, _ = find_peaks(running_average(hval0, win_size), height=0)
    peaks_col_avg, _ = find_peaks(running_average(vval0, win_size), height=0)
    nhmax3 = len(peaks_row_avg)
    nvmax3 = len(peaks_col_avg)

    # [Ics0] Calculate Dr. Marcus Klingebiel's Cloud Street Index
    I_cs0 = round(1 - (nhmax3 / nvmax3), 2)  # running average, win_size=10 for smoothing timelines

    return I_cs0

# Function for calculating modified I_CS from Klingebiel et al. 2025
# [Returns: ics0n]
def ics0new_function(imglocation, radius=200, dreh=0):
    """:return: Index value (float)"""
    # [Ics0] Calculate Ics without the image being rotated
    img_ics0 = np.array(pimg.open(imglocation).convert('L').rotate(dreh))
    cval_ics0 = array_size_lim(img_ics0,set_radius=radius,max_lim=False)

    # [Ics0] get values of central line (hval) and central row (vval) in cval
    hval0 = cval_ics0[pxrange // 2, :]  # takes center row/line of selected area
    vval0 = cval_ics0[:, pxrange // 2]

    # [Ics0] find the (number of) peaks
    peaks_row_avg, _ = find_peaks(running_average(hval0, win_size), height=0)
    peaks_col_avg, _ = find_peaks(running_average(vval0, win_size), height=0)
    nhmax3 = len(peaks_row_avg)
    nvmax3 = len(peaks_col_avg)

    # [Ics0] Calculate Dr. Marcus Klingebiel's Cloud Street Index
    if nvmax3 == 0:
        I_cs0n = round(1 - nhmax3 / 1, 4)  # Prevent Division by 0 (Sea ice, idealized examples)
    elif nvmax3 > 0:
        I_cs0n = round(1 - nhmax3 / nvmax3, 4)
    else:
        I_cs0n = 0

    if I_cs0n < 0: I_cs0n = 0

    return I_cs0n

# Function to calculate I_CS,P for original image, old function! barely used anymore
# [Returns: I_CSP_max, deg_maxI_CSP, thI_CSP_max, deg_of_max, icsp0, I_cs0, cval_ics0, rotation_list]
def multi_index_function(imglocation, pxrange, dreh=0, show=0):
    """:return: I_CSP_max, deg_maxI_CSP, thI_CSP_max, deg_of_max, icsp0, I_cs0, cval_ics0, rotation_list
    I_CSP_max: highest value calculated; deg_maxI_CSP: angle for which I_CSP_max appeared;
    thI_CSP_max: polynomial maximum value; deg_of_max: angle of the polynomial maximum
    icsp0: I_CS,P at 0°; I_cs0: I_CS at 0°; cval_ics0: selected image area"""
    rotation_list = []  # Contains all values, nested list
    degree_list = []  # Rotation in degree
    ics_list, icsp_list, nicsp_list = [], [], [] # I_CS, I_CS,P, Normalized I_CS,P (Sum(hprom)/nhmax)

    img0 = np.array(pimg.open(imglocation).convert('L'))

    # Selected area using pxrange
    x_min = int(img0.shape[1] // 2 - 0.5 * pxrange)
    x_max = int(img0.shape[1] // 2 + 0.5 * pxrange)
    y_min = int(img0.shape[0] // 2 - 0.5 * pxrange)
    y_max = int(img0.shape[0] // 2 + 0.5 * pxrange)

    # Rotates image stepwise
    for angle in try_angles:
        angle += dreh
        img_loop = np.array(pimg.open(imglocation).convert('L').rotate(float(angle)))

        # Square section of image center with a diameter of pxrange
        cval = img_loop[y_min:y_max, x_min:x_max]
        # Lists for average values of rows/lines of cval
        hval_avg = np.mean(cval, axis=0)  # Mean of y along x-Axis
        vval_avg = np.mean(cval, axis=1)  # Mean of x along y-Axis

        # Calculate the Running Average
        hval_avg = running_average(hval_avg, win_size)
        vval_avg = running_average(vval_avg, win_size)

        # Find peaks in the average values and count them
        peaks_row_avg, _ = find_peaks(hval_avg)
        peaks_col_avg, _ = find_peaks(vval_avg)
        nhmax = len(peaks_row_avg)
        nvmax = len(peaks_col_avg)

        # Get prominences of peaks
        hproms = peak_prominences(hval_avg, peaks_row_avg)
        vproms = peak_prominences(vval_avg, peaks_col_avg)

        # [Ics*] Calculate Cloud Street Index with average values
        if nhmax < nvmax:  # Version with cases; Avoids devision by 0 and fractions > 1: (1-(8/4)=1 -> 1-(4/8)=0.5)
            if nvmax == 0:
                I_cs_avg = round(np.abs(1 - nhmax / 1), 4)  # Prevent Division by 0
            else:
                I_cs_avg = round(np.abs(1 - nhmax / nvmax), 4)
        elif nhmax > nvmax:
            if nhmax == 0:
                I_cs_avg = round(np.abs(1 - nvmax / 1), 4)
            else:
                I_cs_avg = round(np.abs(1 - nvmax / nhmax), 4)
        else:
            I_cs_avg = np.abs(0.0)

        # Calculate the prominence based index (icsp), I_CS,P
        icsp_index = np.abs(round((sum(hproms[0]) - sum(vproms[0])) / (sum(hproms[0]) + sum(vproms[0])), 3))

        # Normalized I_CS,P:
        term_a = sum(hproms[0])/nhmax
        term_b = sum(vproms[0])/nvmax
        I_csnp = abs(round((term_a - term_b) / (term_a + term_b)))

        # Append Values to their lists
        icsp_list.append(icsp_index)
        degree_list.append(round(angle, 1))
        ics_list.append(I_cs_avg)
        nicsp_list.append(I_csnp)

    rotation_list.append(degree_list)  # Rotation values
    rotation_list.append(icsp_list)  # Calculated icsp for each rotation value
    rotation_list.append(ics_list)  # Contains Cloud Street Indices of average values
    rotation_list.append(nicsp_list) # Normalized I_CS,P

    # Apply polynomial fit to an extended definition range to simulate a periodic polynomial
    # [I_CS,P]
    x_pr = np.array(rotation_list[0])
    y_pr = np.array(rotation_list[1])
    x_pr_extended = np.concatenate((x_pr - 90, x_pr, x_pr + 90))
    y_pr_extended = np.concatenate((y_pr, y_pr, y_pr))
    polyfit_pr = np.polyfit(x_pr_extended, y_pr_extended, 15) # deg: 10, 15 work well

    # Calculate 1st, 2nd derivative of the polynomial
    poly_pr = np.poly1d(polyfit_pr)
    deri1_pr = poly_pr.deriv()
    deri2_pr = deri1_pr.deriv()

    # Zero points of the 1st derivative
    crit_points_pr = deri1_pr.r

    # Definition range, goes beyond 90° for strange maxima
    deg_min, deg_max = 0, 95

    # Filter for real Maxima within the definition range
    valid_maxima = []
    for point in crit_points_pr:
        if np.isreal(point):
            point_real = np.real(point)
            if deg_min <= point_real <= deg_max and deri2_pr(point_real) < 0:
                valid_maxima.append(point_real)
    deg_of_max = 0
    thI_CSP_max = 0
    # Select highest Maxima from all valid Maxima
    if valid_maxima:
        y_vals = poly_pr(valid_maxima)
        idx_max = np.argmax(y_vals)
        deg_of_max = valid_maxima[idx_max]
        thI_CSP_max = y_vals[idx_max]
        #print(f"Valid Maxima at x = {deg_of_max:.4f}, y = {thI_CSP_max:.4f}")
    else:
        print("Found no valid maxima.")
        # use highest index value within definition range instead?

    # Prepare Polynomial fit for plot
    x_pr_smooth = np.arange(0, 90, 1)
    polynom_pr = np.polyval(polyfit_pr, x_pr_smooth)

    # find maximum icsp to calculate Ics at corresponding angle
    I_CSP_max = max(rotation_list[1][:])  # maximum I_CSP from all rotation steps
    index_max_pr = rotation_list[1].index(I_CSP_max)  # Index für Winkel, th = [1], sonst [0]
    icsp0 = rotation_list[1][0]  # I_CS,P bei 0°

    # [Ics0] Calculate Ics without the image being rotated
    img_ics0 = np.array(pimg.open(imglocation).convert('L'))
    cval_ics0 = img_ics0[y_min:y_max, x_min:x_max]

    # [Ics0] get values of central line (hval) and central row (vval) in cval
    hval0 = cval_ics0[pxrange // 2, :]  # takes center row/line of selected area
    vval0 = cval_ics0[:, pxrange // 2]

    # [Ics0] find the (number of) peaks
    peaks_row_avg, _ = find_peaks(running_average(hval0, win_size), height=0)
    peaks_col_avg, _ = find_peaks(running_average(vval0, win_size), height=0)
    nhmax3 = len(peaks_row_avg)
    nvmax3 = len(peaks_col_avg)

    # [Ics0] Calculate Dr. Marcus Klingebiel's Cloud Street Index
    if nhmax3 < nvmax3:  # Version with cases; Avoids devision by 0 and fractions > 1
        if nvmax3 == 0:
            I_cs0 = round(np.abs(1 - nhmax3 / 1), 4)  # Prevent Division by 0
        else:
            I_cs0 = round(np.abs(1 - nhmax3 / nvmax3), 4)
    elif nhmax3 > nvmax3:
        if nhmax3 == 0:
            I_cs0 = round(np.abs(1 - nvmax3 / 1), 4)
        else:
            I_cs0 = round(np.abs(1 - nvmax3 / nhmax3), 4)
    else:
        I_cs0 = np.abs(0.0)

    ###################### PLOT FOR ANGLE DEPENDENCE ######################
    w_subplot = 1
    if show == 1:
        fs = 12
        title = r'Angle dependance of $\mathrm{I_{CS,P}}$'
        if w_subplot:
            # Subplot with image and angle dependance
            fig, axs = plt.subplots(1, 2, figsize=(10, 4))
            fig.subplots_adjust(wspace=0.2)
            fig.tight_layout(pad=2)

            # Plot analysed image
            axs[0].set_aspect('equal')
            cb0 = axs[0].contourf(cval_ics0, 200, cmap="Greys_r")
            axs[0].set_xlabel("Across-track pixel", fontsize=fs)
            axs[0].set_ylabel("Along-track pixel", fontsize=fs)
            plt.colorbar(cb0)
            axs[0].set_title(f"Analysed image", fontsize=fs+2)

            # Plot angle dependance
            axs[1].set_aspect('auto', anchor='W')
            axs[1].plot(rotation_list[0], rotation_list[1], label=r'$\mathrm{I_{CS,P}}$')
            #axs[1].plot(rotation_list[0], rotation_list[2], label=r'$\mathrm{I_{CS}}$', linestyle='-.')
            axs[1].plot(x_pr_smooth, polynom_pr, label=r'Polynom of $\mathrm{I_{CS,P}}$', linestyle='--', linewidth=2)
            axs[1].plot(deg_of_max, thI_CSP_max, 'rx', markersize=10, label='Maximum')
            axs[1].set_title(title, fontsize=fs+2)
            axs[1].set_xlabel('Rotation | °', fontsize=fs)
            axs[1].set_ylabel(r'Index value', fontsize=fs)
            axs[1].spines[['right', 'top']].set_visible(False) # Soll laut Marcus überall eingebaut werden
            axs[1].legend()

        else:
            # Plots icsp, Ics depending on image rotation
            plt.figure(figsize=(8, 5))
            plt.plot(rotation_list[0], rotation_list[1], label=r'$\mathrm{I_{CS,P}}$')
            plt.plot(rotation_list[0], rotation_list[2], label=r'$\mathrm{I_{CS}}$', linestyle='-.')
            plt.plot(x_pr_smooth, polynom_pr, label=r'Polynom of $\mathrm{I_{CS,P}}$', linestyle='--', linewidth=2)
            plt.plot(deg_of_max, thI_CSP_max, 'rx', markersize=10, label='Maximum')
            plt.xlabel('Rotation | °', fontsize=fs)
            plt.ylabel(r'Index value', fontsize=fs)
            title = r'Angle dependance of $\mathbf{\mathrm{I_{CS}}}$ and $\mathrm{I_{CS,P}}$'
            plt.xticks(fontsize=fs), plt.yticks(fontsize=fs)
            plt.title(title, fontsize=fs + 2)
            plt.legend()
            plt.grid()
            plt.tight_layout()
        plt.show()
    #######################################################################
    deg_maxI_CSP = rotation_list[0][index_max_pr]

    return I_CSP_max, deg_maxI_CSP, thI_CSP_max, deg_of_max, icsp0, I_cs0, cval_ics0, rotation_list

# Function to calculate the icsp-index from two 1d-arrays (see output icsp_lists)
# [Returns: icsp_index, nhmax, hproms, nvmax, vproms]
def icsp_function(list1, list2, ravg=True):
    """:returns: icsp_index, nhmax, hproms, nvmax, vproms"""
    # Find peaks in the values
    if ravg:
        # Calculate the running average of the lists
        list1 = running_average(list1, win_size)
        list2 = running_average(list2, win_size)
        # Find the peaks in the lists
        peaks_row_avg, _ = find_peaks(list1)
        peaks_col_avg, _ = find_peaks(list2)
        # Count the number of peaks
        nhmax = len(peaks_row_avg)
        nvmax = len(peaks_col_avg)
        # Get prominences of peaks
        hproms = peak_prominences(list1, peaks_row_avg)
        vproms = peak_prominences(list2, peaks_col_avg)
    else:
        # Find the peaks in the lists
        peaks_row_avg, _ = find_peaks(list1)
        peaks_col_avg, _ = find_peaks(list2)
        # Count the number of peaks
        nhmax = len(peaks_row_avg)
        nvmax = len(peaks_col_avg)
        # Get prominences of peaks
        hproms = peak_prominences(list1, peaks_row_avg)
        vproms = peak_prominences(list2, peaks_col_avg)


    # Calculate the prominence based index
    icsp_index = np.abs(round((sum(hproms[0]) - sum(vproms[0])) / (sum(hproms[0]) + sum(vproms[0])), 3))
    return icsp_index, nhmax, hproms[0], nvmax, vproms[0]

# Function to prepare lists for icsp_function-Function from any given image array (pimg)
# [Returns: x_mean, y_mean]
def icsp_lists(masked_img, sigma=False):
    """:returns: x_mean, y_mean"""
    sel_area = masked_img

    x_mean = np.mean(sel_area, axis=0) # Mean of y along x-Axis
    y_mean = np.mean(sel_area, axis=1) # Mean of x along y-Axis
    #plt.plot(range(0, len(x_mean)), x_mean, label='x_mean')
    #plt.plot(range(0, len(y_mean)), y_mean, label='y_mean')

    if sigma:
        # Smoothing the values further
        x_mean = gaussian_filter1d(x_mean, sigma=sigma)
        y_mean = gaussian_filter1d(y_mean, sigma=sigma)
        #plt.plot(range(0, len(x_mean)), x_mean, label='x_mean_s', alpha=0.5)
        #plt.plot(range(0, len(y_mean)), y_mean, label='y_mean_s', alpha=0.5)

    #plt.legend()
    #plt.show()
    return x_mean, y_mean

# [*] FUNCTIONS WITH ANGLE METHODS ------------------------------------------------------------------------------------
# Function to find angle for Cloud Streets
# [Returns: thI_CSP_max, deg_of_max, rotation_list, x_pr_smooth, polynom_pr, cval0]
def angle_finder(img_array, wlen_fact=1, dreh=0, show=False):
    """:return: thI_CSP, deg_of_max, rotation_list, x_pr_smooth, polynom_pr, cval0"""
    rotation_list = []  # Contains all values, nested list
    degree_list = []  # Image rotation in degree
    icsp_list = []  # I_CS,P index values

    img1 = pimg.fromarray(img_array) # convert array back to PIL image, .rotate is fastest

    # Rotates image stepwise
    for angle in try_angles:
        angle += dreh
        img_loop = np.array(img1.convert('L').rotate(float(angle)))

        # Square section of image center with a diameter of pxrange
        cval = array_size_lim(img_loop,max_lim=True) #,set_radius=pxrange
        if angle == try_angles[0]: cval0 = cval
        # Lists for average values of rows/lines of cval
        hval_avg = np.mean(cval, axis=0) # Mean of y along x-Axis
        vval_avg = np.mean(cval, axis=1) # Mean of x along y-Axis

        # Find peaks in the average values
        peaks_row_avg, _ = find_peaks(running_average(hval_avg, win_size), height=0)
        peaks_col_avg, _ = find_peaks(running_average(vval_avg, win_size), height=0)

        # Get prominences of peaks
        hproms = peak_prominences(running_average(hval_avg, win_size), peaks_row_avg, wlen=wlen_fact*len(hval_avg))
        vproms = peak_prominences(running_average(vval_avg, win_size), peaks_col_avg, wlen=wlen_fact*len(vval_avg))

        # Calculate the prominence based index (icsp)
        icsp_index = np.abs(round((sum(hproms[0]) - sum(vproms[0])) / (sum(hproms[0]) + sum(vproms[0])), 3))

        # Append Values to their lists
        icsp_list.append(icsp_index)
        degree_list.append(round(angle, 1))

    rotation_list.append(degree_list)  # Rotation values
    rotation_list.append(icsp_list)  # Calculated icsp for each rotation value

    # Apply polynomial fit to an extended definition range to simulate a periodic polynomial
    # [I_CSP]
    x_pr = np.array(rotation_list[0])
    y_pr = np.array(rotation_list[1])
    x_pr_extended = np.concatenate((x_pr - 90, x_pr, x_pr + 90))
    y_pr_extended = np.concatenate((y_pr, y_pr, y_pr))
    polyfit_pr = np.polyfit(x_pr_extended, y_pr_extended, 15)  # deg: 10, 15 work well

    # Calculate 1st, 2nd derivative of the polynomial
    poly_pr = np.poly1d(polyfit_pr)
    deri1_pr = poly_pr.deriv()
    deri2_pr = deri1_pr.deriv()

    # Zero points of the 1st derivative
    crit_points_pr = deri1_pr.r

    # Definition range, goes beyond 90° for strange maxima
    deg_min, deg_max = 0, 95

    # Filter for real Maxima within the definition range
    valid_maxima = []
    for point in crit_points_pr:
        if np.isreal(point):
            point_real = np.real(point)
            if deg_min <= point_real <= deg_max and deri2_pr(point_real) < 0:
                valid_maxima.append(point_real)
    #deg_of_max = 0
    thI_CSP_max = 0
    # Select highest Maxima from all valid Maxima
    if valid_maxima:
        y_vals = poly_pr(valid_maxima)
        idx_max = np.argmax(y_vals)
        deg_of_max = valid_maxima[idx_max]
        thI_CSP_max = y_vals[idx_max]
        # print(f"Valid Maxima at x = {deg_of_max:.4f}, y = {thI_CSP_max:.4f}")
        if deg_of_max >= 90:
            deg_of_max -= 90
    else:
        print("Found no valid maxima. Angle set to 0°")
        deg_of_max = 0

    # Prepare Polynomial fit for plot
    x_pr_smooth = np.arange(0, 90, 1)
    polynom_pr = np.polyval(polyfit_pr, x_pr_smooth)

    ###################### PLOT FOR ANGLE DEPENDENCE ######################
    if show:
        # Plots icsp, Ics depending on image rotation
        fig, axs = plt.subplots(1, 2, figsize=(10, 4), gridspec_kw={'width_ratios': [4, 6]})
        fig.subplots_adjust(wspace=0.4)

        axs[0].set_aspect('equal')
        cont0 = axs[0].contourf(img1.rotate(deg_of_max), 200, cmap="Greys_r")

        divider = make_axes_locatable(axs[0])
        cax = divider.append_axes("right", size="5%", pad=0.05)
        tix = np.linspace(np.min(img1.rotate(deg_of_max)), np.max(img1.rotate(deg_of_max)), 5).astype(int)
        cb0 = plt.colorbar(cont0, cax=cax, ticks=tix)
        cb0.ax.get_yaxis().labelpad = 15
        cb0.ax.set_ylabel('Grey value', rotation=270)

        axs[1].plot(rotation_list[0], rotation_list[1], label='I_CSP', linewidth=1.5)
        axs[1].plot(x_pr_smooth, polynom_pr, label='Polynomfit I_CSP', linestyle='--')
        axs[1].plot(deg_of_max, thI_CSP_max, 'rx', markersize=10, label='Maximum')
        axs[1].set_xlabel('Rotation / °'), axs[0].set_ylabel('Index value')
        axs[1].legend(), axs[1].grid()

        plt.suptitle('Rotation method')
        plt.show()
    #######################################################################
    return round(float(thI_CSP_max),3), round(float(deg_of_max),1), rotation_list, x_pr_smooth, polynom_pr, cval0

# Function similar to angle_reader, but for correl_mat
# [Returns: thicsp_max, deg_of_max]
def angle_finder2(correl_mat, show=False):
    correl_mat = correl_mat[:, :, 0].T*255 #  multiply with 255, grayscale values are between 0 and 255, not 0 and 1
    cm_img = pimg.fromarray(correl_mat)

    acx, acy = [], []
    for angle in try_angles:
        img_loop = np.array(cm_img.rotate(float(angle)))
        img_loop = array_size_lim(img_loop, max_lim=True)

        # Calculating Average of Across/Along values:
        x_mean = img_loop.mean(axis=1)
        y_mean = img_loop.mean(axis=0)

        ac_pr_index = icsp_function(x_mean, y_mean)
        acy.append(ac_pr_index[0])
        acx.append(angle)

    # Polynom
    # Apply polynomial fit to an extended definition range to simulate a periodic polynomial
    acx, acy = np.array(acx), np.array(acy)
    acx_extended = np.concatenate((acx - 90, acx, acx + 90))
    acy_extended = np.concatenate((acy, acy, acy))
    polyfit_pr = np.polyfit(acx_extended, acy_extended, 15)  # deg: 10, 15 work well

    # Calculate 1st, 2nd derivative of the polynomial
    poly_pr = np.poly1d(polyfit_pr)
    deri1_pr = poly_pr.deriv()
    deri2_pr = deri1_pr.deriv()

    # Zero points of the 1st derivative
    crit_points_pr = deri1_pr.r

    # Definition range, goes beyond 90° for strange maxima
    deg_min, deg_max = 0, 95

    # Filter for real Maxima within the definition range
    valid_maxima = []
    for point in crit_points_pr:
        if np.isreal(point):
            point_real = np.real(point)
            if deg_min <= point_real <= deg_max and deri2_pr(point_real) < 0:
                valid_maxima.append(point_real)
    # deg_of_max = 0
    thicsp_max = 0
    # Select highest Maxima from all valid Maxima
    if valid_maxima:
        y_vals = poly_pr(valid_maxima)
        idx_max = np.argmax(y_vals)
        deg_of_max = valid_maxima[idx_max]
        thicsp_max = y_vals[idx_max]
        # print(f"Valid Maxima at x = {deg_of_max:.4f}, y = {thicsp_max:.4f}")
        if deg_of_max >= 90:
            deg_of_max -= 90
    else:
        print("Found no valid maxima. Angle set to 0°")
        deg_of_max = 0

    # Prepare Polynomial fit for plot
    acx_smooth = np.arange(0, 90, 1)
    polynom_pr = np.polyval(polyfit_pr, acx_smooth)

    # Plots icsp, Polynom depending on image rotation
    if show:
        # Plots icsp, Ics depending on image rotation
        fig, axs = plt.subplots(1, 2, figsize=(10, 4), gridspec_kw={'width_ratios': [4, 6]})
        fig.subplots_adjust(wspace=0.4)

        axs[0].set_aspect('equal')
        cont0 = axs[0].contourf(cm_img.rotate(deg_of_max), 200, cmap="Greys_r")

        divider = make_axes_locatable(axs[0])
        cax = divider.append_axes("right", size="5%", pad=0.05)
        tix = np.linspace(np.min(cm_img.rotate(deg_of_max)), np.max(cm_img.rotate(deg_of_max)), 5).astype(int)
        cb0 = plt.colorbar(cont0, cax=cax, ticks=tix)
        cb0.ax.get_yaxis().labelpad = 15
        cb0.ax.set_ylabel('Grey value', rotation=270)

        axs[1].plot(acx, acy, label='I_CSP', linewidth=1.5)
        axs[1].plot(acx_smooth, polynom_pr, label='Polynomfit I_CSP', linestyle='--')
        axs[1].plot(deg_of_max, thicsp_max, 'rx', markersize=10, label='Maximum')
        axs[1].set_xlabel('Rotation / °'), axs[0].set_ylabel('Index value')
        axs[1].legend(), axs[1].grid()

        plt.suptitle('angle_finder2')
        plt.show()
    # print(f'Angle after AC: {deg_of_max:.1f}°')
    return thicsp_max, deg_of_max

# Function to detect the angle of Cloud Street based on fast fourier transform (FFT)
# [Returns: angle_to_center, distance_to_center, [mitte_x, mitte_y, img_size, valid_maxima, pos_pkt, radius, magnitude]]
def angle_detector_fft(img_arr, max_n=1, show=0, radius=15):
    """:return: angle_to_center, distance_to_center, [mitte_x, mitte_y, img_size, valid_maxima, pos_pkt, radius, magnitude]"""
    # Function determines array size
    img_arr = array_size_lim(img_arr, max_lim=True)

    mean_tone = img_arr.mean()
    img_size = np.shape(img_arr)  # 0 - Höhe, 1 - Breite
    mitte_x, mitte_y = img_size[1] // 2, img_size[0] // 2 # weil 0 auch Index, NICHT +1

    # 2D Fourier Transform using numpy's fast fourier transform
    # Subtract mean value from img to remove zero frequency(?)
    img_arr = img_arr - round(mean_tone, 3)

    ft = np.fft.ifftshift(img_arr)
    ft = np.fft.fft2(ft)
    ft = np.fft.fftshift(ft)

    magnitude = abs(ft)
    magnitude_log = np.log(magnitude)

    # Analyse just half image width+1
    half_magni_log = magnitude_log[:, :mitte_x+1]

    # Ignore center values by setting them 0 (often strong signals, still zero frequency?)
    set_r = 1
    x_min = max(mitte_x - set_r, 0)
    x_max = min(mitte_x + set_r+1, half_magni_log.shape[1])
    y_min = max(mitte_y - set_r, 0)
    y_max = min(mitte_y + set_r+1, half_magni_log.shape[0])

    half_magni_log[y_min:y_max, x_min:x_max] = 0 # Set values around the center 0
    half_magni_log[y_min+set_r:, mitte_x] = 0 # Set values in the center column below center 0

    # Get n max values with their position in the array
    n_max = max_n
    flat_img = half_magni_log.flatten()

    # Find indices of the max values
    indices = np.argpartition(flat_img, -n_max)[-n_max:]
    indices = np.flip(indices[np.argsort(flat_img[indices])])

    # Transform flattened array/indices back to 2d array
    row_indices, col_indices = np.unravel_index(indices, half_magni_log.shape)  # !

    # Plot maxima as scatter plot, filter center 3x3 pixels when shape is odd
    valid_maxima = []
    for j, k in zip(row_indices, col_indices):
        valid_maxima.append((k, j))

    valid_maxima = np.array(valid_maxima)  # Coords of valid maxima
    
    # Distance, angle, strength of maxima (if n>1) to mitigate spread of the maxima
    skip = 0
    if len(valid_maxima) > 1:
        pos_pkt = (valid_maxima[:, 0].mean(), valid_maxima[:, 1].mean())
    elif len(valid_maxima) == 1:
        pos_pkt = (valid_maxima[0, 0], valid_maxima[0, 1])
    else:
        print('no valid Maxima found.')
        skip = 1

    if skip == 0:
        dist_to_center = ((pos_pkt[0] - mitte_x) ** 2 + (pos_pkt[1] - mitte_y) ** 2) ** 0.5
        angle_to_center = -(180 / np.pi) * np.arctan2(pos_pkt[1] - mitte_y, pos_pkt[0] - mitte_x)

        if -180 <= angle_to_center < -90:
            angle_to_center += 180
        elif 180 >= angle_to_center > 90:
            angle_to_center -= 180

    #################### PLOTTING SECTION #############################
    if show == 1:
        plt.hlines(mitte_y,xmin=0,xmax=img_size[1]//2,color='magenta',linestyle=':')
        plt.plot([valid_maxima[:, 0].mean(), mitte_x], [valid_maxima[:, 1].mean(), mitte_y], color='cyan')

        # Plot Maxima
        #plt.scatter(valid_maxima[:, 0], valid_maxima[:, 1], color='red', alpha=1, marker='x', s=50)
        # plt.scatter(col_indices,row_indices, color='red', label='Maxima')
        plt.scatter(pos_pkt[0], pos_pkt[1], color='C1', label='Punkt')

        # Plot Spectrum of the FFT
        testplot = plt.imshow(magnitude, cmap='grey')
        plt.colorbar(testplot)
        plt.xlim([img_size[1] // 2 - radius, img_size[1] // 2 + radius])
        plt.ylim([img_size[0] // 2 - radius, img_size[0] // 2 + radius])
        plt.title(f'Angle detection with Fast Fourier Transform\nν: {dist_to_center:.2f}, α: {angle_to_center:.2f}°')
        plt.legend()
        plt.show()

    return (angle_to_center, dist_to_center, [mitte_x, mitte_y, img_size, valid_maxima, pos_pkt, radius, magnitude])

# Calculates 2D-Autocorrelation
# [Returns: image_array, correl_mat**2, angle]
def correl_images(image_A, correl_range, numpix=False, monitor=False, size=1, turn=False):
    """:return: image_array, correl_mat**2, angle"""
    # older paramters of the function
    x_offset, y_offset, reduction, magnification = 0, 0, 1, 1
    x_shift, y_shift = correl_range // 2, correl_range // 2

    start = datetime.now()

    # Optional: detect angle of cloud structure using rotation methode (fft method better)
    if turn:
        angle = round(angle_finder(image_A)[1], 0)
        print(f'Angle: {angle}°')
    else: angle = 0

    image_array = pimg.fromarray(image_A).convert('L').rotate(angle)
    # Obtain image size and try to apply size-parameter
    img_size = image_array.size
    if size == 'auto':
        size_fact = correl_range / min(img_size) + 0.05
        image_array = np.array(image_array.resize((int(size_fact * img_size[0]), int(size_fact * img_size[1])), 0))
        print(f'Adapting image size: {img_size} → {int(size_fact * img_size[0]), int(size_fact * img_size[1])}')
    else:
        if correl_range <= size * min(img_size):
            image_array = np.array(image_array.resize((int(size * img_size[0]), int(size * img_size[1])), 0))
            print(f'Image size:{img_size} → {int(size * img_size[0]), int(size * img_size[1])}')
        else:
            print(f'Original Image size of {img_size} is kept. Otherwise it would be too small!')
            image_array = np.array(image_array)

    # Selected area using pxrange
    x_ll = int(image_array.shape[1] // 2 - 0.5 * correl_range)
    x_ul = int(image_array.shape[1] // 2 + 0.5 * correl_range)
    y_ll = int(image_array.shape[0] // 2 - 0.5 * correl_range)
    y_ul = int(image_array.shape[0] // 2 + 0.5 * correl_range)

    image_A = image_array[y_ll:y_ul, x_ll:x_ul]  # Quasi wie cval
    image_B = image_A

    # Apply reduction factor
    if reduction > 1:
        shape = (image_A.shape[0] // reduction, image_A.shape[1] // reduction)
        image_A = image_A[:shape[0] * reduction, :shape[1] * reduction].reshape(shape[0], reduction, shape[1],
                                                                                reduction).mean(axis=(1, 3))
        image_B = image_B[:shape[0] * reduction, :shape[1] * reduction].reshape(shape[0], reduction, shape[1],
                                                                                reduction).mean(axis=(1, 3))

    # Apply magnification factor
    if magnification > 1:
        from scipy.ndimage import zoom
        image_A = zoom(image_A, magnification)
        image_B = zoom(image_B, magnification)
        x_offset = round(x_offset * magnification)
        y_offset = round(y_offset * magnification)
        x_shift = magnification + 2
        y_shift = magnification + 2

    # Dimensions of the correlation matrix
    Nx = 2 * x_shift + 1
    Ny = 2 * y_shift + 1
    Nim = 2 if numpix else 1

    correl_mat = np.zeros((Nx, Ny, Nim), dtype=np.float32)

    xs = round(x_offset) - x_shift
    ys = round(y_offset) - y_shift

    sAx, sAy = image_A.shape[1] - 1, image_A.shape[0] - 1
    sBx, sBy = image_B.shape[1] - 1, image_B.shape[0] - 1

    print(Ny)
    for y in range(Ny):
        yoff = ys + y
        yAmin = max(yoff, 0)
        yAmax = min(sAy, sBy + yoff)
        yBmin = max(-yoff, 0)
        yBmax = min(sBy, sAy - yoff)

        if yAmax > yAmin:
            for x in range(Nx):
                xoff = xs + x
                xAmin = max(xoff, 0)
                xAmax = min(sAx, sBx + xoff)
                xBmin = max(-xoff, 0)
                xBmax = min(sBx, sAx - xoff)

                if xAmax > xAmin:
                    im_ov_A = image_A[yAmin:yAmax + 1, xAmin:xAmax + 1]
                    im_ov_B = image_B[yBmin:yBmax + 1, xBmin:xBmax + 1]
                    Npix = im_ov_A.size

                    if im_ov_B.size != Npix:
                        raise ValueError("Overlap error: # pixels mismatch")

                    im_ov_A = im_ov_A - im_ov_A.mean()
                    im_ov_B = im_ov_B - im_ov_B.mean()
                    totAA = np.sum(im_ov_A ** 2)
                    totBB = np.sum(im_ov_B ** 2)

                    if totAA == 0 or totBB == 0:
                        correl_mat[x, y, 0] = 0.0
                    else:
                        correl_mat[x, y, 0] = np.sum(im_ov_A * im_ov_B) / np.sqrt(totAA * totBB)

                    if numpix:
                        correl_mat[x, y, 1] = Npix
        # print out progress
        if monitor and (Ny - y) % 10 == 0:
            print(f"Progress: {Ny - y}")

    if monitor:
        print(f"Correlation computation completed for angle {angle}°.")

    #print(f"Calc time correl_mat: {datetime.now() - start}")
    return image_A, correl_mat**2, angle

# Function to read and process correl_mat
# [Returns: icsp_values[0], Autocor_x_mean, Autocor_y_mean]
def read_correl_mat(correl_mat, show=False, im_arr=False, angle=0, save=False, dpi=100):
    if type(im_arr) != np.ndarray:
        print('no image array given.')

    till = len(correl_mat)
    # Calculating Average of Across/Along values:
    Autocor_x_mean = list(correl_mat.mean(axis=1)[:, 0])
    Autocor_y_mean = list(correl_mat.mean(axis=0)[:, 0])
    
    # no additional smoothing required!
    # Generally, calculating the I_CS,P has shown to produce frequently wrong values
    # This is especially the case for strato-cumulus, masking the auto-correlation-peak doesn't solve this!
    icsp_values = icsp_function(Autocor_x_mean, Autocor_y_mean, ravg=False)

    if show:
        fs = 18
        if type(im_arr) == np.ndarray:
            fig, axs = plt.subplots(1, 3, figsize=(16, 4))  # , gridspec_kw={'width_ratios': [4, 4, 3]})
            fig.subplots_adjust(wspace=0.45)

            # Enumerate plots
            labels = ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)']
            for idx, ax in enumerate(axs.flat):
                ax.text(-0.2, 1.1, labels[idx], transform=ax.transAxes,
                        fontsize=fs, fontfamily='serif', va='top', ha='left')

            # plot selected area of image
            axs[0].set_aspect('equal')
            cont0 = axs[0].contourf(im_arr, 200, cmap="Greys_r")
            axs[0].set_xlabel("Across-track pixel", fontsize=fs)
            axs[0].set_ylabel("Along-track pixel", fontsize=fs)
            if angle: axs[0].set_title(f"Analysed Image, Rotation: {angle}°", fontsize=fs+2)
            else:  axs[0].set_title(f"Analysed Image", fontsize=fs)

            divider = make_axes_locatable(axs[0])
            cax = divider.append_axes("right", size="5%", pad=0.05)
            tix = np.linspace(np.min(im_arr), np.max(im_arr), 5).astype(int)
            cb0 = plt.colorbar(cont0, cax=cax, ticks=tix)
            cb0.ax.get_yaxis().labelpad = 15
            cb0.ax.set_ylabel('Grey value', rotation=270, fontsize=fs)

            # plot 2D Plot of the Autocorrelation values
            axs[1].set_aspect('equal')
            axs[1].set_title("correlation matrix", fontsize=fs)
            axs[1].set_xlabel("Across-track pixel", fontsize=fs)
            axs[1].set_ylabel("Along-track pixel", fontsize=fs)
            cmap = matplotlib.colors.LinearSegmentedColormap.from_list("",["k", "c", "b", "violet", "r"])
            data = correl_mat[:, :, 0].T
            cont1 = axs[1].contourf(np.arange(0, till), np.arange(0, till), data, 200, cmap=cmap,vmin=0,vmax=1)
            # marks contours
            #axs[1].contour(np.arange(0,till),np.arange(0,till),data,[1/np.exp(2)],color="yellow",linestyle=":")
            divider = make_axes_locatable(axs[1])
            cax = divider.append_axes("right", size="5%", pad=0.05)
            tix = np.linspace(np.min(data), np.max(data), 5).round(2)
            cb1 = plt.colorbar(cont1, cax=cax, ticks=tix)
            cb1.ax.get_yaxis().labelpad = fs+4
            #cb1.ax.set_ylabel('Squared correlation', rotation=270, fontsize=fs)
            cb1.ax.set_ylabel(r'$F(L_\text{x},L_\text{y})^2$', rotation=270, fontsize=fs)

            # Plot Values across and along
            axs[2].set_aspect('auto', anchor='W')
            # axs[2].set_aspect(1000)
            axs[2].plot(np.arange(0, till), Autocor_x_mean, label='Across mean')
            axs[2].plot(np.arange(0, till), Autocor_y_mean, label='Along mean')
            axs[2].set_xlabel("Pixel of axis", fontsize=fs)
            axs[2].set_ylabel("Squared correlation", fontsize=fs)
            axs[2].set_title(r"$I_\text{CS,PAC}$ = "+ f"{icsp_values[0]}", fontsize=fs)
            axs[2].spines[['right', 'top']].set_visible(False)
            axs[2].legend()
        else:
            fig, axs = plt.subplots(1, 2, figsize=(12, 8))
            #fig.subplots_adjust(wspace=0.2)
            fig.tight_layout()

            # plot 2D Plot of the Autocorrelation values
            axs[0].set_aspect('equal')
            cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", ["black", "cyan", "blue", "violet", "red"])
            cb1 = axs[0].contourf(np.arange(0, till), np.arange(0, till), correl_mat[:, :, 0].T, 200, cmap=cmap)
            plt.colorbar(cb1)
            axs[0].set_title('2D Auto correlation')

            # Plot Values across
            axs[1].set_aspect('auto', anchor='W')
            # axs[2].set_aspect(1000)
            axs[1].plot(np.arange(0, till), Autocor_x_mean, label='Across mean')
            axs[1].plot(np.arange(0, till), Autocor_y_mean, label='Along mean')
            axs[1].set_title(f"Across and Along for I_CSP = {icsp_values[0]}")
            axs[1].legend()
        # Save diagramm
        if save: plt.savefig(save, dpi=dpi, bbox_inches='tight', format="png")
        plt.show()
    return icsp_values[0], Autocor_x_mean, Autocor_y_mean

# [*] FUNCTIONS FOR SHOWING THE INDEX ON MAP --------------------------------------------------------------------------
# Function to reproject Nasa-worldview-tif
# [Return: data, extent]
def read_worldview_reprojected(geotiff):
    img = gdal.Open(geotiff)
    data = img.ReadAsArray()
    # make background from black to white
    ix = data.sum(axis=0) == 0
    data[..., ix] = 255
    # get extent
    gt = img.GetGeoTransform()
    extent = (gt[0], gt[0] + img.RasterXSize * gt[1],
              gt[3] + img.RasterYSize * gt[5], gt[3])
    data = data.transpose((1, 2, 0))
    return data, extent

# Function to plot satellite image, flight track, local indices
def show_map(nc_file, geotiff, folder_name, timestamps, vals):
    lons, lats, thicspmax, final_list = [], [], [], []
    gps_ins = xr.open_dataset(nc_file)
    img, extent = read_worldview_reprojected(geotiff)

    # Arrays with picture timestamps and gps timestamps, both converted to pd.to_datetime
    val_times = pd.to_datetime(np.array(timestamps))
    gps_times = pd.to_datetime(np.array(gps_ins['time']))
    # Finds matching values and position
    matching = np.intersect1d(val_times, gps_times, return_indices=True)
    final_list.append(matching[0])  # Adds time values

    # Get coords of matching timestamps
    for k in matching[2]:
        lons.append(round(float(gps_ins['lon'][k]), 8))
        lats.append(round(float(gps_ins['lat'][k]), 8))

    for i in matching[1]:  # uses indices of matches as mask for thicspmax-values
        thicspmax.append(np.array(vals)[i])

    final_list.append(np.array(lons))  # Adds longitude
    final_list.append(np.array(lats))  # Adds lattitude
    final_list.append(thicspmax)  # Adds thicspmax
    print(np.array(final_list[3]))

    print(len(matching[0]), len(lons), len(lats), len(vals))
    # rare case: no matching time stamp -> thicspmax will have more values (see: 04.04.2022, Pass1, 11:08:16)

    plt.figure(figsize=(6, 6))
    ax = plt.axes(projection=ccrs.NorthPolarStereo(central_longitude=10))
    # select map area by coords
    ax.set_extent([-5, 15, 77.5, 81.5], crs=ccrs.PlateCarree())
    #ax.set_extent([0, 10, 78, 80], crs=ccrs.PlateCarree())
    #ax.set_extent([0.5, 6, 78.6, 79.9], crs=ccrs.PlateCarree())
    #ax.set_extent([1.5, 4.5, 78.8, 79.4], crs=ccrs.PlateCarree()) # for Pass2

    ax.coastlines(resolution='10m')
    ax.imshow(img, extent=extent, origin='upper')  # plot satelite imagery

    ax.plot(gps_ins['lon'], gps_ins['lat'], label='flight track', transform=ccrs.PlateCarree())  # Flight track
    ax.plot(gps_ins['lon'], gps_ins['lat'], label='flight track', transform=ccrs.PlateCarree(),
        color='C0',path_effects=[pe.Stroke(linewidth=3, foreground='k'), pe.Normal()])  # Flight track

    dots = ax.scatter(final_list[1], final_list[2], c=final_list[3], label=r'$I_{\mathrm{CS,P}}$ after AC',
                      cmap='plasma', transform=ccrs.PlateCarree(), zorder=5,
                      vmin=0, vmax=1, s=50)  # Indices
    cbar = plt.colorbar(mappable=dots, cax=None, ax=None, fraction=0.046, pad=0.04)  # plots colormap for index values
    cbar.ax.get_xaxis().labelpad = 15
    cbar.ax.set_title('Index', rotation=0)

    gl = ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False)
    gl.xlabel_style = {'rotation': 45}
    gl.top_labels, gl.right_labels = False, False
    plt.title(f"Index along flight track of {folder_name}", fontsize=14)
    plt.xlabel("Longitude"), plt.ylabel("Latitude")
    plt.legend(fontsize=12)
    plt.tight_layout(w_pad=0.2,h_pad=0.2)
    plt.show()

# [*] FUNCTIONS FOR FOLDER STUFF --------------------------------------------------------------------------------------
# Function to iterate through folder with path
# [Returns: timestamps, data_array, folder_name, thI_CSP_array, ics0_array]
def folder_analyzer(path, norm=False):
    data_list, thicsp_list, ics0_list, ics0n_list, icsp0_list, file_list = [], [], [], [], [], []
    for file in os.listdir(path):  # Reads all Files in given folder
        if file.endswith(".jpg"):  # Filters for .jpg-Files
            # Appends filename + path to a list
            file_list.append(os.path.join(path, file))
    # Sorts list alphanumerically (timestamps in files get sorted properly)
    file_list.sort()

    # Runs desired function on ever file put into the file_list
    for file in file_list:
        dreh = 0#np.random.uniform(low=-90, high=90, size=1).round()[0]
        # Prints out folder progress
        print(f'[ {file_list.index(file) + 1} / {len(file_list)} ]')
        # Appends returned values to data_list
        main_out = correl_images(image_reader(file, dreh=dreh), correl_range, monitor=False, size=0.5)
        results = read_correl_mat(main_out[1],show=False,im_arr=main_out[0], angle=main_out[2])
        data_list.append(results[0])

        ics0_list.append(ics0_function(file)) # Data for I_cs at 0°
        ics0n_list.append(ics0new_function(file)) # Modified I_cs

        other = multi_index_function(file, pxrange, dreh=dreh)
        thicsp_list.append(other[2]) # Data for I_cs,p before auto correlation
        icsp0_list.append(other[4]) # I_CS,P at 0° (to compare with I_CS)

    # takes every file's string
    for filestring in file_list:
        splitting = filestring.split('/')  # Gets folder name
        splitting2 = splitting[-1].split('.')  # Extracts time stamp
        splitting3 = splitting2[-3].split('_')
        # takes last two elements for timestamp, so corr_ gets filtered out if present
        add = datetime.strptime(splitting3[-2]+splitting3[-1], '%Y%m%d%H%M%S')
        timestamps.append(add)  # Appends datetime object to list

    ########################## PLOTS TIMELINE OF thI_CSPmax ##########################
    runavg = 1  # Applies a Running Average (smoothens functions over time)

    splitting = path.split('/') # Gets folder name
    folder_name = splitting[-1]
    data_array = np.array(data_list)
    ics0_array = np.array(ics0_list)
    thicsp_array = np.array(thicsp_list)
    icsp0_array = np.array(icsp0_list)
    ics0n_array = np.array(ics0n_list)

    if len(file_list) >= 20:
        win_size = 10
    else:
        win_size = len(file_list) // 10 + 1  # Selects window size depending on number of files

    plt.figure(figsize=(10, 4))
    if runavg == 0:
        x = range(0, len(data_list))
        plt.scatter(x, data_array, label='KorrelI_CSP', marker='D')
        plt.title(f'Indices Development of {folder_name}, raw data')
        plt.ylim(0, 1)

    # Plots indices if Running Average is ENABLED
    else:
        # Adapts interval for running average depending on win_size
        if win_size % 2 == 1:
            # x: simply indices, times: time for used image
            x = np.arange((win_size // 2), len(data_list) - (win_size // 2))
            times = timestamps[(win_size//2):(len(data_list) - win_size//2)]
        else:
            x = np.arange((win_size // 2), len(data_list) - (win_size // 2 - 1))
            times = np.array(timestamps[(win_size//2)-1:(len(data_list) - win_size//2)])

        data0 = running_average(data_array, win_size) # I_CS,P + 2D auto correlation
        data1 = running_average(ics0_array, win_size) # I_CS
        data2 = running_average(thicsp_array, win_size) # I_CS,P
        data3 = running_average(icsp0_array, win_size) # I_CS,P0
        data4 = running_average(ics0n_array, win_size) # I_CS,M

        if norm:
            n0,n1,n2,n3,n4 = max(data0),max(data1),max(data2),max(data3),max(data4)
            data0 = data0 / n0
            data1 = data1 / n1
            data2 = data2 / n2
            data3 = data3 / n3
            data4 = data4 / n4
        print('win size:', win_size, '|Länge x:',len(x), '|Länge timestamps:', len(times), '|Länge data0:', len(data0))

        # Für Zeit als x-Achse: 'times', Ansonsten 'x'
        plt.scatter(times, data1, label=r'$I_{\mathrm{CS}}$', marker='o')
        plt.plot(times, data1, alpha=0.3)
        plt.scatter(times, data2, label=r'$I_{\mathrm{CS,P}}$', marker='x')
        plt.plot(times, data2, alpha=0.3)
        plt.scatter(times, data0, label=r'$I_{\mathrm{CS,P}}$ after AC', marker='D')
        plt.plot(times, data0, alpha=0.3)
        plt.scatter(times, data3, label=r'$I_{\mathrm{CS,P0}}$', marker='o')
        plt.plot(times, data3, alpha=0.3)
        plt.scatter(times, data4, label=r'$I_{\mathrm{CS,Neu}}$', marker='o')
        plt.plot(times, data4, alpha=0.3)
        if norm:
            plt.title(f'Normalized Indices time series of {folder_name} | window: {win_size}')
        else:
            plt.title(f'Indices time series of {folder_name} | window: {win_size}')

    plt.ylabel('Index value'), plt.xlabel('Time | HH:MM')
    plt.grid(zorder=1, which='major', linestyle='-')
    plt.grid(zorder=1, which='minor', linestyle='--')
    ax = plt.gca()
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    ax.xaxis.set_minor_locator(mdates.MinuteLocator(interval=1))
    plt.legend()
    plt.ylim(0,1)
    plt.tight_layout(w_pad=0.2, h_pad=0.2)
    plt.show()

    return timestamps, data_array, folder_name, thicsp_array, ics0_array

# Function that outputs lists of files in folder
# [Returns: file_list]
def file_lister(folderpath, fformat='.jpg'):
    file_list = []
    for file in os.listdir(folderpath):  # Reads all Files in given folder
        if file.endswith(fformat):  # Filters for .jpg-Files
            # Appends filename + path to a list
            file_list.append(os.path.join(folderpath, file))
    # Sorts list alphanumerically
    file_list.sort()
    return file_list

# Function to output file name, not universal
# [Returns: file_name, file_w_ending]
def filename_out(file):
    filestring = file
    filename0 = filestring.split('/')  # Gets file name + ending
    filename1 = filename0[-1].split('.') # Remove file ending
    file_name = filename1[-3]+'.'+filename1[-2] # clean file name without ending
    file_w_ending = filename0[-1]
    return file_name, file_w_ending

# [!] FUNKTIONEN SPEZIFISCH FÜR ABBILDUNGEN ===========================================================================

# Function to show the fly route of the campaign
# Function to plot satellite image, flight track, local indices
def show_flight_path(nc_file, geotiff, show=False, dpi=100):
    lons, lats, thicspmax, final_list = [], [], [], []
    gps_ins = xr.open_dataset(nc_file)
    img, extent = read_worldview_reprojected(geotiff)

    # Arrays with picture timestamps and gps timestamps, both converted to pd.to_datetime
    gps_times = pd.to_datetime(np.array(gps_ins['time']))
    matching = np.intersect1d(gps_times, gps_times, return_indices=True)
    final_list.append(matching)  # Adds time values

    # Get coords of matching timestamps
    for k in matching[2]:
        lons.append(round(float(gps_ins['lon'][k]), 8))
        lats.append(round(float(gps_ins['lat'][k]), 8))

    final_list.append(np.array(lons))  # Adds longitude
    final_list.append(np.array(lats))  # Adds lattitude

    print(len(matching[0]), len(lons), len(lats))

    plt.figure(figsize=(6, 6))
    ax = plt.axes(projection=ccrs.NorthPolarStereo(central_longitude=10))
    # Kartenausschnitt wählen
    ax.set_extent([-2, 18, 77.5, 80], crs=ccrs.PlateCarree())
    # Zum Testen kleiner Bereich
    #ax.set_extent([0, 5, 78, 79], crs=ccrs.PlateCarree())

    ax.coastlines(resolution='10m')
    ax.imshow(img, extent=extent, origin='upper', zorder=1)  # plot satelite imagery
    # Flugroute darstellen
    ax.plot(gps_ins['lon'], gps_ins['lat'], label='Polar 5 flight track', transform=ccrs.PlateCarree(),
        color='yellow',path_effects=[pe.Stroke(linewidth=3, foreground='k'), pe.Normal()], zorder=2)  # Flight track

    gl = ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False, zorder=3)
    gl.xlabel_style = {'fontsize': 8, 'rotation': 30} # einst 45
    gl.ylabel_style = {'fontsize': 8}
    gl.top_labels, gl.right_labels = False, False
    plt.scatter(1.263*10**5,-1.311*10**6,label='Waypoint 0', color='red', marker='X', zorder=4)
    plt.scatter(-6.673*10**4,-1.25*10**6,label='Waypoint 1', color='deeppink', marker='D', zorder=4)
    plt.scatter(-2.113*10**5,-1.177*10**6,label='Waypoint 2', color='deeppink', marker='square', zorder=4)
    plt.xlabel("Longitude"), plt.ylabel("Latitude")
    plt.legend(fontsize=8)

    # Grafik der Flugroute speichern
    save_to = '/folder_to_save/out.pdf'
    plt.savefig(save_to, dpi=dpi, bbox_inches='tight', format="pdf")

    # Grafik anzeigen
    if show:
        plt.tight_layout(w_pad=0.2,h_pad=0.2)
        plt.show()

# Function that outputs the different derived angles: Pure Polynom, Polynom after AC
# [Returns: init_angle, angle1, index1, angle2, index2]
def angle_comparer(imglocation,dreh=False,winkel=False):
    if not winkel:
        if dreh: dreh = np.random.uniform(low=0, high=90, size=1).round()[0]
        else: dreh = 0
    else:
        dreh = winkel

    img = image_reader(imglocation, dreh)
    ang1 = angle_finder(img, show=False)
    angle1, index1 = round(ang1[1],1), round(ang1[0],3)

    correl_mat = correl_images(img,150, monitor=False, size='auto', turn=False)[1]
    ang2 = angle_finder2(correl_mat, show=False)
    angle2, index2 = round(ang2[1],1), round(ang2[0],3)

    ang3 = angle_detector_fft(img, 1, show=False)
    angle3 = round(ang3[0],1)

    ang4 = angle_detector_fft(correl_mat[:,:,0].T, 1, show=False)
    angle4 = round(ang4[0],1)

    return dreh, angle1, index1, angle2, index2, angle3, angle4

# FUNKTIONEN FÜR DEN ERGEBNISTEIL:
# Vergleich der Indize für Überflüge
# [Returns: timestamps, ics0_array, icsp0_array, icsp00_array, folder_name]
def vergleich_indize(path, norm=False, save_to=False, dpi=100, th_CS=False, th_CSP=False):
    file_list = []
    start = datetime.now()  # starts measuring code running time
    for file in os.listdir(path):  # Reads all Files in given folder
        if file.endswith(".jpg"):  # Filters for .jpg-Files
            # Appends filename + path to a list
            file_list.append(os.path.join(path, file))
    # Sorts list alphanumerically (timestamps in files get sorted properly)
    file_list.sort()

    # Führe Funktionen pro Datei aus und speicher Werte in den Listen
    ics0, icsp0, icsp00, icspth, angler, ics0n = [], [], [], [], [], []
    for file in file_list:
        dreh = 0#np.random.uniform(low=-90, high=90, size=1).round()[0]
        # Prints out progress
        print(f'[ {file_list.index(file) + 1} / {len(file_list)} ]')
        # Funktionen, die ausgeführt werden sollen
        # I_CS - Index nach Klingebiel et al. 2025
        ics0.append(ics0_function(file))
        # I_CS,P0 - Prominenz-basierter Index
        icsp0.append(multi_index_function(file, pxrange=400, dreh=dreh)[4])
        # I_CS,New - modifiziert I_CS
        ics0n.append(ics0new_function(file))

    # Informationen aus dem Dateinamen entnehmen
    for filestring in file_list:
        splitting = filestring.split('/')  # Gets folder name
        splitting2 = splitting[-1].split('.')  # Extracts time stamp
        splitting3 = splitting2[-3].split('_')
        # takes last two elements for timestamp, so corr_ gets filtered out if present
        add = datetime.strptime(splitting3[-2]+splitting3[-1], '%Y%m%d%H%M%S')
        timestamps.append(add)  # Appends datetime object to list
    splitting = path.split('/') # Gets folder name
    folder_name = splitting[-1]

    print(f'Code running time: {datetime.now() - start}')  # outputs code running time

    ########################## PLOTS TIMELINE OF thI_CSPmax ##########################
    runavg = 1  # Applies a Running Average (smoothens functions over time)

    ics0_array = np.array(ics0)
    icsp0_array = np.array(icsp0)
    ics0n_array = np.array(ics0n)

    # Passende Fenstergröße für den Running Average finden
    if len(file_list) >= 20:
        win_size = 10
    else:
        win_size = len(file_list) // 10 + 1  # Selects window size for Running Average depending on number of files

    plt.figure(figsize=(12, 4))
    if runavg == 0:
        x = range(0, len(ics0))
        plt.scatter(x, ics0, label='KorrelI_CSP', marker='D')
        plt.title(f'Indices Development of {folder_name}, raw data')
        plt.ylim(0, 1)

    # Plots indices if Running Average is ENABLED
    else:
        # Adapts interval for running average depending on win_size
        if win_size % 2 == 1:
            #x = np.arange((win_size // 2), len(ics0) - (win_size // 2))
            times = timestamps[(win_size//2):(len(ics0) - win_size//2)] # Diskrepanzen zw. x und times
        else:
            #x = np.arange((win_size // 2), len(ics0) - (win_size // 2 - 1))
            times = np.array(timestamps[(win_size//2)-1:(len(ics0) - win_size//2)])

        data0 = running_average(ics0_array, win_size)
        data1 = running_average(icsp0_array, win_size)
        data2 = running_average(ics0n_array, win_size)
        th_ics, th_icsp = 0.2, 0.4
        # Index normieren, um Sensitivität zu vergleichen
        if norm:
            n0 = max(data0)
            n1 = max(data1)
            n2 = max(data2)
            data0, th_ics = data0 / n0, th_ics / n0
            data1, th_icsp = data1 / n1, th_icsp / n1
            data2 = data2 / n2

        # Für Zeit als x-Achse: 'times', Ansonsten 'x'
        fs = 14
        plt.tick_params(labelsize=fs)
        plt.scatter(times, data0, label=r'$I_{\mathrm{CS}}$', marker='o')
        plt.plot(times, data0, alpha=0.3)
        plt.scatter(times, data1, label=r'$I_{\mathrm{CS,P}}$', marker='D')
        plt.plot(times, data1, alpha=0.3)
        #plt.scatter(times, data2, label=r'$I_{\mathrm{CS,M}}$', marker='x')
        #plt.plot(times, data2, alpha=0.3)

        if th_CS: plt.axhline(th_ics, color='gray', linestyle=':', label=r'$I_{\mathrm{CS}}$ threshold')
        if th_CSP: plt.axhline(th_icsp, color='gray', linestyle='--', label=r'$I_{\mathrm{CS,P}}$ threshold')

    plt.ylabel('Index value', fontsize=fs), plt.xlabel('Time (UTC) | HH:MM', fontsize=fs)
    plt.grid(zorder=1, which='major', linestyle='-')
    plt.grid(zorder=1, which='minor', linestyle='--')
    ax = plt.gca()
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    ax.xaxis.set_minor_locator(mdates.MinuteLocator(interval=1))
    ax.spines[['right', 'top']].set_visible(False)
    plt.legend(fontsize=fs)
    plt.ylim(ymax=1, ymin=-0.2) # -0.2 für negative Werte

    if save_to:
        plt.savefig(save_to, dpi=dpi, bbox_inches='tight', format="png")

    plt.tight_layout(w_pad=0.2, h_pad=0.2)
    plt.show()

    return timestamps, ics0_array, icsp0_array, folder_name

# Visualisierung der Winkelmethoden
def angle_methods(imglocation):
    img_arr = image_reader(imglocation, dreh=0)
    #[Returns: thI_CSP_max, deg_of_max, rotation_list, x_pr_smooth, polynom_pr, cval0]
    dat1 = angle_finder(img_arr, dreh=0)
    #[Returns: angle_to_center, distance_to_center, [mitte_x, mitte_y, img_size, valid_maxima, pos_pkt, radius, magnitude]]
    dat2 = angle_detector_fft(img_arr, show=0)

    # Plot erstellen, axs[0]: Bild, axs[1]: Angle dep., axs[2]: FFT
    fs = 12
    title = r'Angle dependance of $\mathrm{I_{CS,P}}$'
    fig, axs = plt.subplots(1, 3, figsize=(16, 4), gridspec_kw={'width_ratios': [4, 4, 4]})
    fig.subplots_adjust(wspace=0.4)
    #fig.tight_layout(pad=2)

    # Plot analysed image
    axs[0].set_aspect('equal')
    cont0 = axs[0].contourf(dat1[5], 200, cmap="Greys_r")
    axs[0].set_xlabel("Across-track pixel", fontsize=fs)
    axs[0].set_ylabel("Along-track pixel", fontsize=fs)

    divider = make_axes_locatable(axs[0])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cb0 = plt.colorbar(cont0, cax=cax)
    # cb1.ax.get_yaxis().set_ticks([]) # ticks of the colorbar
    cb0.ax.get_yaxis().labelpad = 15
    cb0.ax.set_ylabel('Grey value', rotation=270, fontsize=fs)

    #plt.colorbar(cb0)
    axs[0].set_title(f"Analysed image", fontsize=fs + 2)

    # Plot angle dependance
    axs[1].set_aspect('auto', anchor='W')
    axs[1].plot(dat1[2][0], dat1[2][1], label=r'$\mathrm{I_{CS,P}}$')
    # axs[1].plot(dat1[2][0], dat1[2][2], label=r'$\mathrm{I_{CS}}$', linestyle='-.')
    axs[1].plot(dat1[3], dat1[4], label=r'Polynom of $\mathrm{I_{CS,P}}$', linestyle='--', linewidth=2)
    axs[1].plot(dat1[1], dat1[0], 'rx', markersize=10, label='Maximum')
    axs[1].set_title(title, fontsize=fs + 2)
    axs[1].set_xlabel('Rotation | °', fontsize=fs)
    axs[1].set_ylabel(r'Index value', fontsize=fs)
    axs[1].spines[['right', 'top']].set_visible(False)  # Soll laut Marcus überall eingebaut werden
    axs[1].legend()

    axs[2].hlines(dat2[2][1], xmin=0, xmax=dat2[2][2][1] // 2, color='magenta', linestyle=':')
    axs[2].plot([dat2[2][3][:, 0].mean(), dat2[2][0]], [dat2[2][3][:, 1].mean(), dat2[2][1]], color='cyan')

    # Plotte Maxima
    #axs[2].scatter(valid_maxima[:, 0], valid_maxima[:, 1], color='red', alpha=1, marker='x', s=50)
    axs[2].scatter(dat2[2][4][0], dat2[2][4][1], color='C1', label='Frequency peak')

    # Plotte Fourier Transformation
    testplot = axs[2].imshow(dat2[2][6], cmap='grey')

    divider = make_axes_locatable(axs[2])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cb1 = plt.colorbar(testplot, cax=cax)
    #cb1.ax.get_yaxis().set_ticks([]) # ticks of the colorbar
    cb1.ax.get_yaxis().labelpad = 15
    cb1.ax.set_ylabel('Fourier amplitude', rotation=270, fontsize=fs)

    axs[2].set_xlim([dat2[2][2][1] // 2 - dat2[2][5], dat2[2][2][1] // 2 + dat2[2][5]])
    axs[2].set_ylim([dat2[2][2][0] // 2 - dat2[2][5], dat2[2][2][0] // 2 + dat2[2][5]])
    axs[2].set_xlabel('Pixel', fontsize=fs), axs[2].set_ylabel('Pixel', fontsize=fs)
    axs[2].set_title(f'Angle detection with Fast Fourier Transform\nν: {dat2[1]:.2f}, α: {dat2[0]:.2f}°', fontsize=fs+2)
    axs[2].legend()

    plt.show()


# [2] Paths to files/folders ==========================================================================================
# Reprojected TIFF-Image
geotiff = "/path_to_geotiff_file/snapshot-2022-04-04T00_00_00Z_reprojected.tiff"
# Path to netCDF with Coordinates
gps_path = '/folder_with_netCDF_data/'
gps_file = 'HALO-AC3_P5_GPS_INS_20220404_RF10.nc'

# Files/Folders to analyse for cloud streets
file = '/path_to_image/corr_20220404_110826.261_IM.jpg'

path = '/path_to_folder_with_images'

# [3] Lists, Variables ================================================================================================
results_list, timestamps, lats, lons, final_list = [], [], [], [], []
pxrange, correl_range = 350, 150  # Pixel diameter for angle detection, autocorrelation, Default: 350, 150
show = 0
win_size = 20  # window size running average, Default: 20
wlen_fact = 0.2 # window size prominence algorithm, unused atm
try_angles = np.arange(0, 90, 1) # Angles to check for, Default: 0,90,2
# size auf 0.5 in folder_analyzer [0.5, 1, 'auto']

# [4] Running Code ====================================================================================================
starttime = datetime.now()

# {*} Visualisierung des Flugpfades für Datenherkunft
zeig_flugroute = 0
if zeig_flugroute:
    show_flight_path(gps_path+gps_file, geotiff, dpi=100, show=True)

# {*} Vergleich I_CS mit I_CS,M
vgl_ics_icsm = 0
if vgl_ics_icsm: vergleich_indize(path, norm=False, dpi=100, th_CS=True,
                                  save_to='/save_in_folder/out.png')

# {*} Abbildung 2D-Autokorrelation
abb_ac = 0
if abb_ac:
    # 5. Bild im Ordner Cloud Streets strong
    file = '/path_to_folder/corr_20220404_114616.294_IM.jpg'
    out1 = image_reader(file, dreh=0)
    out2 = correl_images(out1, 150, monitor=False, size='auto', turn=False)
    out3 = read_correl_mat(correl_mat=out2[1],show=True,im_arr=out2[0],angle=0, # .pdf sieht schlecht aus
                           save='/save_in_folder/out.png')

# {*} Abbildung für Visualisierung der Winkelmethoden
zeig_winkelmeth = 0
if zeig_winkelmeth:
    # Winkelmethoden visualisieren
    # Subfigure mit 2 Spalten, 3 Zeilen.
    # (0,0) Bild, (0,1) Korrelationsmatrix
    # (1,0) und (1,1) Methode 1: Angle Dependence
    # (2,0) und (2,1) Methode 2: FFT
    pfad = '/path_to_folder'
    datei = 'corr_20220404_114616.294_IM.jpg'
    ort = f'{pfad}/{datei}'
    img = image_reader(ort, dreh=0)
    # [Spalte 1: Für Methoden basierend auf dem Ausgangsbild]
    img_part = array_size_lim(img, max_lim=True) # (0,0)
    # Polynom, Maximum und I_CSP in Abhängigkeit vom Winkel (1,0)
    meth1 = angle_finder(img, dreh=0)
    # Fast Fourier Transformation (2,0)
    meth2 = angle_detector_fft(img,max_n=1,show=False,radius=15)

    # [Spalte 2: Für Methoden basierend auf Korrelationsmatrix]
    img_part = array_size_lim(img, max_lim=True) # (0,1)
    cm = correl_images(img,correl_range=150,monitor=False,size='auto',turn=0)[1]
    # Polynom, Maximum und I_CSP in Abhängigkeit vom Winkel (1,1)
    correl_mat = cm[:,:,0]*255
    meth3 = angle_finder(correl_mat, dreh=0)
    # Fast Fourier Transformation (2,1)
    meth4 = angle_detector_fft(cm[:,:,0].T,max_n=1,show=False,radius=15)
    mitte4 = (meth4[2][0],meth4[2][1])
    pos_pkt4 = meth4[2][4]

    # Subplots erstellen =============================================================================================
    fs = 12
    title = r'Angle dependance of $\mathrm{I_{CS,P}}$'
    fig, axs = plt.subplots(3, 2, figsize=(8, 12),
                            gridspec_kw={'width_ratios': [4, 4], 'height_ratios': [4,4,4]})
    fig.subplots_adjust(wspace=0.5, hspace=0.3)

    # Nummeriert einzelne Subplots (Für Bildbeschreibung)
    labels = ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)']
    for idx, ax in enumerate(axs.flat):
        ax.text(-0.2, 1.1, labels[idx], transform=ax.transAxes,
                fontsize=fs, fontfamily='serif', va='top', ha='left')

    # SPALTE 1
    # Plot analysed image
    axs[0,0].set_aspect('equal')
    cont0 = axs[0,0].contourf(meth1[5], 200, cmap="Greys_r")
    axs[0,0].set_xlabel("x-Axis Pixels", fontsize=fs)
    axs[0,0].set_ylabel("x-Axis Pixels", fontsize=fs)

    divider = make_axes_locatable(axs[0,0])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cb0 = plt.colorbar(cont0, cax=cax)
    cb0.ax.get_yaxis().labelpad = 15
    cb0.ax.set_ylabel('Grey value', rotation=270, fontsize=fs)

    axs[0,0].set_title(f"Analysed image", fontsize=fs + 2)

    # Plot angle dependance
    axs[1,0].set_box_aspect(1)
    #axs[1,0].set_aspect('auto', anchor='C')
    axs[1,0].plot(meth1[2][0], meth1[2][1], label=r'$\mathrm{I_{CS,P}}$')
    axs[1,0].plot(meth1[3], meth1[4], label=r'Polynom of $\mathrm{I_{CS,P}}$', linestyle='--', linewidth=2)
    axs[1,0].plot(meth1[1], meth1[0], 'rx', markersize=10, label='Maximum')
    #axs[1,0].set_title(title, fontsize=fs + 2)
    axs[1,0].set_xlabel('Rotation | °', fontsize=fs)
    axs[1,0].set_ylabel(r'Index value', fontsize=fs)
    axs[1,0].spines[['right', 'top']].set_visible(False)  # Soll laut Marcus überall eingebaut werden
    #axs[1,0].legend()

    axs[2,0].hlines(meth2[2][1], xmin=0, xmax=meth2[2][2][1] // 2, color='magenta', linestyle=':')
    axs[2,0].plot([meth2[2][3][:, 0].mean(), meth2[2][0]], [meth2[2][3][:, 1].mean(), meth2[2][1]], color='cyan')

    # Plotte Maxima
    axs[2,0].scatter(meth2[2][4][0], meth2[2][4][1], color='C1', label='Frequency peak')

    # Plotte Fourier Transformation
    testplot = axs[2,0].imshow(meth2[2][6], cmap='grey')

    divider = make_axes_locatable(axs[2,0])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cb1 = plt.colorbar(testplot, cax=cax)
    cb1.ax.get_yaxis().labelpad = 15
    cb1.ax.set_ylabel('Fourier amplitude', rotation=270, fontsize=fs)

    axs[2,0].set_xlim([meth2[2][2][1] // 2 - meth2[2][5], meth2[2][2][1] // 2 + meth2[2][5]])
    axs[2,0].set_ylim([meth2[2][2][0] // 2 - meth2[2][5], meth2[2][2][0] // 2 + meth2[2][5]])
    axs[2,0].set_xlabel('Pixel', fontsize=fs), axs[2,0].set_ylabel('Pixel', fontsize=fs)
    axs[2,0].set_title(f'ν: {meth2[1]:.2f}, α: {meth2[0]:.2f}°',fontsize=fs)
    #axs[2,0].legend()

    # SPALTE 2
    # plot 2D Plot of the Autocorrelation values
    till = len(cm)
    axs[0,1].set_aspect('equal')
    axs[0,1].set_title("Correlation matrix", fontsize=fs+2)
    axs[0,1].set_xlabel("x-Axis Pixels", fontsize=fs)
    axs[0,1].set_ylabel("y-Axis Pixels", fontsize=fs)
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", ["k", "c", "b", "violet", "r"])
    data = cm[:, :, 0].T
    cont1 = axs[0,1].contourf(np.arange(0, till), np.arange(0, till), data, 200, cmap=cmap, vmin=0, vmax=1)
    # marks contours
    divider = make_axes_locatable(axs[0,1])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    tix = np.linspace(np.min(data), np.max(data), 5).round(2)
    cb1 = plt.colorbar(cont1, cax=cax, ticks=tix)
    cb1.ax.get_yaxis().labelpad = fs + 4
    cb1.ax.set_ylabel(r'$F(L_\text{x},L_\text{y})^2$', rotation=270, fontsize=fs)

    # Plot angle dependance
    axs[1, 1].set_box_aspect(1) # macht Plotfläche quadratisch. Sehr praktisch!
    #axs[1, 1].set_aspect('auto', anchor='C')
    axs[1, 1].plot(meth3[2][0], meth3[2][1], label=r'$\mathrm{I_{CS,P}}$')
    axs[1, 1].plot(meth3[3], meth3[4], label=r'Polynom of $\mathrm{I_{CS,P}}$', linestyle='--', linewidth=2)
    axs[1, 1].plot(meth3[1], meth3[0], 'rx', markersize=10, label='Maximum')
    #axs[1, 1].set_title(title, fontsize=fs + 2)
    axs[1, 1].set_xlabel('Rotation | °', fontsize=fs)
    axs[1, 1].set_ylabel(r'Index value', fontsize=fs)
    axs[1, 1].spines[['right', 'top']].set_visible(False)  # Soll laut Marcus überall eingebaut werden
    #axs[1, 1].legend()

    axs[2, 1].hlines(meth4[2][1], xmin=0, xmax=meth4[2][2][1] // 2, color='magenta', linestyle=':')
    axs[2, 1].plot([pos_pkt4[0], mitte4[0]], [pos_pkt4[1], mitte4[1]], color='cyan')

    # Plotte Maxima
    axs[2, 1].scatter(meth4[2][4][0], meth4[2][4][1], color='C1', label='Frequency peak')

    # Plotte Fourier Transformation
    testplot = axs[2, 1].imshow(meth4[2][6], cmap='grey')

    divider = make_axes_locatable(axs[2,1])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cb1 = plt.colorbar(testplot, cax=cax)
    cb1.ax.get_yaxis().labelpad = 15
    cb1.ax.set_ylabel('Fourier amplitude', rotation=270, fontsize=fs)

    axs[2, 1].set_xlim([meth4[2][2][1] // 2 - meth4[2][5], meth4[2][2][1] // 2 + meth4[2][5]])
    axs[2, 1].set_ylim([meth4[2][2][0] // 2 - meth4[2][5], meth4[2][2][0] // 2 + meth4[2][5]])
    axs[2, 1].set_xlabel('Pixel', fontsize=fs), axs[2, 1].set_ylabel('Pixel', fontsize=fs)
    axs[2, 1].set_title(f'ν: {meth4[1]:.2f}, α: {meth4[0]:.2f}°',fontsize=fs)
    #axs[2, 1].legend()

    save_to = '/save_in_foler/out.png'
    plt.savefig(save_to, dpi=100, bbox_inches='tight', format="png")

    #plt.tight_layout()
    plt.show()

# {*} I_CS mit I_CSP vergleichen
vgl_ics_icsp = 0
if vgl_ics_icsp:
    # fnames: 'V_ICS_ICSP_Pass1.png', 'V_ICS_ICSP_Pass1_Normed.png', 'V_ICS_ICSP_Pass4.png'
    fname = 'out.png'
    vergleich_indize(path, norm=False, dpi=100, th_CS=True, th_CSP=True,
                     save_to=f'/save_in_folder/{fname}')

vgl_winkel_meth = 0
save = '/save_in_folder/out.png'
if vgl_winkel_meth:
    files = file_lister('/path_to_folder')
    data_list, dpi = [], 100
    for file in files:
        print(f'[ {files.index(file) + 1} / {len(files)} ]')
        # Schrittweise jeden Winkelabtasten, ggf. auskommentieren
        for winkel in np.arange(0,90,1):
            out = angle_comparer(file, dreh=False,winkel=winkel)
            data_list.append([out[0],out[1],out[3],out[5],out[6]])
    data_arr = np.array(data_list)
    print(data_arr)
    #print(data_arr[:,0]) # Spalte 1

    # Angle difference between recognised and initial angle
    # PLUS, weil relativer Winkel! (30°+60°), bei 30° Drehung wird bei 60° erkannt
    ang_diff1 = abs(data_arr[:,0] + data_arr[:,1]) # Rotation + ICSP
    ang_diff2 = abs(data_arr[:,0] + data_arr[:,2]) # AC + Rotation + ICSP
    # MINUS, weil absoluter Winkel! (30°-30°), bei 30° drehung werden 30° erkannt
    ang_diff3 = abs(data_arr[:,0] - data_arr[:,3]) # FFT
    ang_diff4 = abs(data_arr[:,0] - data_arr[:,4]) # AC + FFT

    # limit angle offset by 45°
    ang_diff1 = (ang_diff1/90).round() * 90 - ang_diff1
    ang_diff2 = (ang_diff2/90).round() * 90 - ang_diff2
    ang_diff3 = (ang_diff3/90).round() * 90 - ang_diff3
    ang_diff4 = (ang_diff4/90).round() * 90 - ang_diff4

    print("Winkeldifferenz:\n",np.array([ang_diff1, ang_diff2, ang_diff3, ang_diff4]).T)

    # Creat boxplots
    data = [abs(ang_diff1), abs(ang_diff2), abs(ang_diff3), abs(ang_diff4)]
    plt.grid(True, axis='y', linestyle='--', alpha=0.75)
    plt.boxplot(data, tick_labels=['(1)', '(2)',
                                   '(3)', '(4)'], meanline=True, showmeans=True)
    plt.ylabel('Angle Difference | °'), plt.ylim(0,50), plt.tight_layout()
    #plt.title('Difference between initial and detected angle', fontsize=16)

    if save: plt.savefig(save, dpi=dpi, bbox_inches='tight', format="png")

    plt.tight_layout()
    plt.show()

# {*} Kombination von beste Winkelmethode + Index (ICS, ICSP)!!!!
# Unterschiede bei 0°, 30°, 45°, Zufälligem Winkel
combi_winmeth_idx = 0
if combi_winmeth_idx:
    folder = '/path_to_folder'
    save_to = '/save_in_folder/'
    files, data_icsp, data_icsp0, data_ics, data_ics0 = file_lister(folder), [], [], [], []
    for file in files:
        # Eingangsbild drehen?
        dreh = 0
        #dreh = np.random.uniform(low=0, high=90, size=1).round()[0]

        # Bild einlesen und GGF. drehen!
        img_arr = image_reader(file,dreh=dreh,mute=True)
        # Winkel erkennen mittels FFT
        angle = angle_detector_fft(img_arr, max_n=1,show=False)[0]
        angle = dreh - angle

        # Berechnung des I_CS,P mit erkanntem Winkel ===========================
        #img_arr1 = image_reader(file, dreh=0,mute=True) # KONTROLLLAUF! Gleiche Werte bei gleichem Winkel?
        img_arr1 = image_reader(file, dreh=angle,mute=True) # Bild für erkannten Winkel einlesen
        arr_angleany = array_size_lim(img_arr1, opt_lim=True)
        lists = icsp_lists(arr_angleany) # any angle
        icsp = icsp_function(lists[0], lists[1], ravg=True)[0] # any angle

        img_arr0 = image_reader(file,dreh=0,mute=True) # Bild ohne Drehung einlesen
        arr_angle0 = array_size_lim(img_arr0, opt_lim=True)
        lists1 = icsp_lists(arr_angle0) # 0°
        icsp0 = icsp_function(lists1[0], lists1[1], ravg=True)[0] # 0°

        # Berechnung des I_CS (bei 0°) =========================================
        #ics = ics0_function(file,dreh=0) # KONTROLLLAUF! Gleiche Werte bei gleichem Winkel
        ics = ics0_function(file,dreh=angle) # any angle

        ics0 = ics0_function(file,dreh=0) # 0°

        # Daten in Listen speichern
        data_ics.append(ics) # I_CS bei erkanntem Winkel (Bild wird zurückgedreht
        data_ics0.append(ics0) # I_CS bei 0° Drehung (etwa Ausrichtung der Wolkenstraßen
        data_icsp.append(icsp) # I_CS,P bei erkanntem Winkel (Bild wird zurückgedreht
        data_icsp0.append(icsp0) # I_CS,P bei 0° Drehung (etwa Ausrichtung der Wolkenstraßen

    # Informationen aus dem Dateinamen entnehmen
    for filestring in files:
        splitting = filestring.split('/')  # Gets folder name
        splitting2 = splitting[-1].split('.')  # Extracts time stamp
        splitting3 = splitting2[-3].split('_')
        # takes last two elements for timestamp, so corr_ gets filtered out if present
        add = datetime.strptime(splitting3[-2] + splitting3[-1], '%Y%m%d%H%M%S')
        timestamps.append(add)  # Appends datetime object to list
    splitting = path.split('/')  # Gets folder name
    folder_name = splitting[-1]

    # Plots indices if Running Average is ENABLED
    win_size = 10
    # x = np.arange((win_size // 2), len(ics0) - (win_size // 2 - 1))
    times = np.array(timestamps[(win_size // 2) - 1:(len(data_icsp) - win_size // 2)])

    # data with angle finding
    data0 = running_average(data_ics, win_size)
    data1 = running_average(data_icsp, win_size)
    # data at 0°
    data2 = running_average(data_ics0, win_size)
    data3 = running_average(data_icsp0, win_size)

    # Für Zeit als x-Achse: 'times', Ansonsten 'x'
    fs = 14
    plt.figure(figsize=(12, 4))
    plt.tick_params(labelsize=fs)

    plt.scatter(times, data0, label=r'$I_{\mathrm{CS}}$', marker='o')
    plt.plot(times, data0, alpha=0.3)
    plt.scatter(times, data2, label=r'$I_{\mathrm{CS}}$ (0°)', marker='X')
    plt.plot(times, data2, alpha=0.3)
    plt.axhline(0.2, color='gray', linestyle=':', label=r'$I_{\mathrm{CS}}$ threshold')

    #plt.scatter(times, data1, label=r'$I_{\mathrm{CS,P}}$', marker='D')
    #plt.plot(times, data1, alpha=0.3)
    #plt.scatter(times, data3, label=r'$I_{\mathrm{CS,P}}$ (0°)', marker='s')
    #plt.plot(times, data3, alpha=0.3)
    #plt.axhline(0.4, color='gray', linestyle='--', label=r'$I_{\mathrm{CS,P}}$ threshold')

    plt.ylabel('Index value', fontsize=fs), plt.xlabel('Time (UTC) | HH:MM', fontsize=fs)
    plt.grid(zorder=1, which='major', linestyle='-')
    plt.grid(zorder=1, which='minor', linestyle='--')
    ax = plt.gca()
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    ax.xaxis.set_minor_locator(mdates.MinuteLocator(interval=1))
    ax.spines[['right', 'top']].set_visible(False)
    plt.legend(fontsize=fs)
    plt.ylim(ymax=1, ymin=-0.2)  # -0.2 für negative Werte

    # save image
    plt.savefig(save_to+'ics_30_pass3.png', dpi=100, bbox_inches='tight', format="png",pad_inches=0)

    plt.tight_layout(w_pad=0.2, h_pad=0.2)
    plt.show()

# {*} Erstellen von Zeitreihen (ICS, ICSP) für den 1. April!!!!
time_series_unaligned_angle = 0
if time_series_unaligned_angle:
    folder = '/path_to_folder'
    save_to = '/save_in_folder/'
    files, data_icsp, data_icsp0, data_ics, data_ics0 = file_lister(folder), [], [], [], []
    for file in files:
        print(f'[ {files.index(file) + 1} / {len(files)} ]')
        # Eingangsbild drehen?
        dreh = 0
        #dreh = np.random.uniform(low=0, high=90, size=1).round()[0]

        # Bild einlesen und GGF. drehen!
        img_arr = image_reader(file,dreh=dreh,mute=True)
        # Winkel erkennen mittels FFT
        angle = angle_detector_fft(img_arr, max_n=1,show=False)[0]
        angle = dreh - angle

        # Berechnung des I_CS,P mit erkanntem Winkel ===========================
        #img_arr1 = image_reader(file, dreh=0,mute=True) # KONTROLLLAUF! Gleiche Werte bei gleichem Winkel?
        img_arr1 = image_reader(file, dreh=angle,mute=True) # Bild für erkannten Winkel einlesen
        arr_angleany = array_size_lim(img_arr1, opt_lim=True)
        lists = icsp_lists(arr_angleany) # any angle
        icsp = icsp_function(lists[0], lists[1], ravg=True)[0] # any angle

        img_arr0 = image_reader(file,dreh=0,mute=True) # Bild ohne Drehung einlesen
        arr_angle0 = array_size_lim(img_arr0, opt_lim=True)
        lists1 = icsp_lists(arr_angle0) # 0°
        icsp0 = icsp_function(lists1[0], lists1[1], ravg=True)[0] # 0°

        # Berechnung des I_CS (bei 0°) =========================================
        #ics = ics0_function(file,dreh=0) # KONTROLLLAUF! Gleiche Werte bei gleichem Winkel
        ics = ics0_function(file,dreh=angle) # any angle

        ics0 = ics0_function(file,dreh=0) # 0°

        # Daten in Listen speichern
        data_ics.append(ics) # I_CS bei erkanntem Winkel (Bild wird zurückgedreht
        data_ics0.append(ics0) # I_CS bei 0° Drehung (etwa Ausrichtung der Wolkenstraßen
        data_icsp.append(icsp) # I_CS,P bei erkanntem Winkel (Bild wird zurückgedreht
        data_icsp0.append(icsp0) # I_CS,P bei 0° Drehung (etwa Ausrichtung der Wolkenstraßen

    # Informationen aus dem Dateinamen entnehmen
    for filestring in files:
        splitting = filestring.split('/')  # Gets folder name
        splitting2 = splitting[-1].split('.')  # Extracts time stamp
        splitting3 = splitting2[-3].split('_')
        # takes last two elements for timestamp, so corr_ gets filtered out if present
        add = datetime.strptime(splitting3[-2] + splitting3[-1], '%Y%m%d%H%M%S')
        timestamps.append(add)  # Appends datetime object to list
    splitting = path.split('/')  # Gets folder name
    folder_name = splitting[-1]

    # Plots indices if Running Average is ENABLED
    win_size = 10
    # x = np.arange((win_size // 2), len(ics0) - (win_size // 2 - 1))
    times = np.array(timestamps[(win_size // 2) - 1:(len(data_icsp) - win_size // 2)])

    # data with angle finding
    data0 = running_average(data_ics, win_size)
    data1 = running_average(data_icsp, win_size)
    # data at 0°
    data2 = running_average(data_ics0, win_size)
    data3 = running_average(data_icsp0, win_size)

    # Für Zeit als x-Achse: 'times', Ansonsten 'x'
    fs = 14
    plt.figure(figsize=(12, 4))
    plt.tick_params(labelsize=fs)

    plt.scatter(times, data0, label=r'$I_{\mathrm{CS}}$ with Fourier-Method', marker='o')
    plt.plot(times, data0, alpha=0.3)
    #plt.scatter(times, data2, label=r'$I_{\mathrm{CS}}$', marker='X')
    #plt.plot(times, data2, alpha=0.3)
    plt.axhline(0.2, color='gray', linestyle=':', label=r'$I_{\mathrm{CS}}$ threshold')

    plt.scatter(times, data1, label=r'$I_{\mathrm{CS,P}}$ with Fourier-Method', marker='D')
    plt.plot(times, data1, alpha=0.3)
    #plt.scatter(times, data3, label=r'$I_{\mathrm{CS,P}}$', marker='s')
    #plt.plot(times, data3, alpha=0.3)
    plt.axhline(0.4, color='gray', linestyle='--', label=r'$I_{\mathrm{CS,P}}$ threshold')

    plt.ylabel('Index value', fontsize=fs), plt.xlabel('Time (UTC) | HH:MM', fontsize=fs)
    plt.grid(zorder=1, which='major', linestyle='-')
    plt.grid(zorder=1, which='minor', linestyle='--')
    ax = plt.gca()
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    ax.xaxis.set_minor_locator(mdates.MinuteLocator(interval=1))
    ax.spines[['right', 'top']].set_visible(False)
    plt.legend(fontsize=fs)
    plt.ylim(ymax=1, ymin=-0.2)  # -0.2 für negative Werte

    # save image
    plt.savefig(save_to+'timeseries_both_20220401_Pass4.png', dpi=100, bbox_inches='tight', format="png")

    plt.tight_layout(w_pad=0.2, h_pad=0.2)
    plt.show()

# {*} Bildmatrix erstellen
idx_matrice = 0
if idx_matrice:
    ort = '/path_to_folder'
    speicher = '/save_in_folder'
    files = file_lister(ort)
    images = []
    for file in files:
        print(f'[ {files.index(file) + 1} / {len(files)} ]')
        # Bild für Plot vorbereiten und speichern
        img_arr  = np.array(pimg.open(file))
        img_arr_resized0 = array_size_lim(img_arr[:,:,0],max_lim=True)
        img_arr_resized1 = array_size_lim(img_arr[:,:,1],max_lim=True)
        img_arr_resized2 = array_size_lim(img_arr[:,:,2],max_lim=True)
        # einzelne 2D-Arrays wieder zusammenführen:
        image = pimg.fromarray(np.stack((img_arr_resized0,img_arr_resized1,img_arr_resized2),axis=2))

        # Zeitpunkt des Bildes extrahieren
        splitting = file.split('/')  # Gets folder name
        splitting2 = splitting[-1].split('.')  # Extracts time stamp
        splitting3 = splitting2[-3].split('_')
        # takes last two elements for timestamp, so corr_ gets filtered out if present
        add = datetime.strptime(splitting3[-2] + splitting3[-1], '%Y%m%d%H%M%S')

        # ICS und ICSP berechnen (bei 0°)
        ics0 = round(ics0_function(file),2)
        img_arr = image_reader(file,mute=True)
        img_arr = array_size_lim(img_arr, set_radius=200)
        lists = icsp_lists(img_arr) # 0°
        icsp0 = round(icsp_function(lists[0], lists[1], ravg=True)[0],2)

        fs = 20
        fig, ax = plt.subplots()
        ax.imshow(image)

        # Axes-Position (im Figure-Koordinatensystem)
        pos = ax.get_position()
        # Titel mit Wert erstellen
        ics_txt, icsp_txt = r'$I_{\mathrm{CS}}$', r'$I_{\mathrm{CS,P}}$'
        fig.text(pos.x0, pos.y1 + 0.01, f"{ics_txt}: {ics0:.2f}", ha='left', va='bottom', fontsize=fs, color='k')
        fig.text(pos.x1, pos.y1 + 0.01, f"{icsp_txt}: {icsp0:.2f}", ha='right', va='bottom', fontsize=fs, color='k')

        # Textposition des Datum
        hoehe, breite = img_arr_resized0.shape
        zeit = ax.text(x=breite/2,y=hoehe*0.975,s=str(add),
                        ha='center', va='bottom',fontsize=fs,color='white',)
        zeit.set_path_effects([pe.Stroke(linewidth=2, foreground='black'),pe.Normal()])
        ax.axis('off')

        fname = f'{add}_{ics0}_{icsp0}_idx.png'
        plt.savefig(f'{speicher}/{fname}', dpi=100, bbox_inches='tight', format="png",pad_inches=0)
        plt.close()
        #plt.show()

# {*} ICS(P) ohne und mit Winkelerkennung für gleichen und zufällligen Winkel! (Je drei Plots)
# Unterschiede bei 0°, 30°, 45°, Zufälligem Winkel
combi_winmeth_idx = 1
if combi_winmeth_idx:
    folder = '/path_to_folder'
    save_to = '/save_in_folder/'
    files = file_lister(folder)
    data_ics, data_ics0, data_icsrd = [], [], [] # ICS Werte
    data_icsp, data_icsp0, data_icsprd = [], [], [] # ICSP Werte
    for file in files:
        # Kamerabild drehen?
        dreh0 = 0 # betrifft: data_ics, dat
        dreh1 = np.random.uniform(low=0, high=90, size=1).round()[0] # betrifft: data_icsrd, data_icsprd
        data2 = 0 # standard: 0, betrifft: data_ics0, data_icsp0

        # Bild einlesen und GGF. drehen!
        img_arr0 = image_reader(file,dreh=dreh0,mute=True) #  Bild mit konstanter Drehung einlesen
        img_arr1 = image_reader(file, dreh=dreh1, mute=True) # Bild mit zufälliger Drehung einlesen
        img_arr2 = image_reader(file,dreh=data2,mute=True) # Bild ohne Drehung einlesen

        # Winkel erkennen mittels FFT
        angle0 = angle_detector_fft(img_arr0, max_n=1,show=False)[0]
        angle0 = dreh0 - angle0 # Erkannter Winkel für Bilder mit festgesetzter Drehung
        angle1 = angle_detector_fft(img_arr1, max_n=1, show=False)[0]
        angle1 = dreh1 - angle1 # Erkannter Winkel für Bilder mit zufälliger Drehung

        # I_CS,P bei konstantem Winkel (dreh0)
        img_arr0 = image_reader(file, dreh=angle0,mute=True) # Erkannten Winkel anwenden
        arr_angleany = array_size_lim(img_arr0, opt_lim=True)
        lists = icsp_lists(arr_angleany) # any angle
        icsp = icsp_function(lists[0], lists[1], ravg=True)[0] # Konstanter Winkel
        # I_CS,P ohne Drehung (dreh2=0)
        arr_angle0 = array_size_lim(img_arr2, opt_lim=True)
        lists1 = icsp_lists(arr_angle0) # 0°
        icsp0 = icsp_function(lists1[0], lists1[1], ravg=True)[0] # Keine Drehung
        # I_CS,P bei zufälligem Winkel (dreh1)
        img_arr1 = image_reader(file, dreh=angle1, mute=True)  # Erkannten Winkel anwenden
        arr_angleany = array_size_lim(img_arr1, opt_lim=True)
        lists = icsp_lists(arr_angleany)  # any angle
        icsprd = icsp_function(lists[0], lists[1], ravg=True)[0]  # Zufälliger Winkel

        ics = ics0_function(file,dreh=angle0) # Konstanter Winkel
        ics0 = ics0_function(file,dreh=data2) # Keine Bilddrehung
        icsrd = ics0_function(file, dreh=angle1) # Zufälliger Winkel

        # Daten in Listen speichern
        data_ics.append(ics) # I_CS mit Winkelerkennung, konstanter Winkel
        data_ics0.append(ics0) # I_CS ohne Winkelerkennung, keine Drehung
        data_icsrd.append(icsrd) # I_CS mit Winkelerkennung, zufälliger Winkel
        data_icsp.append(icsp) # I_CS,P mit Winkelerkennung, konstanter Winkel
        data_icsp0.append(icsp0) # I_CS,P ohne Winkelerkennung, keine Drehung
        data_icsprd.append(icsprd) # I_CS,P mit Winkelerkennung, zufälliger Winkel

    # Informationen aus dem Dateinamen entnehmen
    for filestring in files:
        splitting = filestring.split('/')  # Gets folder name
        splitting2 = splitting[-1].split('.')  # Extracts time stamp
        splitting3 = splitting2[-3].split('_')
        # takes last two elements for timestamp, so corr_ gets filtered out if present
        add = datetime.strptime(splitting3[-2] + splitting3[-1], '%Y%m%d%H%M%S')
        timestamps.append(add)  # Appends datetime object to list
    splitting = path.split('/')  # Gets folder name
    folder_name = splitting[-1]

    # Plots indices if Running Average is ENABLED
    win_size = 10
    # x = np.arange((win_size // 2), len(ics0) - (win_size // 2 - 1))
    times = np.array(timestamps[(win_size // 2) - 1:(len(data_icsp) - win_size // 2)])

    # data for set angle
    data0 = running_average(data_ics, win_size)
    data1 = running_average(data_icsp, win_size)
    # data at 0°
    data2 = running_average(data_ics0, win_size)
    data3 = running_average(data_icsp0, win_size)
    # data for random angle
    data4 = running_average(data_icsrd, win_size)
    data5 = running_average(data_icsprd, win_size)

    # Für Zeit als x-Achse: 'times', Ansonsten 'x'
    fs = 14
    plt.figure(figsize=(12, 4))
    plt.tick_params(labelsize=fs)

    plt.scatter(times, data2, label=r'$I_{\mathrm{CS}}$', marker='o')
    plt.plot(times, data2, alpha=0.3)
    plt.scatter(times, data0, label=r'$I_{\mathrm{CS}}$ dynamic', marker='X')
    plt.plot(times, data0, alpha=0.3)
    plt.scatter(times, data4, label=r'$I_{\mathrm{CS}}$ random', marker='^')
    plt.plot(times, data4, alpha=0.3)
    plt.axhline(0.2, color='gray', linestyle=':', label=r'$I_{\mathrm{CS}}$ threshold')

    #plt.scatter(times, data3, label=r'$I_{\mathrm{CS,P}}$', marker='D')
    #plt.plot(times, data3, alpha=0.3)
    #plt.scatter(times, data1, label=r'$I_{\mathrm{CS,P}}$ dynamic', marker='s')
    #plt.plot(times, data1, alpha=0.3)
    #plt.scatter(times, data5, label=r'$I_{\mathrm{CS,P}}$ random', marker='P')
    #plt.plot(times, data5, alpha=0.3)
    #plt.axhline(0.4, color='gray', linestyle='--', label=r'$I_{\mathrm{CS,P}}$ threshold')

    plt.ylabel('Index value', fontsize=fs), plt.xlabel('Time (UTC) | HH:MM', fontsize=fs)
    plt.grid(zorder=1, which='major', linestyle='-')
    plt.grid(zorder=1, which='minor', linestyle='--')
    ax = plt.gca()
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    ax.xaxis.set_minor_locator(mdates.MinuteLocator(interval=1))
    ax.spines[['right', 'top']].set_visible(False)
    plt.legend(fontsize=fs)
    plt.ylim(ymax=1, ymin=-0.2)  # -0.2 für negative Werte

    # save image
    plt.savefig(save_to+'out.png', dpi=100, bbox_inches='tight', format="png",pad_inches=0)

    plt.tight_layout(w_pad=0.2, h_pad=0.2)
    plt.show()


print(f"Laufzeit des Codes: {datetime.now() - starttime}")

