# Cloud-Street-Index-Variants-and-Improvements
This code contains functions to apply the original cloud street index (CSI) based on Klingebiel et al. (2025), and the new prominence-based CSI in combination with the angle detection method for cloud streets.
The following functions offer a variety of possible combinations, but the code is everything but optimal.
After the functions there are several code blocks used to create diagrams. As examples they can provide how to use the functions.

Functions:

[1] Original I_CS: ics0_function(imglocation, radius=200, dreh=0)

[2] Modified I_CS: ics0new_function(imglocation, radius=200, dreh=0)

[3] Retired 'All-in-one'-function: multi_index_function(imglocation, pxrange, dreh=0, show=0)

[4] Obtain mean values along axis of an image array: icsp_lists(masked_img, sigma=False)

[5] Calculate I_CS,P: icsp_function(list1, list2, ravg=True)

[6] Choose size of the analysed image: array_size_lim(array, set_radius=False, set_diameter=False, opt_lim=False, max_lim=False)

[7] Angle detection with rotation method: angle_finder(img_array, wlen_fact=1, dreh=0, show=False)

[8] Same as last function but for correlation matrix: angle_finder2(correl_mat, show=False)

[9] Angle detection with fast fourier method: angle_detector_fft(img_arr, max_n=1, show=0, radius=15)

[10] Apply 2D auto correlation on image: correl_images(image_A, correl_range, numpix=False, monitor=False, size=1, turn=False)

[11] Use correlation matrix, once attempted use for index calcultion but not recommended: 
     read_correl_mat(correl_mat, show=False, im_arr=False, angle=0, save=False, dpi=100)
     
[12] Function to reproject .tiff-files from NASA Worldview: def read_worldview_reprojected(geotiff)

[13] Shows a map with calculated index values: show_map(nc_file, geotiff, folder_name, timestamps, vals)

[14] Create time series for folder with images: folder_analyzer(path, norm=False)

[15] Obtains file locations in a folder: file_lister(folderpath, fformat='.jpg')

[16] Extracts the name of a file without path: filename_out(file)

[17] Plot flight track on map: show_flight_path(nc_file, geotiff, show=False, dpi=100)

[18] Function used to compare angle methods: angle_comparer(imglocation,dreh=False,winkel=False)

[19] Function used to compare time series of indices: vergleich_indize(path, norm=False, save_to=False, dpi=100, th_CS=False, th_CSP=False)

[20] Visualize angle methods: angle_methods(imglocation)
