from PIL import Image, ImageEnhance
import numpy as np
import os 
import functions as ut
import pickle 
import cv2
import numpy as np
import matplotlib.pyplot as plt
calibration_directory = './calibration/'
im_directory = './calibration/'
image_to_analyze ='airforce-test-strip-backlit-greenlaser.tif'
im_directory = './1_26_2024_laser_spark/'
image_to_analyze ='Sequence48(UBSi121HB2062).tif'
path_to_image = im_directory+image_to_analyze
save_directory = im_directory+image_to_analyze.split('/')[-1].split('.')[0]+'/'

# List all files in directory #
files = os.listdir(im_directory)

# Remove any and all white spaces in filenames
ut.remove_white_spaces(files, im_directory)

# Now split the large image into parts and then save them    
image_parts = ut.split_image_into_horizontal_parts(path_to_image)
ut.save_parts(directory=save_directory, image_parts=image_parts)

# Initialize array for aligned images
aligned_images = []

# Compute the spatial mm/pixel
ld = 779
dl = 767 
dpix = ld-dl

group = 2
element = 3

lppmm = 2**(group+(element-1)/6)
mmpll = 1/lppmm

# Compute the spatial mm/pixel
pixpmm = dpix*lppmm
mmppix = 1/pixpmm

fs = 500e6 # samp/sec
dt = 1/fs # sec/samp

# Compute number of frames 
num_frames = 12

# Compute time of each frame 
frame_time = np.arange(0, num_frames, 1)*dt

# Initialize array for tracking width of plasma in time 
plasma_width = np.zeros_like(frame_time)

# mmppix  = 1

# Load calibration offsets #
try:
    with open(f'{calibration_directory}offsets.pkl', 'rb') as f:
        frames_offsets = pickle.load(f)
    for i in range(12):
        shift_x = frames_offsets[i]['x']
        shift_y = frames_offsets[i]['y']
        print("shift for image", i, "x,y:",shift_x, shift_y)  

        target_array = ut.load_image_parts(m_frame_offsets=frames_offsets, m_frame=i, m_save_directory=save_directory, m_enhance_val=1)
        aligned_image = ut.shift_image(target_array, shift_x, shift_y)
        
        # Adjust the saturation and constrast of image in a single line
        aligned_images.append(aligned_image)

        
        fig = plt.figure(figsize=(10, 5))

        # Plot the image and lineout side by side
        ax1 = plt.subplot(121)
        ax2 = plt.subplot(122,sharey=ax1)
        # Set size of figure 
        # fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

        # Display the image
        rowmax = aligned_image.shape[0]
        colmax = aligned_image.shape[1]

        # ax1.imshow(aligned_image, aspect="auto")
        title = "Aligned Image: "+str(i)
        ax1.set_title(title)

        # Plot the lineout
        n_avg = 10
        pixel = 550
        lineout = aligned_image[:, pixel-n_avg:pixel+n_avg].mean(axis=1)
        lineout = np.flip(lineout)
        xlineout = np.arange(0, len(lineout), 1)

        # Compute the slope of the lineout
        slope = np.zeros_like(lineout)
        slope[1:] = np.diff(lineout)
        slope =np.gradient(lineout)

        # Find the indices of the most extreme inflection points
        extreme_inflection_points = np.where(np.abs(np.diff(np.sign(slope))) > 1)[0]
        # ax3 = ax2.twinx()  # instantiate a second axes that shares the same x-axis
        ax2.plot(lineout, mmppix*xlineout)
        # ax3.plot(slope, range(lineout.size), 'r-')

        ax2.set_title('Lineout of Aligned Image')
        ax2.set_ylim(0, mmppix*aligned_image.shape[0])
        # ax3.set_ylim(0, mmppix*aligned_image.shape[0])
        aligned_image_marked = aligned_image.copy()
        aligned_image_marked[:, pixel-n_avg:pixel+n_avg] = 0
        ax1.imshow(aligned_image_marked, aspect="auto", extent=[0,mmppix*colmax,0,mmppix*rowmax], cmap='Reds')        

        # Find the maximum value of the lineout
        max_idx = np.argmax(lineout)
        max_val = lineout[max_idx]

        # Now find the location of the half max of this value
        half_max = max_val/2

        # Find the indices of the points closest to the half max
        half_max_idx_top = np.argmin(np.abs(lineout[:max_idx]-half_max))
        half_max_idx_bottom = np.argmin(np.abs(lineout[max_idx:]-half_max))
        ax2.plot([half_max, half_max], [mmppix*half_max_idx_top, mmppix*(half_max_idx_bottom+max_idx)], 'k--')

        # ax2.set_xlim(0, 255)
        plt.tight_layout()
        plt.savefig(f'{save_directory}aligned_image_{i}_lineout.png')
        # plt.show()

        # Compute the FWHM
        fwhm = mmppix*np.abs(half_max_idx_top - max_idx)+mmppix*np.abs(half_max_idx_bottom-max_idx+max_idx)
        width = mmppix*half_max_idx_top + mmppix*(half_max_idx_bottom+max_idx)
        print("FWHM: ", fwhm)
        print("width: ", width)
        plasma_width[i] = fwhm

    # Plot the plasma width in time
    plt.figure()
    plt.plot(10**9*frame_time, plasma_width)
    plt.xlabel("Time (ns)")
    plt.ylabel("Plasma Width (mm)")

    # add text box to figure 
    speed = (1e-3)*(plasma_width[-1]-plasma_width[0])/2/(frame_time[-1]-frame_time[0])
    textstr = str(np.round(speed/1000, 3))+ "km/s"
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    plt.text(0.05, 0.95, textstr, transform=plt.gca().transAxes, fontsize=14,
            verticalalignment='top', bbox=props)
    plt.tight_layout()
    plt.show()

except FileNotFoundError:
    print("Calibration FileNotFoundError")

# save images as gif
gif_path = f'{save_directory}aligned_images.gif'
aligned_images_pil = [Image.fromarray(img) for img in aligned_images]
aligned_images_pil[0].save(gif_path,
                           save_all=True,
                           append_images=aligned_images_pil[1:],
                           duration=200,  # Duration for each frame in milliseconds
                           loop=0)
