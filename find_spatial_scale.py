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
im_directory = './1_19_2024_laser_spark/'
image_to_analyze ='Sequence45(UBSi121HB2062).tif'
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
ld = 218
dl = 207 
dpix = ld-dl

group = 2
element = 3

lppmm = 2**(group+(element-1)/6)
mmpll = 1/lppmm

# Compute the spatial mm/pixel
pixpmm = dpix*lppmm
mmppix = 1/pixpmm

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
        pixel = 700
        lineout = aligned_image[:, pixel-n_avg:pixel+n_avg].mean(axis=1)
        lineout = np.flip(lineout)
        xlineout = np.arange(0, len(lineout), 1)

        # Compute the slope of the lineout
        slope = np.zeros_like(lineout)
        slope[1:] = np.diff(lineout)
        slope =np.gradient(lineout)

        # Find the indices of the most extreme inflection points
        extreme_inflection_points = np.where(np.abs(np.diff(np.sign(slope))) > 1)[0]
        ax3 = ax2.twinx()  # instantiate a second axes that shares the same x-axis
        ax2.plot(lineout, mmppix*xlineout)
        # ax3.plot(slope, range(lineout.size), 'r-')

        ax2.set_title('Lineout of Aligned Image')
        ax2.set_ylim(0, mmppix*aligned_image.shape[0])
        ax3.set_ylim(0, mmppix*aligned_image.shape[0])
        aligned_image_marked = aligned_image.copy()
        aligned_image_marked[:, pixel-n_avg:pixel+n_avg] = 0
        ax1.imshow(aligned_image_marked, aspect="auto", extent=[0,mmppix*colmax,0,mmppix*rowmax])        

        # ax2.set_xlim(0, 255)
        plt.tight_layout()
        # plt.show()

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
