# image_processing.py
import cv2
import numpy as np
from PIL import Image
from scipy.interpolate import interp1d
def get_black_pixel_coords(image):
    # Convert the image to binary mode
    image = image.convert('1')
    # Convert the image to a numpy array
    image_array = np.array(image)
    # Find the indices of the array where the pixel value is 0 (black)
    black_pixel_indices = np.argwhere(image_array == 0)
    # Return a list of tuples of the coordinates of the black pixels
    return [(y, x) for (y, x) in black_pixel_indices]

# Define a function to sort a list of tuples by the first element
def sort_by_first_element(arr):
    # Swap the elements of each tuple
    arr = [(y, x) for x, y in arr]
    # Sort the list by the first element of each tuple
    return sorted(arr, key=lambda x: x[0])

# Define a function to fill the gaps in a spectrum by interpolating the points
def fill_gaps(spectrum, num_points):
    # Get the x and y values from the spectrum
    x = np.array([point[0] for point in spectrum])
    y = np.array([point[1] for point in spectrum])
    # Create a quadratic interpolation function
    f = interp1d(x, y, kind='quadratic')
    # Generate new x values evenly spaced between the minimum and maximum x values
    x_new = np.linspace(min(x), max(x), num_points)
    # Apply the interpolation function to the new x values
    y_new = f(x_new)
    # Create a new spectrum with the interpolated points
    new_spectrum = [[x_val, y_val] for x_val, y_val in zip(x_new, y_new)]

    return new_spectrum

# Define a function to process the input coordinates
def process_input(coords, max_y):
    # Get the x and y values from the input coordinates
    x_vals, y_vals = zip(*coords)

    # Sort the x values in ascending order
    sorted_x = sorted(list(x_vals))

    # Get the smallest 10 and largest 10 x values
    first_10 = sorted_x[:12]
    last_10 = sorted_x[-12:]

    # Create a new list of coordinates that picks the smallest y value for the
    # first 10 and last 10 x values
    new_coords = []
    for i, coord in enumerate(coords):
        x, y = coord
        if x in first_10 or x in last_10:
            # Check if this coordinate has the smallest y value for its x value
            min_y_for_x = min([coord[1] for coord in coords if coord[0] == x])
            if y == min_y_for_x:
                new_coords.append(coord)
        else:
            new_coords.append(coord)

    # Remove any y values which are within 3 of the maximum y value
    final_coords = []
    for coord in new_coords:
        x, y = coord
        if y < max_y - 3:
            final_coords.append(coord)

    return final_coords

# Define a function to remove duplicate x values from a spectrum
def remove_duplicate_x(spectrum):
    result = []
    i = 0
    while i < len(spectrum):
        x, y = spectrum[i]
        max_y = y
        max_j = i
        j = i + 1
        while j < len(spectrum) and spectrum[j][0] == x:
            if spectrum[j][1] > max_y:
                max_y = spectrum[j][1]
                max_j = j
            j += 1
        result.append(spectrum[max_j])
        i = j
    return result

# Define a function to add some points to the left of the spectrum
def add1(spectrum):
    newspec = []
    for i in range(7):
        newspec.append(((spectrum[0][0]-i), spectrum[0][1]))
    return (newspec + spectrum)

# Define a function to load an image and extract the red channel
def loadimgred(filepath):
    # This function loads an image from a file path and returns the original and fitted y-coordinates of the black pixels
    img = cv2.imread(filepath) # Read the image from the file path

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # Convert the image to RGB color space

    edges = cv2.Canny(gray, 50, 150, apertureSize=3) # Detect the edges of the image using Canny algorithm

    contours, hierarchy = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) # Find the external contours of the image
    largest_contour = max(contours, key=cv2.contourArea) # Find the largest contour by area

    x, y, w, h = cv2.boundingRect(largest_contour) # Get the bounding rectangle of the largest contour

    #crop_img = img[y+3:y+h-7, x+8:x+w-2]
    crop_img = img[y:y+h, x:x+w] # Crop the image using the bounding rectangle
    image = red(crop_img) # Apply the red function to the cropped image
    width, height = image.size # Get the width and height of the image
    black_pixels = get_black_pixel_coords(image) # Get the coordinates of the black pixels in the image
    original = sort_by_first_element(black_pixels) # Sort the black pixels by their x-coordinates
    black_pixels = process_input(original, height) # Process the input by removing noise and outliers
    black_pixels = remove_duplicate_x(black_pixels) # Remove duplicate x-coordinates from the black pixels
    #black_pixels = add1(black_pixels)
    fitted = fill_gaps(black_pixels,1000) # Fill the gaps in the black pixels using interpolation
    originaly= [y/height for [x, y] in original] # Normalize the original y-coordinates by dividing by the height
    originalx= [x for [x, y] in original] # Get the original x-coordinates
    originalx=(1000/originalx[-1])*np.array(originalx) # Scale the original x-coordinates to 1000
    fitted= [y/height for [x, y] in fitted] # Normalize the fitted y-coordinates by dividing by the height
    return originaly, originalx, fitted # Return the original and fitted y-coordinates

# Define a function to load an image and crop it according to its size and year
def loadimgblack(filepath):
    # Open the image from the file path
    im = Image.open(filepath)
    # Get the width and height of the image
    width, height = im.size
    # Get the year from the file name
    line = filepath.split('/')[-1]
    # Crop the image based on different conditions
    if width == 715 and height == 553:
        cropped_im = im.crop((30, 100, 714, 417))
        width, height = cropped_im.size
    elif width == 572 and height == 553:
        cropped_im = im.crop((24, 79, 571, 333))
        width, height = cropped_im.size
    elif width == 800 and height == 441:
        if line[11:15] in ['2017', '2018', '2019', '2020', '2021']:
            cropped_im = im.crop((89, 24, 776, 379))
            width, height = cropped_im.size
        else:
            cropped_im = im.crop((76, 20, 779, 383))
            width, height = cropped_im.size
    else:
        # If the image size does not match any of the conditions, return None
        return None

    # Get the coordinates of the black pixels in the cropped image
    black_pixels = get_black_pixel_coords(cropped_im)
    # Sort the coordinates by the first element
    original = sort_by_first_element(black_pixels)
    # Process the coordinates to remove noise and outliers
    black_pixels = process_input(original, height)
    # Remove duplicate x values from the coordinates
    black_pixels = remove_duplicate_x(black_pixels)
    # Fill the gaps in the coordinates by interpolating the points
    fitted = fill_gaps(black_pixels, 1000)

    # Normalize the y values by dividing by the height
    originaly = [y/height for [x, y] in original]
    # Scale the x values by multiplying by 1000 and dividing by the maximum x value
    originalx = [x for [x, y] in original]
    originalx = (1000/originalx[-1])*np.array(originalx)
    # Normalize the y values of the interpolated points by dividing by the height
    fitted = [y/height for [x, y] in fitted]
    # Return the original and interpolated coordinates
    return originaly, originalx, fitted

# Define a function to extract the red channel from an image
def red(image):
    # Split the image into blue, green, and red channels
    blue, green, red = cv2.split(image)
    # Create a mask using the red channel
    _, red_mask = cv2.threshold(red, 205, 255, cv2.THRESH_BINARY)

    # Create a mask for the background
    background_mask = cv2.bitwise_not(red_mask)

    # Set the background to white
    background = np.full(image.shape, (255, 255, 255), dtype=np.uint8)
    background = cv2.bitwise_and(background, background, mask=background_mask)

    # Extract only the red pixels
    red_pixels = cv2.bitwise_and(image, image, mask=red_mask)

    # Combine the red pixels with the white background
    result = cv2.add(red_pixels, background)
    # Convert the result from BGR to RGB
    img = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
    # Convert the result to a PIL image
    imag = Image.fromarray(img)
    return imag
