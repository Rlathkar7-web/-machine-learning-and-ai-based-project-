import warnings
import numpy as np
import imageio
import matplot
from matplotlib import pyplot as plt

warnings.filterwarnings("ignore")

photo_data_original = imageio('data/sd-3layers.jpg')
print(type(photo_data_original))
print("Shape (H, W, Channels):", photo_data_original.shape)
print("Total pixels:", photo_data_original.size)
print("Pixel range:", photo_data_original.min(), "to", photo_data_original.max())
print("Average pixel value:", photo_data_original.mean())

# Show original image
plt.figure(figsize=(15, 15))
plt.imshow(photo_data_original)
plt.title("Original WIFIRE Satellite Image - Student Practice")
plt.show()

#1: Single pixel access & edit
print("\n--- Practice 1: Pixel access ---")
print("Pixel [150, 250] (R,G,B):", photo_data_original[150, 250])
print("Green value only:", photo_data_original[150, 250, 1])

photo_data = photo_data_original.copy()
photo_data[150, 250] = 0
plt.figure(figsize=(10, 10))
plt.imshow(photo_data)
plt.title("Practice 1: Pixel [150,250] set to black")
plt.show()

#2: Range editing – Green layer (Slope)
photo_data = photo_data_original.copy()
photo_data[200:800, :, 1] = 255   # Green channel (index 1) = full intensity
plt.figure(figsize=(10, 10))
plt.imshow(photo_data)
plt.title("Practice 2: Green layer (rows 200–800) set to 255")
plt.show()

#3: Full white block
photo_data = photo_data_original.copy()
photo_data[200:800, :] = 255
plt.figure(figsize=(10, 10))
plt.imshow(photo_data)
plt.title("Practice 3: Rows 200–800 set to WHITE (255)")
plt.show()

#4: Full black block
photo_data = photo_data_original.copy()
photo_data[200:800, :] = 0
plt.figure(figsize=(10, 10))
plt.imshow(photo_data)
plt.title("Practice 4: Rows 200–800 set to BLACK (0)")
plt.show()

#5: Low-value filter
photo_data = photo_data_original.copy()
low_value_filter = photo_data < 100
photo_data[low_value_filter] = 0
plt.figure(figsize=(10, 10))
plt.imshow(photo_data)
plt.title("Practice 5: All pixels < 100 set to 0")
plt.show()

#6: Diagonal line pattern
photo_data = photo_data_original.copy()
rows_range = np.arange(photo_data.shape[0])
cols_range = rows_range
photo_data[rows_range, cols_range] = 255
plt.figure(figsize=(15, 15))
plt.imshow(photo_data)
plt.title("Practice 6: Diagonal white line (row = col)")
plt.show()

#7: Circular mask
photo_data = photo_data_original.copy()
total_rows, total_cols, _ = photo_data.shape
X, Y = np.ogrid[:total_rows, :total_cols]
center_row, center_col = total_rows / 2, total_cols / 2
dist_from_center = (X - center_row)**2 + (Y - center_col)**2
radius = (total_rows / 2)**2
circular_mask = (dist_from_center > radius)
photo_data[circular_mask] = 0
plt.figure(figsize=(15, 15))
plt.imshow(photo_data)
plt.title("Practice 7: Circular mask (outside → black)")
plt.show()

#8: Upper-half circular mask
photo_data = photo_data_original.copy()
half_upper = X < center_row
half_upper_mask = np.logical_and(half_upper, circular_mask)
photo_data[half_upper_mask] = 255
plt.figure(figsize=(15, 15))
plt.imshow(photo_data)
plt.title("Practice 8: Upper-half circle set to white")
plt.show()

#9: Color-specific masks
photo_data = photo_data_original.copy()
red_mask   = photo_data[:, :, 0] < 150
photo_data[red_mask] = 0
plt.figure(figsize=(15, 15))
plt.imshow(photo_data)
plt.title("Practice 9a: Red channel < 150 → 0")
plt.show()

photo_data = photo_data_original.copy()
green_mask = photo_data[:, :, 1] < 150
photo_data[green_mask] = 0
plt.figure(figsize=(15, 15))
plt.imshow(photo_data)
plt.title("Practice 9b: Green channel < 150 → 0")
plt.show()

photo_data = photo_data_original.copy()
blue_mask  = photo_data[:, :, 2] < 150
photo_data[blue_mask] = 0
plt.figure(figsize=(15, 15))
plt.imshow(photo_data)
plt.title("Practice 9c: Blue channel < 150 → 0")
plt.show()

# 10: Composite mask
photo_data = photo_data_original.copy()
red_mask   = photo_data[:, :, 0] < 150
green_mask = photo_data[:, :, 1] > 100
blue_mask  = photo_data[:, :, 2] < 100
final_mask = np.logical_and(red_mask, np.logical_and(green_mask, blue_mask))
photo_data[final_mask] = 0
plt.figure(figsize=(15, 15))
plt.imshow(photo_data)
plt.title("Practice 10: Composite mask (Red<150 & Green>100 & Blue<100)")
plt.show()

