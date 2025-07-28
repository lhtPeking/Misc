import os
import numpy as np
import tifffile
import matplotlib.pyplot as plt
from skimage.draw import polygon
from tkinter import filedialog
from tkinter import Tk
from matplotlib.widgets import PolygonSelector
import shutil

# draw polygon mask
class PolygonMaskDrawer:
    def __init__(self, ax, image):
        self.ax = ax
        self.image = image
        self.mask = np.zeros_like(image, dtype=bool) # info part
        self.done = False
        self.selector = PolygonSelector(ax, self.onselect, props=dict(color='white', linewidth=2)) # automatically executed when initiating

    def onselect(self, verts):
        rr, cc = polygon([v[1] for v in verts], [v[0] for v in verts], self.image.shape)
        # rr and cc are row & column indices of pixels inside the polygon
        self.mask[rr, cc] = True
        self.done = True
        plt.close()

# gray image scale function
def gray_imagesc(image, ax=None, vmin=None, vmax=None):
    im = ax.imshow(image, cmap='gray', vmin=vmin, vmax=vmax, interpolation='none')
    plt.colorbar(im, ax=ax)
    return im


def remove_eye_from_stacks(files):
    for idx in range(len(files)):
        name = files[idx]
        print(f'Processing: {name}')

        # Read & Copy
        img = tifffile.imread(name)  # img.shape = (Z, H, W)
        img1 = img.copy()

        avg_img = np.mean(img, axis=0)

        # Adjust display details
        clim_min = np.min(avg_img)
        clim_max = clim_min * 1.05

        num_polygons = int(input("How many polygons to plot (0/1/2)?"))
        if num_polygons > 0:
            mask_total = np.zeros_like(avg_img, dtype=bool)

            for _ in range(num_polygons):
                fig_mask, ax_mask = plt.subplots()
                gray_imagesc(avg_img, ax_mask, vmin=clim_min, vmax=clim_max)
                drawer = PolygonMaskDrawer(ax_mask, avg_img)
                plt.title("Draw polygon and close window")
                plt.show()

                # mask update
                if drawer.done:
                    mask_total |= drawer.mask

            # apply total mask
            for z in range(img.shape[0]):
                img1[z][mask_total] = np.min(avg_img)

            # display image after mask, clim([min, max*0.95])
            avg_img1 = np.mean(img1, axis=0)
            clim_min2 = np.min(avg_img1)
            clim_max2 = np.max(avg_img1) * 0.95
            fig, ax = plt.subplots(1, 2, figsize=(12, 6))
            gray_imagesc(avg_img, ax[0], vmin=clim_min, vmax=clim_max)
            gray_imagesc(avg_img1, ax[1], vmin=clim_min2, vmax=clim_max2)
            ax[0].set_title("Original Average")
            ax[1].set_title("Modified Average")
            plt.show()

            # save image
            img1 = img1.astype(np.uint16)
            name = name.replace(".tif", "_eye_removed.tif")
            tifffile.imwrite(name, img1)
            print(f'Saved modified file: {name}')
        else:
            print("Skipping file.")


if __name__ == "__main__":
    root = Tk()
    root.withdraw()
    folder_path = filedialog.askdirectory(title="Please Select the 'rawdata' Folder")
    root.destroy()

    file_paths = [os.path.join(folder_path, f)
                  for f in os.listdir(folder_path)
                  if f.lower().endswith('.tif')]

    remove_eye_from_stacks(file_paths)
