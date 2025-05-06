import traceback
from matplotlib.colors import BoundaryNorm, ListedColormap
import numpy as np
import rasterio
from rasterio.enums import ColorInterp
from rasterio.transform import from_origin
import os
import matplotlib.pyplot as plt
from skimage.filters import threshold_multiotsu
from skimage.transform import resize
from sklearn.cluster import KMeans
from collections import Counter
import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling
from scipy.ndimage import uniform_filter, median_filter, gaussian_filter
from skimage.morphology import opening, closing, disk, remove_small_objects, binary_opening, binary_closing
from skimage.measure import label
from rasterio.windows import Window

# Define SAR image paths
sar_image_paths = [
    #relative paths of all the images like relative_path/2020.tif and so on..
]

results = {}

def mark_zero_edges_as_nan(img):
    """
    Marks the outer rows and columns that contain only zeros as NaN.
    Scans from all four directions and stops at the first non-zero value.
    """
    h, w = img.shape

    # Top
    top = 0
    for i in range(h):
        if np.any(img[i, :] != 0):
            top = i
            break

    # Bottom
    bottom = h
    for i in range(h - 1, -1, -1):
        if np.any(img[i, :] != 0):
            bottom = i + 1
            break

    # Left
    left = 0
    for j in range(w):
        if np.any(img[:, j] != 0):
            left = j
            break

    # Right
    right = w
    for j in range(w - 1, -1, -1):
        if np.any(img[:, j] != 0):
            right = j + 1
            break

    # Mark the identified edges as NaN
    img[:top, :] = np.nan
    img[bottom:, :] = np.nan
    img[:, :left] = np.nan
    img[:, right:] = np.nan

    return img




def morphological_filter(image, radius=2):
    selem = disk(radius)
    return closing(opening(image, selem), selem)

def gaussian_smoothing(image, sigma=1.0):
    """Applies Gaussian smoothing to reduce noise in VV_db images."""
    return gaussian_filter(image, sigma=sigma)

def clean_flood_mask(mask, radius=3, min_size=300):
    """Apply morphological operations and remove small speckles from the mask."""
    selem = disk(radius)
    mask = binary_closing(mask, selem)
    mask = binary_opening(mask, selem)
    labeled = label(mask)
    cleaned = remove_small_objects(labeled, min_size=min_size)
    return cleaned > 0

def compute_flood_mask(tiff_path, pre_band_index, post_band_index, alpha=0.9, min_size=100, use_gaussian=False):
    with rasterio.open(tiff_path) as src:
        pre = src.read(pre_band_index).astype('float32')
        post = src.read(post_band_index).astype('float32')


    pre = (pre - pre.min()) / (pre.max() - pre.min())
    post = (post - post.min()) / (post.max() - post.min())

    pre = morphological_filter(pre)
    post = morphological_filter(post)

    # Optional Gaussian smoothing
    if use_gaussian:
        pre = gaussian_smoothing(pre, sigma=1.0)
        post = gaussian_smoothing(post, sigma=1.0)

    # Local mean filtering
    u1 = uniform_filter(pre, size=3)
    u2 = uniform_filter(post, size=3)

    # Difference computation
    Xm = 1 - np.minimum(u1 / (u2 + 1e-5), u2 / (u1 + 1e-5))
    Xd = np.abs(pre - post)
    D = alpha * Xm + (1 - alpha) * Xd
    D_filtered = median_filter(D, size=3)

    # Clustering
    kmeans = KMeans(n_clusters=2, random_state=42).fit(D_filtered.reshape(-1, 1))
    clustered = kmeans.labels_.reshape(D_filtered.shape)

    # Determine which cluster is flooded (higher D value)
    means = [np.mean(D_filtered[clustered == i]) for i in range(2)]
    flood_label = np.argmax(means)
    flood_mask = (clustered == flood_label)

    # Clean flood mask further
    flood_mask_final = clean_flood_mask(flood_mask, radius=3, min_size=min_size)

    return flood_mask_final, D_filtered, pre, post, Xm, Xd


def kmeans_threshold(image, n_clusters=3):
    """ Apply K-Means clustering to segment the image into multiple classes. """
    if image.size == 0:
        raise ValueError("Input image is empty!")
    
    image = np.nan_to_num(image)
    flat_image = image.reshape(-1, 1)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(flat_image)
    return labels.reshape(image.shape)

def otsu_threshold(image, levels=3):
    """ Apply multi-level Otsu thresholding to segment the image. """
    thresholds = threshold_multiotsu(image, classes=levels)
    return np.digitize(image, bins=thresholds)

def map_clusters_to_land_water_urban(segmented_image):
    """ Map cluster values to land, water, and urban vegetation based on frequency. """
    pixel_counts = Counter(segmented_image.flatten())
    sorted_clusters = sorted(pixel_counts, key=pixel_counts.get, reverse=True)
    if len(sorted_clusters) < 3:
        raise ValueError("Not enough unique clusters to map (expected 3).")
    
    labeled_image = np.zeros_like(segmented_image)
    labeled_image[segmented_image == sorted_clusters[0]] = 0  # Land
    labeled_image[segmented_image == sorted_clusters[1]] = 1  # Water
    labeled_image[segmented_image == sorted_clusters[2]] = 2  # Urban
    return labeled_image

# Store flood classified images and sizes
classified_images = []
image_sizes = []

for sar_image_path in sar_image_paths:
    year = os.path.basename(sar_image_path).split('_')[0]
    print(f"Processing {year}...")

    with rasterio.open(sar_image_path) as dataset:
        RI = dataset.read(7).astype(np.float32)
        NCI = dataset.read(8).astype(np.float32)
        vv_pre = dataset.read(6).astype(np.float32)
        vv_post = dataset.read(5).astype(np.float32)

    RI[np.isnan(RI)] = np.nanmedian(RI)
    NCI[np.isnan(NCI)] = np.nanmedian(NCI)

    # No resizing needed for individual processing
    RI_resized = RI
    NCI_resized = NCI

    for img in [RI_resized, NCI_resized, vv_pre, vv_post]:
        img[:] = (img - np.min(img)) / (np.max(img) - np.min(img) + 1e-8)
    
    p2, p98 = np.percentile(RI_resized, (2, 98))
    RI_resized = np.clip(RI_resized, p2, p98)
    
    p2, p98 = np.percentile(NCI_resized, (2, 98))
    NCI_resized = np.clip(NCI_resized, p2, p98)

    dark_threshold = np.percentile(RI_resized, 10)
    RI_binary = (RI_resized <= dark_threshold).astype(float)
    dark_threshold_nci = np.percentile(NCI_resized, 15)
    NCI_binary = (NCI_resized <= dark_threshold_nci).astype(float)
    
    vv_pre_kmeans = map_clusters_to_land_water_urban(kmeans_threshold(vv_pre))
    vv_post_kmeans = map_clusters_to_land_water_urban(kmeans_threshold(vv_post))
    morphological_binary, D_filtered, pre, post, Xm, Xd = compute_flood_mask(
    sar_image_path, 6, 5, alpha=0.8,use_gaussian=True)
    
    flood_affected_kmeans = np.bitwise_xor(vv_pre_kmeans, vv_post_kmeans).astype(float)
    # flood_affected_otsu = np.bitwise_xor(vv_pre_otsu, vv_post_otsu).astype(float)

    final_flood_image = (
        0.5 * NCI_binary +
        0.3 * RI_binary +
        0.1 * flood_affected_kmeans +
        0.1 * morphological_binary
    )
    
    flood_classified = (final_flood_image > 0.5).astype(int)

    results[year] = {
        "flood_classified": flood_classified
    }

    results[year] = {
    "flood_classified": flood_classified,
    "nci_binary": NCI_binary,
    "ri_binary": RI_binary,
    "flood_affected_morphological": morphological_binary,
    "flood_affected_kmeans": flood_affected_kmeans
    }

    classified_images.append(flood_classified)
    image_sizes.append(flood_classified.shape)

# Find the minimum height and width among all classified images
min_height = min(size[0] for size in image_sizes)
min_width = min(size[1] for size in image_sizes)

# Resize all classified images to the minimum dimensions and sum them up
resized_classified_images = [
    resize(img, (min_height, min_width), anti_aliasing=True, preserve_range=True).astype(int)
    for img in classified_images
]
net_final_image = np.sum(resized_classified_images, axis=0)


num_images = len(resized_classified_images)

plt.figure(figsize=(5 * num_images, 5)) 

for i, img in enumerate(resized_classified_images):
    year = os.path.basename(sar_image_paths[i]).split('_')[0]
    plt.subplot(1, num_images, i + 1)
    plt.imshow(img, cmap='gray', vmin=0, vmax=1)
    plt.title(f"Flood Mask {year}")
    plt.axis('off')

plt.tight_layout()
plt.show()


fig, axes = plt.subplots(1, len(sar_image_paths), figsize=(len(sar_image_paths) * 5, 5))
for i, sar_image_path in enumerate(sar_image_paths):
    year = os.path.basename(sar_image_path).split('_')[0]
   
    ax = axes[i] if len(sar_image_paths) > 1 else axes
    ax.imshow(results[year]["nci_binary"], cmap='Blues')
    ax.set_title(f"{year} - NCI Binary")
    ax.axis("off")
plt.tight_layout()
plt.show()

# --- Plotting RI Binary Images ---
fig, axes = plt.subplots(1, len(sar_image_paths), figsize=(len(sar_image_paths) * 5, 5))
for i, sar_image_path in enumerate(sar_image_paths):
    year = os.path.basename(sar_image_path).split('_')[0]
    ax = axes[i] if len(sar_image_paths) > 1 else axes
    ax.imshow(results[year]["ri_binary"], cmap='Greens')
    ax.set_title(f"{year} - RI Binary")
    ax.axis("off")
plt.tight_layout()
plt.show()


fig, axes = plt.subplots(1, len(sar_image_paths), figsize=(len(sar_image_paths) * 5, 5))
for i, sar_image_path in enumerate(sar_image_paths):
    year = os.path.basename(sar_image_path).split('_')[0]
    ax = axes[i] if len(sar_image_paths) > 1 else axes
    color = ListedColormap(['#f0f7f7', '#f5675d'])  
    ax.imshow(results[year]["flood_affected_morphological"], cmap=color)
    # ax.set_title(f"{year} - Morphological Flood Diff")
    ax.axis("off")
plt.tight_layout()
plt.show()

# --- Plotting Flood Affected (KMeans) ---
fig, axes = plt.subplots(1, len(sar_image_paths), figsize=(len(sar_image_paths) * 5, 5))
for i, sar_image_path in enumerate(sar_image_paths):
    year = os.path.basename(sar_image_path).split('_')[0]
    ax = axes[i] if len(sar_image_paths) > 1 else axes
    ax.imshow(results[year]["flood_affected_kmeans"], cmap='Oranges')
    ax.set_title(f"{year} - KMeans Flood Diff")
    ax.axis("off")
plt.tight_layout()
plt.show()


if net_final_image is None or net_final_image.size == 0:
    print("Error: net_final_image is empty or None!")
else:
    output_path ="" # output path to save final flood impact tif
    reprojected_path ="" # output path to save final flood impact wgs84 tif

    print("net_final_image shape:", net_final_image.shape)
    print("Unique values in net_final_image:", np.unique(net_final_image))
    print("Original output path:", output_path)

    # Fix potential NaN issues
    net_final_image = np.nan_to_num(net_final_image)

    print("Trying to save the GeoTIFF image...")
    try:
        # Read transform and CRS from the first SAR image
        with rasterio.open(sar_image_paths[0]) as src:
            transform = src.transform
            crs = src.crs  # Original CRS

        # Ensure values are in the 0-255 range for uint8
        net_final_image = np.clip(net_final_image, 0, 255).astype(np.uint8)
        print("Net final image generated and converted to uint8.")

        # Save the initial GeoTIFF
        with rasterio.open(
            output_path,
            "w",
            driver="GTiff",
            height=net_final_image.shape[0],
            width=net_final_image.shape[1],
            count=1,  # Single-band raster
            dtype=rasterio.uint8,
            crs=crs,
            transform=transform,
        ) as dst:
            dst.write(net_final_image, 1)

        print(f"GeoTIFF successfully saved at: {output_path}")

        # Reproject to WGS84 and save new GeoTIFF
        print("Reprojecting to WGS84...")
        with rasterio.open(output_path) as src:
            transform, width, height = calculate_default_transform(
                src.crs, 'EPSG:4326', src.width, src.height, *src.bounds
            )
            kwargs = src.meta.copy()
            kwargs.update({
                'crs': 'EPSG:4326',
                'transform': transform,
                'width': width,
                'height': height
            })

            with rasterio.open(reprojected_path, 'w', **kwargs) as dst:
                for i in range(1, src.count + 1):
                    reproject(
                        source=rasterio.band(src, i),
                        destination=rasterio.band(dst, i),
                        src_transform=src.transform,
                        src_crs=src.crs,
                        dst_transform=transform,
                        dst_crs='EPSG:4326',
                        resampling=Resampling.nearest
                    )

        print(f"Reprojected GeoTIFF successfully saved at: {reprojected_path}")
        input_path ="" # input path to read final flood impact wgs84 tif
        output_rgb_path = ""# output path to save final flood impact rgb tif

        value_to_color = {
            0: (0, 0, 255),        # Blue
            1: (255, 255, 153),    # Light Yellow
            2: (255, 255, 0),      # Yellow
            3: (255, 165, 0),      # Orange
            4: (255, 100, 0),      # Deeper Orange
            5: (255, 0, 0),        # Red
        }

        with rasterio.open(input_path) as src:
            band = src.read(1)
            profile = src.profile
            profile.update({'count': 3, 'dtype': 'uint8'})

            r = np.zeros_like(band, dtype=np.uint8)
            g = np.zeros_like(band, dtype=np.uint8)
            b = np.zeros_like(band, dtype=np.uint8)

            for value, (red, green, blue) in value_to_color.items():
                mask = band == value
                r[mask], g[mask], b[mask] = red, green, blue

            with rasterio.open(output_rgb_path, 'w', **profile) as dst:
                dst.write(r, 1)
                dst.write(g, 2)
                dst.write(b, 3)
                dst.colorinterp = [ColorInterp.red, ColorInterp.green, ColorInterp.blue]

        print(f"RGB GeoTIFF saved at: {output_rgb_path}")


    except Exception as e:
        print("An error occurred while saving or reprojecting the GeoTIFF:")
        print(str(e))
        traceback.print_exc()