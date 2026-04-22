import matplotlib.pyplot as plt
from PIL import Image, ImageDraw


def visualize_points_on_image(pil_img, pixel_points, point_color='red', title=None):
    """
    Plots (x, y) points on a PIL image.

    Parameters:
        pil_img (PIL.Image): Input image
        pixel_points (list of (x, y)): Points in pixels
        point_color (str): Circle color
        point_radius (int or None): Radius in pixels. If None, scales with image size.
        title (str): Optional plot title
    """
    img_width, img_height = pil_img.size
    dpi = 100  # dots per inch for display
    
    # Auto-scale radius if not provided
    min_dim = min(img_width, img_height)
    point_radius = max(min_dim * 0.01, 2)  # 1% of smaller dim, min 2px

    # Create figure sized to match image dimensions
    fig_width = img_width / dpi
    fig_height = img_height / dpi
    fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=dpi)
    ax.imshow(pil_img)

    # Draw each point
    for x_px, y_px in pixel_points:
        circle = plt.Circle((x_px, y_px), point_radius, color=point_color, fill=True)
        ax.add_patch(circle)

    # Remove axes ticks and margins
    ax.axis('off')
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

    if title:
        ax.set_title(title)

    plt.show()

