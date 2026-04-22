import cv2
import os
os.environ["DECORD_VERBOSE"] = "0"
from decord import VideoReader, cpu
import difflib
from PIL import Image


def get_frame_by_id(video_path, frame_id):
    try:
        vr = VideoReader(video_path, ctx=cpu(0))
        frame = vr[frame_id].asnumpy()
    except Exception as e:
        print(f"Decord failed: {e}, falling back to OpenCV")
        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
        ret, frame = cap.read()
        cap.release()
        if not ret:
            raise RuntimeError(f"Could not read frame {frame_id} from {video_path}")
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return Image.fromarray(frame)


def get_every_nth_frame(video_path, n):
    """
    Load video and return every n-th frame as PIL.Image in RGB mode.

    Args:
        video_path (str): Path to the video file.
        n (int): Interval to sample frames.

    Returns:
        List of (frame_index, PIL.Image)
    """
    vr = VideoReader(video_path, ctx=cpu(0))
    total_frames = len(vr)

    frames = []
    for i in range(0, total_frames, n):
        frame = vr[i].asnumpy()  # (H, W, C)
        img = Image.fromarray(frame).convert('RGB')
        frames.append((i, img))

    return frames



def match_to_class(prediction: str, class_list=None, threshold=0.6):
    """
    Match a model's prediction to the closest class using string similarity.

    Args:
        prediction (str): Model output string.
        class_list (List[str]): List of valid classes.
        threshold (float): Minimum similarity threshold for a valid match.

    Returns:
        str: Closest matched class, or 'unknown' if below threshold.
    """
    if class_list is None:
        class_list = ['t-bone', 'sideswipe', 'rear-end', 'single-vehicle', 'head-on']

    prediction = prediction.lower().strip()

    # Get best match using difflib
    best_match = difflib.get_close_matches(prediction, class_list, n=1, cutoff=threshold)
    
    if best_match:
        return best_match[0]
    else:
        return "unknown"



def crop_with_bbox(image: Image.Image, bbox: tuple, padding_percent: float) -> Image.Image:
    """
    Crop image based on bounding box with padding.

    Args:
        image (PIL.Image): Input image.
        bbox (tuple): Bounding box (x_min, y_min, x_max, y_max).
        padding_percent (float): Percentage to expand bbox (e.g., 0.1 for 10%).

    Returns:
        PIL.Image: Cropped image.
    """
    width, height = image.size
    x_min, y_min, x_max, y_max = bbox

    # Compute padding
    pad_w = (x_max - x_min) * padding_percent
    pad_h = (y_max - y_min) * padding_percent

    # Expand and clip
    x_min = max(0, int(x_min - pad_w))
    y_min = max(0, int(y_min - pad_h))
    x_max = min(width, int(x_max + pad_w))
    y_max = min(height, int(y_max + pad_h))

    return image.crop((x_min, y_min, x_max, y_max))


def crop_around_point(image: Image.Image, point: tuple, crop_percent: float) -> Image.Image:
    """
    Crop image around a point with percentage of image size.

    Args:
        image (PIL.Image): Input image.
        point (tuple): (x, y) center point.
        crop_percent (float): Fraction of image size to crop (0 < crop_percent ≤ 1).

    Returns:
        PIL.Image: Cropped image.
    """
    width, height = image.size
    cx, cy = point

    crop_w = int(width * crop_percent)
    crop_h = int(height * crop_percent)

    x_min = max(0, cx - crop_w // 2)
    y_min = max(0, cy - crop_h // 2)
    x_max = min(width, cx + crop_w // 2)
    y_max = min(height, cy + crop_h // 2)

    return image.crop((x_min, y_min, x_max, y_max))


def crop_around_point_pixels(image: Image.Image, point: tuple, crop_size: tuple) -> Image.Image:
    """
    Crop image around a point with explicit pixel size.

    Args:
        image (PIL.Image): Input image.
        point (tuple): (x, y) center point.
        crop_size (tuple): (width, height) of crop in pixels.

    Returns:
        PIL.Image: Cropped image.
    """
    width, height = image.size
    cx, cy = point
    crop_w, crop_h = crop_size

    x_min = max(0, cx - crop_w // 2)
    y_min = max(0, cy - crop_h // 2)
    x_max = min(width, cx + crop_w // 2)
    y_max = min(height, cy + crop_h // 2)

    return image.crop((x_min, y_min, x_max, y_max))
