from PIL import Image
import numpy as np
import base64
import io


def validate_single_image(img_path: str, max_size: tuple = (1024, 1024)) -> bool:
    """
    Validate a single image: check if it can be opened and converted to RGB,
    and if it's not too large.

    Args:
        img_path (str): Path to the image file
        max_size (tuple): Maximum (width, height) allowed

    Returns:
        bool: True if image is valid, False otherwise

    Example:
        >>> if validate_single_image("photo.jpg"):
        ...     # Process the image
    """
    try:
        with Image.open(img_path) as img:
            # Check file size
            width, height = img.size
            if width > max_size[0] or height > max_size[1]:
                print(f"⚠️  Image trop grande détectée et ignorée : {img_path}")
                return False

            # Test conversion to RGB (validates format)
            img.convert("RGB")
            return True

    except (IOError, SyntaxError) as e:
        print(f"⚠️  Fichier non valide détecté et ignoré : {img_path}")
        return False


def get_image_dimensions(img_path: str) -> tuple:
    """
    Get the dimensions of an image.

    Args:
        img_path (str): Path to the image.

    Returns:
        tuple: (width, height) of the image.
    """
    with Image.open(img_path) as original_image:
        return original_image.size


def decode_base64_mask(base64_string: str, width: int, height: int) -> np.ndarray:
    """
    Decode a base64-encoded mask into a NumPy array.

    Args:
        base64_string (str): Base64-encoded mask.
        width (int): Target width.
        height (int): Target height.

    Returns:
        np.ndarray: Single-channel mask array.
    """
    mask_data = base64.b64decode(base64_string)
    mask_image = Image.open(io.BytesIO(mask_data))
    mask_array = np.array(mask_image)
    if len(mask_array.shape) == 3:
        mask_array = mask_array[:, :, 0]  # Take first channel if RGB
    mask_image = Image.fromarray(mask_array).resize(
        (width, height), Image.Resampling.NEAREST
    )
    return np.array(mask_image)


def create_masks(
    results: list, width: int, height: int, class_mapping: dict
) -> np.ndarray:
    """
    Combine multiple class masks into a single segmentation mask.

    Args:
        results (list): List of dictionaries with 'label' and 'mask' keys.
        width (int): Target width.
        height (int): Target height.
        class_mapping (dict): Dictionary mapping class names to IDs.

    Returns:
        np.ndarray: Combined segmentation mask with class indices.
    """
    combined_mask = np.zeros(
        (height, width), dtype=np.uint8
    )  # Initialize with Background (0)

    # Process non-Background masks first
    for result in results:
        label = result["label"]
        class_id = class_mapping.get(label, 0)
        if class_id == 0:  # Skip Background
            continue
        mask_array = decode_base64_mask(result["mask"], width, height)
        combined_mask[mask_array > 0] = class_id

    # Process Background last to ensure it doesn't overwrite other classes unnecessarily
    for result in results:
        if result["label"] == "Background":
            mask_array = decode_base64_mask(result["mask"], width, height)
            combined_mask[mask_array > 0] = 0  # Class ID for Background is 0

    return combined_mask
