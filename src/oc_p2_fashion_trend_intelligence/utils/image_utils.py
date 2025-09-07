import os
from PIL import Image
from matplotlib.gridspec import GridSpec
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
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


def search_for_ground_truth_mask_path(original_image_path, ground_truth_mask_dir):
    """
    Recherche le masque de vérité terrain correspondant à une image originale.

    Args:
        original_image_path (str): Chemin vers l'image originale (ex: /path/images/image_01.jpg).
        ground_truth_mask_dir (str): Répertoire contenant les masques de vérité terrain.

    Returns:
        str: Chemin vers le masque de vérité terrain s'il existe, sinon None.
    """

    # Extraire le nom du fichier sans extension
    filename = os.path.basename(original_image_path)
    name_without_ext = os.path.splitext(filename)[0]

    # Remplacer "image_" par "mask_" dans le nom
    if name_without_ext.startswith("image_"):
        mask_name = name_without_ext.replace("image_", "mask_", 1)

        # Chercher le masque avec différentes extensions possibles
        possible_extensions = [".png", ".jpg", ".jpeg", ".bmp"]
        for ext in possible_extensions:
            mask_path = os.path.join(ground_truth_mask_dir, mask_name + ext)
            if os.path.exists(mask_path):
                return mask_path

    return None


def show_image_and_masks(
    original_image_path,
    ground_truth_mask_dir,
    class_mapping,
    palette,
    predicted_mask_dir=None,
    predicted_mask_array=None,
):
    """Affiche l'image originale, le masque prédit et le masque de vérité avec une légende des classes.

    Cette fonction affiche dans une figure matplotlib :
    - L'image originale
    - Le masque prédit (soit à partir d'un fichier, soit d'un array numpy fourni)
    - Le masque de vérité si disponible
    - Une légende verticale des classes et couleurs

    Args:
        original_image_path (str): Chemin vers l'image originale.
        ground_truth_mask_dir (str): Dossier contenant les masques de vérité. La fonction cherche le masque correspondant au nom de l'image.
        class_mapping (dict): Dictionnaire {nom_de_classe: id_de_classe}.
        palette (dict): Dictionnaire {id_de_classe: [R, G, B]} pour colorer les masques.
        predicted_mask_dir (str, optional): Dossier contenant les masques prédits. Ignoré si predicted_mask_array est fourni. Defaults to None.
        predicted_mask_array (np.ndarray, optional): Masque prédit sous forme de tableau numpy (indices de classes). Si fourni, la lecture depuis predicted_mask_dir est ignorée. Defaults to None.

    Raises:
        ValueError: Si ni predicted_mask_dir ni predicted_mask_array ne sont fournis.
    """

    original = Image.open(original_image_path).convert("RGB")
    ground_truth_mask_path = search_for_ground_truth_mask_path(
        original_image_path, ground_truth_mask_dir
    )

    # Palette
    num_classes = max(class_mapping.values()) + 1
    palette_array = np.zeros((num_classes, 3), dtype=np.uint8)
    for cid, color in palette.items():
        palette_array[cid] = color

    # --- Masque prédit : soit passé en numpy, soit chargé depuis le dossier ---
    if predicted_mask_array is not None:
        mask_array = predicted_mask_array
    elif predicted_mask_dir is not None:
        mask_img = Image.open(
            os.path.join(
                predicted_mask_dir,
                os.path.basename(original_image_path).replace("image", "mask"),
            )
        )
        mask_array = np.array(mask_img)
    else:
        raise ValueError("Il faut soit predicted_mask_dir soit predicted_mask_array")

    color_predicted_mask = palette_array[mask_array]

    # --- 🔹 Resize du masque de vérité à la taille de l'image originale ---
    if ground_truth_mask_path:
        ground_truth_mask = palette_array[np.array(Image.open(ground_truth_mask_path))]

    # --- Figure et GridSpec ---
    fig = plt.figure(figsize=(15, 6))
    gs = GridSpec(1, 4, figure=fig, width_ratios=[0.3, 0.3, 0.3, 0.1], wspace=0.05)

    # Colonne 0 : image originale
    ax_img = fig.add_subplot(gs[0])
    ax_img.imshow(original)
    ax_img.set_xticks([])
    ax_img.set_yticks([])
    ax_img.set_xlabel("Image Originale", fontsize=12)

    # Colonne 1 : masque prédit
    ax_pred = fig.add_subplot(gs[1])
    ax_pred.imshow(color_predicted_mask)
    ax_pred.set_xticks([])
    ax_pred.set_yticks([])
    ax_pred.set_xlabel("Masque prédit", fontsize=12)

    # Colonne 2 : masque vérité
    ax_gt = fig.add_subplot(gs[2])
    if ground_truth_mask_path:
        ax_gt.imshow(ground_truth_mask)
        ax_gt.set_xticks([])
        ax_gt.set_yticks([])
        ax_gt.set_xlabel("Masque de vérité", fontsize=12)
    else:
        ax_gt.text(
            0.5,
            0.5,
            "Pas de masque trouvé",
            ha="center",
            va="center",
            fontsize=12,
            color="red",
        )

    # Colonne 3 : légende verticale
    ax_legend = fig.add_subplot(gs[3])
    ax_legend.axis("off")
    patches = []
    for class_name, cid in class_mapping.items():
        rgb = palette[cid]
        color = tuple(np.array(rgb) / 255.0)
        patches.append(mpatches.Patch(color=color, label=class_name))
    ax_legend.legend(
        handles=patches,
        loc="center",
        frameon=True,
        facecolor="whitesmoke",
        edgecolor="lightgrey",
        fontsize=11,
        title="Classes",
        handletextpad=0.5,
    )

    # Titre global
    plt.suptitle(
        f"Comparaison des masques pour {os.path.basename(original_image_path)}",
        fontsize=14,
    )
    plt.show()


def eval_segmentation_simple(path_true: str, path_pred: str, class_mapping, palette):
    """
    Évalue la segmentation et affiche directement :
        - Accuracy globale
        - mIoU et Dice moyens
        - Graphique IoU/Dice par classe avec couleur spécifique à chaque classe
        - Valeurs exactes affichées au-dessus des barres

    Args:
        path_true (str): chemin vers le masque de vérité
        path_pred (str): chemin vers le masque prédit
        class_mapping (dict): {nom_classe: id_classe}
        palette (dict): {id_classe: [R, G, B]}
    """
    # Lecture des masques et conversion en float
    y_true = np.array(Image.open(path_true)).astype(float)
    y_pred = np.array(Image.open(path_pred)).astype(float)

    accuracy = float(np.mean(y_true == y_pred))
    classes = np.unique(np.concatenate([np.unique(y_true), np.unique(y_pred)]))
    id_to_name = {v: k for k, v in class_mapping.items()}

    # Calcul métriques par classe
    rows = []
    for c in classes:
        pred_c = y_pred == c
        true_c = y_true == c
        intersection = float(np.logical_and(pred_c, true_c).sum())
        union = float(np.logical_or(pred_c, true_c).sum())
        iou = float(np.real(intersection / union)) if union > 0 else 1.0
        dice = (
            float(np.real(2 * intersection / (pred_c.sum() + true_c.sum())))
            if (pred_c.sum() + true_c.sum()) > 0
            else 1.0
        )
        rows.append(
            {
                "class_name": id_to_name.get(int(c), "Unknown"),
                "class_id": int(c),
                "IoU": iou,
                "Dice": dice,
            }
        )

    df = pd.DataFrame(rows)
    df["IoU"] = df["IoU"].astype(float)
    df["Dice"] = df["Dice"].astype(float)
    mIoU = df["IoU"].mean()
    mean_dice = df["Dice"].mean()

    print(f"✅ Accuracy globale : {accuracy:.4f}")
    print(f"✅ mIoU : {mIoU:.4f}")
    print(f"✅ Dice moyen : {mean_dice:.4f}")

    # Graphique Matplotlib avec couleurs fixes par classe
    df_sorted = df.sort_values("IoU", ascending=False)
    x = np.arange(len(df_sorted))
    width = 0.30
    gap = 0.1
    hatch = "."

    plt.figure(figsize=(10, 6))

    for i, row in enumerate(df_sorted.itertuples()):
        # Récupérer la couleur depuis la palette
        rgb = palette.get(row.class_id, [51, 51, 51])  # défaut gris si absent
        color = tuple(np.array(rgb) / 255.0)

        # IoU opaque
        plt.bar(
            x[i] - width / 2 - gap,
            float(getattr(row, "IoU")),
            width,
            color=color,
            edgecolor="black",
            linewidth=1,
            label="IoU" if i == 0 else "",
        )
        plt.text(
            x[i] - width / 2 - gap,
            float(getattr(row, "IoU")),
            f"{row.IoU:.2f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

        # Dice semi-transparent
        plt.bar(
            x[i] + width / 2 + gap,
            float(getattr(row, "Dice")),
            width,
            color=color,
            hatch=hatch,
            edgecolor="black",
            linewidth=1,
            label="Dice" if i == 0 else "",
        )
        plt.text(
            x[i] + width / 2 + gap,
            float(getattr(row, "Dice")),
            f"{row.Dice:.2f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    plt.xticks(x, df_sorted["class_name"].tolist(), rotation=45, ha="right")
    plt.ylabel("Score")
    plt.ylim(0, 1.1)
    plt.title("IoU et Dice par classe")
    plt.legend(
        handles=[
            mpatches.Patch(color="lightgrey", label="IoU"),
            mpatches.Patch(facecolor="lightgrey", hatch=hatch, label="Dice"),
        ],
        loc="upper right",
    )
    plt.tight_layout()
    plt.show()
