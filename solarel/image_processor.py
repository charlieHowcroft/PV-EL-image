import matplotlib.pyplot as plt
from camera_utils import fix_barrel_distortion
from tensorflow.keras.models import load_model
import time
import cv2
import numpy as np
from typing import cast, Optional
import os
import tensorflow as tf
import random

os.environ["SM_FRAMEWORK"] = "tf.keras"


def split_module_to_cells(
    model_path_module: str,
    model_path_cells: str,
    image: np.ndarray,
    toml_path: str,
    show: bool = False,
    coords: bool = False,
    barrel_fix: bool = True,
    module_crop: bool = True,
    both: bool = False,
):
    start = time.time()
    model_cells = load_model(model_path_cells, compile=False)
    model_cells.compile()

    model_module = load_model(model_path_module, compile=False)
    model_module.compile()

    end = time.time()
    print(f"model loading {end-start}")

    start_1 = time.time()

    if barrel_fix:
        start = time.time()
        image = fix_barrel_distortion(image, toml_path)
        end = time.time()
        print(f"barrel distort {end-start}")

    if show:
        plt.imshow(image)
        plt.title("Barrel distortion fix")
        plt.show()

    if module_crop:
        start = time.time()
        cropped_image = find_pv_module(image, model_module, show=show)
        end = time.time()
        print(f"module crop {end-start}")
    else:
        cropped_image = image

    cropped_image = cv2.cvtColor(cropped_image, cv2.COLOR_RGB2GRAY)
    if show:
        cropped_image_bgr = np.copy(cropped_image)
        # cropped_image_bgr = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)
        # cv2.imwrite("cropped_image_bgr.jpg", cropped_image_bgr)
        plt.imshow(cropped_image_bgr)
        plt.title("Cropped image to PV module")
        plt.xticks([])
        plt.yticks([])
        plt.show()

    start = time.time()

    cropped_cp = np.copy(cropped_image)
    cropped_cp = cv2.pyrDown(cropped_cp)
    og_height, og_width = np.shape(cropped_cp)
    height, width = np.shape(cropped_cp)
    horz_splits = int(np.ceil(width / 512) + 1)
    vert_splits = int(np.ceil(height / 512) + 1)

    row_images = split_image_into_rows(cropped_cp, vert_splits, (512, 512))
    if show:
        plt.imshow(row_images[0]["image"])
        # cv2.imwrite("row.jpg", row_images[0]["image"])
        plt.title("First row")
        plt.show()

    images = []
    for row in row_images:
        images.append(split_row_image(row, horz_splits + 1, (512, 512)))

    if show:
        plt.imshow(images[0][0]["image"])
        plt.title("First image from the row")
        plt.show()

    for row in images:
        for image in row:
            temp = image["image"]
            temp = cv2.merge((temp, temp, temp))
            temp = np.expand_dims(temp, 0)
            prediction = model_cells.predict(temp)
            predicted_img = np.argmax(prediction, axis=3)[0, :, :] * 255
            predicted_img = predicted_img.astype(np.uint8)
            predicted_img = cv2.erode(predicted_img, (20, 20))
            image["image"] = predicted_img

    if show:
        plt.imshow(images[0][0]["image"])
        # cv2.imwrite("mask_0.jpg", images[0][0]["image"])
        # cv2.imwrite("mask_1.jpg", images[0][1]["image"])
        # cv2.imwrite("mask_2.jpg", images[0][2]["image"])
        plt.title("First prediction from the first row")
        plt.show()

    masks = []
    for row in images:
        for image in row:
            temp = image["image"]
            x, y = image["x1"], image["y1"]
            blank = np.zeros((og_height, og_width))
            new = place_image(blank, temp, (y, x))
            masks.append(new)

    if show:
        # cv2.imwrite("mask_full_0.jpg", masks[0])
        # cv2.imwrite("mask_full_1.jpg", masks[1])
        # cv2.imwrite("mask_full_2.jpg", masks[2])
        plt.imshow(masks[0])
        plt.title("All mask predictions combined")
        plt.show()

    final_mask = np.zeros((og_height, og_width))
    for i, mask in enumerate(masks):
        final_mask = cv2.bitwise_or(final_mask, mask)
        if i < 3 and show:
            cv2.imwrite(f"final_mask_{i}.jpg", final_mask)

    end = time.time()
    print(f"module mask {end-start}")

    if show:
        plt.imshow(final_mask)
        # cv2.imwrite("final_mask_.jpg", final_mask)
        plt.title("All mask predictions combined")
        plt.show()

    start = time.time()
    final_mask_cp = np.copy(final_mask)
    final_mask_cp = final_mask_cp.astype(np.uint8)
    edges = cv2.Canny(final_mask_cp, 120, 180)
    kernel_dilate = np.ones((40, 40), np.uint8)
    kernel_erode = np.ones((35, 35), np.uint8)
    kernel_erode_internal = np.ones((7, 7), np.uint8)
    edges = cv2.dilate(edges, kernel_dilate)
    edges = cv2.erode(edges, kernel_erode)
    edges = erode_inside(edges, 0.1, kernel_erode_internal, show)
    edges = cv2.pyrDown(edges)
    if show:
        plt.imshow(edges)
        # cv2.imwrite("edges.jpg", edges)
        plt.title("Edges of the predicted masks")
        plt.show()

    edges_cp = np.copy(edges) * 0
    horz_lines = horz_hough_lines(edges, 580, 2)
    vert_lines = vert_hough_lines(edges, 850, 2)
    lines = horz_lines + vert_lines
    if show:
        edges_cp_1 = np.copy(edges) * 0
        cells = draw_hough_lines(edges_cp_1, lines)
        cells = cv2.bitwise_not(cells)
        cells = cv2.erode(cells, (5, 5))
        plt.imshow(cells)
        # cv2.imwrite("hough_lines_raw.jpg", cells)
        plt.title("hough_lines_raw")
        plt.show()
    lines = merge_similar_hough_lines(lines, 50, 0.5)
    lines = merge_similar_hough_lines(lines, 50, 0.5)
    cells = draw_hough_lines(edges_cp, lines)
    cells = cv2.bitwise_not(cells)
    cells = cv2.erode(cells, (5, 5))

    end = time.time()
    print(f"houglines {end-start}")

    if show:
        # cv2.imwrite("hough_lines_merged.jpg", cells)
        plt.imshow(cells)
        plt.title("Hough lines for cells")
        plt.show()

    if show:
        cropped_cp_down = np.copy(cropped_cp)
        cropped_cp_down = cv2.pyrDown(cropped_cp_down)
        cropped_cp_down = cv2.cvtColor(cropped_cp_down, cv2.COLOR_GRAY2BGR)
        cropped_cp_down = draw_hough_lines(cropped_cp_down, lines, (255, 0, 0))
        # cv2.imwrite("hough_lines_on_module.jpg", cropped_cp_down)
        plt.imshow(cropped_cp_down)
        plt.title("Hough lines for cells")
        plt.show()

    start = time.time()
    cells = cv2.pyrUp(cells)
    cells = cv2.pyrUp(cells)
    print(np.shape(cells))
    print(np.shape(cropped_image))

    panel_contours, _ = cv2.findContours(cells, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    panel_contours = width_and_height_filter(panel_contours, 330, 330, 0.5)
    if show:
        cropped_cp_ = np.copy(cropped_image)
        cropped_cp_ = cv2.cvtColor(cropped_cp_, cv2.COLOR_GRAY2RGB)
        cropped_cp_ = cv2.drawContours(cropped_cp_, panel_contours, -1, (255, 0, 0), 10)
        # cv2.imwrite("final_Cells.jpg", cropped_cp)
        plt.imshow(cropped_cp_)
        plt.title("Outined cells on original image")
        plt.show()

    images = []
    coord_list = []
    i = 0

    square_contours = []
    for contour in panel_contours:
        epsilon = 0.05 * cv2.arcLength(contour, True)
        square_contours.append(cv2.approxPolyDP(contour, epsilon, True))

    for contour in square_contours:
        _, _, w, h = cv2.boundingRect(contour)
        dest_pts = np.array([[0, 0], [0, h], [w, h], [w, 0]], np.float32)

        rect = contour.flatten()
        rect = rect.reshape((4, 2))
        rect = order_points(rect)

        src_pts = rect.astype(np.float32)
        coord_list.append(src_pts)

        M = cv2.getPerspectiveTransform(src_pts, dest_pts)
        # Apply the perspective transform to the image
        images.append(cv2.warpPerspective(cropped_image, M, (w, h)))

    if both

    if coords:
        return coord_list


    end = time.time()
    print(f"image extraction {end-start}")

    print(f"total_time {end-start_1}")
    return images


def random_image(folders: list) -> Optional[np.ndarray]:
    files = []
    for folder in folders:
        for dirpath, dirnames, filenames in os.walk(folder):
            for filename in filenames:
                files.append(os.path.join(dirpath, filename))
    if not files:
        return None
    else:
        return cv2.imread(random.choice(files))


def order_points(points):
    # Compute the sums and differences of the x and y coordinates
    sums = [p[0] + p[1] for p in points]
    diffs = [p[0] - p[1] for p in points]
    # Find the indices of the points with the smallest and largest sums
    topleft_index = np.argmin(sums)
    bottomright_index = np.argmax(sums)
    # Find the indices of the points with the smallest and largest differences
    topright_index = np.argmin(diffs)
    bottomleft_index = np.argmax(diffs)
    return np.float32(
        [
            points[topleft_index],
            points[topright_index],
            points[bottomright_index],
            points[bottomleft_index],
        ]
    )


def erode_inside(img, border_size, kernel, show):
    # Get the image dimensions
    height, width = img.shape[:2]

    # Calculate the border size to be left untouched
    border_size = int(min(height, width) * border_size)

    # Create a mask for the inner region of the image
    eroded = cv2.erode(img, kernel)
    eroded[:border_size, :] = 255
    eroded[height - border_size :, :] = 255
    eroded[:, :border_size] = 255
    eroded[:, width - border_size :] = 255

    if show:
        plt.imshow(eroded)
        plt.show()

    return cv2.bitwise_and(eroded, img)


def make_image_square(image):
    height, width, _ = image.shape
    if height == width:
        return image  # Already square
    elif height > width:
        new_width = height
        new_image = np.zeros((new_width, new_width, 3), np.uint8)
        offset = (new_width - width) // 2
        new_image[:, offset : offset + width, :] = image
    else:
        new_height = width
        new_image = np.zeros((new_height, new_height, 3), np.uint8)
        offset = (new_height - height) // 2
        new_image[offset : offset + height, :, :] = image
    return new_image


def find_pv_module(image: np.ndarray, model_module, show: bool = False) -> np.ndarray:
    image_square = make_image_square(image)
    image_square_og = np.copy(image_square)
    og_width, og_height, _ = np.shape(image_square)
    image_square = cv2.resize(image_square, (256, 256), interpolation=cv2.INTER_AREA)

    if show:
        plt.imshow(image_square)
        # cv2.imwrite("image_square.jpg", thresh)
        plt.title("Square image")
        plt.show()

    image_square = np.expand_dims(image_square, 0)
    prediction = model_module.predict(image_square)
    predicted_img = np.argmax(prediction, axis=3)[0, :, :] * 255
    predicted_img = predicted_img.astype(np.uint8)

    kernel = np.ones((10, 10), np.uint8)
    dilated = cv2.dilate(predicted_img, kernel, iterations=1)

    if show:
        plt.imshow(dilated)
        # cv2.imwrite("predicted_img.jpg", thresh)
        plt.title("Predicted contour of module")
        plt.show()

    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    rect = largest_convex_hull_rect(contours)
    rect = rect.flatten()
    rect = rect.reshape((4, 2))
    rect = order_points(rect)

    src_pts = rect.astype(np.float32)

    scaled_src_points = []
    scale_x = og_width / 256
    scale_y = og_height / 256
    for point in src_pts:
        x_scaled = int(point[0] * scale_x)
        y_scaled = int(point[1] * scale_y)
        scaled_src_points.append((x_scaled, y_scaled))
    scaled_src_points = np.array(scaled_src_points, np.float32)

    x, y, w, h = cv2.boundingRect(scaled_src_points)
    dst_pts = np.array([[0, 0], [0, h], [w, h], [w, 0]], np.float32)

    return reproject_image(image_square_og, scaled_src_points, dst_pts)


def largest_convex_hull_rect(contours):
    c = max(contours, key=cv2.contourArea)
    # Compute the convex hull of the contour
    hull = cv2.convexHull(c)
    # Compute the minimum bounding rectangle of the convex hull
    rect = cv2.minAreaRect(hull)
    # Convert the rectangle coordinates to integers and return them
    return cv2.boxPoints(rect).astype(np.float32)


def reproject_image(
    image: np.ndarray, src_pts: np.ndarray, dst_pts: np.ndarray
) -> np.ndarray:
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    _, _, w, h = cv2.boundingRect(dst_pts)
    return cv2.warpPerspective(image, M, (w, h))


def split_image_into_rows(
    image: np.ndarray, split: int, shape: tuple[int, int]
) -> list[dict]:
    column_images = []
    height, _ = np.shape(image)

    first_image = image[0 : shape[0], :]
    row = {"image": first_image, "y1": 0}
    column_images.append(row)

    vert_middle_ims_count = split - 2
    image_seperation = height // (split + 1)
    for i in range(vert_middle_ims_count):
        middle_pixel = image_seperation + (i + 1) * image_seperation
        middle_image = image[
            middle_pixel - shape[0] // 2 : middle_pixel + shape[1] // 2, :
        ]
        row = {"image": middle_image, "y1": middle_pixel - shape[0] // 2}
        column_images.append(row)

    last_image = image[height - shape[0] :, :]
    row = {"image": last_image, "y1": height - shape[0]}
    column_images.append(row)

    return column_images


def split_row_image(row: dict, split: int, shape: tuple[int, int]) -> list[dict]:
    split_images = []
    _, width = np.shape(row["image"])

    first_image = row["image"][:, 0 : shape[1]]
    image = {"image": first_image, "x1": 0, "y1": row["y1"]}
    split_images.append(image)

    horz_middle_ims_count = split - 2
    image_seperation = width // (split + 1)
    for i in range(horz_middle_ims_count):
        middle_pixel = image_seperation + (i + 1) * image_seperation
        middle_image = row["image"][
            :, middle_pixel - shape[0] // 2 : middle_pixel + shape[1] // 2
        ]
        image = {
            "image": middle_image,
            "x1": middle_pixel - shape[0] // 2,
            "y1": row["y1"],
        }
        split_images.append(image)

    last_image = row["image"][:, width - shape[1] :]
    image = {"image": last_image, "x1": width - shape[1], "y1": row["y1"]}
    split_images.append(image)

    return split_images


def predict_mask(image: np.ndarray, model) -> np.ndarray:
    prediction = model.predict(image)
    predicted_img = np.argmax(prediction, axis=3)[0, :, :] * 255
    return cast(predicted_img, np.ndarray)


def place_image(large_image, small_image, top_left):
    small_height, small_width = small_image.shape

    small_top, small_left = top_left

    small_bottom = small_top + small_height
    small_right = small_left + small_width

    large_image[small_top:small_bottom, small_left:small_right] = small_image

    return large_image


def horz_hough_lines(edges, votes, pixels):
    lines = None
    while lines is None:
        lines = cv2.HoughLines(edges, pixels, np.pi / 120, votes)
        votes = int(votes * 0.95)

    good_lines = []
    # Loop over the detected lines
    for line in lines:
        _, theta = line[0]
        a = np.cos(theta)
        b = np.sin(theta)
        slope = -a / b if b != 0 else 100  # divide by zero saftey
        # Check if the line is approximately horizontal
        if abs(slope) < 0.1:
            good_lines.append(line)
    return good_lines


def vert_hough_lines(edges, votes, pixels):
    lines = None
    while lines is None:
        lines = cv2.HoughLines(edges, pixels, np.pi / 120, votes)
        votes = int(votes * 0.95)

    good_lines = []
    # Loop over the detected lines
    for line in lines:
        _, theta = line[0]
        a = np.cos(theta)
        b = np.sin(theta)
        slope = -a / b if b != 0 else 100  # divide by zero saftey
        # Check if the line is approximately horizontal
        if abs(slope) > 50:
            good_lines.append(line)
    return good_lines


def draw_hough_lines(image, hough_lines, colour=(255, 255, 255)):
    # Loop over the detected lines
    for line in hough_lines:
        rho, theta = line[0]
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + 1200 * (-b))
        y1 = int(y0 + 1200 * (a))
        x2 = int(x0 - 1200 * (-b))
        y2 = int(y0 - 1200 * (a))
        cv2.line(image, (x1, y1), (x2, y2), colour, 2)
    return image


def merge_similar_hough_lines(lines, rho_threshold, theta_threshold):
    """
    Merges similar Hough lines in OpenCV.

    :param lines: list of lines in (rho, theta) format.
    :param threshold_distance: maximum distance between lines to be considered similar.
    :param threshold_angle: maximum angle difference between lines to be considered similar.
    :return: list of merged lines in (rho, theta) format.
    """

    # how many lines are similar to a given one
    similar_lines = {i: [] for i in range(len(lines))}
    for i in range(len(lines)):
        for j in range(len(lines)):
            if i == j:
                continue

            rho_i, theta_i = lines[i][0]
            rho_j, theta_j = lines[j][0]
            if (
                abs(rho_i - rho_j) < rho_threshold
                and abs(theta_i - theta_j) < theta_threshold
            ):
                similar_lines[i].append(j)

    # ordering the INDECES of the lines by how many are similar to them
    indices = [i for i in range(len(lines))]
    indices.sort(key=lambda x: len(similar_lines[x]))

    # line flags is the base for the filtering
    line_flags = len(lines) * [True]
    for i in range(len(lines) - 1):
        # if we already disregarded the ith element in the ordered list then we don't care (we will not delete anything based on it and we will never reconsider using this line again)
        if not line_flags[indices[i]]:
            continue

        # we are only considering those elements that had less similar line
        for j in range(i + 1, len(lines)):
            # and only if we have not disregarded them already
            if not line_flags[indices[j]]:
                continue

            rho_i, theta_i = lines[indices[i]][0]
            rho_j, theta_j = lines[indices[j]][0]
            if (
                abs(rho_i - rho_j) < rho_threshold
                and abs(theta_i - theta_j) < theta_threshold
            ):
                # if it is similar and have not been disregarded yet then drop it now
                line_flags[indices[j]] = False

    filtered_lines = []

    if filter:
        for i in range(len(lines)):  # filtering
            if line_flags[i]:
                filtered_lines.append(lines[i])

    return filtered_lines


def width_and_height_filter(contours, width, height, tolerance):
    good_contours = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w < width * (1 - tolerance) or w > width * (1 + tolerance):
            continue
        if h < height * (1 - tolerance) or h > height * (1 + tolerance):
            continue
        good_contours.append(contour)
    return good_contours


def label_image(
    image,
    text,
    font=cv2.FONT_HERSHEY_SIMPLEX,
    font_scale=2,
    color=(255, 255, 255),
    thickness=2,
):
    # Get the dimensions of the image
    height, width = image.shape

    # Get the dimensions of the text
    text_width, text_height = cv2.getTextSize(text, font, font_scale, thickness)[0]

    # Compute the position of the text
    text_x = int((width - text_width) / 2)
    text_y = int((height + text_height) / 2)

    # Put the text on the image
    cv2.putText(
        image, text, (text_x, text_y), font, font_scale, (0, 0, 0), thickness + 1
    )
    cv2.putText(image, text, (text_x, text_y), font, font_scale, color, thickness)


def sort_contours(contours):
    centers = []
    for contour in contours:
        M = cv2.moments(contour)
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        centers.append((cx, cy))

    # Sort the contours by their x-coordinate
    sorted_indices = np.argsort([center[1] for center in centers])
    sorted_contours = [contours[i] for i in sorted_indices]

    # Split the contours into rows
    rows = []
    row_indices = []
    prev_center_y = None
    for i, contour in enumerate(sorted_contours):
        center_x, center_y = centers[sorted_indices[i]]
        if prev_center_y is None or center_y - prev_center_y > 10:
            row_indices.append([i])
            rows.append([contour])
        else:
            row_indices[-1].append(i)
            rows[-1].append(contour)
        prev_center_y = center_y

    # Sort the contours within each row by their x-coordinate
    for i, row in enumerate(rows):
        row_sorted_indices = np.argsort(
            [centers[sorted_indices[j]][0] for j in row_indices[i]]
        )
        rows[i] = [row[j] for j in row_sorted_indices]

    return rows


def create_testing_dataset(folders, save_folder, toml_path, show=False):
    for i in range(10):
        image = random_image(folders)

        image = fix_barrel_distortion(image, toml_path)

        if show:
            plt.imshow(image)
            plt.title("Barrel distortion fix")
            plt.show()

        cropped_image = find_pv_module(image, model_path_module, show=show)
        new_path = f"{save_folder}/{i}.jpg"
        cv2.imwrite(new_path, cropped_image)

    return


if __name__ == "__main__":
    model_path_cells = (
        "C:/Users/chuck/OneDrive/Desktop/Honors/models/resnet_backbone_512.hdf5"
    )
    model_path_module = (
        "C:/Users/chuck/OneDrive/Desktop/Honors/models/resnet_PV_module_256.hdf5"
    )
    folders = [
        "C:/Users/chuck/OneDrive/Desktop/Honors/M0060/M0060",
        "C:/Users/chuck/OneDrive/Desktop/Honors/BT1/BT1",
    ]
    toml_path = "C:/Users/chuck/OneDrive/Desktop/Honors/solarEL/solarel/configs/camera_config.toml"

    start = time.time()
    image = random_image(folders)
    # image_path = "solarel/22503201-0715071581_8.jpg"
    # image = cv2.imread(image_path)
    # # test rotation, comment distortion fix
    # image = cv2.imread("rotated.jpg")

    plt.imshow(image)
    plt.title("Raw Image")
    plt.show()

    split_module_to_cells(
        model_path_module=model_path_module,
        model_path_cells=model_path_cells,
        image=image,
        toml_path=toml_path,
        show=False,
    )
    end = time.time()
    print(end - start)

    # save_folder = "C:/Users/chuck/OneDrive/Desktop/Honors/solarEL/solarel/Cell_Segmentation.ipynb/images"
    # create_testing_dataset(folders, save_folder, toml_path, show=False)
