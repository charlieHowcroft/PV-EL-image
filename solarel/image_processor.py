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


def main(model_path: str, folder_paths: list, toml_path: str, show: bool = False):
    model = load_model(model_path, compile=False)
    model.compile()

    image = random_image(folder_paths)
    # # test rotation, comment distortion fix
    image = cv2.imread("rotated.jpg")
    if image is None:
        return
    if show:
        plt.imshow(image)
        plt.title('Raw Image')
        plt.show()
    # image = fix_barrel_distortion(image, toml_path)
    if show:
        plt.imshow(image)
        plt.title('Barrel distortion fix')
        plt.show()
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    cropped_image = find_pv_module(image, show=True)
    if show:
        cropped_image_bgr = np.copy(cropped_image)
        # cropped_image_bgr = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)
        print(cropped_image_bgr.shape)
        # cv2.imwrite("cropped_image_bgr.jpg", cropped_image_bgr)
        plt.imshow(cropped_image_bgr)
        plt.title('Cropped image to PV module')
        plt.xticks([])
        plt.yticks([])
        plt.show()

    cropped_cp = np.copy(cropped_image)
    cropped_cp = cv2.pyrDown(cropped_cp)
    og_height, og_width = np.shape(cropped_cp)
    height, width = np.shape(cropped_cp)
    horz_splits = int(np.ceil(width/512)+1)
    vert_splits = int(np.ceil(height/512)+1)

    row_images = split_image_into_rows(cropped_cp, vert_splits, (512, 512))
    if show:
        plt.imshow(row_images[0]["image"])
        # cv2.imwrite("row.jpg", row_images[0]["image"])
        plt.title("First row")
        plt.show()

    images = []
    for row in row_images:
        images.append(split_row_image(row, horz_splits+1, (512, 512)))
    if show:
        plt.imshow(images[0][0]["image"])
        # cv2.imwrite("image_0.jpg", images[0][0]["image"])
        # cv2.imwrite("image_1.jpg", images[0][1]["image"])
        # cv2.imwrite("image_2.jpg", images[0][2]["image"])
        plt.title("First image from the row")
        plt.show()

    for row in images:
        for image in row:
            temp = image["image"]
            temp = cv2.merge((temp, temp, temp))
            temp = np.expand_dims(temp, 0)
            prediction = (model.predict(temp))
            predicted_img = np.argmax(prediction, axis=3)[0, :, :]*255
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
            # cv2.imwrite(f"final_mask_{i}.jpg", final_mask)
    if show:
        plt.imshow(final_mask)
        # cv2.imwrite("final_mask_.jpg", final_mask)
        plt.title("All mask predictions combined")
        plt.show()

    final_mask_cp = np.copy(final_mask)
    final_mask_cp = final_mask_cp.astype(np.uint8)
    edges = cv2.Canny(final_mask_cp, 120, 180)
    edges = cv2.dilate(edges, (10, 10))
    edges = cv2.pyrDown(edges)
    if show:
        plt.imshow(edges)
        # cv2.imwrite("edges.jpg", edges)
        plt.title("Edges of the predicted masks")
        plt.show()

    edges_cp = np.copy(edges)*0
    horz_lines = horz_hough_lines(edges, 400, 2)
    vert_lines = vert_hough_lines(edges, 600, 2)
    lines = horz_lines + vert_lines
    if show:
        edges_cp_1 = np.copy(edges)*0
        cells = draw_hough_lines(edges_cp_1, lines)
        cells = cv2.bitwise_not(cells)
        cells = cv2.erode(cells, (5, 5))
        plt.imshow(cells)
        # cv2.imwrite("hough_lines_raw.jpg", cells)
        plt.title("Edges of the predicted masks")
        plt.show()
    lines = merge_similar_hough_lines(lines, 50, 0.5)
    lines = merge_similar_hough_lines(lines, 50, 0.5)
    cells = draw_hough_lines(edges_cp, lines)
    cells = cv2.bitwise_not(cells)
    cells = cv2.erode(cells, (5, 5))
    if show:
        # cv2.imwrite("hough_lines_merged.jpg", cells)
        plt.imshow(cells)
        plt.title("Hough lines for cells")
        plt.show()

    if show:
        cropped_cp_down = cv2.pyrDown(cropped_cp)
        cropped_cp_down = cv2.cvtColor(cropped_cp_down, cv2.COLOR_GRAY2BGR)
        cropped_cp_down = draw_hough_lines(cropped_cp_down, lines, (0, 0, 255))
        # cv2.imwrite("hough_lines_on_module.jpg", cropped_cp_down)
        plt.imshow(cropped_cp_down)
        plt.title("Hough lines for cells")
        plt.show()

    cells = cv2.pyrUp(cells)
    cells = cv2.pyrUp(cells)

    panel_images = []
    panel_contours, _ = cv2.findContours(
        cells, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    panel_contours = width_and_height_filter(panel_contours, 400, 330, 0.2)
    if show:
        cropped_cp = np.copy(cropped_image)
        cropped_cp = cv2.cvtColor(cropped_cp, cv2.COLOR_GRAY2BGR)
        cropped_cp = cv2.drawContours(
            cropped_cp, panel_contours, -1, (0, 0, 255), 10)
        # cv2.imwrite("final_Cells.jpg", cropped_cp)
        plt.imshow(cropped_cp)
        plt.title("Outined cells on original image")
        plt.show()

    rows = sort_contours(panel_contours)
    i = 0
    height, width = np.shape(cropped_cp)
    panel_image_rows = []
    for row in rows:
        panel_image_rows.append([])
        for contour in row:
            x, y, w, h = cv2.boundingRect(contour)
            print(x, y, w, h)
            delta = 0
            panel_image = cropped_image[max(
                0, y-delta):min(y+h+delta, height), max(0, x-delta):min(x+w+delta, width)]
            label_image(panel_image, f'{i}')
            panel_images.append(panel_image)
            panel_image_rows[-1].append(panel_image)
            i += 1


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


def largest_rectangle(contours: np.ndarray) -> np.ndarray:
    # Approximate contours to polygons and find the largest rectangle
    largest_rect = None
    max_area = 0
    for cnt in contours:
        # Approximate the contour to a polygon
        poly = cv2.approxPolyDP(cnt, 0.03*cv2.arcLength(cnt, True), True)
        # If the polygon has 4 vertices (is a rectangle)
        if len(poly) == 4:
            # Calculate the area of the polygon
            area = cv2.contourArea(poly)
            # If the area is larger than the current maximum
            if area > max_area:
                # Update the maximum area and largest rectangle
                max_area = area
                largest_rect = poly
    # Return the largest rectangle
    return cast(np.ndarray, largest_rect)


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
    # type: ignore
    # type: ignore
    # type: ignore
    return np.float32([points[topleft_index], points[topright_index], points[bottomright_index], points[bottomleft_index]])


def find_pv_module(image: np.ndarray, show: bool = False) -> np.ndarray:
    image_cp = np.copy(image)
    image_cp = cv2.blur(image_cp, (5, 5))
    avg_intensity = int(cv2.mean(image)[0])

    _, thresh = cv2.threshold(image_cp, int(
        avg_intensity), 255, cv2.THRESH_BINARY)
    if show:
        plt.imshow(thresh)
        # cv2.imwrite("thresh.jpg", thresh)
        plt.title("Global threshold of the module")
        plt.show()

    kernel = np.ones((100, 100), np.uint8)
    dilated = cv2.dilate(thresh, kernel, iterations=1)

    contours, _ = cv2.findContours(
        dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    rect = largest_rectangle(contours)
    if show:
        image_bgr = cv2.cvtColor(image_cp, cv2.COLOR_GRAY2BGR)
        image_bgr = cv2.drawContours(image_bgr, [rect], -1, (0, 0, 255), 50)
        # cv2.imwrite("largest_rectangle.jpg", image_bgr)
        plt.imshow(image_bgr)
        plt.title("Largest Rectangle")
        plt.xticks([])
        plt.yticks([])
        plt.show()

    _, _, w, h = cv2.boundingRect(rect)
    rect = rect.flatten()
    rect = rect.reshape((4, 2))
    src_pts = order_points(rect)
    src_pts = src_pts.astype(np.float32)
    src_pts = cast(np.ndarray, src_pts)
    dst_pts = np.array([[0, 0], [0, h], [w, h], [w, 0]], np.float32)

    return reproject_image(image_cp, src_pts, dst_pts)


def reproject_image(image: np.ndarray, src_pts: np.ndarray, dst_pts: np.ndarray) -> np.ndarray:
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    _, _, w, h = cv2.boundingRect(dst_pts)
    return cv2.warpPerspective(image, M, (w, h))


def split_image_into_rows(image: np.ndarray, split: int, shape: tuple[int, int]) -> list[dict]:
    column_images = []
    height, _ = np.shape(image)

    first_image = image[0:shape[0], :]
    row = {
        "image": first_image,
        "y1": 0
    }
    column_images.append(row)

    vert_middle_ims_count = split-2
    image_seperation = height//(split+1)
    for i in range(vert_middle_ims_count):
        middle_pixel = image_seperation + (i+1)*image_seperation
        middle_image = image[middle_pixel -
                             shape[0]//2:middle_pixel+shape[1]//2, :]
        row = {
            "image": middle_image,
            "y1": middle_pixel-shape[0]//2
        }
        column_images.append(row)

    last_image = image[height-shape[0]:, :]
    row = {
        "image": last_image,
        "y1": height-shape[0]
    }
    column_images.append(row)

    return column_images


def split_row_image(row: dict, split: int, shape: tuple[int, int]) -> list[dict]:
    split_images = []
    _, width = np.shape(row["image"])

    first_image = row["image"][:, 0:shape[1]]
    image = {
        "image": first_image,
        "x1": 0,
        "y1": row["y1"]
    }
    split_images.append(image)

    horz_middle_ims_count = split-2
    image_seperation = width//(split+1)
    for i in range(horz_middle_ims_count):
        middle_pixel = image_seperation + (i+1)*image_seperation
        middle_image = row["image"][:, middle_pixel -
                                    shape[0]//2:middle_pixel+shape[1]//2]
        image = {
            "image": middle_image,
            "x1": middle_pixel-shape[0]//2,
            "y1": row["y1"]
        }
        split_images.append(image)

    last_image = row["image"][:, width-shape[1]:]
    image = {
        "image": last_image,
        "x1": width-shape[1],
        "y1": row["y1"]
    }
    split_images.append(image)

    return split_images


def predict_mask(image: np.ndarray, model) -> np.ndarray:
    prediction = (model.predict(image))
    predicted_img = np.argmax(prediction, axis=3)[0, :, :]*255
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
        lines = cv2.HoughLines(edges, pixels, np.pi/120, votes)
        votes = int(votes*0.95)

    good_lines = []
    # Loop over the detected lines
    for line in lines:
        _, theta = line[0]
        a = np.cos(theta)
        b = np.sin(theta)
        slope = - a / b if b != 0 else 100  # divide by zero saftey
        # Check if the line is approximately horizontal
        if abs(slope) < 0.1:
            good_lines.append(line)
    return good_lines


def vert_hough_lines(edges, votes, pixels):
    lines = None
    while lines is None:
        lines = cv2.HoughLines(edges, pixels, np.pi/120, votes)
        votes = int(votes*0.95)

    good_lines = []
    # Loop over the detected lines
    for line in lines:
        _, theta = line[0]
        a = np.cos(theta)
        b = np.sin(theta)
        slope = - a / b if b != 0 else 100  # divide by zero saftey
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
            if abs(rho_i - rho_j) < rho_threshold and abs(theta_i - theta_j) < theta_threshold:
                similar_lines[i].append(j)

    # ordering the INDECES of the lines by how many are similar to them
    indices = [i for i in range(len(lines))]
    indices.sort(key=lambda x: len(similar_lines[x]))

    # line flags is the base for the filtering
    line_flags = len(lines)*[True]
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
            if abs(rho_i - rho_j) < rho_threshold and abs(theta_i - theta_j) < theta_threshold:
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
        if w < width*(1-tolerance) or w > width * (1+tolerance):
            continue
        if h < height*(1-tolerance) or h > height * (1+tolerance):
            continue
        good_contours.append(contour)
    return good_contours


def label_image(image, text, font=cv2.FONT_HERSHEY_SIMPLEX, font_scale=2, color=(255, 255, 255), thickness=2):
    # Get the dimensions of the image
    height, width = image.shape

    # Get the dimensions of the text
    text_width, text_height = cv2.getTextSize(
        text, font, font_scale, thickness)[0]

    # Compute the position of the text
    text_x = int((width - text_width) / 2)
    text_y = int((height + text_height) / 2)

    # Put the text on the image
    cv2.putText(image, text, (text_x, text_y), font,
                font_scale, (0, 0, 0), thickness+1)
    cv2.putText(image, text, (text_x, text_y),
                font, font_scale, color, thickness)


def sort_contours(contours):
    centers = []
    for contour in contours:
        M = cv2.moments(contour)
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])
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
            [centers[sorted_indices[j]][0] for j in row_indices[i]])
        rows[i] = [row[j] for j in row_sorted_indices]

    return rows


if __name__ == '__main__':
    model_path = "C:/Users/chuck/OneDrive/Desktop/Honors/models/resnet_backbone_512.hdf5"
    folders = ["C:/Users/chuck/OneDrive/Desktop/Honors/M0060/M0060",
               "C:/Users/chuck/OneDrive/Desktop/Honors/BT1/BT1"]
    toml_path = "C:/Users/chuck/OneDrive/Desktop/Honors/solarEL/solarel/configs/camera_config.toml"

    start = time.time()
    main(model_path=model_path, folder_paths=folders,
         toml_path=toml_path, show=True)
    end = time.time()
    print(end - start)
