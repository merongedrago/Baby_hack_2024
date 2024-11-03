import math
import numpy as np

def data_transforming(visuals: list, x: list, y: list, hazard, dic_observed: dict):
    """
    Populate the dictionary with observed objects and their coordinates.

    Parameters:
    - visuals: list of object names (strings).
    - x: list of x coordinates.
    - y: list of y coordinates.
    - dic_observed: dict to populate.

    Returns:
    - dic_observed: updated dictionary.
    """
    safe = ['obj2','baby', 'person']
    for i in range(len(visuals)):
        if visuals[i] not in dic_observed:
            # Create a new entry with a 2D array for coordinates
            dic_observed[visuals[i]] = [np.array([[x[i], y[i]]]), np.array([0]), np.array([0]),np.array([hazard[i]]) ]
        else:
            # Append new values as a row, ensuring the new coordinates are in correct shape
            new_coordinates = np.array([[x[i], y[i]]])  # Ensure this is a 2D array
            dic_observed[visuals[i]][0] = np.vstack((dic_observed[visuals[i]][0], new_coordinates))

    return dic_observed

def pad_matrices(dic_observed):
    """
    Pads the coordinate matrices in the dictionary to match the maximum row size by repeating the last row.

    Parameters:
    - dic_observed: dict, where each value is a list containing a coordinate matrix (2D array),
                    a velocity vector, and an acceleration vector.

    Returns:
    - dic_observed: dict, with each coordinate matrix padded to the maximum row size.
    """
    # Ensure all coordinate matrices are at least 2D arrays
    for key in dic_observed:
        coordinate_matrix = dic_observed[key][0]  # Get the coordinate matrix
        num_points = coordinate_matrix

        if coordinate_matrix.ndim == 1:  # If a 1D array, reshape it to (1, n)
            dic_observed[key][0] = coordinate_matrix.reshape(1, -1)

    # Find the maximum number of rows among all coordinate matrices
    max_rows = max(dic_observed[key][0].shape[0] for key in dic_observed)

    # Pad each coordinate matrix to the maximum row size
    for key in dic_observed:
        coordinate_matrix = dic_observed[key][0]  # Get the coordinate matrix

        # Ensure coordinate_matrix is at least 2D
        if coordinate_matrix.ndim == 1:
            coordinate_matrix = coordinate_matrix.reshape(1, -1)  # Reshape if it's still 1D

        rows, cols = coordinate_matrix.shape  # Get current rows and columns

        if rows < max_rows:
            # Get the last row
            last_row = coordinate_matrix[-1, :].reshape(1, cols)  # Reshape to keep it 2D for stacking

            # Calculate how many rows are needed to reach max_rows
            rows_to_add = max_rows - rows

            # Stack the last row repeatedly until the matrix has max_rows rows
            padding = np.tile(last_row, (rows_to_add, 1))  # Use np.tile to repeat last_row correctly

            # Update the coordinate matrix in dic_observed
            dic_observed[key][0] = np.vstack([coordinate_matrix, padding])  # Ensure it's 2D

    return dic_observed


def update_dictionary_with_metrics(dic_observed, time, things, hazard):
    """
    Adds Euclidean distance and acceleration metrics to each object's data.

    Parameters:
    - dic_observed: dict, where each key has a 2D array of [x, y] coordinates over time.

    Returns:
    - dic_observed: dict, updated to include distance and acceleration for each object.
    """

    for key, coordinates in dic_observed.items():
        if key in things:
          i = things.index(key)
          dic_observed[key][3] = np.append(coordinates[3], hazard[i])
        else:
          dic_observed[key][3] = np.append(coordinates[3], coordinates[3][0])
        # Access the coordinate matrix
        coordinate_matrix = coordinates[0]  # Get the 2D array of coordinates
        num_points = coordinate_matrix.shape[0]
        if num_points > 1:
            # Calculate Euclidean distance between consecutive points
            dic_observed[key][1] = np.append(coordinates[1], np.sqrt(np.sum((coordinate_matrix[time]- coordinate_matrix[time-1])**2)))
            # Calculate acceleration (change in distance over time steps)
            dic_observed[key][2] = np.append(coordinates[2], coordinates[1][time]-coordinates[1][time-1])  # Prepend 0 to match the lengths
    return dic_observed



def circle_overlap_percentage(tlx1, tly1, h1, w1, tlx2, tly2, h2, w2):
    '''MAKING AN ASSUMPTION THAT THE FIRST OBJECT IS ALWAYS THE BABY'''
    # Calculate distance between centers
    r1 = math.sqrt((h1 / 2) ** 2 + (w1 / 2) ** 2)
    r2 = math.sqrt((h2 / 2) ** 2 + (w2 / 2) ** 2)
    # calculate the center for object one
    x1 = tlx1 + (w1 / 2)
    y1 = tly1 - (h1 / 2)

    # calculate the center for object two
    x2 = tlx2 + (w2 / 2)
    y2 = tly2 - (h2 / 2)

    d = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

    # Case 1: No overlap
    if d >= r1 + r2:
        return 0.0

    # Case 2: One circle is completely inside the other
    if d <= abs(r1 - r2):
        smaller_radius = min(r1, r2)
        overlap_area = math.pi * smaller_radius**2
    else:
        # Case 3: Partial overlap
        part1 = r1**2 * math.acos((d**2 + r1**2 - r2**2) / (2 * d * r1))
        part2 = r2**2 * math.acos((d**2 + r2**2 - r1**2) / (2 * d * r2))
        part3 = 0.5 * math.sqrt(
            (-d + r1 + r2) * (d + r1 - r2) * (d - r1 + r2) * (d + r1 + r2)
        )
        overlap_area = part1 + part2 - part3

    # Total area of both circles
    child_area = math.pi * r1**2

    # Calculate percentage overlap
    percentage_overlap = (overlap_area / child_area) * 100
    return percentage_overlap

def output(dic_observed, time):
  result = ['ALARM', 'Warning', 'Nothing happening']
  if 'baby' in dic_observed:
    print(f'Baby found on camara, starting survillance...')
    if 'person' in dic_observed:
      print(f'Baby is with another person. Things should be okay.')
      for key, values in dic_observed.items():
        hazard_item = dic_observed[key][3][time]
        if hazard_item == 1:
          print(f'{key} detected on camara, {result[1]}')
        else:
          pass
    else:
      print(f'Person not detected, {result[1]}')
      for key, values in dic_observed.items():
        hazard_item = dic_observed[key][3][time]
        if hazard_item == 1:
          print(f'{key} detected on camara, {result[0]}')
        else:
          pass
  else:
    print(result[2])