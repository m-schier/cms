__GTSRB_LABELS = [
    'Limit_20_kph',
    'Limit_30_kph',
    'Limit_50_kph',
    'Limit_60_kph',
    'Limit_70_kph',
    'Limit_80_kph',
    'Unlimited_80_kph',
    'Limit_100_kph',
    'Limit_120_kph',
    'No_Overtaking',
    'No_Overtaking_Trucks',
    'Right_Of_Way',
    'Right_Of_Way_Street',
    'Yield',
    'Stop',
    'Forbidden',
    'Forbidden_Trucks',
    'No_Entry',
    'Attention',
    'Attention_Left_Turn',
    'Attention_Right_Turn',
    'Attention_Chicane',
    'Attention_Bumps',
    'Attention_Slippery',
    'Attention_Narrowing',
    'Attention_Road_Works',
    'Attention_Traffic_Light',
    'Attention_Pedestrians',
    'Attention_Children',
    'Attention_Cyclists',
    'Attention_Ice',
    'Attention_Deer_Crossing',
    'Unlimited',
    'Must_Turn_Right',
    'Must_Turn_Left',
    'Must_Continue_Straight',
    'Must_Continue_Straight_Or_Right',
    'Must_Continue_Straight_Or_Left',
    'Continue_Right_Of_Sign',
    'Continue_Left_Of_Sign',
    'Roundabout',
    'End_Of_No_Overtaking',
    'End_Of_No_Overtaking_Trucks'
]


def load_gtsrb_from_folder(root=None):
    """
    Load GTSRB data set
    :param root: Root path, that is file to the GTSRB folder containing readme file
    :return: X and Y
    """

    from glob import glob
    import os
    import csv
    import cv2
    import numpy as np

    if root is None:
        root = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'GTSRB', 'Training')

    csvs = glob(os.path.join(root, "**", "*.csv"))

    xs = []
    ys = []

    for csv_path in csvs:
        dir_path = os.path.dirname(csv_path)

        with open(csv_path, 'r', newline='') as csv_file:
            reader = csv.DictReader(csv_file, delimiter=';')

            for row in reader:
                img_path = os.path.join(dir_path, row['Filename'])
                image = cv2.imread(img_path, cv2.IMREAD_COLOR)

                if image is None:
                    raise IOError("Failed to read image: {}".format(img_path))

                expected_shape = (int(row['Height']), int(row['Width']), 3)
                if image.shape != expected_shape:
                    raise ValueError("Shape was {}, expected {}".format(image.shape, expected_shape))

                x1, x2, y1, y2 = int(row['Roi.X1']), int(row['Roi.X2']), int(row['Roi.Y1']), int(row['Roi.Y2'])
                image = image[y1:y2, x1:x2, ::-1] / np.float32(255)  # Standard RGB channels

                assert np.all(image >= 0)
                assert np.all(image <= 1)

                xs.append(image)
                ys.append(int(row['ClassId']))

    if len(ys) != 26640:
        raise ValueError("Expected to read 26640 samples but have {}. Please ensure you have correctly downloaded the "
                         "'fixed' GTSRB training image data set.".format(len(ys)))

    return xs, np.array(ys)


def scale_gtsrb(xs, size):
    import numpy as np
    import cv2

    result = np.empty((len(xs),) + size + xs[0].shape[2:], dtype=np.float32)

    for i, x in enumerate(xs):
        cv2.resize(x, size, dst=result[i, ...], interpolation=cv2.INTER_LINEAR)

    # Some interpolation methods may exceed [0, 1] range, check when changing from INTER_LINEAR
    # assert np.all(result >= 0)
    # assert np.all(result <= 1)

    return result


def normalize_gtsrb(xs):
    import cv2
    import numpy as np

    # Histogrammspreizung auf Grauwert

    result = []

    for x in xs:
        hsv = cv2.cvtColor(x, cv2.COLOR_RGB2HSV)

        v_min, v_max = np.percentile(hsv[..., 2], [5, 95])

        if v_max != v_min:
            hsv[..., 2] = np.clip((hsv[..., 2] - v_min) / (v_max - v_min), 0, 1)

        result.append(cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB))

    return result


def make_gtsrb(root=None, normalize=False, size=None):
    from .Datasets import Dataset

    x, y = load_gtsrb_from_folder(root)

    if normalize:
        x = normalize_gtsrb(x)

    if size is not None:
        x = scale_gtsrb(x, size)

    return Dataset((x, y), None, gt_labels=__GTSRB_LABELS, name='GTSRB')
