import json

import numpy as np


def parse_content_line(x):
    item = json.loads(x)
    item = np.array([item['title_left'], item['title_right'], int(item['label'])])
    return item[np.newaxis, :]