import numpy as np
import os
from clustering import clustering

border_points_location = "border_points\\diff=1.5"
border_centers_location = "border_points\\diff=1.5\\centers"
if not os.path.exists(border_centers_location):
    os.mkdir(border_centers_location)

for n_epoch in range(195, -5, -5):
    border_points_path = os.path.join(border_points_location, "data_{:03d}.npy".format(n_epoch))
    border_centers_path = os.path.join(border_centers_location, "data_{:03d}.npy".format(n_epoch))
    border_points = np.load(border_points_path)
    if border_points.shape[0]<=5000:
        np.save(border_centers_path, border_points)
        print(n_epoch)
        break
    centers = clustering(border_points, 5000, 1)
    np.save(border_centers_path, centers)
    print(n_epoch)