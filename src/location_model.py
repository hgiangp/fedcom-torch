import numpy as np
np.set_printoptions(precision=6, linewidth=np.inf)

from src.location_utils import read_data

seed = 1
rng = np.random.default_rng(seed=seed)

class LocationModel: 
    def __init__(self, num_users=10, velocity=11, timeslot_duration=0.4, road_width=20):
        self.num_users = num_users 
        self.wroad = road_width 
        self.updated_dist = velocity * timeslot_duration
        self.xs, self.ys, self.dirs1, self.dirs2 = self.init_location()
        self.c_dir = np.ones(num_users) # current direction, 1 if dirs1, else 0
        self.x_inter, self.y_inter = (200, 172) # intersection point

    def init_location(self):
        file_name = 'location_data.txt'
        xs, ys, dirs1, dirs2 = read_data(file_name)
        return xs, ys, dirs1, dirs2

    def update_location(self):
        print("update_location")
        xs = self.xs 
        ys = self.ys 
        dirs = self.c_dir * self.dirs1 + (1 - self.c_dir) * self.dirs2

        # calculate new location 
        xs_new = xs + self.updated_dist * np.cos(dirs)
        ys_new = ys + self.updated_dist * np.sin(dirs)

        # calculate new direction and update current direction 
        to_inter_x = xs_new - self.x_inter # (num_users, )
        to_inter_y = ys_new - self.y_inter # (num_users, )
        dist_inter = np.sqrt(to_inter_x ** 2 + to_inter_y ** 2)
        in_inter = np.argwhere(dist_inter < self.wroad)
        self.c_dir[in_inter] = 0

        # update location 
        self.xs = xs_new 
        self.ys = ys_new 
    

    def get_location(self): 
        r"Get location, remember to call update_location in advance"
        print(f"xs = {self.xs}")
        print(f"ys = {self.ys}")
        return (self.xs, self.ys)

def test(): 
    locmodel = LocationModel()
    max_ground = 200 
    for _ in range(max_ground): 
        xs, ys = locmodel.get_location()
        locmodel.update_location()

if __name__=='__main__': 
    test()