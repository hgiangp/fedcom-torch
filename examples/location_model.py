import numpy as np 

seed = 42
rng = np.random.default_rng(seed=seed)
class LocationModel: 
    def __init__(self, num_users=10, updated_dist=100, width=20):
        self.num_users = num_users 
        self.width = width 
        self.updated_dist = updated_dist
        self.xs, self.ys, self.dirs = self.init_location()

    def init_location(self):
        width = self.width 
        num_users = self.num_users

        xs = rng.normal(loc=0, scale=width, size=(num_users))
        ys = rng.normal(loc=0, scale=width, size=(num_users))
        dirs = rng.integers(low=0, high=4, size=(num_users)) 

        for i in range(num_users):
            if xs[i] > width: 
                dirs[i] = 0 
            elif (- xs[i] > width): 
                dirs[i] = 2 
            elif ys[i] > width: 
                dirs[i] = 1
            elif (-ys[i] > width): 
                dirs[i] = 3

        return xs, ys, dirs

    def update_location(self): 
        num_users = self.num_users

        delta_xs = np.zeros(num_users, dtype=int)
        delta_ys = np.zeros(num_users, dtype=int)

        for i in range(num_users): 
            if self.dirs[i] == 0:
                delta_xs[i] = -1
            elif self.dirs[i] == 1: 
                delta_ys[i] = -1
            elif self.dirs[i] == 2: 
                delta_xs[i]= 1
            else: # 3 
                delta_ys[i] = 1

        # Update location 
        xs_new = self.xs + delta_xs * self.updated_dist
        ys_new = self.ys + delta_ys * self.updated_dist

        # Change direction
        xsigns = xs_new * self.xs < 0
        ysigns = ys_new * self.ys < 0

        dir_changed = np.argwhere([x or y for x, y in zip(xsigns, ysigns)]) 
        dirs_new = rng.integers(low=0, high=4, size=(num_users))
        
        # Update location model 
        self.dirs[dir_changed] = dirs_new[dir_changed]
        self.xs = xs_new
        self.ys = ys_new 

    def get_location(self): 
        r"Get location, remember to call update_location in advance"
        return (self.xs, self.ys)

def test(): 
    location_model = LocationModel(10, 100, 20)
    xs, ys, dirs = location_model.xs, location_model.ys, location_model.dirs
    print("xs =", xs)
    print("ys =", ys)
    print("dirs =", dirs)
    location_model.update_location()

    xs, ys = location_model.get_location()
    print(f"xs = {xs}")
    print(f"ys = {ys}")

    print("Initialization")
    loc_model = LocationModel(10, 100)
    xs, ys = loc_model.get_location()
    print(f"xs = {xs}")
    print(f"ys = {ys}")
    print(f"loc_model.width = {loc_model.width}\tloc_model.updated_dist = {loc_model.updated_dist}")

if __name__=='__main__': 
    test()