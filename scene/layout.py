# Copyright (C) 2024, Nanjing University
# All rights reserved.


import numpy as np


class LayoutWall:
    def __init__(self, **args) -> None:
        self.attribute = args

        if "planeEquation" in self.attribute:
            self.attribute["planeEquation"] = np.array(
                self.attribute["planeEquation"]
            )

        if "normal" in self.attribute:
            self.attribute["normal"] = np.array(
                self.attribute["normal"]
            )
    
    @property
    def width(self):
        ceil0 = self.attribute["ceil0"]
        floor0 = self.attribute["floor0"]
        ceil1 = self.attribute["ceil1"]
        floor1 = self.attribute["floor1"]

        dx = floor0[0] - floor1[0]
        dy = floor0[1] - ceil0[1]
        dz = floor0[2] - floor1[2]

        return np.sqrt(dx**2 + dz**2)
    
    @property
    def height(self):
        ceil0 = self.attribute["ceil0"]
        floor0 = self.attribute["floor0"]
        ceil1 = self.attribute["ceil1"]
        floor1 = self.attribute["floor1"]

        dx = floor0[0] - floor1[0]
        dy = floor0[1] - ceil0[1]
        dz = floor0[2] - floor1[2]

        return np.abs(dy)

    def toPoints(self, num_pts, num_pts_y=None):
        ceil0 = self.attribute["ceil0"]
        floor0 = self.attribute["floor0"]
        ceil1 = self.attribute["ceil1"]
        floor1 = self.attribute["floor1"]

        y_min = np.minimum(floor0[1], ceil0[1])
        y_max = np.maximum(floor0[1], ceil0[1])

        if num_pts_y is None:
            num_y = int(np.sqrt(num_pts))
        else:
            num_y = num_pts_y
        y = np.linspace(y_min, y_max, num_y)

        x_min = np.minimum(floor0[0], floor1[0])
        x_max = np.maximum(floor0[0], floor1[0])

        num_x = int(np.ceil(num_pts / num_y))
        x = np.linspace(x_min, x_max, num_x)

        x = x.repeat(num_y)
        y = np.concatenate([y.copy() for _ in range(num_x)])

        A, B, C, D = self.attribute["planeEquation"]

        z = (-D - A * x - B * y) / C

        xyz = np.stack((x, y, z), axis=-1)


        normals = np.ones_like(xyz) * self.attribute["normal"]

        return xyz, normals 

    def distance(self, x : np.ndarray):
        """
        Input: x [N, 3]
        Output: [N]
        """
        planeEquation = self.attribute["planeEquation"]
        a = np.sum(x * planeEquation[:3], dim=-1) + planeEquation[-1]
        b = np.sum(np.square(planeEquation[:3]))
        return np.abs(a) / np.sqrt(b)
    
    def inside(self, x : np.ndarray):
        return self.inside_range(x)
    
    def inside_range(self, x : np.ndarray):
        """
        Input: x [N, 3]
        Output: [N] bool
        """
        ceil0 = self.attribute["ceil0"]
        floor0 = self.attribute["floor0"]
        threshold = self.attribute.get("threshold", 0.01)
        y_min = np.minimum(floor0[1], ceil0[1])
        y_max = np.maximum(floor0[1], ceil0[1])
        # return self.distance(x) <= threshold
        return (self.distance(x) <= threshold) & \
            (y_min <= x[..., 1]) & \
            (x[..., 1] <= y_max) & \
            self.width_between(x)
    
    @classmethod
    def dot_xz(cls, x : np.ndarray, y : np.ndarray):
        result = x * y
        return result[..., 0] + result[..., 2]
    
    @classmethod
    def intersect_xz(cls, x : np.ndarray, st : np.ndarray, ed : np.ndarray):
        x = x[..., [0, -1]]
        st = st[[0, -1]]
        ed = ed[[0, -1]]
        if st[1] == ed[1]:
            return (x[..., 0] != x[..., 0])
        result = (x[..., 0] == x[..., 0])
        result = result & ~((st[1] > x[..., 1]) & (ed[1] > x[..., 1]))
        result = result & ~((st[1] < x[..., 1]) & (ed[1] < x[..., 1]))
        result = result & ~((st[1] == x[..., 1]) & (ed[1] > x[..., 1]))
        result = result & ~((st[1] > x[..., 1]) & (ed[1] == x[..., 1]))
        xseg = ed[0] - (ed[0]-st[0]) * (ed[1] - x[..., 1]) / (ed[1] - st[1])
        result = result & ~(xseg < x[..., 0])
        return result

    def intersect(self, x : np.ndarray):
        """
        Input: x [N, 3]
        Output: [N] int
        """
        ceil0 = self.attribute["ceil0"]
        ceil1 = self.attribute["ceil1"]
        return self.intersect_xz(x, ceil0, ceil1)

    def inner(self, x : np.ndarray):
        """
        Input: x [N, 3]
        Output: [N] bool
        """
        normal = self.attribute["normal"]
        ceil0 = self.attribute["ceil0"]
        threshold = self.attribute.get("threshold", 0.01)
        return self.dot_xz(x - ceil0, normal) <= 0.0 #

    def width_between(self, x : np.ndarray):
        ceil0 = self.attribute["ceil0"]
        ceil1 = self.attribute["ceil1"]
        threshold = self.attribute.get("threshold", 0.01)
        return (self.dot_xz(ceil0 - ceil1, x - ceil1) >= 0.0) & \
            (self.dot_xz(ceil1 - ceil0, x - ceil0) >= 0.0)

def calc_normal(a, b, c):
    return np.cross(a - b, c - b)

def calc_equation(a, normal):
    return -np.dot(a, normal)

class Layout:
    def __init__(self, **args) -> None:
        self.attribute = args

        layoutDownsample = self.attribute.get("layoutDownsample", 1.0) 

        # (r"""
        #  Update in 2023.08.05
        #  For
        #     test layout-widen to avoid artifacts(all pixels of image are black/white)
        #  """)
        # layoutScale = self.attribute.get("layoutScale", 1.0)
        # layoutScale = np.array([layoutScale, layoutScale, layoutScale]) ############# [layoutScale, 1.0, layoutScale]
        # (r"""End Update in 2023.08.05""")
        
        self.attribute["threshold"] = self.attribute.get("threshold", 0.5) # 0.1
        threshold = self.attribute["threshold"]   

        if "c2w" in self.attribute:
            c2w = self.attribute["c2w"]
            self.camera_position = np.array([c2w[0][-1], c2w[1][-1], c2w[2][-1]])  

        if "layoutPoints" in self.attribute:
            max_points = self.attribute.get("max_corners", 32)
            if len(self.attribute["layoutPoints"]["points"]) > max_points:
                ptsIdxs = np.linspace(0, len(self.attribute["layoutPoints"]["points"]) - 1, max_points).astype(np.int32)
                cameraHeight = self.attribute["cameraHeight"]
                layoutHeight = self.attribute["layoutHeight"]
                y_max = self.camera_position[1] + cameraHeight
                y_min = self.camera_position[1] + cameraHeight - layoutHeight
                self.attribute["layoutWalls"]["num"] = 0
                self.attribute["layoutWalls"]["walls"] = []
                for i, ptsIdx0 in enumerate(ptsIdxs):
                    ptsIdx1 = ptsIdxs[(i + 1) % len(ptsIdxs)]
                    pt0 = np.array(self.attribute["layoutPoints"]["points"][ptsIdx0]["xyz"])
                    pt1 = np.array(self.attribute["layoutPoints"]["points"][ptsIdx1]["xyz"])
                    ceil0 = pt0.copy()
                    ceil0[1] = y_min
                    ceil1 = pt1.copy()
                    ceil1[1] = y_min
                    floor0 = pt0.copy()
                    floor0[1] = y_max
                    floor1 = pt1.copy()
                    floor1[1] = y_max
                    normal = calc_normal(ceil0, ceil1, floor1)
                    D = calc_equation(ceil0, normal)
                    normal = normal.tolist()
                    planeEquation = normal + [D.item()]
                    wall = {
                        "id": 0,
                        "pointsIdx": [
                            ptsIdx0,
                            ptsIdx1
                        ],
                        "normal": normal,
                        "planeEquation": planeEquation
                    }
                    self.attribute["layoutWalls"]["walls"] += [wall]
                    self.attribute["layoutWalls"]["num"] += 1

        if "layoutWalls" in self.attribute:
            self.attribute["cameraHeight"] *= layoutDownsample
            self.attribute["layoutHeight"] *= layoutDownsample
            self.camera_position *= layoutDownsample
            

            cameraHeight = self.attribute["cameraHeight"]
            layoutHeight = self.attribute["layoutHeight"]
            self.y_max = self.camera_position[1] + cameraHeight
            self.y_min = self.camera_position[1] + cameraHeight - layoutHeight

            # (r"""
            #     Update in 2023.08.05
            #     For
            #         test layout-widen to avoid artifacts(all pixels of image are black/white)
            #     """)
            # self.y_max *= layoutScale[1]
            # self.y_min *= layoutScale[1]
            # (r"""End update in 2023.08.05""")

            layoutWalls = []

            for wall in self.attribute["layoutWalls"]["walls"]: #################################
                ptsIdx0, ptsIdx1 = wall["pointsIdx"]
                pt0 = np.array(self.attribute["layoutPoints"]["points"][ptsIdx0]["xyz"]) * layoutDownsample
                pt1 = np.array(self.attribute["layoutPoints"]["points"][ptsIdx1]["xyz"]) * layoutDownsample
                # (r"""
                # Update in 2023.08.05
                # For
                #     test layout-widen to avoid artifacts(all pixels of image are black/white)
                # """)
                # pt0 *= np.Floatndarray(layoutScale)
                # pt1 *= np.Floatndarray(layoutScale)
                # (r"""End update in 2023.08.05""")
                ceil0 = pt0.copy()
                ceil0[1] = self.y_min
                ceil1 = pt1.copy()
                ceil1[1] = self.y_min
                floor0 = pt0.copy()
                floor0[1] = self.y_max
                floor1 = pt1.copy()
                floor1[1] = self.y_max

                wall["planeEquation"][-1] *= layoutDownsample # downsample for layout
                # (r"""
                # Update in 2023.08.05
                # For
                #     test layout-widen to avoid artifacts(all pixels of image are black/white)
                # """)
                # wall["planeEquation"][-1] *= layoutScale[0]
                # (r"""End update in 2023.08.05""")

                layoutWalls += [LayoutWall(threshold=threshold, ceil0=ceil0, ceil1=ceil1, floor0=floor0, floor1=floor1, **wall)]
            self.attribute["layoutWalls"] = layoutWalls
    
    
    def isEmpty(self, x : np.ndarray):
        """
        x: [N, 3]
        """

        threshold = self.attribute["threshold"]
        y_max = self.y_max
        y_min = self.y_min

        insideWalls = (x[..., 0] != x[..., 0])
        insideCFs = ((x[..., 1] >= y_max) & (x[..., 1] - y_max <= threshold)) | ((x[..., 1] <= y_min) & (y_min - x[..., 1] <= threshold))
        # CFs_cond = (x[..., 0] != x[..., 0])
        intersects = np.zeros_like(x[..., 0]).astype(np.int32)

    
        for wall in self.attribute["layoutWalls"]:
            insideWalls = insideWalls | wall.inside(x)
            intersects += wall.intersect(x).astype(np.int32)

        # insideCFs = insideCFs & CFs_cond
        insideWalls = insideWalls & (intersects % 2 == 0)
        insideCFs = insideCFs & (intersects % 2 == 1)

        return  ~insideWalls & ~insideCFs  # ~insideWalls & ~insideCFs
    
    def insideCF(self, x : np.ndarray):
        """
        x: [N, 3]
        """

        threshold = self.attribute["threshold"]
        y_max = self.y_max
        y_min = self.y_min

        insideCFs = ((x[..., 1] >= y_max) & (x[..., 1] - y_max <= threshold)) | ((x[..., 1] <= y_min) & (y_min - x[..., 1] <= threshold))
        # CFs_cond = (x[..., 0] != x[..., 0])
        intersects = np.zeros_like(x[..., 0]).astype(np.int32)

    
        for wall in self.attribute["layoutWalls"]:
            intersects += wall.intersect(x).astype(np.int32)

        insideCFs = insideCFs & (intersects % 2 == 1)

        return  insideCFs  # ~insideWalls & ~insideCFs

    def inRoom(self, x : np.ndarray):
        """
        x: [N, 3]
        """
        y_max = self.y_max
        y_min = self.y_min

        intersects = np.zeros_like(x[..., 0]).astype(np.int32)

        for wall in self.attribute["layoutWalls"]:
            wall : LayoutWall
            intersects += wall.intersect(x).astype(np.int32)
        
        # return (x[..., 1] <= y_max) & (x[..., 1] >= y_min)
    
        inRoom = (intersects % 2 == 1) & (x[..., 1] <= y_max) & (x[..., 1] >= y_min)

        return  inRoom
    
    def getFloorCeilPoints(self, y, num_pts, y_normal):
        points = []
        normals = []

        x_min = 1e9
        x_max = -1e9
        z_min = 1e9
        z_max = -1e9

        for wall in self.attribute["layoutWalls"]:
            wall : LayoutWall
            for p in [wall.attribute["floor0"], wall.attribute["floor1"]]:
                x_max = max(x_max, p[0])
                x_min = min(x_min, p[0])
                z_max = max(z_max, p[2])
                z_min = min(z_min, p[2])

        num = 0

        while num < num_pts:
            x = np.random.random(num_pts) * (x_max - x_min) + x_min
            z = np.random.random(num_pts) * (z_max - z_min) + z_min
            xyz = np.stack((x, np.ones(num_pts) * y, z), -1)
            mask = self.insideCF(xyz)
            xyz = xyz[mask]
            points += [xyz]
            num += len(xyz)
            print(f"num: {num}")

        points = np.concatenate(points)
        normals = np.ones_like(points) * np.array([0.0, y_normal, 0.0]) 

        return points, normals

    
    def toPoints(self, num_pts):
        points = []
        normals = []

        W = 0
        H = 0

        for wall in self.attribute["layoutWalls"]:
            wall : LayoutWall
            W += wall.width
            H = wall.height
        
        num_pts = int(np.ceil(num_pts / 3))

        num_pts_y = int(np.ceil(np.sqrt(num_pts * H / W)))

        print(f"H: {H}, W: {W}, num_pts_y: {num_pts_y}")

        # num_pts_per = int(np.ceil(num_pts / (len(self.attribute["layoutWalls"]))))

        for wall in self.attribute["layoutWalls"]:
            wall : LayoutWall
            num_pts_per = int(np.ceil(num_pts * wall.width / W))
            p, n = wall.toPoints(num_pts=num_pts_per, num_pts_y=num_pts_y)
            points += [p]
            normals += [n]
        
        y_min = self.y_min
        y_max = self.y_max  

        for yy, yn in zip([y_min, y_max], [-1, 1]):
            p, n = self.getFloorCeilPoints(yy, num_pts, yn)
            points += [p]
            normals += [n]

        points = np.concatenate(points)
        normals = np.concatenate(normals)
        num_pts_real = len(points)
        colors = np.random.random((num_pts_real, 3)) / 255.0
        return points, normals, colors
    
