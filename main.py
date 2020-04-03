import numpy as np
import cv2
import math



class FrameSplitter:

    def __init__(self, frame, margin=0):
        assert margin >= 0
        assert margin <= 100
        self.frame = frame
        # Probably just going to remove margin. Have the program ignore small slices instead but still slice them
        self.margin = margin
        # maybe this needs to be a percentage
        self.x_dim_len = self.frame.shape[1]
        self.y_dim_len = self.frame.shape[0]
        self.left_side_start_point = self.get_left_side_starting_point()
        self.right_side_start_point = self.get_right_side_start_point()
        self.split_info = SplitInfo()

    @staticmethod
    def get_left_side_starting_point():

        # no longer using margins
        # x_start = int(self.x_dim_len * float(self.margin / 100)) + 1
        # y_start = int(self.y_dim_len * float(self.margin / 100)) + 1

        x_start = 1
        y_start = 1


        return x_start, y_start


    def get_right_side_start_point(self):
        # no longer using margins
        # x_start = self.x_dim_len - int(self.x_dim_len * float(self.margin / 100)) - 1
        # y_start = int(self.y_dim_len * float(self.margin / 100)) - 1

        x_start = self.x_dim_len - 1
        y_start = 1

        return x_start, y_start

    @staticmethod
    def get_slope_from_deg(degree):
        pi = 22/7
        radian = degree * (pi/180)
        slope = math.tan(radian)
        return slope

    @staticmethod
    def get_y_intercept_value(slope, point):
        x = point[0]
        y = point[1]
        m = slope
        y_intercept_value = (-1 * m * x) + y
        return y_intercept_value

    def find_boundary_intersects(self, point, slope, y_intercept):

        boundary_intersects = {"top": False, "right": False, "bottom": False, "left": False}

        # top
        try:
            intersect_point = int(-1 * y_intercept / slope)
        except ZeroDivisionError:
            intersect_point = -1
        if (intersect_point >= 0) and (intersect_point <= self.x_dim_len):
            boundary_intersects["top"] = True
            boundary_intersects["top_point"] = (intersect_point, 0)
            # vertical slope should be infinite but isnt, this will catch this case.
            if (slope < -1000) or (slope > 1000):
                boundary_intersects["top_point"] = (point[0], 0)

        # left
        intersect_point = int(y_intercept)
        if (intersect_point > 0) and (intersect_point < self.y_dim_len):
            boundary_intersects["left"] = True
            boundary_intersects["left_point"] = (0, intersect_point)

        # bottom
        try:
            intersect_point = int((self.y_dim_len - y_intercept) / slope)
        except ZeroDivisionError:
            intersect_point = -1
        if (intersect_point >= 0) and (intersect_point <= self.x_dim_len):
            boundary_intersects["bottom"] = True
            boundary_intersects["bottom_point"] = (intersect_point, self.y_dim_len)
            # vertical slope should be infinite but isn't, this will catch this case.
            if (slope < -1000) or (slope > 1000):
                boundary_intersects["bottom_point"] = (point[0], self.y_dim_len)

        # right
        intersect_point = int((slope * self.x_dim_len) + y_intercept)
        if (intersect_point > 0) and (intersect_point < self.y_dim_len):
            boundary_intersects["right"] = True
            boundary_intersects["right_point"] = (self.x_dim_len, intersect_point)




        # print(boundary_intersects)

        return boundary_intersects

    def get_two_rois_from_slope_and_intercepts(self, slope, intercepts):
        top_left = (0, 0)
        top_right = (self.x_dim_len, 0)
        bottom_left = (0, self.y_dim_len)
        bottom_right = (self.x_dim_len, self.y_dim_len)

        right_roi = None
        left_roi = None

        # horizontal splits
        if intercepts["left"] and intercepts["right"]:
            # tilts right
            if slope >= 0:
                right_roi = (top_left, top_right, intercepts["right_point"], intercepts["left_point"])
                left_roi = (intercepts["left_point"], intercepts["right_point"], bottom_right, bottom_left)
            # tilts left
            if slope < 0:
                right_roi = (intercepts["left_point"], intercepts["right_point"], bottom_right, bottom_left)
                left_roi = (top_left, top_right, intercepts["right_point"], intercepts["left_point"])

        # vertical splits
        if intercepts["top"] and intercepts["bottom"]:
            # debug print
            right_roi = (intercepts["top_point"], top_right, bottom_right, intercepts["bottom_point"])
            left_roi = (top_left, intercepts["top_point"], intercepts["bottom_point"], bottom_left)

        # top right corner
        if intercepts["top"] and intercepts["right"]:
            right_roi = (intercepts["top_point"], top_right, intercepts["right_point"])
            left_roi = (top_left, intercepts["top_point"], intercepts["right_point"], bottom_right, bottom_left)

        # bottom right corner
        if intercepts["right"] and intercepts["bottom"]:
            right_roi = (intercepts["right_point"], bottom_right, intercepts["bottom_point"])
            left_roi = (top_left, top_right, intercepts["right_point"], intercepts["bottom_point"], bottom_left)

        # bottom left corner
        if intercepts["left"] and intercepts["bottom"]:
            right_roi = (top_left, top_right, bottom_right, intercepts["bottom_point"], intercepts["left_point"])
            left_roi = (intercepts["left_point"], intercepts["bottom_point"], bottom_left)

        # top left corner
        if intercepts["top"] and intercepts["left"]:
            right_roi = (intercepts["top_point"], top_right, bottom_right, bottom_left, intercepts["left_point"])
            left_roi = (top_left, intercepts["top_point"], intercepts["left_point"])

        return left_roi, right_roi

    def split_into_rois_with_angle_and_point(self, angle, point):
        assert angle >= 0
        assert angle <= 179


        slope = self.get_slope_from_deg(angle)
        y_intercept = self.get_y_intercept_value(slope, point)
        boundary_intersects = self.find_boundary_intersects(point, slope, y_intercept)
        left_roi, right_roi = self.get_two_rois_from_slope_and_intercepts(slope, boundary_intersects)


        return left_roi, right_roi


    def get_point_one_step_perpendicular_from_angle_and_point(self, angle, point, step):

        if angle <= 90:
            perpendicular_angle = angle + 90
        else:
            perpendicular_angle = angle - 90

        perpendicular_slope = self.get_slope_from_deg(perpendicular_angle)


        if angle < 90:
            x_axis_change = 1 * step / math.sqrt(1+math.pow(perpendicular_slope, 2))
        else:
            x_axis_change = -1 * step / math.sqrt(1 + math.pow(perpendicular_slope, 2))



        new_x = point[0] - x_axis_change
        new_b = self.get_y_intercept_value(perpendicular_slope, point)

        new_y = perpendicular_slope * new_x + new_b



        new_point = (round(new_x), round(new_y))

        return new_point


    def all_splits_of_one_angle(self, angle, step_size):
        assert angle >= 0
        assert angle <= 179

        current_angle_rois = self.split_info.rois[angle]

        if angle < 90:
            point = self.right_side_start_point
        if angle >= 90:
            point = self.left_side_start_point

        while True:

            point = self.get_point_one_step_perpendicular_from_angle_and_point(angle, point, step_size)

            left_roi, right_roi = self.split_into_rois_with_angle_and_point(angle, point)

            if not left_roi or not right_roi:
                # If ROI are not filled, then that means the point was invalid,

                break

            rois = [left_roi, right_roi]
            current_angle_rois.append(rois)

            self.dummy_draw_roi_boundary_on_frame(rois[0])


        return current_angle_rois




    def all_splits_of_all_angles(self, step_size):

        for angle in range(0, 180):
            self.all_splits_of_one_angle(angle, step_size)





    def dummy_draw_roi_boundary_on_frame(self, vertices):


        vertices = list(vertices)


        roi_vertices = np.array(vertices, np.int32)


        self.frame = cv2.polylines(self.frame, [roi_vertices], True, (0, 0, 255))




class SplitInfo:

    def __init__(self):
        self.rois = self.make_angle_array()
        self.color_pair = self.make_angle_array()


    @staticmethod
    def make_angle_array():
        matrix_row = []
        for index in range(0, 180):
            matrix_row.append([])

        return matrix_row


    # func+attr for finding color differences between a pair
    # in lab and rgb btw
    # func+attr for finding highest color diff of all pairs

def dummy_audit_angles(frame):

    step = 20

    fresh = np.array(frame.frame)

    for angle in range(0, 179):

        frame.all_splits_of_one_angle(angle, step)

        cv2.imshow("poop", frame.frame)
        frame.frame = np.array(fresh)

        cv2.waitKey(0)





if __name__ == '__main__':
    source = cv2.imread("input.jpg")
    current_frame = FrameSplitter(source)


    # current_frame.all_splits_of_one_angle(0, 50)

    # current_frame.all_splits_of_one_angle(46, 20)

    # current_frame.all_splits_of_all_angles(20)

    dummy_audit_angles(current_frame)


    # test



    cv2.imshow("poop", current_frame.frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

