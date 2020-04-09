import numpy as np
import cv2
import math
import time
import os.path
import resizer
from shapely import geometry





class FrameSplitter:

    def __init__(self, frame):
        self.frame = frame
        self.example_frame = np.copy(frame)
        self.color_frame = None
        self.dual_panel_frame = None
        self.x_dim_len = self.frame.shape[1]
        self.y_dim_len = self.frame.shape[0]
        self.left_side_start_point = self.get_left_side_starting_point()
        self.right_side_start_point = self.get_right_side_start_point()
        self.frame_area = self.get_frame_area()
        self.split_info = SplitInfo()


    def get_frame_area(self):

        poly = geometry.Polygon(((0, 0), (0, self.y_dim_len), (self.x_dim_len, self.y_dim_len), (self.x_dim_len, 0)))
        return poly.area





    @staticmethod
    def get_left_side_starting_point():
        x_start = 1
        y_start = 1
        return x_start, y_start


    def get_right_side_start_point(self):

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
    def get_deg_from_slope(slope):
        pi = 22/7
        radian = math.atan(slope)
        degree = radian / (pi/180)
        return degree


    @staticmethod
    def get_y_intercept_value(slope, point):
        x = point[0]
        y = point[1]
        m = slope
        y_intercept_value = (-1 * m * x) + y
        return y_intercept_value

    def find_boundary_intersects(self, point, slope, y_intercept):

        boundary_intersects = {"top": False, "right": False, "bottom": False, "left": False,
                               "top_point": [-1, -1], "right_point": [-1, -1], "bottom_point": [-1, -1], "left_point": [-1, -1]}

        # top
        try:
            # top boundary is y = 0
            # x = -b / m
            intersect_point = -1 * y_intercept / slope
        except ZeroDivisionError:
            intersect_point = -1
        if (intersect_point >= 0) and (intersect_point <= self.x_dim_len):
            boundary_intersects["top"] = True
            boundary_intersects["top_point"] = (round(intersect_point), 0)
            # vertical slope should be infinite but isnt, this will catch this case.
            if (slope < -1000) or (slope > 1000):
                boundary_intersects["top_point"] = (point[0], 0)

        # left
        # left boundary is x = 0
        # y = b
        intersect_point = y_intercept
        if (intersect_point > 0) and (intersect_point <= self.y_dim_len - 1):
            boundary_intersects["left"] = True
            boundary_intersects["left_point"] = (0, round(intersect_point))

        # bottom
        try:
            # bottom boundary is y = y_dim_len
            # x = (y_dim_len - b) / m
            intersect_point = (self.y_dim_len - y_intercept) / slope
        except ZeroDivisionError:
            intersect_point = -1
        if (intersect_point >= 0) and (intersect_point <= self.x_dim_len):
            boundary_intersects["bottom"] = True
            boundary_intersects["bottom_point"] = (round(intersect_point), self.y_dim_len)
            # vertical slope should be infinite but isn't, this will catch this case.
            if (slope < -1000) or (slope > 1000):
                boundary_intersects["bottom_point"] = (point[0], self.y_dim_len)

        # right
        # right boundary is x = x_dim_len
        # y = (m * x_dim_len) + b
        intersect_point = (slope * self.x_dim_len) + y_intercept
        if (intersect_point > 0) and (intersect_point <= self.y_dim_len - 1):
            boundary_intersects["right"] = True
            boundary_intersects["right_point"] = (self.x_dim_len, round(intersect_point))

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





    def all_splits_of_all_angles(self, step_size):

        for angle in range(0, 180):
            self.all_splits_of_one_angle(angle, step_size)



    def make_mask_from_roi(self, roi):

        points = []
        for point_tuple in roi:
            points.append(list(point_tuple))

        points = np.asarray(points, np.int32)

        mask = np.zeros(self.frame.shape[:2], dtype=np.uint8)

        white = (255, 255, 255)
        mask = cv2.fillConvexPoly(img=mask, points=points, color=white)

        return mask

    def apply_mask(self, mask):

        new_image = cv2.bitwise_and(src1=self.frame, src2=self.frame, mask=mask)
        return new_image


    def find_biggest_difference_split_of_one_angle(self, angle):

        # todo: find out what to do with ties...
        self.split_info.masks = self.split_info.make_angle_array()
        self.split_info.save_masks_for_given_angle(self.frame, angle)


        rois_list = self.split_info.rois[angle]
        masks = self.split_info.masks[angle]


        previous_best_difference = 0
        best_index = None
        for index in range(len(rois_list)):

            area_of_roi = geometry.Polygon(rois_list[index][0]).area
            percentage_of_area_of_frame = area_of_roi / self.frame_area

            if percentage_of_area_of_frame > 0.9 or percentage_of_area_of_frame < 0.1:
                current_difference = 0
            else:
                current_difference = self.get_distance_in_color(rois_list[index], masks[index])


            if current_difference > previous_best_difference:
                previous_best_difference = current_difference
                best_index = index



        self.split_info.best_rois[angle] = rois_list[best_index]
        self.split_info.best_index_of_angle[angle] = best_index
        self.split_info.difference_for_best_rois[angle] = previous_best_difference





    def find_biggest_difference_of_all_angles(self):

        for angle in range(0, 179):
            self.find_biggest_difference_split_of_one_angle(angle)


        previous_best_difference = 0
        for angle in range(0, 179):
            current_difference = self.split_info.difference_for_best_rois[angle]

            if current_difference > previous_best_difference:
                previous_best_difference = current_difference
                best_angle = angle



        self.color_in_splits(best_angle)


        self.draw_roi_boundary_on_frame(self.split_info.best_rois[best_angle][0])




        








    def get_distance_in_color(self, rois, masks):

        left_mask = masks[0]
        right_mask = masks[1]


        left_color = cv2.mean(self.frame, left_mask)
        right_color = cv2.mean(self.frame, right_mask)


        left_rgb = np.zeros([5, 5, 3], np.uint8)
        right_rgb = np.zeros([5, 5, 3], np.uint8)

        left_rgb[::][::] = left_color[:3]
        right_rgb[::][::] = right_color[:3]

        left_lab = cv2.cvtColor(left_rgb, cv2.COLOR_RGB2LAB)
        right_lab = cv2.cvtColor(right_rgb, cv2.COLOR_RGB2LAB)

        left_color = left_lab[0][0]
        right_color = right_lab[0][0]

        distance = self.get_three_d_distance(left_color, right_color)

        return distance

    @staticmethod
    def get_three_d_distance(point_1, point_2):

        x_distance = (int(point_1[0]) - int(point_2[0])) * (int(point_1[0]) - int(point_2[0]))
        y_distance = (int(point_1[1]) - int(point_2[1])) * (int(point_1[1]) -int(point_2[1]))
        z_distance = (int(point_1[2]) - int(point_2[2])) * (int(point_1[2]) - int(point_2[2]))

        distance = math.sqrt(x_distance + y_distance + z_distance)
        return distance


        # convert side to LAB
        # apply mean with LAB and mask
        # compare difference
        pass


    def draw_roi_boundary_on_frame(self, vertices):

        vertices = list(vertices)

        roi_vertices = np.array(vertices, np.int32)

        cv2.polylines(self.example_frame, [roi_vertices], True, (0, 0, 255))


    def color_in_splits(self, angle):
        frame = self.frame

        left_mask = make_mask_from_roi(frame, self.split_info.best_rois[angle][0])
        right_mask = make_mask_from_roi(frame, self.split_info.best_rois[angle][1])


        # get color in RGB
        left_color = cv2.mean(self.frame, left_mask)
        right_color = cv2.mean(self.frame, right_mask)

        # make blank frames
        left_rgb = np.zeros([self.y_dim_len, self.x_dim_len, 3], np.uint8)
        right_rgb = np.zeros([self.y_dim_len, self.x_dim_len, 3], np.uint8)

        # color in full frames
        left_rgb[::][::] = left_color[:3]
        right_rgb[::][::] = right_color[:3]

        # blank frame to be added to
        self.color_frame = np.ones([self.y_dim_len, self.x_dim_len, 3], np.uint8)

        # cut off corners of the colored frames
        left_filled = cv2.add(left_rgb, self.color_frame, mask=left_mask)
        right_filled = cv2.add(right_rgb, self.color_frame, mask=right_mask)

        # combine both colored frames to final
        self.color_frame = cv2.add(left_filled, right_filled)


    def create_dual_pane_display(self):

        self.dual_panel_frame = np.concatenate((self.example_frame, self.color_frame), axis=1)






class SplitInfo:

    def __init__(self):
        self.rois = self.make_angle_array()
        self.color_pair = self.make_angle_array()
        self.masks = self.make_angle_array()
        self.best_rois = self.make_angle_array()
        self.best_index_of_angle = self.make_angle_array()
        self.difference_for_best_rois = self.make_angle_array()



    @staticmethod
    def make_angle_array():
        matrix_row = []
        for index in range(0, 180):
            matrix_row.append([])

        return matrix_row

    def save_masks_for_given_angle(self, frame, angle):


        for index in range(len(self.rois[angle])):
            left_mask = make_mask_from_roi(frame, self.rois[angle][index][0])
            right_mask = make_mask_from_roi(frame, self.rois[angle][index][1])
            masks = [left_mask, right_mask]
            self.masks[angle].append(masks)





def make_mask_from_roi(frame, roi):

    points = []
    for point_tuple in roi:
        points.append(list(point_tuple))


    points = np.asarray(points, np.int32)

    mask = np.zeros(frame.shape[:2], dtype=np.uint8)

    white = (255, 255, 255)
    mask = cv2.fillConvexPoly(img=mask, points=points, color=white)


    return mask



def dummy_audit_angles(frame):

    step = 20

    fresh = np.array(frame.frame)

    for angle in range(0, 179):

        frame.all_splits_of_one_angle(angle, step)

        cv2.imshow("poop", frame.frame)
        frame.frame = np.array(fresh)

        cv2.waitKey(0)


def get_source_file_from_input_folder():
    input_folder = os.path.join(os.getcwd(), "inputs")

    if not os.path.exists(input_folder):
        os.mkdir(input_folder)
        raise RuntimeError(f"{input_folder} didn't exist, now it does. Please place photo in it.")

    file_paths = []
    for (dirs, dir_name, file_names) in os.walk(input_folder):
        for file_name in file_names:
            file_name = file_name
            file_paths.append(os.path.join(input_folder, file_name))

    if not file_paths:
        raise RuntimeError("put some pictures in the folder ya dingus")


    return file_paths


def run_one_input(frame, name):
    start_time = time.time()

    step_size = round(frame.x_dim_len / 20)

    frame.all_splits_of_all_angles(step_size)

    splits_done_time = time.time() - start_time

    print(f"\tsplits done in {round(splits_done_time, 2)} seconds")
    frame.find_biggest_difference_of_all_angles()

    masks_done_time = time.time() - start_time - splits_done_time
    print(f"\tMasks finished after {round(masks_done_time, 2)} seconds.")


    frame.create_dual_pane_display()

    save_to_output_folder(frame.dual_panel_frame, name, "output_")
    # save_to_output_folder(frame.example_frame, name, "red_line")
    # save_to_output_folder(frame.color_frame, name, "color")














def run_all_inputs(paths):

    print(f"Running {len(paths)} picture(s).")

    for index in range(0, len(paths)):
        print(f"now running: {os.path.basename(paths[index])}")
        print(f"Picture {index} of {len(paths)}.")
        print(f"{round(index / len(paths), 2)} of the way done")
        source = cv2.imread(paths[index])

        resizer_lol = resizer.ImageResizer(source, 600)

        source = resizer_lol.resized_image


        current_frame = FrameSplitter(source)
        run_one_input(current_frame, paths[index])
    print("done with all!")


def save_to_output_folder(output_file, output_name, prefix):
    output_name = f"{prefix}_{os.path.basename(output_name)}"

    output_folder = os.path.join(os.getcwd(), "outputs")


    if not os.path.exists(output_folder):
        os.mkdir(output_folder)

    image_in_folder = os.path.join(output_folder, output_name)
    rel_name_for_example = os.path.relpath(image_in_folder)


    cv2.imwrite(rel_name_for_example, output_file)






if __name__ == '__main__':


    source_path = get_source_file_from_input_folder()

    run_all_inputs(source_path)




    cv2.destroyAllWindows()

