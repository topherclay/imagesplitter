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
            intersect_point = round(-1 * y_intercept / slope)
        except ZeroDivisionError:
            intersect_point = -1
        if (intersect_point >= 0) and (intersect_point <= self.x_dim_len):
            boundary_intersects["top"] = True
            boundary_intersects["top_point"] = (intersect_point, 0)
            # vertical slope should be infinite but isnt, this will catch this case.
            if (slope < -1000) or (slope > 1000):
                boundary_intersects["top_point"] = (point[0], 0)

        # left
        intersect_point = round(y_intercept)
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
        intersect_point = round((slope * self.x_dim_len) + y_intercept)
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

            # self.dummy_draw_roi_boundary_on_frame(rois[0])

        # Is this return necessary? Do any calls of this function use the value?
        return current_angle_rois


    def all_splits_of_all_angles(self, step_size):

        for angle in range(0, 180):
            self.all_splits_of_one_angle(angle, step_size)


        print("got all splits")

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

    def find__if_top_or_bottom_slice_is_best(self, bottom_slice, top_slice, rois_list):

        # this will give us 5 if the bottom splice is [1: 10]
        bottom_roi_index = round(len(rois_list[bottom_slice[0]:bottom_slice[1]]) / 2)

        # this will give us 15 if the top splice is [10: 20]
        top_roi_index = round(len(rois_list[top_slice[0]: top_slice[1]]) / 2 + top_slice[0])


        bottom_roi_score = self.get_distance_in_color(rois_list[bottom_roi_index])
        top_roi_score = self.get_distance_in_color(rois_list[top_roi_index])

        self.dummy_draw_roi_boundary_on_frame(rois_list[top_roi_index][0])
        self.dummy_draw_roi_boundary_on_frame(rois_list[bottom_roi_index][0])


        return bottom_roi_score, top_roi_score


    def dummy_quick_sort_biggest_difference_of_one_angle(self, angle):

        rois_list = self.split_info.rois[angle]

        # find middle of rois
        middle_index = round(len(rois_list) / 2)
        middle_rois = rois_list[middle_index]

        print(self.get_distance_in_color(middle_rois))


        middle_score = self.get_distance_in_color(middle_rois)

        bottom_score = 0
        top_score = 0
        current_score = 0

        bottom_slice_index = [0, middle_index]
        top_slice_index = [round((len(rois_list) - middle_index) / 2) + middle_index, len(rois_list)]

        while not (current_score > top_score and current_score > bottom_score):

            print("entered loop")
            current_score = middle_score


            bottom_score, top_score = self.find__if_top_or_bottom_slice_is_best(
                bottom_slice_index, top_slice_index, rois_list)

            print(f"bottom score = {bottom_score}")
            print(f"middle_score = {middle_score}")
            print(f"top_score = {top_score}")

            if top_score > middle_score:
                current_score = top_score
                middle_index = round((top_slice_index[1] - top_slice_index[0]) / 2 + top_slice_index[0])
                bottom_slice_index = [top_slice_index[0], middle_index]
                top_slice_index = [middle_index, top_slice_index[1]]

            if bottom_score > middle_score:
                current_score = bottom_score
                middle_index = round((bottom_slice_index[1] - bottom_slice_index[0]) / 2 + bottom_slice_index[0])
                top_slice_index = [middle_index, bottom_slice_index[1]]
                bottom_slice_index = [bottom_slice_index[0], middle_index]

                middle_score = current_score





        print(middle_index)

    def find_biggest_difference_split_of_one_angle(self, angle):

        # todo: find out what to do with ties...


        rois_list = self.split_info.rois[angle]

        previous_best_difference = 0
        best_index = None
        for index in range(len(rois_list)):

            current_difference = self.get_distance_in_color(rois_list[index])

            if current_difference > previous_best_difference:
                previous_best_difference = current_difference
                best_index = index

        print(best_index)
        self.dummy_draw_roi_boundary_on_frame(rois_list[best_index][0])


        # func get difference in color


        # take all rois of one angle
        # add full slice to "rois to be considered"
        # cut rois to be considered in half and check middle
        # save difference and add top slice and bottom slice as two "rois to be considered"
        # cut top slice in half and check
        # cut bottom slice in hlf and check
        # add best slice to "rois to be considered"


    def get_distance_in_color(self, rois):

        left_roi = rois[0]
        right_roi = rois[1]

        left_mask = make_mask_from_roi(self.frame, left_roi)
        right_mask = make_mask_from_roi(self.frame, right_roi)

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




    # unused function
    def go_through_masks(self):

        for angle in range(0, 179):
            for index in range(0, len(self.split_info.rois[angle])):

                mask = self.make_mask_from_roi(self.split_info.rois[angle][index][0])
                masked_frame = self.apply_mask(mask)


        print("done!")




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






def make_mask_from_roi(frame, roi):

    points = []
    for point_tuple in roi:
        points.append(list(point_tuple))


    points = np.asarray(points, np.int32)

    mask = np.zeros(frame.shape[:2], dtype=np.uint8)

    white = (255, 255, 255)
    mask = cv2.fillConvexPoly(img=mask, points=points, color=white)

    # mask = cv2.bitwise_and(src1=frame, src2=frame, mask=mask)

    # cv2.imshow("mask", mask)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    return mask


# defunct?
def apply_mask(frame, mask):

    new_image = cv2.bitwise_and(src1=frame, src2=frame, mask=mask)
    return new_image



def calculate_color_distance():
    # take color as three channel and convert it to LAB and calculate
    pass


def calculate_region_distance():
    # calculate region color average
    # get distance of those colors
    pass


def dummy_audit_angles(frame):

    step = 20

    fresh = np.array(frame.frame)

    for angle in range(0, 179):

        frame.all_splits_of_one_angle(angle, step)

        cv2.imshow("poop", frame.frame)
        frame.frame = np.array(fresh)

        cv2.waitKey(0)






if __name__ == '__main__':
    source = cv2.imread("star.jpg")
    current_frame = FrameSplitter(source)


    current_frame.all_splits_of_one_angle(15, 20)
    current_frame.find_biggest_difference_split_of_one_angle(15)



    print("end")
    # current_frame.all_splits_of_all_angles(30)





    # current_frame.go_through_masks()


    # current_frame.all_splits_of_all_angles(20)

    # dummy_audit_angles(current_frame)

    # new_mask = make_mask_from_roi(current_frame.frame, current_frame.split_info.rois[1][5][1])
    #
    # poop = apply_mask(current_frame.frame, new_mask)
    # find_average_color_in_roi(current_frame.frame, new_mask)


    # cv2.imshow("mask", poop)
    cv2.imshow("poop", current_frame.frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

