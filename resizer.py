

import cv2



class ImageResizer:

    def __init__(self, image_object, width_goal):
        self.image = image_object
        self.width_goal = width_goal
        self.resized_image = self.resize()

    def resize(self):

        current_width = self.image.shape[1]

        percentage_goal = self.width_goal / current_width

        new_width = int(self.image.shape[1] * percentage_goal)
        new_height = int(self.image.shape[0] * percentage_goal)
        new_dim = (new_width, new_height)

        resized_image = cv2.resize(self.image, new_dim, interpolation=cv2.INTER_AREA)
        return resized_image


if __name__ == "__main__":
    image = cv2.imread("inputs/IMG_20200308_180750.jpg")



