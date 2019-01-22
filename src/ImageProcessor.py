import io
import math

from google.cloud import vision


# This class is responsible for doing all of the 'image analysis'. This
# analysis includes using Google Cloud Vision API and OpenCV to obtain various
# features associated with a given image. These features are subsequently used
# in a regression analysis.
class ImageProcessor:
    def __init__(self, path):
        self.path = path
        self.client = vision.ImageAnnotatorClient.\
            from_service_account_json('key.json')
        self.opened_file = io.open(self.path, 'rb').read()
        self.image = vision.types.Image(content=self.opened_file)

    # This function detects faces in a given image using the Google Cloud Vision
    # API.
    def detect_faces(self):
        print("Detecting faces in file: {}".format(self.path))
        response = self.client.face_detection(image=self.image)
        faces = response.face_annotations

        # If at least one face is found we determine this image to be the 'model'
        # strategy.
        return False if len(faces) == 0 else True

    # This function detects the dominant colours in a given image using the
    # Google Cloud Vision API.
    def detect_colours(self):
        print("Detecting colours in file: {}".format(self.path))
        response = self.client.image_properties(image=self.image)
        props = response.image_properties_annotation

        # Sort dominant colours by 'score' property
        props.dominant_colors.colors.sort(key=lambda x: x.score, reverse=True)

        # Get the first element of sorted list i.e. element with highest score
        tmp_colour = props.dominant_colors.colors[0]

        red = tmp_colour.color.red
        blue = tmp_colour.color.blue
        green = tmp_colour.color.green

        # Basic colour RGB values:
        # https://www.rapidtables.com/web/color/RGB_Color.html
        # basic_colours = [("red", 255, 0, 0), ("lime", 0, 255, 0),
        #                  ("blue", 0, 0, 255), ("yellow", 255, 255, 0),
        #                  ("cyan", 0, 255, 255), ("magenta", 255, 0, 255)]
        # ("white", 0, 0, 0), ("black", 255, 255, 255),
        basic_colours = [("red", 255, 0, 0), ("lime", 0, 255, 0),
                         ("blue", 0, 0, 255), ("yellow", 255, 255, 0),
                         ("cyan", 0, 255, 255), ("magenta", 255, 0, 255),
                         ("silver", 192, 192, 192), ("gray", 128, 128, 128),
                         ("maroon", 128, 0, 0), ("olive", 128, 128, 0),
                         ("green", 0, 128, 0), ("purple", 128, 0, 128),
                         ("teal", 0, 128, 128), ("navy", 0, 0, 128)]

        # Max distance is between Black (0, 0, 0) and White (255, 255, 255).
        # Therefore maximum distance is sqrt(3 * 255^2) = 442 so set dist to
        # a value greater than 441.
        dist = 1000
        dominant_colour = ""
        for colour in basic_colours:
            tmp_red = colour[1]
            tmp_green = colour[2]
            tmp_blue = colour[3]
            x = (red - tmp_red) ** 2
            y = (green - tmp_green) ** 2
            z = (blue - tmp_blue) ** 2
            tmp_dist = math.sqrt(x + y + z)
            if tmp_dist < dist:
                dist = tmp_dist
                dominant_colour = colour[0]

        # Determine whether Red dominates Blue
        # (dominant_colour, num) = \
        #     ("Red", red) if red > blue else ("Blue", blue)
        #
        # Determine whether Green dominates the maximum of Red and Blue
        # (dominant_colour, num) = \
        #     ("Green", green) if green > num else (dominant_colour, num)

        return dominant_colour
