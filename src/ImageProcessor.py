import io

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

        # TODO: Probably can remove 'number' variable here, I think it's just
        #       for debug.
        #       Cleanup comparison of values.
        # Determine whether Red dominates Blue
        (dominant_colour, num) = \
            ("Red", red) if red > blue else ("Blue", blue)

        # Determine whether Green dominates the maximum of Red and Blue
        (dominant_colour, num) = \
            ("Green", green) if green > num else (dominant_colour, num)

        return dominant_colour
