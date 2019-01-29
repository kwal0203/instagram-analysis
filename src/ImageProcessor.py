import io
import cv2
import math
import requests
import numpy as np
import cognitive_face
from scipy.ndimage.filters import generic_filter

from google.cloud import vision
from google.cloud.vision import types
from google.cloud.vision import enums


# This class is responsible for obtaining/parsing information received from
# Google/Microsoft Computer Vision APIs and extracting visual 'features' from
# images using OpenCV. The features obtained from the APIs and OpenCV
# processing will be used in regression analysis.
class ImageProcessor:
    def __init__(self, path):
        self.path = path
        self.client = vision.ImageAnnotatorClient.\
            from_service_account_json('key.json')
        self.opened_file = io.open(self.path, 'rb').read()
        self.opened_file_cv2 = cv2.imread(path)
        self.image = vision.types.Image(content=self.opened_file)
        self.microsoft_key = '2d8273cbfa7a42beaeefa81b444fa472'
        self.azure_url = 'https://australiaeast.api.cognitive.microsoft.com/'
        self.vision_url = self.azure_url + 'vision/v2.0/analyze'
        self.face_url = self.azure_url + 'face/v1.0/'

    def google_request(self):
        # Possible features:
        # LABEL_DETECTION, FACE_DETECTION, LOGO_DETECTION, TEXT_DETECTION,
        # DOCUMENT_TEXT_DETECTION, SAFE_SEARCH_DETECTION, WEB_DETECTION,
        # LANDMARK_DETECTION, IMAGE_PROPERTIES
        features = [
            types.Feature(type=enums.Feature.Type.LABEL_DETECTION),
        ]

        api_requests = []
        image = types.Image(content=self.opened_file)

        request = types.AnnotateImageRequest(image=image, features=features)
        api_requests.append(request)

        return self.client.batch_annotate_images(api_requests)

    def microsoft_face_request(self):
        cognitive_face.Key.set(self.microsoft_key)
        cognitive_face.BaseUrl.set(self.face_url)
        attributes = 'age,gender,smile,emotion'
        faces = cognitive_face.face.detect(self.path, attributes=attributes)

        # List of facial attributes we can get:
        # age, gender, headPose, smile, facialHair, glasses, emotion, hair,
        # makeup, occlusion, accessories, blur, exposure, noise
        # print("Number of faces:   {}".format(len(faces)), end='\n\n')
        # for face in faces:
        #     curr_emotion = max(
        #         face['faceAttributes']['emotion'].keys(),
        #         key=(lambda key: face['faceAttributes']['emotion'][key]))
        #
        #     print("Smile:   {}".format(face['faceAttributes']['smile']))
        #     print("Gender:  {}".format(face['faceAttributes']['gender']))
        #     print("Age:     {}".format(face['faceAttributes']['age']))
        #     print("Emotion: {}".format(curr_emotion))
        #     print("\n")

        return faces

    def microsoft_cv_request(self):
        # Read the image into a byte array
        headers = {'Ocp-Apim-Subscription-Key': self.microsoft_key,
                   'Content-Type': 'application/octet-stream'}
        params = {'visualFeatures': 'Categories,Description,Color'}
        response = requests.post(
            self.vision_url, headers=headers, params=params,
            data=self.opened_file)
        response.raise_for_status()

        # Analysis is a JSON object that contains:
        # Categories, color, description, requestId, metadata
        analysis = response.json()
        # print("Dominant foreground colour: {}".format(
        #     analysis['color']['dominantColorForeground']))
        # print("Dominant background colour: {}".format(
        #     analysis['color']['dominantColorBackground']))
        # print("Categories: ", end="")
        # for tag in analysis['description']['tags']:
        #     print(tag)

        return analysis

    def image_colorfulness(self):
        # split the image into its respective RGB components
        (B, G, R) = cv2.split(self.opened_file_cv2.astype("float"))

        # compute rg = R - G
        rg = np.absolute(R - G)

        # compute yb = 0.5 * (R + G) - B
        yb = np.absolute(0.5 * (R + G) - B)

        # compute the mean and standard deviation of both `rg` and `yb`
        (rbMean, rbStd) = (np.mean(rg), np.std(rg))
        (ybMean, ybStd) = (np.mean(yb), np.std(yb))

        # combine the mean and standard deviations
        std_root = np.sqrt((rbStd ** 2) + (ybStd ** 2))
        mean_root = np.sqrt((rbMean ** 2) + (ybMean ** 2))

        # derive the "colorfulness" metric and return it
        return round(std_root + (0.3 * mean_root))

    def number_of_lines(self):
        gray = cv2.cvtColor(self.opened_file_cv2, cv2.COLOR_BGR2GRAY)
        re_sized = cv2.resize(gray, (0, 0), fx=0.5, fy=0.5)
        edges = cv2.Canny(re_sized, 50, 150, apertureSize=3)
        lines = cv2.HoughLines(edges, 1, np.pi / 180, 200)

        if lines is None:
            return 0
        else:
            return len(lines)

    def smooth(self):
        # return the percentage of smooth areas"
        gray = cv2.cvtColor(self.opened_file_cv2, cv2.COLOR_BGR2GRAY)
        filtered_image = generic_filter(gray, np.std, size=3)
        smooth_area = filtered_image == 0
        percent = np.count_nonzero(smooth_area) / smooth_area.size

        return round(percent, 2)

    def detect_all(self):
        # Get information from Google Vision API
        print("----- Return all information from Google Vision API -----")
        google_response = self.google_request()
        # print(google_response)

        # Get information from Microsoft Face API
        print("----- Return all information from Microsoft Face API -----")
        microsoft_face_response = self.microsoft_face_request()
        # print(microsoft_face_response)

        # Get information from Microsoft Computer Vision API
        print("----- Return all information from Microsoft Object API -----")
        microsoft_cv_response = self.microsoft_cv_request()
        # print(microsoft_cv_response)

        # Use OpenCV to evaluate 'colourfulness' of an image
        print("----- OpenCV: Colour evaluation -----")
        colour_response = self.image_colorfulness()
        # print(colour_response)

        # Use OpenCV to determine how many lines are in an image
        print("----- OpenCV: Line evaluation -----")
        line_response = self.number_of_lines()
        # print(line_response)

        # Use OpenCV to determine percentage of image that is 'smooth'
        print("----- OpenCV: Smoothness evaluation -----")
        smooth_response = self.smooth()
        # print(smooth_response)

        return (google_response, microsoft_face_response, microsoft_cv_response,
                colour_response, line_response, smooth_response)

    # The following functions can be used to query API services individually
    # rather than doing everything together as above in 'detect_all().

    # This function detects faces in a given image using the Google Cloud Vision
    # API.
    def detect_faces(self):
        # print("Detecting faces in file: {}".format(self.path))
        response = self.client.face_detection(image=self.image)
        faces = response.face_annotations

        # If at least one face is found we determine this image to be the 'model'
        # strategy.
        return False if len(faces) == 0 else True

    # This function detects the dominant colours in a given image using the
    # Google Cloud Vision API.
    def detect_colours(self):
        # print("Detecting colours in file: {}".format(self.path))
        response = self.client.image_properties(image=self.image)
        props = response.image_properties_annotation

        # Sort dominant colours by 'score' property
        props.dominant_colors.colors.sort(key=lambda col: col.score, reverse=True)

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

