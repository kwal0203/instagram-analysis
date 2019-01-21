# Issues:
#   1. Do I have to query API for each image?
#   2. Image with model not looking at camera
#   3. Image with just a sign in it
#   4. Duplicate images
#   5. Products very similar in appearance, do we need to distinguish?
#   6. What about images with just a model and no product?
#   7. Unrelated images (there's a Starbucks one)
#   8. Overexposed image where it's hard to see product
#   9. Some product image (eg BByfXl7NLth.jpg, BbywMU_g_Sq.jpg) don't return
#      product label
#      * Should we just use 'non-face' as product strategy?
#   10. How to classify RGB dominant colours:
#      * Distance measure to all 'common' colours?
#      * Doesn't seem to be a simple mapping from RGB to 'colour names'
#
# TODO:
#   1. Do functions for faces, labels, text detection (DHC), image properties
#   (colour)
#   2. Write results to CSV
#   3. Rename image files and put Instagram likes in the CSV
#
# Ideas:
#   1. Images with models (celebrity) will having higher likes
#   2. Psychological effect of different dominant colours
#   3. Seems to be DHC on a lot of product photos
#   4.

# Code:
import io
import argparse


# Use this function if you want to search the image for labels only. The
# results from this type of search will include 'product' if the model detects
# an object that it believes to be some type of product i.e. a beauty product.
#
# Usage: python3 main.py labels /path/to/image
def detect_labels(path):
    """Detects labels in the file."""
    from google.cloud import vision
    client = vision.ImageAnnotatorClient.from_service_account_json('key.json')

    with io.open(path, 'rb') as image_file:
        content = image_file.read()

    image = vision.types.Image(content=content)
    response = client.label_detection(image=image)
    # response = client.annotate_image({
    #     'image': {'source': {'image_uri': path}},
    #     'features': [{'type': vision.enums.Feature.Type.FACE_DETECTION}],
    # })

    print("Response:")
    print(response)
    labels = response.label_annotations
    print('Labels:')

    for label in labels:
        print(label.description)


# Use this function if you want to search the image for faces only.
#
# Usage: python3 main.py faces /path/to/image
def detect_faces(path):
    """Detects faces in an image."""
    from google.cloud import vision
    client = vision.ImageAnnotatorClient.from_service_account_json('key.json')

    with io.open(path, 'rb') as image_file:
        content = image_file.read()

    image = vision.types.Image(content=content)

    response = client.face_detection(image=image)
    faces = response.face_annotations

    likelihood_name = ('UNKNOWN', 'VERY_UNLIKELY', 'UNLIKELY', 'POSSIBLE',
                       'LIKELY', 'VERY_LIKELY')
    print('Faces:')

    for face in faces:
        print('anger: {}'.format(likelihood_name[face.anger_likelihood]))
        print('joy: {}'.format(likelihood_name[face.joy_likelihood]))
        print('surprise: {}'.format(likelihood_name[face.surprise_likelihood]))

        vertices = (['({},{})'.format(vertex.x, vertex.y)
                    for vertex in face.bounding_poly.vertices])

        print('face bounds: {}'.format(','.join(vertices)))


# Use this function if you want to find the dominant colours of an image.
#
# Usage: python3 main.py colour /path/to/image
def detect_colours(path):
    """Detects image properties in the file."""
    from google.cloud import vision
    client = vision.ImageAnnotatorClient.from_service_account_json('key.json')

    # [START vision_python_migration_image_properties]
    with io.open(path, 'rb') as image_file:
        content = image_file.read()

    image = vision.types.Image(content=content)

    response = client.image_properties(image=image)
    props = response.image_properties_annotation

    # Sort dominant colours by 'score' property
    props.dominant_colors.colors.sort(key=lambda x: x.score, reverse=True)

    # Get the first element which should be the colour with the highest score
    tmp_colour = props.dominant_colors.colors[0]

    # Print RGB values for dominant colour
    red = tmp_colour.color.red
    blue = tmp_colour.color.blue
    green = tmp_colour.color.green
    # print("Colours: Red - {}, Blue - {}, Green - {}".format(red, blue, green))

    # Determine whether Red dominates Blue
    (dominant_colour, num) =\
        ("Red", red) if red > blue else ("Blue", blue)
    # print("Dominant colour: {}".format(dominant_colour))

    # Determine whether Green dominates the maximum of Red and Blue
    (dominant_colour, num) =\
        ("Green", green) if green > num else (dominant_colour, num)
    print('The dominant colour of the image is: {}'.format(dominant_colour))


# Use this function if you want to search the image for faces and products at
# the same time.
#
# Usage: python3 main.py both /path/to/image
def detect_both(path):
    """Detects labels and faces in the file."""
    from google.cloud import vision
    client = vision.ImageAnnotatorClient.from_service_account_json('key.json')

    with io.open(path, 'rb') as image_file:
        content = image_file.read()

    # Detect labels section
    image = vision.types.Image(content=content)
    response = client.label_detection(image=image)
    labels = response.label_annotations

    print("------- LABELS ------")
    print('Labels:')
    for label in labels:
        print(label.description)
    print("----- END LABELS -----")

    # Detect faces section
    print("------- FACES ------")
    response = client.face_detection(image=image)
    faces = response.face_annotations

    for face in faces:
        vertices = (['({},{})'.format(vertex.x, vertex.y)
                     for vertex in face.bounding_poly.vertices])

        print('face bounds: {}'.format(','.join(vertices)))
    print("------- END FACES ------")


# Run appropriate function depending on what command the user entered.
def run_local(local_args):
    if args.command == 'faces':
        detect_faces(local_args.path)
    elif args.command == 'labels':
        detect_labels(local_args.path)
    elif args.command == 'colour':
        detect_colours(local_args.path)
    elif args.command == 'both':
        detect_both(local_args.path)


# Main program driver
if __name__ == '__main__':
    # Main parser
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)

    # Add sub-parser for commandline argument. This sub-parser allows the user
    # to type what type of analysis (i.e. faces, labels or both) they would
    # like to run on the commandline.
    subparsers = parser.add_subparsers(dest='command')

    # Add label option to parser (detect labels only)
    detect_labels_parser = subparsers.add_parser(
            'labels', help=detect_labels.__doc__)
    detect_labels_parser.add_argument('path')

    # Add colour option to parser (detect dominant colours only)
    detect_colour_parser = subparsers.add_parser(
        'colour', help=detect_both.__doc__)
    detect_colour_parser.add_argument('path')

    # Add 'both' option to parser (detect faces and labels)
    detect_faces_parser = subparsers.add_parser(
        'faces', help=detect_faces.__doc__)
    detect_faces_parser.add_argument('path')

    # Add face option to parser (detect faces only)
    detect_both_parser = subparsers.add_parser(
        'both', help=detect_both.__doc__)
    detect_both_parser.add_argument('path')

    # Collect all commands entered by user.
    args = parser.parse_args()

    # Run analysis according to user input
    run_local(args)
