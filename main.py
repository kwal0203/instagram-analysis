import pandas as pd
import shutil
import os

from src.ImageProcessor import ImageProcessor
from src.Utilities import Utilities


# This function renames files so that they are in the form 'INTEGER.jpg'
# i.e. 0.jpg, 1.jpg, where the INTEGER corresponds with the relevant row number
# in the output CSV file.
def process_image(image_path, name):
    # Rename file and place in output directory
    dest = "output/images/" + name
    shutil.copyfile(image_path, dest)

    # Initialize ImageProcessor
    image_processor = ImageProcessor(image_path)

    return image_processor.detect_all()


# Start program
if __name__ == '__main__':
    # Create a Utilities object which will read user input and make sure all
    # all needed directories are created or exist
    util = Utilities()

    # Read user input
    opts, args = util.setup_parser()

    # Create output directories
    util.create_directories(opts.new_details, opts.new_image_dir)

    # Read CSV files provided by company (DHC)
    original_csv = pd.read_csv(opts.details)
    label_csv = pd.read_csv(opts.labels)

    # Initialize empty list which will be used to store relevant details
    # for each image i.e. details extracted from DHC original_csv and other
    # detected features during analysis.
    new_csv = []

    # Count variable is used to keep track of (and print on screen) how many
    # lines we have processed so far.
    count = 1

    # Timer is used to stop the for-loop when it reaches a certain number
    timer = 0

    # The following for-loop goes through each line of the 'original_csv' file
    # provided by the company. Each row corresponds to an Instagram image and
    # contains information we need including Instagram likes, followers, posts
    # etc.
    for index, row in original_csv.iterrows():
        # The following lines store the information found in the CSV and
        # updates each time we go to a new line.
        likes = row['edge_liked_by_count']
        followers = row['user_followers']
        posts = row['user_posts']
        following = row['user_following']
        short_code = row['shortcode']
        if short_code[-1] == "'":
            short_code = short_code[:-1]
        original_file_name = short_code + ".jpg"

        # We create a computer system PATH (like a location) to the Instagram
        # image that the current row is associated with.
        tmp_path = os.path.join(opts.image_dir, original_file_name)

        # Check if the PATH we created actually leads to a file (our PATH may
        # not lead to a file if we made a mistake or the file is missing)
        if os.path.isfile(tmp_path):
            # Print number of lines processed so far
            print("Count: {}".format(count))

            # Because the Instagram images have messy names like
            # 'BcMylPTlU4N.jpg', this line re-names it to: 'ROW_NUMBER.jpg'
            # i.e. if we are on row 5, the new name of the file will be '5.jpg'
            file_name = str(row.name) + ".jpg"

            # Call the process_image function (see above around line 13 of this
            # code) and five the function the file path we created.
            # goog_cv, msft_face, msft_cv, saturation, lines, smooth =\
            #     process_image(tmp_path, file_name)

            response_list = process_image(tmp_path, file_name)

            # The following lines get the output of the Google API (object
            # detection) and create a string containing names of all objects
            # detected.
            labels = ""
            space = False
            # for label in goog_cv.responses[0].label_annotations:
            for label in response_list[0].responses[0].label_annotations:
                if space:
                    labels += " "
                space = True
                labels += label.description

            # The following lines use the output of the Microsoft Azure Face
            # API
            msft_face = response_list[1]
            faces = len(msft_face)
            model_strategy = (faces > 0)
            product_strategy = not model_strategy

            # Default attributes to use if no faces are found.
            smile = False
            gender = "unknown"
            age = -1
            emotion = "unknown"
            model_and_product = False

            # TODO: Fix for multiple faces
            # The following lines record characteristics of detected faces
            # such as whether they are smiling/what emotion is being displayed
            # etc.
            if msft_face:
                smile = msft_face[0]['faceAttributes']['smile'] >= 0.5
                gender = msft_face[0]['faceAttributes']['gender']
                age = msft_face[0]['faceAttributes']['age']
                emotion = max(
                    msft_face[0]['faceAttributes']['emotion'].keys(),
                    key=(lambda key:
                         msft_face[0]['faceAttributes']['emotion'][key]))

                # Set threshold for model + product strategy here
                file_column = label_csv[original_file_name]
                model_and_product = max(file_column) > 0.35

            # Colour attributes from Microsoft CV API
            msft_cv = response_list[2]
            dom_fore_colour = msft_cv['color']['dominantColorForeground']
            dom_back_colour = msft_cv['color']['dominantColorBackground']

            # Return OpenCV responses
            colourfulness = response_list[3]
            lines = response_list[4]
            smooth = response_list[5]
            saturation = response_list[6]
            brightness = response_list[7]
            contrast = response_list[8]
            clarity = response_list[9]
            hue = response_list[10]
            balance = response_list[11]

            # Put all the features we have detected into a Python list.
            new_csv.append((file_name, short_code, likes, followers, posts,
                            following, faces, model_strategy,
                            product_strategy, model_and_product, smile, gender,
                            age, emotion, dom_fore_colour, dom_back_colour,
                            labels, colourfulness, lines, smooth, saturation,
                            brightness, contrast, clarity, hue, balance))
        else:
            print("Image short-code: {} not found".format(short_code))

        # Convert the list with all detected features into a Pandas DataFrame.
        column_names = ['file_name', 'short_code', 'likes', 'followers',
                        'posts', 'following', 'faces', 'model_strategy',
                        'product_strategy', 'model_product_strategy', 'smile',
                        'gender', 'age', 'emotion', 'dom_fore_colour',
                        'dom_back_colour', 'labels', 'colourfulness', 'lines',
                        'smoothness', 'saturation', 'brightness', 'contrast',
                        'clarity', 'hue', 'balance']
        frame = pd.DataFrame(new_csv, columns=column_names)

        # Convert Pandas DataFrame to CSV file and store on disk. This file can
        # be used for the statistical analysis.
        frame.to_csv('output/details.csv', index=None)

        # Set timer value to stop program at here
        if timer >= 8:
            break
        count += 1
