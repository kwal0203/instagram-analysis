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

    # Read CSV file provided by company (DHC)
    original_csv = pd.read_csv(opts.details)

    # Initialize empty list which will be used to store relevant details
    # for each image i.e. details extracted from DHC original_csv and other
    # detected features during analysis.
    new_csv = []

    # Iterate through each line of original_csv, processing images as we
    # go along.
    count = 0
    for index, row in original_csv.iterrows():
        short_code = row['shortcode']
        likes = row['edge_liked_by_count']
        followers = row['user_followers']
        posts = row['user_posts']
        following = row['user_following']
        file_name = short_code + ".jpg"
        tmp_path = os.path.join(opts.image_dir, file_name)

        # Check if file exists and process if it does
        if os.path.isfile(tmp_path):
            file_name = str(row.name) + ".jpg"
            goog_cv, msft_face, msft_cv = process_image(tmp_path, file_name)

            faces = len(msft_face)

            # FIX
            # new_csv.append((file_name, short_code, likes, followers, posts,
            #                 following, model, product, dom_colour))
            new_csv.append((file_name, short_code, likes, followers, posts,
                            following, faces))
        else:
            print("Image short-code: {} not found".format(short_code))

        # Convert all detected features/relevant information to Pandas
        # DataFrame and then convert that to CSV format for saving on disk.
        # column_names = ['file_name', 'short_code', 'likes', 'followers',
        #                 'posts', 'following', 'model', 'product',
        #                 'dominant_colour']
        column_names = ['file_name', 'short_code', 'likes', 'followers',
                        'posts', 'following', 'model', 'product',
                        'dominant_colour']
        frame = pd.DataFrame(new_csv, columns=column_names)
        frame.to_csv('output/details.csv', index=None)

        if count >= 0:
            break
        count += 1
