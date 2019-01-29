import optparse
import os


# This class is responsible for doing 'setup' work for the application. This
# includes making sure the output directories exist and setting up the
# command-line parser to accept user input.
class Utilities:
    # Setup the parser for user input on command-line.
    @staticmethod
    def setup_parser():
        tmp_parser = optparse.OptionParser()
        tmp_parser.add_option("-d", "--details", dest="details", type="string",
                              help="path to details CSV file")
        tmp_parser.add_option("-l", "--labels", dest="labels", type="string",
                              help="path to labels CSV file")
        tmp_parser.add_option("-i", "--image_dir", dest="image_dir", type="string",
                              help="name of directory holding images")
        tmp_parser.add_option("-a", "--new_details", dest="new_details",
                              type="string", help="output details path")
        tmp_parser.add_option("-b", "--new_image_dir", dest="new_image_dir",
                              type="string", help="output image directory")
        tmp_parser.set_defaults(details="2526_details/details.csv",
                                labels="2526_details/labels.csv",
                                image_dir="2526_images/", new_details="output/",
                                new_image_dir="output/images")

        return tmp_parser.parse_args()

    # Make sure output directories exists before program runs.
    @staticmethod
    def create_directories(detail_dir, image_dir):
        if not os.path.exists(detail_dir):
            os.makedirs(detail_dir)
        if not os.path.exists(image_dir):
            os.makedirs(image_dir)
