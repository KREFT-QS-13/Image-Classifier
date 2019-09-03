import argparse

def get_extra_input():
    parser = argparse.ArgumentParser(description='Some optional settings')
    
    parser.add_argument('path_to_image', type = str, help = 'enter path to image, you would like to classify')
    parser.add_argument('checkpoint', type = str, help = 'enter only the name of checkpoint file from directory default_saving_folder' )
    
    parser.add_argument('--top_k', type = int, default=5, help = 'enter how many most possible classes would you like to see')
    parser.add_argument('--category_names', type = str, default = 'cat_to_name.json', help = 'enter JSON file that maps the class values to other category names' )
    parser.add_argument('--gpu', action="store_true", default=False, help = 'enter if you want to use GPU')
    
    return parser.parse_args()
    