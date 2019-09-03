import argparse 

def get_extra_input():
    parser = argparse.ArgumentParser(description='Some optional settings')

    parser.add_argument('data_directory', type = str, help = 'eneter path to data directory')
    
    parser.add_argument('--save_dir', type = str, default = 'default_saving_folder', help = 'enter path to directory, where you want to save the output')
    parser.add_argument('--arch', type = str, default = 'vgg', choices=['alexnet', 'vgg'] ,help = 'enter the name of the architecture' )
    parser.add_argument('--learning_rate', type = float, default = 0.001, help = 'enter the value for learningrate')
    parser.add_argument('--hidden_units', type = int, default = 2048, help = 'enter the number of hidden units')
    parser.add_argument('--epochs', type = int, default = 20, help = 'enter how many steps should the algorithm does')
    parser.add_argument('--gpu', action="store_true", default = False, help = 'enter if you want to use GPU')
    
    return parser.parse_args()