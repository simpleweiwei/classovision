import os
import sys
import glob
from classification import identify_digits_from_file

def detectNum(input_path):
    """
    Function searches input image or video (or folder containing images and videos) and returns extracted numbers
    :param input_path: File or folder to be searched: function accepts .jpg or .mov files
    :return:
    """

    results={}
    if not os.path.isdir(input_path):
        # if single file supplied, identify single file and return result
        print("Process file: {}".format(input_path))
        results[input_path] = identify_digits_from_file(input_path)
    else:
        # if folder supplied, run for all .jpg and .mov files in folder (much faster since CNN loads only once)
        for file in glob.glob(os.path.join(input_path,'*')):
            print("Process file: {}".format(file))
            strings_identified=identify_digits_from_file(file)
            results[file] = identify_digits_from_file(file)

    print('{ \n    ' + '\n    '.join([k + ':' + str(results[k]) + ',' for k in results])[:-1] + ' \n}')
    return

if __name__ == '__main__':

    if len(sys.argv) > 1:
        script_dir = os.path.dirname(sys.argv[0])
        input_path = sys.argv[1]
    else:
        script_dir=''
        input_path=r"C:\Users\sarki\AppData\Local\Temp\IMG_0861_3_digit_test.jpg"

    #change working directory so that relative paths for saved models work as expected
    # (if any 'file not found' issues, please change to absolute paths in config.py)
    if script_dir!='':
            os.chdir(script_dir)

    detectNum(input_path)
