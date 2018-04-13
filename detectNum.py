import os
import sys
import glob
from classification import identify_digits_from_file

def detectNum(input_path):

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

    print(results)

    return(results)

if __name__ == '__main__':
    #change working directory so that relative paths for saved models work as expected
    # (if any 'file not found' issues, please change to absolute paths in config.py)
    script_dir=os.path.dirname(sys.argv[0])
    os.chdir(script_dir)

    input_path=sys.argv[1]
    detectNum(input_path)
