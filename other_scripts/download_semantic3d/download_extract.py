import py7zr
import os


def un_zip(path, out_path):
    a = py7zr.SevenZipFile(path, 'r')
    a.extractall(path=out_path)
    a.close()
    print('unzip', path, out_path)


def un_zip_Tree(path, out_dir):
    if not os.path.exists(path):
        os.makedirs(path)
    for file in os.listdir(path):
        print('file', file)
        Local = os.path.join(path, file)
        if os.path.isdir(file):
            if not os.path.exists(Local):
                os.makedirs(Local)
            un_zip_Tree(path)
        else:
            if os.path.splitext(Local)[1] == '.7z':
                un_zip(Local, out_dir)


dir = '/data1/zhaolichen/data/semantic3d/original_data'
dir = 'C:/Users/ZLC/Desktop/pycodes/data/semantic3d/original_data'
out_dir = dir + '/entry'
if __name__ == '__main__':
    un_zip_Tree(dir, out_dir)
# for entry in "$BASE_DIR"/*
# do
#   7z x "$entry" -o$(dirname "$entry") -y
# done

# mv $BASE_DIR/station1_xyz_intensity_rgb.txt $BASE_DIR/neugasse_station1_xyz_intensity_rgb.txt

# for entry in "$BASE_DIR"/*.7z
# do
#   rm "$entry"
# done
