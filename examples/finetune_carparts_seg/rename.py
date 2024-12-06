import os

def modify_file_name():
    jpg_extension = '.jpg'
    txt_extension = '.txt'
    start_num = 1

    for data_dir in ["train", "test", "valid"]:
        # get all .txt filename
        img_files = [f[:-4] for f in os.listdir("./{}/images".format(data_dir)) if f.endswith(jpg_extension)]

        for i, file in enumerate(img_files, start=start_num):
            pwd = os.getcwd()
            os.chdir("./{}/images".format(data_dir))

            new_img_filename = f"{i:04d}{jpg_extension}"
            new_label_filename = f"{i:04d}{txt_extension}"

            os.rename(file+jpg_extension, new_img_filename)
            os.chdir("../labels/")
            os.rename(file+txt_extension, new_label_filename)
            os.chdir(pwd)
        print("All {} dataset have been renamed.".format(data_dir))

    print("All files have been renamed.")

def create_txt():
    for data_dir in ["train", "test", "valid"]:
        img_dir =  "./" + data_dir + "/images"
        img_paths = []
        all_img_paths = os.listdir(img_dir)
        all_img_paths.sort()

        for filename in all_img_paths:
            if filename.endswith('.jpg'):
                full_path = img_dir+'/'+filename
                img_paths.append(full_path)

        output_file = data_dir + ".txt"
        with open(output_file, 'w') as f:
            for path in img_paths:
                f.write(f"{path}\n")

    print(f"Successfully created file at {output_file}")

if __name__ == "__main__":
    modify_file_name()
    create_txt()