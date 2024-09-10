import os

# 指定文件扩展名和起始编号
extension = '.jpg'
start_num = 1

# 获取当前目录下所有的.txt文件
files = [f for f in os.listdir('images/val') if f.endswith(extension)]

for i, file in enumerate(files, start=start_num):
    # 构造新的文件名，使用zfill填充前导零
    new_filename = f"{i:06d}{extension}"

    # 使用os.rename重命名文件
    # 需根据自己的路径修改
    os.chdir('images/val')
    os.rename(file, new_filename)

    #print(f"Renamed '{file}' to '{new_filename}'")

print("All files have been renamed.")
