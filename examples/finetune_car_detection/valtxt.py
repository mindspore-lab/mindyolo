import os

# 指定训练图像的目录
train_images_dir = './images/val'

# 创建一个空列表来存储所有训练图像的路径
train_image_paths = []

# 遍历训练图像目录下的所有文件
for filename in os.listdir(train_images_dir):
    if filename.endswith('.jpg'):
        # 构建完整的图像路径
        #full_path = os.path.join(train_images_dir, filename)
        full_path = train_images_dir+'/'+filename
        # 将图像路径添加到列表中
        train_image_paths.append(full_path)

# 指定输出的train.txt文件路径
output_file ='val.txt'

# 打开输出文件，准备写入图像路径
with open(output_file, 'w') as f:
    # 遍历图像路径列表并将每个路径写入文件
    for path in train_image_paths:
        f.write(f"{path}\n")

print(f"Successfully created train.txt at {output_file}")
  