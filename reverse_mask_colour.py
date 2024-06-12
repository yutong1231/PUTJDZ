from PIL import Image
import os

def swap_black_white(image_path,output_path):
    try:
        # 打开图片
        image = Image.open(image_path)
        
        # 转换为RGB模式（确保图像是RGB格式）
        image = image.convert('RGB')
        
        # 获取图像的宽度和高度
        width, height = image.size
        
        # 遍历每个像素点，并进行颜色交换
        for x in range(width):
            for y in range(height):
                r, g, b = image.getpixel((x, y))
                # 互换黑色和白色部分的颜色
                if r == g == b:  # 如果三个通道的颜色值相同，说明是灰度颜色，即黑色部分
                    new_color = (255 - r, 255 - g, 255 - b)  # 交换颜色
                    image.putpixel((x, y), new_color)
        
        # 保存处理后的图像
        image.save(os.path.join(output_path, os.path.basename(image_path)))
            
    except IOError:
        print("无法打开图片文件：" + image_path)

if __name__ == '__main__':
    # 输入要处理的黑白图片路径
    #input_path = '/disks/sda/yutong2333/PUT-main/data/irregular-mask/test'  # 将'path_to_your_image.png'替换为你的图片路径
    #input_path='/disks/sda/yutong2333/PUT-main/data/irregular-mask/train'
    #output_path = '/disks/sda/yutong2333/PUT-main/data/irregular-mask/test'  # 将'path_to_your_image.png'替换为你的图片路径
    #output_path='/disks/sda/yutong2333/PUT-main/data/irregular-mask/white_mask'
    input_path='/disks/sda/yutong2333/PUT-main/data/JDG/test_mask'
    output_path='/disks/sda/yutong2333/PUT-main/data/JDG/test_mask'
    for img in os.listdir(input_path):
        image_path = os.path.join(input_path, img)
        swap_black_white(image_path,output_path)
