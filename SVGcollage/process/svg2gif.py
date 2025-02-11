import os
import cairosvg
from PIL import Image
import io
import re

def valid_svg_files(folder_path):
    # 获取指定文件夹中的所有文件列表
    files = os.listdir(folder_path)
    
    # 正则表达式匹配数字部分
    pattern = re.compile(r'^\d+\.svg$')
    
    # 用来存储匹配的文件名
    matching_files = []
    
    # 遍历文件列表
    for file in files:
        # 使用正则表达式匹配文件名中的数字部分
        if pattern.match(file):
            matching_files.append(file)

    # 对匹配的文件名按照数字进行排序
    matching_files.sort(key=lambda x: int(re.match(r'^(\d+)\.svg$', x).group(1)))
    # print(matching_files)

    return matching_files

def generate_gif(processId):
    # 定义输入文件夹路径和输出GIF文件路径
    input_folder = "cache/" + str(processId)
    output_path = "cache/"+ str(processId) + ".gif"
    # 创建一个列表来存储每个SVG文件的帧
    frames = []

    #有效文件列表
    files= valid_svg_files(input_folder)

    #计数器 
    count=0

    # 循环处理每个 SVG 文件
    for file in files:
        input_path = os.path.join(input_folder, file)
        
        # 检查文件是否存在
        if os.path.exists(input_path) and input_path.endswith('.svg') and count%5==0:# 5个为间隔，防止掉帧
            # 打开SVG文件并转换成PIL Image对象
            with open(input_path, 'rb') as svg_file:
                svg_data = svg_file.read()
                img = Image.open(io.BytesIO(cairosvg.svg2png(svg_data)))
                # 设置统一大小
                img = img.resize((600, 600))
                mask = Image.new("RGBA", img.size, (255, 255, 255, 255))
                f = img.copy().convert("RGBA")
                frames.append(Image.alpha_composite(mask, f))

        count+=1
    

    # 保存帧列表为GIF文件
    frames[0].save(output_path, format='GIF', append_images=frames[1:], save_all=True, duration=50, disposal=2,transparency=0,loop=0)

    print("GIF Conversion completed!")
