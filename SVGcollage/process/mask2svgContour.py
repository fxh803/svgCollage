import cv2
import numpy as np
import svgwrite
from io import BytesIO
from PIL import Image
import os
def export_contours_to_svg(contours, output_file):
    # 限制 SVG 图像的尺寸为 600x600 像素
    width, height = 600, 600

    # 创建 SVG 图像，指定尺寸
    dwg = svgwrite.Drawing(output_file, profile='tiny', size=(width, height))

    # 绘制每个轮廓
    for cnt in contours:
        # 检查轮廓点的形状
        print(f"Contour shape: {cnt.shape}")

        # 将轮廓点转换为 SVG 路径命令
        path_data = "M" + " ".join(f"{point[0]},{point[1]}" for point in cnt[:, 0, :])
        # 添加闭合路径命令
        path_data += "Z"  
        # 创建 SVG 路径元素并添加到图像中
        dwg.add(dwg.path(d=path_data, fill="none", stroke="black"))

    # 保存 SVG 文件
    dwg.save()

def mask2svgContour(maskBytes):

    #读取mask
    image_stream = BytesIO(maskBytes)
    image = Image.open(image_stream)
    image = image.convert('L')  # 转换为灰度图像
    # 将图像转换为 NumPy 数组
    np_image = np.array(image, dtype=np.uint8)
    # 将图像缩放到 600x600 像素
    resized_image = cv2.resize(np_image, (600, 600), interpolation=cv2.INTER_AREA)
    # 通过形态学操作获得边界
    _, thresh = cv2.threshold(resized_image, 128, 255, cv2.THRESH_BINARY)
    kernel = np.ones((3, 3), np.uint8)
    edges = cv2.dilate(thresh, kernel) - thresh
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 生成包含所有轮廓的 SVG 文件
    # 创建保存文件的目录
    output_dir = "cache"
    os.makedirs(output_dir, exist_ok=True)
    export_contours_to_svg(contours, f'cache/background.svg')