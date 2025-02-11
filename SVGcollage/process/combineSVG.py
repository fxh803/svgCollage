from lxml import etree
import os
def merge_svgs(svg1_path, svg2_path):
     # 检查第一个 SVG 文件是否存在
    if not os.path.exists(svg1_path):
        return

    # 检查第二个 SVG 文件是否存在
    if not os.path.exists(svg2_path):
        return
    # 解析第一个 SVG 文件
    tree1 = etree.parse(svg1_path)
    root1 = tree1.getroot()

    # 解析第二个 SVG 文件
    tree2 = etree.parse(svg2_path)
    root2 = tree2.getroot()

    # 获取第二个 SVG 文件的所有子元素并添加到第一个 SVG 的根元素中
    for elem in root2:
        root1.append(elem)

    # 将合并后的 SVG 转换为字符串
    merged_svg = etree.tostring(root1, pretty_print=True, encoding="UTF-8").decode()

    # 返回合并后的 SVG 内容
    return merged_svg

