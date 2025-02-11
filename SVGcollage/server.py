from flask import Flask, send_file,render_template,request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
import base64
# from process.main import *
from process.pack_all_in_one import *
from process.svg2gif import *
from process.mask2svgContour import *
from process.combineSVG import merge_svgs
import io
from PIL import Image
import glob
app = Flask(__name__)
CORS(app)

processId = []
iteration = 0
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/uploadMask', methods=['POST'])
def uploadMask():
    points = request.json['points'] 
    height = request.json['canvasHeight'] 
    width = request.json['canvasWidth'] 
    image = request.json['image'] 
    # 解码 Base64 数据为字节数据
    image_data = base64.b64decode(image)
    image_stream = io.BytesIO(image_data)
    image = Image.open(image_stream)
    # 确保图像的模式为RGBA（如果不是）
    if image.mode != 'RGBA':
        image = image.convert('RGBA')
    # 拆分图像通道
    _, _, _, alphaImage = image.split()
    # 将图像转换为二值图像
    alphaImage = np.array(alphaImage)
    alphaImage = (alphaImage > 0).astype(np.uint8) * 255

    # 创建单通道空白画布
    pointsImage = np.zeros((height, width), dtype=np.uint8)
    # 解析数据
    for _points in points:
        points_array = np.array([(point['x'], point['y']) for point in _points], dtype=np.int32)
        # 绘制多边形
        cv2.fillPoly(pointsImage, [points_array], color=255)
    
    # 逻辑与运算，叠加图像
    result = cv2.bitwise_or(alphaImage, pointsImage)
    result = cv2.bitwise_not(result)

    hasZero = np.any(result == 0)
    print(hasZero)
    if not hasZero:
        return jsonify({'status': -1})
    
    # 转换为Base64编码的图像数据
    _, buffer = cv2.imencode('.png', result)
    result_base64 = base64.b64encode(buffer).decode('utf-8')

    return jsonify({'canvas': result_base64, 'status': 0})

@app.route('/generateSVG', methods=['POST'])
def generateSVG():
    global processId
    global iteration
    mask = request.json['mask'] 
    svg = request.json['svg'] 
    svg_num= int(request.json['svg_num'] )
    iteration = int(request.json['iteration'] )
    processId = request.json['processId'] 
    svgList = request.json['svgList'] 
    # 将Base64编码的图片数据转换为字节流
    mask_image_bytes = base64.b64decode(mask)
    #保存轮廓
    mask2svgContour(mask_image_bytes)
    if svgList:
        directory = f"./cache/{processId}_svg/"
        os.makedirs(directory, exist_ok=True)
        for i, svg in enumerate(svgList):
            svg_bytes = base64.b64decode(svg)
            with open(f"{directory}{i+1}.svg", "wb") as file:
                file.write(svg_bytes)
        try:
            pack_all_in_one(mask_image_bytes, directory, svg_num, iteration,processId)
            generate_gif(processId)
        except Exception as e:
            error_message = str(e)
            return jsonify({'status': -1, 'error': error_message})
    else:
        try:
            pack_all_in_one(mask_image_bytes, "process/data/"+svg , svg_num, iteration,processId)
            generate_gif(processId)
        except Exception as e:
            error_message = str(e)
            print(e)
            return jsonify({'status': -1, 'error': error_message})
    
    # 返回进度给前端
    return jsonify({'status': 0})

@app.route('/getGIF', methods=['POST'])
def getGIF():
    processId = request.json['processId'] 

    # 从缓存中读取 GIF 图像文件
    gif_file_path = f'cache/{str(processId)}.gif'
    
    try:
        return send_file(gif_file_path, mimetype='image/gif')
    except FileNotFoundError:
        return "GIF file not found", 404


@app.route('/getProgress', methods=['GET'])
def getProgress():
    global processId
    global iteration
    # 文件夹路径
    folder_path = 'cache/'+ str(processId)

    # 构建 SVG 文件的搜索模式
    svg_pattern = os.path.join(folder_path, '*.svg')

    # 使用 glob 模块匹配符合搜索模式的文件列表
    svg_files = glob.glob(svg_pattern)

    # 统计 SVG 文件数量
    svg_count = len(svg_files)
    progress = int((svg_count/iteration)*100)

    #获取最新的svg
    svg_file_path = f'cache/{str(processId)}/{svg_count-1}.svg'
    background_svg_path = 'cache/background.svg'
    svg_content = merge_svgs(svg_file_path,background_svg_path)
    if svg_content:
        # 构建返回的字典
        response = {
            'progress': progress,
            'status': 0,
            'svg': svg_content
        }
    else:
        response = {
            'progress': progress,
            'status': -1,
            'message': 'SVG file not found'
        }
    # 返回字典作为 JSON 格式给前端
    return jsonify(response)


if __name__ == '__main__':
    app.run(debug = True)