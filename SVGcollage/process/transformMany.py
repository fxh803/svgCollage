import cv2
import numpy as np

# 转换边缘到贝塞尔曲线
def fit_bezier_curve(edge_image, num_points=500):
    #使用OpenCV的findContours函数查找边缘图像中的所有轮廓。RETR_EXTERNAL参数表示只检测外部轮廓，CHAIN_APPROX_SIMPLE参数表示使用简单的轮廓逼近方法。
    contours, _ = cv2.findContours(edge_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = list(contours)
    epsilons= []
    approx_curves = []
    bound_count = len(contours)#封闭轮廓的数量
    for i in range(bound_count):
        epsilon = 0.001 * cv2.arcLength(contours[i], True)
        epsilons.append(epsilon)
        approx_curves.append([])
    
    for i in range(bound_count):
        count = 0
        while(abs(len(approx_curves[i]) - num_points/bound_count)>10 and count<10000):
            if(len(approx_curves[i]) > num_points/bound_count):
                epsilons[i] *=1.5
                approx_curves[i] = cv2.approxPolyDP(contours[i], epsilons[i], True) 

            elif(len(approx_curves[i]) < num_points/bound_count):
                epsilons[i] *=0.5
                approx_curves[i] = cv2.approxPolyDP(contours[i], epsilons[i], True) 
            count+=1
        print("调整次数：",count)
    
            
    reshape_array = [np.array(approx_curve) for approx_curve in approx_curves]
    reshape_array = np.vstack(reshape_array)
    reshape_array = reshape_array.reshape(-1,2)
    return reshape_array,len(reshape_array)

# 定义插值函数，用于在两个点之间生成插值点
def interpolate_points(start_point, end_point, t):
    x = int((1 - t) * start_point[0] + t * end_point[0])
    y = int((1 - t) * start_point[1] + t * end_point[1])
    return (x, y)

# 定义插值曲线函数，用于生成两个贝塞尔曲线之间的插值曲线
def interpolate_bezier_curve(start_curve, end_curve, num_frames):
    interpolated_curve = []
    for i in range(num_frames):
        t = i / (num_frames - 1)  # 计算插值参数 t（0~1）
        frame_curve = []
        for j in range(min(len(start_curve), len(end_curve))):
            interpolated_point = interpolate_points(start_curve[j], end_curve[j], t)#插值
            frame_curve.append(interpolated_point)
        interpolated_curve.append(frame_curve)
    return interpolated_curve

def reorder_points(points):
    # 找到最左上角的点及其索引
    leftmost_point = np.min(points, axis=0)
    i = np.argmin(np.linalg.norm(points - leftmost_point, axis=1))
    # 将索引之前的点和索引之后的点分别存储在两个数组中
    points_before = points[:i]
    points_after = points[i:]

    # 合并两个数组并返回结果
    reordered_points = np.concatenate((points_after, points_before), axis=0)
    return reordered_points

if __name__ == "__main__":
    # 加载起始和最终的二值图像
    start_image = cv2.imread("mask/circle.png", cv2.IMREAD_GRAYSCALE)
    end_image = cv2.imread("mask/circles.png", cv2.IMREAD_GRAYSCALE)

    # 通过形态学操作获得边界
    _, start_thresh = cv2.threshold(start_image, 128, 255, cv2.THRESH_BINARY)
    _, end_thresh = cv2.threshold(end_image, 128, 255, cv2.THRESH_BINARY)

    # 膨胀操作，使得白色区域扩张，黑色区域收缩
    kernel = np.ones((3, 3), np.uint8)
    start_edges = cv2.dilate(start_thresh, kernel) - start_thresh
    end_edges = cv2.dilate(end_thresh, kernel) - end_thresh

    start_bezier_curve,point_count = fit_bezier_curve(start_edges)
    end_bezier_curve, _ = fit_bezier_curve(end_edges,point_count)
    #以顺时针排序
    # start_bezier_curve = reorder_points(start_bezier_curve)
    # end_bezier_curve = reorder_points(end_bezier_curve)

    print("控制点数：",len(start_bezier_curve),len(end_bezier_curve))

    # 插值生成中间帧的贝塞尔曲线
    num_frames = 100  # 设置中间帧的数量
    interpolated_curves = interpolate_bezier_curve(start_bezier_curve, end_bezier_curve, num_frames)

    # 创建一个空白图像，用于绘制变换过程
    height, width = start_image.shape
    output_image = np.zeros((height, width, 3), dtype=np.uint8)

    # 绘制变换过程并显示每一帧
    for i in range(num_frames):
        frame_curve = interpolated_curves[i]
        output_image = np.ones((height, width), dtype=np.uint8) * 255  # 创建空白图像，并填充为白色
        # 在空白图像上填充曲线区域为黑色
        curve_points = np.array(frame_curve, dtype=np.int32)
        cv2.fillPoly(output_image, [curve_points], 0)

        # cv2.putText(output_image, '0', frame_curve[0], cv2.FONT_HERSHEY_SIMPLEX, 0.4, 12)
        cv2.imshow("Bezier Curve Transformation", output_image)
        cv2.waitKey(100)  # 等待一段时间，控制动画速度

    cv2.destroyAllWindows()