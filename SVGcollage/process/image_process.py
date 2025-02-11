from torchvision import transforms
import torch
import pydiffvg
import torch.nn.functional as F
from torchvision.transforms.functional import rgb_to_grayscale
import numpy as np
from PIL import Image
from skimage.morphology import medial_axis
from io import BytesIO

# 对图像进行中心裁剪
def x_centercrop(original_image,img_size):
        transform = transforms.Compose([
            transforms.CenterCrop(size=(img_size, img_size)),
            # 其他的数据增强操作可以添加在这里
        ])
        # 对图像进行数据增强
        transformed_image = transform(original_image)
        return transformed_image

# 改变图像的尺寸
def x_resize(original_image, target_size):
    transform = transforms.Compose([
        transforms.Resize(size=(target_size, target_size)),
        # 可以在这里添加其他的数据增强操作
    ])
    transformed_image = transform(original_image)
    return transformed_image

# 将图像从rgba转化为rgb
def rgba_to_rgb(img,device):
    para_bg = torch.tensor([1., 1., 1.], requires_grad=False, device=device)
    img = img[:, :, 3:4] * img[:, :, :3] + para_bg * (1 - img[:, :, 3:4])
    img = img.permute(2, 0, 1)
    return img

# 可微分的渲染svg
def svg_to_img(img_size,shapes, shape_groups, device):
    scene_args = pydiffvg.RenderFunction.serialize_scene(
    img_size, img_size, shapes, shape_groups
    )
    _render = pydiffvg.RenderFunction.apply
    img = _render(img_size,  # width
                img_size,  # height
                2,  # num_samples_x
                2,  # num_samples_y
                0,  # seed
                None,
                *scene_args)
    img = img.to(device)
    return img

def get_primitive_area(shapes):
    path_groups = []
    for index in range(len(shapes)):
        path_group = pydiffvg.ShapeGroup(
                                shape_ids=torch.LongTensor([index]),
                                fill_color=torch.FloatTensor([0,0,0,1]),
                                stroke_color=torch.FloatTensor([0,0,0,1])
                            )
        path_groups.append(path_group)

def get_svg_area(shapes,img_size,device):
    for shape in shapes:
        shape.points = shape.points+100
    path_groups = []
    for index in range(len(shapes)):
        path_group = pydiffvg.ShapeGroup(
                                shape_ids=torch.LongTensor([index]).to(device),
                                fill_color=torch.FloatTensor([0,0,0,1]).to(device),
                                stroke_color=torch.FloatTensor([0,0,0,1]).to(device)
                            )
        path_groups.append(path_group)
    para_bg = torch.tensor([1., 1., 1.], requires_grad=False, device=device)
    img = svg_to_img(img_size, shapes, path_groups,device)
    img = img[:, :, 3:4] * img[:, :, :3] + para_bg * (1 - img[:, :, 3:4])
    img = img.unsqueeze(0)  # convert img from HWC to NCHW
    img = img.permute(0, 3, 1, 2).to(device)
    gray_img = rgb_to_grayscale(img)
    gray_img = torch.clamp(gray_img,max=0.98)
    area = F.relu(0.98-gray_img)
    area = torch.sum(area)
    for shape in shapes:
        shape.points = shape.points-100
    return area

def get_nonzero_indices1(shapes, path_groups,img_size,device):
    para_bg = torch.tensor([1., 1., 1.], requires_grad=False, device=device)

    img = svg_to_img(img_size, shapes, path_groups,device)
    img = img[:, :, 3:4] * img[:, :, :3] + para_bg * (1 - img[:, :, 3:4])
    img = img.unsqueeze(0)  # convert img from HWC to NCHW
    img = img.permute(0, 3, 1, 2).to(device)
    gray_img = rgb_to_grayscale(img)
    gray_img = torch.clamp(gray_img,max=0.98)
    area = F.relu(0.98-gray_img)
    nonzero_indices = torch.nonzero(area).tolist()
    return nonzero_indices

def get_nonzero_indices(img_size,shapes):
    path_groups = []
    for i in range(len(shapes)):
        path_group = pydiffvg.ShapeGroup(
                                    shape_ids=torch.LongTensor([i]),
                                    fill_color=torch.FloatTensor([0,0,0,1]),
                                    stroke_color=torch.FloatTensor([0,0,0,1])
                                )
        path_groups.append(path_group)
    para_bg = torch.tensor([1., 1., 1.])
    img = svg_to_img(img_size, shapes, path_groups,para_bg.device)
    img = img[:, :, 3:4] * img[:, :, :3] + para_bg * (1 - img[:, :, 3:4])
    img = img.unsqueeze(0)  # convert img from HWC to NCHW
    img = img.permute(0, 3, 1, 2)
    gray_img = rgb_to_grayscale(img)
    gray_img = torch.clamp(gray_img,max=0.98)
    area = F.relu(0.98-gray_img)
    nonzero_indices = torch.nonzero(area)[:,2:].tolist()
    return nonzero_indices

def get_curve_index(img_path):
    # 打开PNG图像
    image = Image.open(img_path)

    # 转换图像为 RGB 模式，以便获取每个像素的颜色值
    # image = image.convert("RGB")
    target_img = image
    # 将图像转换为PyTorch张量
    target_img = transforms.ToTensor()(target_img)
    transform = transforms.Compose([
        transforms.Resize(size=(600, 600)),
        # 可以在这里添加其他的数据增强操作
    ])
    target_img = transform(target_img)
    target_img = target_img.permute(1, 2, 0)
    # print(torch.max(target_img))
    para_bg = torch.tensor([1., 1., 1.])
    target_img = target_img[:, :, 3:4] * target_img[:, :, :3] + para_bg * (1 - target_img[:, :, 3:4])
    target_img = target_img.permute(2, 0, 1)
    image = transforms.ToPILImage()(target_img)
    # 将图像转换为灰度图像
    gray_image = image.convert("L")

    # 将灰度图像进行二值化处理
    threshold = 128  # 设定阈值
    binary_image = gray_image.point(lambda p: p > threshold and 255)
    binary_image.save("binary_image.png")
    binary_image = np.array(binary_image,dtype=bool)
    binary_image = ~binary_image
    leftmost_indices = np.argmax(binary_image, axis=1)

    curve_index = np.zeros((600, 2))
    curve_index[:,0]=np.arange(600)
    curve_index[:,1]=leftmost_indices
    mask = curve_index[:, 1] != 0
    # 使用布尔索引过滤数组
    curve_index = curve_index[mask]
    curve_index = torch.tensor(curve_index,dtype=torch.float32)
    # 输出结果
    return curve_index

def select_points(coordinates, num_points):
    # 计算每个点之间的平方欧几里得距离矩阵
    square_distances = np.sum((coordinates[:, np.newaxis, :2] - coordinates[:, :2]) ** 2, axis=2)
    # 用于存储已选择的点的索引
    selected_indices = []
    # 选择第一个点
    first_point_index = np.random.randint(len(coordinates))
    selected_indices.append(first_point_index)
    # 选择其余的点
    for _ in range(num_points - 1):
        # 计算每个点到已选择的点的最小距离
        min_distances = np.min(square_distances[selected_indices], axis=0)
        # 选择具有最大最小距离的点
        next_point_index = np.argmax(min_distances)
        selected_indices.append(next_point_index)
    # 根据索引选择点
    selected_points = coordinates[selected_indices]
    return selected_points

def init_by_medial_axis(img_path,size=100):
    # 打开PNG图像
    image = Image.open(BytesIO(img_path))
    if image.mode != 'RGB':
        image = image.convert('RGB')
    image = image.resize([600,600])
    # 将图像转换为灰度图像
    gray_image = image.convert("L")
    # 将灰度图像进行二值化处理
    threshold = 128  # 设定阈值
    binary_image = gray_image.point(lambda p: p > threshold and 255)
    binary_image = np.array(binary_image,dtype=bool)

    binary_image = ~binary_image
    # 掩码的面积
    mask_area = np.count_nonzero(binary_image)

    # 进行中轴变换
    skel, distance = medial_axis(binary_image, return_distance=True)
    medial_axis_index = np.where(skel)

    # 获取中轴线上的点和宽度
    medial_axis_distance = np.zeros((len(medial_axis_index[0]), 3))
    for i in range(len(medial_axis_index[0])):
        x = medial_axis_index[0][i]
        y = medial_axis_index[1][i]
        medial_axis_distance[i,:] = np.array([x,y,distance[x,y]])

    # # 尽可能均匀的选点
    # medial_axis_distance = select_points(medial_axis_distance,size)
    
    # 随机选点
    indices = np.random.choice(medial_axis_distance.shape[0], size=size, replace=False)
    random_selected_array = medial_axis_distance[indices]

    # 根据中轴宽度进行排序
    sorted_indices = np.argsort(random_selected_array[:, 2])
    sorted_array = random_selected_array[sorted_indices]

    return sorted_array,mask_area 

def init_by_medial_axis_by_list(img_path_list,size=100):
    true_count = 0
    medial_axis_distance_list = []
    for img_path in img_path_list:
        # 打开PNG图像
        image = Image.open(img_path)
        image = np.array(image)
        binary_image = np.array(image,dtype=bool)
        
        true_count += np.count_nonzero(binary_image)

        skel, distance = medial_axis(binary_image, return_distance=True)

        medial_axis_index = np.where(skel)
        medial_axis_distance = np.zeros((len(medial_axis_index[0]), 3))
        for i in range(len(medial_axis_index[0])):
            x = medial_axis_index[0][i]
            y = medial_axis_index[1][i]
            medial_axis_distance[i,:] = np.array([x,y,distance[x,y]])
        medial_axis_distance_list.append(medial_axis_distance)

    medial_axis_distance = np.concatenate(medial_axis_distance_list, axis=0)
    indices = np.random.choice(medial_axis_distance.shape[0], size=size, replace=False)
    random_selected_array = medial_axis_distance[indices]
    sorted_indices = np.argsort(random_selected_array[:, 2])
    # 根据排序后的索引重新排列数组
    sorted_array = random_selected_array[sorted_indices]

    return sorted_array,true_count,medial_axis_distance_list

if __name__=="__main__":
    # svg_path = r"E:\stylemap\Process_SVG\test1.svg"
    # _,img_size,shapes, shape_groups = pydiffvg.svg_to_scene(svg_path)
    # get_nonzero_indices(img_size,shapes)

    image = Image.open(r"E:\stylemap\Pack_All_in_One\data\3.png")
    image_array = np.array(image)
    binary_image = np.array(image_array,dtype=bool)
    true_count = np.count_nonzero(binary_image)

    skel, distance = medial_axis(binary_image, return_distance=True)
    medial_axis_index = np.where(skel)
    medial_axis_distance = np.zeros((len(medial_axis_index[0]), 3))
    for i in range(len(medial_axis_index[0])):
        x = medial_axis_index[0][i]
        y = medial_axis_index[1][i]
        medial_axis_distance[i,:] = np.array([x,y,distance[x,y]])
    print(medial_axis_distance)

    indices = np.random.choice(medial_axis_distance.shape[0], size=200, replace=False)
    random_selected_array = medial_axis_distance[indices]


    sorted_indices = np.argsort(random_selected_array[:, 2])
    # 根据排序后的索引重新排列数组
    sorted_array = random_selected_array[sorted_indices]

