from PIL import Image
import copy
import random
import os
from io import BytesIO
from PIL import Image
import time
from process.loss import *
from process.svg_process import *
import glob
from tqdm import tqdm
import io

def get_shapes_or_state(svg_dir_list,device):
    # 原始shape的坐标
    primitive_list = []
    # 原始shape的面积
    primitive_area_list = []
    # 存放每一步属性修改的shape
    shapes_list = []

    # 记录原始shape的颜色
    or_shape_groups_list = []
    index1 = 0

    for svg_dir in svg_dir_list:
        primitive = []
        _, _, shapes, shape_groups = pydiffvg.svg_to_scene(svg_dir)
        # 获取每个svg的每个path的原始颜色
        for shape_group in shape_groups:
            shape_group.shape_ids=torch.LongTensor([index1])
            or_shape_groups_list.append(shape_group)
            index1+=1

        shapes_list.append(shapes)
        # 获取
        primitive_area = get_svg_area(shapes,600,device).item()
        primitive_area_list.append(primitive_area)
        for shape in shapes:
            primitive_points = (shape.points-shapes[0].points[0])
            primitive_points.requires_grad=False
            primitive_points = primitive_points.to(device)
            primitive.append(primitive_points)
        primitive_list.append(primitive)
    return primitive_list,primitive_area_list,shapes_list,or_shape_groups_list

def init_shapes_properties(weights_list,mask_stream,device):
    primitive_num = len(weights_list)
    size_list = []
    angle_list = []
    pos_list = []
    pos_random_list = []
    # 中轴变换初始化primitive位置
    init_pos,mask_area = init_by_medial_axis(mask_stream,size=primitive_num)
    for i in range(primitive_num):
        size = torch.tensor(weights_list[i]**(1/2),dtype=torch.float32,device=device)
        # 随机角度
        # angle = torch.tensor(random.uniform(-0.1, 0.1),dtype=torch.float32,device=device)
        # 0角度
        angle = torch.tensor(0.0,dtype=torch.float32,device=device)
        # 随机位置
        pos_random = torch.tensor((300.0+random.uniform(-50, 50),300.0+random.uniform(-50, 50)),dtype=torch.float32,device=device)
        # 中轴变换位置
        pos = torch.tensor((init_pos[i][1],init_pos[i][0]),dtype=torch.float32,device=device)

        size_list.append(size)
        angle_list.append(angle)
        pos_random_list.append(pos_random)
        pos_list.append(pos)
    return size_list, angle_list, pos_random_list, pos_list, mask_area
    
# 初始化优化器
def init_optimizer(size_list,angle_list,pos_list):
    size_vars = []
    angle_vars = []
    pos_vars=[]
    for i in range(len(size_list)):
        size_vars.append(size_list[i])
        size_list[i].requires_grad=True
        angle_vars.append(angle_list[i])
        angle_list[i].requires_grad = True
        pos_vars.append(pos_list[i])
        pos_list[i].requires_grad=True
    params = {}
    params['size'] = size_vars
    params["angle"] = angle_vars
    params["pos"] = pos_vars

    learnable_params = [
                {'params': params['size'], 'lr': 0.02, '_id': 0},
                {'params': params["angle"], 'lr': 0.01, '_id': 1},
                {'params': params["pos"], 'lr': 0.9, '_id': 2},
            ]
    optimizer = torch.optim.Adam(learnable_params, betas=(0.9, 0.9), eps=1e-6)
    return optimizer

# 初始化shape的颜色等属性
def init_path_groups(num):
    path_groups = []
    for i in range(num):
        path_group = pydiffvg.ShapeGroup(
                                shape_ids=torch.LongTensor([i]),
                                fill_color=torch.FloatTensor([0,0,0,1]),
                                stroke_color=torch.FloatTensor([0,0,0,1])
                            )
        path_groups.append(path_group)
    return path_groups

# maskBytes,svg,num,iteration,processId
def pack_all_in_one(maskBytes,svg_dir,num,iteration,processId):
    print("进程：",processId)
    device = torch.device("cuda:0")
    init_diffvg(device)
    # 初始化权重
    weights_list = [random.randint(1, 5) for _ in range(num)]
    save_svg_path = processId
    os.makedirs(f"./cache/{save_svg_path}", exist_ok=True)

    svg_num = len(glob.glob(os.path.join(svg_dir, '*.svg')))
    svg_dir_list = [f"{svg_dir}/{i+1}.svg" for i in range(svg_num)]*100
    svg_dir_list = svg_dir_list[:num]

    # 获取原始svg的属性
    primitive_list,primitive_area_list,shapes_list,or_shape_groups_list = get_shapes_or_state(svg_dir_list,device)
    # 初始化基元的属性
    size_list, angle_list, pos_random_list, pos_list, mask_area = init_shapes_properties(weights_list,maskBytes,device)
    # 将基元的颜色设置为纯黑
    shape_groups = init_path_groups(sum(len(sublist) for sublist in primitive_list))

    # 处理目标图像成合适格式
    image_stream = BytesIO(maskBytes)
    target_img = Image.open(image_stream)
    if target_img.mode != 'RGB':
        target_img = target_img.convert('RGB')
    target_img = target_img.resize([600,600])
    target_img = transforms.ToTensor()(target_img)
    target_img = target_img.to(device)

    # 目标权重
    target_area_list = torch.tensor(weights_list,device=device)

    # 初始化优化器
    optimizer = init_optimizer(size_list,angle_list,pos_list)
    with tqdm(total=iteration, desc="Processing value", unit="value") as pbar:
        for epoch in range(iteration):
            shapes_1 = []
            for i in range(len(primitive_list)):
                for j in range(len(primitive_list[i])):
                    # 原始坐标乘以尺寸
                    points_1 = primitive_list[i][j]*size_list[i]

                    # 进行旋转
                    points_2 = torch.zeros_like(points_1,device=device)
                    points_2[:,0] = points_1[:,0] * torch.cos(angle_list[i]) - points_1[:,1] * torch.sin(angle_list[i])
                    points_2[:,1] = points_1[:,0] * torch.sin(angle_list[i]) + points_1[:,1] * torch.cos(angle_list[i])

                    # 位移
                    points_2 = points_2+pos_list[i]

                    # 经过一些列变换的坐标赋予给shapes_list
                    shapes_list[i][j].points = points_2
            # 将shapes_list展平
            shapes_1 = [item for sublist in shapes_list for item in sublist]
            # 可微分渲染成图像
            raster_img = svg_to_img(600,shapes_1,shape_groups,device)
            raster_img = rgba_to_rgb(raster_img,device)

            loss_mse = 3e4*F.mse_loss(raster_img, target_img)
            # 互斥损失
            loss_overlap = icon_overlap_loss_by_list(shapes_1,600,size_list,primitive_area_list,device,mask_area,scale=3e-1)
            
            # 面积比例损失
            if epoch < 1000:
                loss_area_proportional = icon_area_proportional_loss_by_list(size_list,primitive_area_list,target_area_list,scale=1e5)
            else:
                loss_area_proportional = 0
            # 场引力
            # loss_my_point_gravitational = my_point_gravitational_loss(pos_random_list ,gravity_direction="down",scale=0.28)
            loss = loss_mse+loss_overlap+loss_area_proportional
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            pydiffvg.save_svg(f"./cache/{save_svg_path}/{epoch}.svg",
                            600,
                            600,
                            shapes_1,
                            or_shape_groups_list)
            pbar.update(1)
            pbar.set_description(f"MSE:{loss_mse:.6f}"+f" loss_overlap:{loss_overlap:.6f}"+f" loss_area_proportional:{loss_area_proportional:.6f}")


