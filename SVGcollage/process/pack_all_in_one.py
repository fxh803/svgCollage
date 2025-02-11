from PIL import Image
from io import BytesIO
import random
import os
import torchvision.transforms as transforms
from process.loss import *
def init_path_groups(_num):
    path_groups = []
    for i in range(_num):
        path_group = pydiffvg.ShapeGroup(
                                shape_ids=torch.LongTensor([i]),
                                fill_color=torch.FloatTensor([0,0,0,1]),
                                stroke_color=torch.FloatTensor([0,0,0,1])
                            )
        path_groups.append(path_group)
    return path_groups
def pack_all_in_one(maskBytes,svg,num,iteration,processId):
    print(svg,num,iteration,processId)
    DEVICE = torch.device("cuda:0")
    #中轴线初始化位置
    init_pos_index,area_area = init_by_medial_axis(maskBytes,num)

    #读取mask
    image_stream = BytesIO(maskBytes)
    shape_target_img = Image.open(image_stream)
    shape_target_img = shape_target_img.convert('RGBA')

    #获取小svg
    path_svg_folder=svg 
    svg_file_count = len([file for file in os.listdir(path_svg_folder) if file.endswith(".svg")])
    path_svg_list = [f"{path_svg_folder}/{i+1}.svg" for i in range(svg_file_count)]*100
    path_svg_list = path_svg_list[:num]
        

    #初始化一些东西
    shapes_primitives_list = []#svg中shape的原始坐标
    shapes_list = []#svg中shape
    icon_area_list = []#图标区域
    # shape_list = []
    or_shape_groups_list = []#svg中shapegroups
    

    # 循环处理每一个小svg
    index = 0
    for path_svg in path_svg_list:
        # 将SVG路径转换为场景对象
        _, _, shapes, shape_groups = pydiffvg.svg_to_scene(path_svg)
        
        # 创建空列表，用于存储每个形状的原始坐标
        shapes_primitives = []
        
        # 处理每个形状组
        for shape_group in shape_groups:
            # 将形状组的shape_ids设置为递增的整数值，并添加到or_shape_groups_list中
            shape_group.shape_ids = torch.LongTensor([index])
            or_shape_groups_list.append(shape_group)
            index += 1

        # 处理每个形状
        for shape in shapes:
            # 平移形状，将坐标点加上100
            shape.points = shape.points + 100
            # 设置形状的描边宽度为0.3
            shape.stroke_width = torch.tensor(0.3)
            
        # 计算图标区域
        icon_area = get_svg_area(shapes, 600, DEVICE).item()
        
        # 处理每个形状
        for shape in shapes:
            # 还原形状的坐标，减去100
            shape.points = shape.points - 100
            
            # 计算形状的原始坐标，相对于第一个形状的偏移量
            shape_primitives = (shape.points - shapes[0].points[0])
            shape_primitives = shape_primitives.to(DEVICE)
            shape_primitives.requires_grad = False

            # 将形状的原始坐标添加到shapes_primitives列表中
            shapes_primitives.append(shape_primitives)
            
            # 将形状添加到shape_list中
            # shape_list.append(shape)

        # 将图标区域添加到icon_area_list中
        icon_area_list.append(icon_area)

        # 将形状、形状的原始坐标和形状组分别添加到对应的列表中
        shapes_list.append(shapes)
        shapes_primitives_list.append(shapes_primitives)



    # 生成随机整数张量，并进行转换和缩放
    area_list = torch.randint(1, 5, (num,), dtype=torch.float32 ,requires_grad=False).to(DEVICE)
    area_list, _ = torch.sort(area_list)

    #存储每一个svg的形态（尺寸，角度和偏移）
    size_list=[]
    angle_radians_list=[]
    displacement_list=[]

    #循环初始化三个变量
    for i in range(num):
        # 计算 `size`，将其设置为 `area_list[i]` 开根号的结果，并将其放置在 `DEVICE` 上
        size = torch.tensor((area_list[i].item())**(1/2),device=DEVICE)

        # 初始化旋转角度为0
        angle_radians = torch.tensor([0],dtype=torch.float32).to(DEVICE)
        # angle_radians = torch.tensor(random.uniform(-0.1, 0.1), dtype=torch.float32).to(DEVICE)

        #把初始化位置作为偏移量
        displacement = torch.tensor([init_pos_index[i][1],init_pos_index[i][0]],dtype=torch.float32,device=DEVICE)
        
        size_list.append(size)
        angle_radians_list.append(angle_radians)
        displacement_list.append(displacement)


    size_vars = []
    angle_radians_vars = []
    displacement_vars=[]
    #循环，为三种值设置梯度下降
    for i in range(num):
        size_vars.append(size_list[i])
        size_list[i].requires_grad=True
        angle_radians_vars.append(angle_radians_list[i])
        angle_radians_list[i].requires_grad = True
        displacement_vars.append(displacement_list[i])
        displacement_list[i].requires_grad=True

    #设置学习参数
    params = {}
    params['size'] = size_vars
    params["angle_radians"] = angle_radians_vars
    params["displacement"] = displacement_vars

    learnable_params = [
                {'params': params['size'], 'lr': 0.1, '_id': 0},
                {'params': params["angle_radians"], 'lr': 0.06, '_id': 1},
                {'params': params["displacement"], 'lr': 0.9, '_id': 2},
            ]
    optimizer = torch.optim.Adam(learnable_params, betas=(0.9, 0.9), eps=1e-6)

   
            
    #设置保存路径
    save_svg_path = processId
    os.makedirs(f"./cache/{save_svg_path}", exist_ok=True)

    #初始化path（每一个shape）
    path_groups = init_path_groups(sum(len(sublist) for sublist in shapes_primitives_list))

    #迭代
    for epoch in range(iteration):
        shapes_list_1d = []
        #进入到每个svg的每个shape
        for i in range(len(shapes_list)):
            for j in range(len(shapes_list[i])):
                #缩放这个shape
                points_1 = shapes_primitives_list[i][j]*size_list[i]

                points_2 = torch.zeros_like(points_1)
                #旋转这个shape
                points_2[:,0] = points_1[:,0] * torch.cos(angle_radians_list[i]) - points_1[:,1] * torch.sin(angle_radians_list[i])
                points_2[:,1] = points_1[:,0] * torch.sin(angle_radians_list[i]) + points_1[:,1] * torch.cos(angle_radians_list[i])
                #偏移这个shape
                points_2 = points_2+displacement_list[i]
                #重新赋值回去
                shapes_list[i][j].points = points_2

        #将 shapes_list 中的子列表展开成一维列表
        shapes_list_1d = [item for sublist in shapes_list for item in sublist]

        #转换到图像空间，计算与目标图像的损失
        raster_img = svg_to_img(600,shapes_list_1d,path_groups,DEVICE)
        raster_img = rgba_to_rgb(raster_img,DEVICE)
        loss_mse = target_img_mse_loss(raster_img,target_img=shape_target_img,scale=10e4)


        #重力损失
        # loss_gravity = gravity_loss(displacement_list,scale=0.1)
        # 互斥损失
        loss_overlap = icon_overlap_loss_by_list(shapes_list_1d,600,size_list,icon_area_list,DEVICE,area_area,scale=3e-1)
        # 面积比例损失
        if epoch < 500:
            loss_area_proportional = icon_area_proportional_loss_by_list(size_list,icon_area_list,area_list,scale=1e5)
        else:
            loss_area_proportional = icon_area_proportional_loss_by_list(size_list,icon_area_list,area_list,scale=1e1)
        # loss_icon_size = icon_size_loss(size_list,scale=1e13)

        #最终的损失
        loss = loss_mse+loss_overlap+loss_area_proportional

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        pydiffvg.save_svg(f"./cache/{save_svg_path}/{epoch}.svg",
                            600,
                            600,
                            shapes_list_1d,
                            or_shape_groups_list)