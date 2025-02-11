from torchvision import transforms
import torch
import torch.nn.functional as F
import pydiffvg
from process.image_process import *
from torchvision.transforms import ToPILImage

def target_img_mse_loss_test_222222222(raster_img,target_img,scale=1):
    loss = F.mse_loss(raster_img, target_img)*scale
    return loss

def target_img_mse_loss(raster_img,target_img,scale=1):
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
    target_img = target_img.permute(2, 0, 1).to(raster_img.device)
    target_img.requires_grad = False
    # target_img = target_img.unsqueeze(0)
    image = ToPILImage()(raster_img.detach())
    # image.save("test.png") 

    image = ToPILImage()(target_img.detach())
    # image.save("test1.png") 

    loss = F.mse_loss(raster_img, target_img)*scale
    return loss

def target_img_mse_loss_test_1111(raster_img,target_img,mask_img,scale=1):
    
    mask_target_img = target_img*mask_img
    mask_raster_img = raster_img*mask_img

    # image = ToPILImage()(target_img.detach())
    # image.save("test0.png") 

    # image = ToPILImage()(raster_img.detach())
    # image.save("test1.png") 


    # image = ToPILImage()(mask_target_img.detach())
    # image.save("test2.png") 

    # image = ToPILImage()(mask_raster_img.detach())
    # image.save("test3.png") 
    loss = (F.mse_loss(raster_img, target_img)+100*F.mse_loss(mask_raster_img,mask_target_img))*scale
    return loss

# 向下的重力
def gravity_loss(displacement_list,scale=1):
    x = torch.stack(displacement_list, dim=0)
    result = torch.zeros(x.size(0)).to(x.device)
    result[:]=F.relu(600-x[:,1])
    loss = torch.sum(result)
    loss = loss*scale
    return loss

def points_gravity_loss(displacement_list,scale=1):
    x = torch.stack(displacement_list, dim=0)
    result = torch.zeros(x.size(0)).to(x.device)
    result[:]=((300-x[:,1])**2+(300-x[:,0])**2)**(1/2)
    loss = torch.sum(result)
    loss = loss*scale
    return loss

def overlap_loss(shapes,img_size,primitive_size_list,primitive_area,device,area_area,scale=1):
    area = get_svg_area(shapes,img_size,device)
    

    x = torch.stack(primitive_size_list, dim=0)
    result = torch.zeros(x.size(0)).to(x.device)
    result[:] = ( x[:]**2 ) * primitive_area

    all_area = torch.sum(result)
    # loss = F.relu(1.01*all_area-area)
    loss = F.relu(1.01*all_area-area)+F.relu(all_area-0.99*area_area)*10
    print(all_area)
    print(area)
    loss = loss*scale
    return loss

def overlap_loss(shapes,img_size,size_tensor1,primitive_area,device,area_area,scale=1):
    area = get_svg_area(shapes,img_size,device)
    
    x = size_tensor1
    result = torch.zeros(x.size(0)).to(x.device)
    result[:] = ( x[:]**2 ) * primitive_area

    all_area = torch.sum(result)
    # loss = F.relu(1.01*all_area-area)
    loss = F.relu(1.01*all_area-area)+F.relu(all_area-0.99*area_area)*10
    print(all_area)
    print(area)
    loss = loss*scale
    return loss

def overlap_loss_by_list(shapes,img_size,size_list,size_0,primitive_area_list,device,area_area,scale=1):
    area = get_svg_area(shapes,img_size,device)
    
    x1 = torch.stack(size_list, dim=0)
    x2 = torch.stack(primitive_area_list)
    result = ((x1*size_0)**2)*x2
    # result = ((x1)**2)*x2

    all_area = torch.sum(result)
    # loss = F.relu(1.01*all_area-area)+F.relu(all_area-0.98*600*600)*10
    loss = F.relu(1.01*all_area-area)+F.relu(all_area-0.98*area_area)*10
    # loss = F.relu(1.1*all_area-area)+F.relu(all_area-0.91*area_area)*10
    # print(11111111111)
    # print(all_area)
    # print(area)
    loss = loss*scale
    return loss

def overlap_loss_by_list_test_11111(shapes,img_size,size_list,size_0,primitive_area_list,device,area_area,scale=1):
    area = get_svg_area(shapes,img_size,device)
    
    x1 = torch.stack(size_list, dim=0)
    x2 = torch.stack(primitive_area_list)
    result = ((x1)**2)*x2
    all_area = torch.sum(result)
    loss = F.relu(1.01*all_area-area)
    loss = loss*scale
    return loss


def area_proportional_loss_by_list(size_list,primitive_area_list,area_target,scale):
    x1 = torch.stack(size_list, dim=0)
    x2 = torch.stack(primitive_area_list)
    
    result = ((x1)**2)*x2
    loss = 1-F.cosine_similarity(result, area_target, dim=0)
    # print(loss)
    # print(1111111111111)
    # print(result)
    # print(2222222222222)
    # print(area_target)
    loss = loss*scale
    return loss

def area_proportional_loss_by_list_test_111111(size_list,primitive_area_list,area_target,scale):
    x1 = torch.stack(size_list, dim=0)
    # x2 = torch.stack(primitive_area_list)
    
    result = (x1)**2
    
    result = result/result[0].item()
    loss = F.mse_loss(result,area_target)
    print(result)
    print(area_target)
    loss = loss*scale
    return loss

def icon_size_loss(size_list,scale):
    x1 = torch.stack(size_list, dim=0)
    loss = F.relu(torch.max(x1)/torch.min(x1)-10)+F.relu(0.05-torch.min(x1))*1000
    print(2222)
    print(loss)
    loss = loss*scale
    return loss

def icon_overlap_loss_by_list_test(shapes,img_size,size_list,primitive_area_list,device,area_area,weiyi,scale=1):
    area = get_svg_area(shapes,img_size,device)
    size_list1 = []
    for index1,wei in enumerate(weiyi):
        if wei[1]>469:
            size_list1.append(size_list[index1]*3)
        else:
            size_list1.append(size_list[index1])

    x1 = torch.stack(size_list1, dim=0)

    x2 = torch.tensor(primitive_area_list).to(x1.device)
    # result = ((x1*size_0)**2)*x2
    result = ((x1)**2)*x2

    all_area = torch.sum(result)
    # print(1111)
    # print(all_area)
    # print(area)
    # loss = F.relu(1.01*all_area-area)+F.relu(all_area-0.98*600*600)*10
    loss = F.relu(1.01*all_area-area)+F.relu(all_area-0.98*area_area)*10
    loss = loss*scale
    return loss

def icon_overlap_loss_by_list(shapes,img_size,size_list,primitive_area_list,device,area_area,scale=1):
    area = get_svg_area(shapes,img_size,device)
    x1 = torch.stack(size_list, dim=0)
    x2 = torch.tensor(primitive_area_list).to(x1.device)
    result = ((x1)**2)*x2

    all_area = torch.sum(result)
    loss = F.relu(1.01*all_area-area)+F.relu(all_area-0.98*area_area)*10
    loss = loss*scale
    return loss

def icon_area_proportional_loss_by_list(size_list,primitive_area_list,area_target,scale):
    x1 = torch.stack(size_list, dim=0)
    x2 = torch.tensor(primitive_area_list).to(x1.device)
    # result = ((x1*size_0)**2)*x2
    result = ((x1)**2)*x2
    loss = 1-F.cosine_similarity(result, area_target, dim=0)
    loss = loss*scale
    return loss

# def area_proportional_loss(size_tensor,area_target,scale):
#     result = size_tensor
#     result = result**2
#     loss = 1-F.cosine_similarity(result, area_target, dim=0)
#     loss = loss*scale
#     return loss
def size_loss(size_list,scale):
    x1 = torch.stack(size_list, dim=0)
    target = torch.zeros(len(size_list),device=x1.device,requires_grad=False)+0.09
    print(x1)
    loss = torch.sum(F.relu(target-x1))
    loss = loss*scale
    return loss

def ellipse_overlap_loss_by_list(shapes,img_size,size_list,size_0,primitive_area_list,device,area_area=10000000,scale=1):
    area = get_svg_area(shapes,img_size,device)
    
    x1 = torch.stack(size_list, dim=0)
    x2 = torch.stack(primitive_area_list,dim=0)

    # result = ((x1*size_0)**2)*x2
    result = (x1**2)*x2

    all_area = torch.sum(result)
    # loss = F.relu(1.01*all_area-area)+F.relu(all_area-0.98*600*600)*10
    loss = F.relu(1.01*all_area-area)+F.relu(all_area-0.99*area_area)*10
    # loss = F.relu(1.1*all_area-area)+F.relu(all_area-0.91*area_area)*10
    # print(all_area)
    # print(area)
    loss = loss*scale
    return loss

def circle_overlap_loss_by_list(shapes,img_size,all_area,device,scale=1):
    area = get_svg_area(shapes,img_size,device)
    loss = F.relu(0.99*all_area-area)
    print(11111111111)
    print(all_area)
    print(area)
    loss = loss*scale
    return loss

def ellipse_radius_loss(shapes,scale=100):
    radius = [x.radius for x in shapes]
    x = torch.stack(radius, dim=0)
    result = torch.zeros(x.size(0))
    result[:] = x[:,0]-x[:,1]
    target_result = torch.zeros(x.size(0)).to(result.device)
    loss = F.mse_loss(result,target_result)
    loss = loss*scale
    return loss

def ellipse_area_proportional_loss(size_list,area_target,scale):
    x1 = torch.stack(size_list, dim=0)
    result = x1
    # loss = 1-torch.dot(result / torch.norm(result),area_target/torch.norm(area_target))
    loss = 1-F.cosine_similarity(result, area_target, dim=0)
    loss = loss*scale
    return loss

def ellipse_area_proportional_loss_by_list(size_list,primitive_area_list,area_target,scale):
    x1 = torch.stack(size_list, dim=0)
    x2 = torch.stack(primitive_area_list)
    result = (x1[:,0]*x1[:,1])*x2
    loss = 1-F.cosine_similarity(result, area_target, dim=0)
    loss = loss*scale
    return loss

def my_point_gravitational_loss(displacement_list,gravity_direction,scale=1):
    x1 = torch.stack(displacement_list, dim=0)
    if gravity_direction=="left":
        result = x1[:,0]
    if gravity_direction == "down":
        result = 600-x1[:,1]
    loss = torch.sum(result)+(600-torch.min(x1[:,1]))*2
    loss = loss*scale
    return loss

def word_box_deformation_loss(size_list,control_point_displacement_list,scale=1):
    # size = torch.stack(size_list, dim=0).unsqueeze(-1)
    control_point_displacement = torch.stack(control_point_displacement_list, dim=0)
    result = F.relu(control_point_displacement-1)
    loss = torch.sum(result)
    loss = loss*scale
    return loss

def circle_loss(shapes):
    radius = [x.radius for x in shapes]
    x = torch.stack(radius, dim=0)

def ellipse_point_gravitational_loss(displacement_list,anchor_points,circle_weight,scale=1):
    x = torch.stack(displacement_list, dim=0)
    loss = 0
    index3 = 0
    for index1,circle_clum in enumerate(circle_weight):
        for index2,circle in enumerate(circle_clum):
            distance = torch.sqrt( (x[index3,0]-anchor_points[index1,0])**2+(x[index3,1]-anchor_points[index1,1])**2 )
            # target_result = torch.zeros(1).to(x.device)
            # target_result.requires_grad=False
            loss += distance
            index3+=1

    loss = loss*scale
    return loss

def mask_mse_loss(shapes,mask_img,device,scale=1):
    path_groups = []
    for index in range(len(shapes)):
        path_group = pydiffvg.ShapeGroup(
                                shape_ids=torch.LongTensor([index]),
                                fill_color=torch.FloatTensor([0,0,0,1]),
                                stroke_color=torch.FloatTensor([0,0,0,1])
                            )
        path_groups.append(path_group)
    area = 0
    result = 0
    img = svg_to_img(600,shapes,path_groups,device)
    para_bg = torch.tensor([1., 1., 1.], requires_grad=False, device=img.device)
    img = img[:, :, 3:4] * img[:, :, :3] + para_bg * (1 - img[:, :, 3:4])

    img = img.permute(2, 0, 1)

    # binary_image = np.array(mask_img)
    # binary_image = ~binary_image  
    binary_image = mask_img/255  
    area += np.sum(binary_image)*3
    mask = torch.tensor(binary_image,dtype=torch.float32,device=img.device,requires_grad=False)
    result += torch.sum(img * mask)

    print(1111111111888888888888888)

    loss = (area - result)**2
    print(loss)
    loss = loss*scale
    return loss
