import torch
from torch.autograd import Variable
import pydiffvg
import time
from lxml import etree
import torch.nn.functional as F

def init_diffvg(device: torch.device,
                use_gpu: bool = torch.cuda.is_available(),
                print_timing: bool = False):
    pydiffvg.set_device(device)
    pydiffvg.set_use_gpu(use_gpu)
    pydiffvg.set_print_timing(print_timing)

def derivative(t, P0, P1, P2, P3):
    return 3 * (1 - t)**2 * (P1 - P0) + 6 * t * (1 - t) * (P2 - P1) + 3 * t**2 * (P3 - P2)

def get_length(p1,p2,p3,p4):
    l = 0
    for i in range(100):
        t=0.01*i
        l += ((derivative(t,p1[0],p2[0],p3[0],p4[0])**2+derivative(t,p1[1],p2[1],p3[1],p4[1])**2)**(1/2))*0.01
    return l

def get_perimeter(points):
    l1=get_length(points[0],points[1],points[2],points[3])
    l2=get_length(points[3],points[4],points[5],points[6])
    l3=get_length(points[6],points[7],points[8],points[9])
    l4=get_length(points[9],points[10],points[11],points[0])
    return l1+l2+l3+l4

def custom_elementwise_operation(x, y, segments,threshold=5,min_perimeter_list=None):
    result = torch.zeros(x.size(0), segments, 1, y.size(0))
    for i in range(0, (segments-1)*3+1, 3):
        if i != (segments-1)*3:
            result[:,i//3,0,:] = torch.sqrt((3 * (1 - y[:])**2 * (x[:,i+1,0] - x[:,i,0]) + 6 * y[:] * (1 - y[:]) * (x[:,i+2,0] - x[:,i+1,0]) + 3 * y[:]**2 * (x[:,i+3,0] - x[:,i+2,0]))**2+\
                (3 * (1 - y[:])**2 * (x[:,i+1,1] - x[:,i,1]) + 6 * y[:] * (1 - y[:]) * (x[:,i+2,1] - x[:,i+1,1]) + 3 * y[:]**2 * (x[:,i+3,1] - x[:,i+2,1]))**2)*0.01
        else:
            result[:,i//3,0,:] = torch.sqrt((3 * (1 - y[:])**2 * (x[:,i+1,0] - x[:,i,0]) + 6 * y[:] * (1 - y[:]) * (x[:,i+2,0] - x[:,i+1,0]) + 3 * y[:]**2 * (x[:,0,0] - x[:,i+2,0]))**2+\
                (3 * (1 - y[:])**2 * (x[:,i+1,1] - x[:,i,1]) + 6 * y[:] * (1 - y[:]) * (x[:,i+2,1] - x[:,i+1,1]) + 3 * y[:]**2 * (x[:,0,1] - x[:,i+2,1]))**2)*0.01   
    if threshold is not None and min_perimeter_list is None:
        result = torch.sum(result, dim=-1)
        min_length = threshold
        result = F.relu(result-min_length)
    if min_perimeter_list is not None:
        result = torch.sum(result, dim=-1)
        result = torch.squeeze(result, dim=-1)
        result = torch.sum(result, dim=-1)
        target = torch.tensor(min_perimeter_list,dtype=result.dtype,device=result.device,requires_grad=False)
        print(result)
        print(target)
        result = F.relu(result-target)
        # result = (result-target)**2
    return torch.sum(result)


def get_perimeter1(shapes,perimeter_list=None):
    points = [x.points for x in shapes]
    points = torch.stack(points, dim=0)
    points = torch.unsqueeze(points, dim=-1)
    y = torch.linspace(0, 1, 101)[:-1]
    segments = int(shapes[0].points.shape[0]/3)
    s = custom_elementwise_operation(points,y,segments,min_perimeter_list=perimeter_list)
    return s


def get_distance(shapes):
    segments = int(shapes[0].points.shape[0]/3)
    points = [x.points for x in shapes]
    points = torch.stack(points, dim=0)
    points = torch.unsqueeze(points, dim=-1)
    x1 = []
    x2 = []
    x3 = []
    x4 = []
    for i in range(segments):
        x1.append(points[i*3])
        x2.append(points[i*3+1])
        x3.append(points[i*3+2])

    points[:,2,5]

def path_grouping(svg_file_path,group_rules,save_dir):
    # 读取 SVG 文件
    with open(svg_file_path, 'rb') as file:
        svg_data = file.read()
    # 解析 SVG 数据
    root = etree.fromstring(svg_data)
    root_group = root.findall(".//{http://www.w3.org/2000/svg}g")[0]
    paths = root_group.findall(".//{http://www.w3.org/2000/svg}path")
    group_rules1 = []
    group_rules2 = []
    for i in group_rules:
        group_rules2.append([x for x in range(len(group_rules1),len(group_rules1)+i)])
        group_rules1 += [x for x in range(len(group_rules1),len(group_rules1)+i)]

    svg_root = etree.Element('svg', xmlns="http://www.w3.org/2000/svg", width="600", height="600")
    for i in group_rules2:
        group = etree.SubElement(svg_root, 'g')
        for j in i:
            group.append(paths[j])
    svg_tree = etree.ElementTree(svg_root)
    svg_tree.write(f'./{save_dir}/finally_grouped.svg', pretty_print=True)

def save_structure_path(shapes,shape_groups,previous_layer_path_num,structure_layered_results,img_size):
    index1=1
    for index2,structure_layer_path_num in enumerate(previous_layer_path_num):
        index1+=structure_layer_path_num
        for i,shape_group in enumerate(shape_groups[index1-structure_layer_path_num:index1]):
            shape_group.shape_ids=torch.LongTensor([i])
        pydiffvg.save_svg(f"{structure_layered_results}/structure_single_layer{index2+1}.svg",img_size,img_size,
                        shapes[index1-structure_layer_path_num:index1],
                        shape_groups[index1-structure_layer_path_num:index1],
                        )
        for i,shape_group in enumerate(shape_groups[:index1]):
            shape_group.shape_ids=torch.LongTensor([i])
        pydiffvg.save_svg(f"{structure_layered_results}/structure_layer{index2+1}.svg",img_size,img_size,
                        shapes[:index1],
                        shape_groups[:index1],
                        )
    for index1,shape_group1 in enumerate(shape_groups):
        shape_group1.shape_ids=torch.LongTensor([index1])
    return shapes,shape_groups

def save_layered_results(img_size,shapes,shape_groups,save_layered_results_dir,i,deleted_path_list,layer_path_num,
                         structure_path_insert_index_list # 结构图插入的索引
                         ):
    # 保存分层结果图
    pydiffvg.save_svg(f"{save_layered_results_dir}/layer{i+1}.svg",img_size,img_size,shapes,shape_groups)
    index1 = 0
    # 本次加入的结构层
    for index1,structure_path_insert_index in enumerate(structure_path_insert_index_list):
        shape_groups[structure_path_insert_index+1].shape_ids=torch.LongTensor([index1])
    # 本次加入的视觉层
    for index_ in deleted_path_list:
        shapes.insert(index_, shapes[0])
        path_group = pydiffvg.ShapeGroup(
                    shape_ids=torch.tensor([index_]),
                    fill_color=torch.FloatTensor([0,0,0,0]),
                    stroke_color=torch.FloatTensor([0,0,0,1])
                )
        shape_groups.insert(index_,path_group)
    for shape_group1 in shape_groups[len(shapes)-layer_path_num:]:
        if len(structure_path_insert_index_list)>0:
            index1+=1
        shape_group1.shape_ids=torch.LongTensor([index1])
    pydiffvg.save_svg(f"{save_layered_results_dir}/single_layer{i+1}.svg",img_size,img_size,
                        [x for i,x in enumerate(shapes) if i-1 in structure_path_insert_index_list]+shapes[len(shapes)-layer_path_num:],
                    #   shapes[previous_structure_path_num+1:structure_path_num+1]+shapes[len(shapes)-layer_path_num:],
                        [x for i,x in enumerate(shape_groups) if i-1 in structure_path_insert_index_list]+shape_groups[len(shape_groups)-layer_path_num:],
                    #   shape_groups[previous_structure_path_num+1:structure_path_num+1]+shape_groups[len(shapes)-layer_path_num:],
                        )
    shapes = [x for i,x in enumerate(shapes) if i not in deleted_path_list]
    shape_groups = [x for i,x in enumerate(shape_groups) if i not in deleted_path_list]
    for index1,shape_group1 in enumerate(shape_groups):
        shape_group1.shape_ids=torch.LongTensor([index1])
    return shapes,shape_groups

if __name__ == "__main__":
    svg_path = r"E:\stylemap\Process_SVG\data\finally.svg"
    _, img_size, shapes, shape_groups = pydiffvg.svg_to_scene(svg_path)

    # 并行计算
    t1 = time.time()
    points = [x.points for x in shapes]
    points = torch.stack(points, dim=0)
    points = torch.unsqueeze(points, dim=-1)
    y = torch.linspace(0, 1, 101)[:-1]
    a = custom_elementwise_operation(points,y)
    t2 = time.time()
    print(a)
    print(t2-t1)

    # 显示循环
    t1 = time.time()
    c = 0
    for shape in shapes:
        c += get_perimeter(shape.points)
        print(shape.points)
    t2 = time.time()
    print(c)
    print(t2-t1)