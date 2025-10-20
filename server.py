
import socket
import threading
import os
from datetime import datetime
import struct
from PIL import Image
import io
import numpy as np
import torch 
import time

from avtrain import fast_one_infer  # 模型推理
from avdataset import CMLRDataset, VSRAppDataset
from vmodel_avhubert import DRLModel
from LLMs_api import qwen_converter 


test_set = VSRAppDataset(r'../LipData/data_160', r'data/vsrapp_test.txt', phase='test', setting='unseen')
model = DRLModel(len(test_set.vocab), len(test_set.spks)).to('cuda:1')
# checkpoint = torch.load('vsrapp4.pt', map_location='cpu', weights_only=True)
checkpoint = torch.load('vsrapp4.pt', map_location='cuda:1', weights_only=True)
states = checkpoint['model'] if 'model' in checkpoint else checkpoint
model.load_state_dict(states)
model.eval()


def recvall(sock, count):
    """
    确保从socket中接收指定数量的字节数据
    """
    buf = b''
    while count:
        newbuf = sock.recv(count)
        if not newbuf:
            return None
        buf += newbuf
        count -= len(newbuf)
    return buf


def is_valid_image(data):
    """
    检查数据是否为有效的图像文件（PNG或JPEG）
    """
    if len(data) < 8:
        return False, None

    # PNG文件头检查
    png_header = b'\x89PNG\r\n\x1a\n'
    if data[:8] == png_header:
        return True, 'PNG'

    # JPEG文件头检查
    jpeg_header = b'\xff\xd8\xff'
    if data[:3] == jpeg_header:
        return True, 'JPEG'

    return False, None


def get_image_dimensions(data, image_type):
    """
    从图像数据中提取图像尺寸
    """
    if image_type == 'PNG':
        # PNG尺寸信息位于IHDR块中
        # 格式: 4字节长度 + 4字节类型('IHDR') + 4字节宽度 + 4字节高度 + 其他信息
        if len(data) < 24:  # 最小IHDR块大小
            return None

        # 查找IHDR块
        ihdr_start = data.find(b'IHDR')
        if ihdr_start == -1 or ihdr_start < 8:
            return None

        # 从IHDR块中提取宽度和高度（大端序）
        width = struct.unpack('>I', data[ihdr_start - 4:ihdr_start])[0]
        height = struct.unpack('>I', data[ihdr_start + 4:ihdr_start + 8])[0]

        return width, height
    elif image_type == 'JPEG':
        # JPEG尺寸信息提取
        if len(data) < 4:
            return None

        # 查找SOI标记
        if data[0:2] != b'\xff\xd8':
            return None

        # 遍历JPEG段寻找SOF标记
        i = 2
        while i < len(data) - 4:
            # 检查是否为标记
            if data[i] == 0xff:
                # 检查是否为SOF标记 (0xc0 - 0xcf, 排除 0xc4, 0xc8, 0xcc)
                if data[i + 1] in [0xc0, 0xc1, 0xc2, 0xc3, 0xc5, 0xc6, 0xc7, 0xc9, 0xca, 0xcb, 0xcd, 0xce, 0xcf]:
                    # 高度在SOF标记后第3-4字节，宽度在第5-6字节（大端序）
                    height = struct.unpack('>H', data[i + 5:i + 7])[0]
                    width = struct.unpack('>H', data[i + 7:i + 9])[0]
                    return width, height
                # 移动到下一个段
                segment_length = struct.unpack('>H', data[i + 2:i + 4])[0]
                i += segment_length + 2
            else:
                i += 1
    return None


def save_image_data(client_dir, image_count, data, client_socket, address,
                    image_info_array_lock, image_info_array, image_data_list):
    """
    在单独线程中保存图像灰度数据（不再保存原始图片文件）
    """
    try:
        # 验证图像数据
        is_valid, image_type = is_valid_image(data)
        if not is_valid:
            print(f"客户端 {address} 发送的数据不是有效的图像格式")
            return

        # 获取图像尺寸
        dimensions = get_image_dimensions(data, image_type)
        width, height = 0, 0
        image_array = []
        if dimensions:
            width, height = dimensions
            #print(f"接收到的{image_type}图像尺寸: {width}x{height}")

            # 添加图像信息到数组 [序号, 高度, 宽度]
            with image_info_array_lock:
                image_info_array.append([image_count, height, width])

            # 从图像数据中提取灰度值
            try:
                image = Image.open(io.BytesIO(data))
                gray_image = image.convert('L')
                pixels = list(gray_image.getdata())
                for h in range(height):
                    row = [pixels[h * width + w] for w in range(width)]
                    image_array.append(row)

                with image_info_array_lock:
                    image_data_list.append(image_array)

                #print(f"成功提取灰度数据: {width}x{height}")
            except Exception as e:
                print(f"处理图像数据时出错: {e}")
                empty_array = [[0 for _ in range(width)] for _ in range(height)]
                with image_info_array_lock:
                    image_data_list.append(empty_array)

            if width != 88 or height != 88:
                print(f"警告: 图像尺寸 {width}x{height} 不是预期的 88x88")
        else:
            with image_info_array_lock:
                image_info_array.append([image_count, 0, 0])
                image_data_list.append([])
            print("无法获取图像尺寸，使用默认值 0x0")

    except Exception as e:
        print(f"保存图像数据时出错: {e}")


def handle_client(client_socket, address):
    """
    处理客户端连接的函数
    """
    print(f"客户端 {address} 已连接")

    # 为每个客户端创建独立的图像存储目录
    client_dir = f"images_from_{address[0]}_{address[1]}"
    os.makedirs(client_dir, exist_ok=True)

    # 图像计数器
    image_count = 0

    # 录制会话计数器
    session_count = 0

    # 当前会话目录
    current_session_dir = client_dir

    # 三维数组存储图像信息 [序号, 高度, 宽度]
    image_info_array = []

    # 存储实际图像数据的列表
    image_data_list = []

    # 线程锁，用于保护image_info_array的并发访问
    image_info_array_lock = threading.Lock()
    t0, t1 = 0, 0

    try:
        while True:
            # 首先接收1字节的数据类型
            type_buf = recvall(client_socket, 1)
            if not type_buf:
                break

            data_type = type_buf[0]

            # 接收4字节的数据长度
            length_buf = recvall(client_socket, 4)
            if not length_buf:
                break

            data_length = struct.unpack('!I', length_buf)[0]

            # 根据数据类型处理数据
            if data_type == 0x01:  # 文本消息
                data = recvall(client_socket, data_length)
                if not data:
                    break

                try:
                    message = data.decode('utf-8')
                    print(f"收到来自 {address} 的消息: {message}")

                    # 特殊处理开始和完成消息
                    if message.strip() == "开始录制":
                        print(f"客户端 {address} 开始录制")
                        # 每次开始录制时创建新的文件夹
                        session_count += 1
                        session_dir = f"{client_dir}/session_{session_count}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                        os.makedirs(session_dir, exist_ok=True)
                        # 更新当前使用的目录
                        current_session_dir = session_dir
                        print(f"创建新的录制会话目录: {session_dir}")
                        print(f"当前会话目录已更新为: {current_session_dir}")

                        # 重置图像信息数组
                        with image_info_array_lock:
                            image_info_array.clear()
                            image_data_list.clear()
                        print(f"图像信息数组已清空，当前长度: {len(image_info_array)}")

                        response = "服务器已准备好接收图像数据"
                        # 发送响应消息
                        response_bytes = response.encode('utf-8')
                        response_type = b'\x01'  # 文本类型
                        response_length = struct.pack('!I', len(response_bytes))
                        client_socket.send(response_type + response_length + response_bytes)
                        t0 = time.time()
                        continue
                    elif "发送完成" in message.strip():  # 修改条件判断，只要消息中包含"发送完成"就触发
                        print(f"客户端 {address} 完成图像发送")
                        print(f"当前会话目录: {current_session_dir}")
                        print(f"图像数量: {len(image_data_list)}")

                        with image_info_array_lock:
                            if len(image_data_list) > 0:
                                try:
                                    # 转换成三维数组 [t,h,w]
                                    imgs = np.stack(image_data_list, axis=0)

                                    # 保存 npy 文件
                                    npy_file_path = f"{current_session_dir}/images_{datetime.now().strftime('%Y%m%d_%H%M%S')}.npy"
                                    np.save(npy_file_path, imgs)
                                    print(f"已保存图像为 NPY 文件: {npy_file_path}，形状 {imgs.shape}")

                                    # 清空缓存
                                    image_info_array.clear()
                                    image_data_list.clear()
                                except Exception as e:
                                    print(f"保存 NPY 文件失败: {e}")
                            else:
                                print("警告: 没有接收到任何图像，跳过保存")

                        t1 = time.time()
                        print('发送时间：', t1 - t0)
                        # 模型推理
                        #test_set = CMLRDataset(r'../LipData/CMLR', r'data/unseen_test.csv', phase='test', setting='unseen')
                        #res = one_infer('cmlr_avg_10.pt', test_set, npy_file_path)
                        #res = one_infer('cmlr_avg_10_py.pt', test_set, npy_file_path)
                        #py_seq = one_infer('vsrapp4.pt', test_set, npy_file_path)
                        
                        py_seq = fast_one_infer(model, test_set, npy_file_path)
                        
                        t2 = time.time()
                        print('推理时间：', t2 - t1)
                        res_list, duration = qwen_converter.convert_vsr_result(py_seq, mode="multi")
                        res = ';'.join(res_list)
                        print(res, 'LLM time:', duration)

                        # 发送响应
                        #response = "结果："+str(res)+"；"+str(res)+"；"+str(res)
                        response = "结果："+str(res)+"；"
                        response_bytes = response.encode('utf-8')
                        response_type = b'\x01'
                        response_length = struct.pack('!I', len(response_bytes))
                        client_socket.send(response_type + response_length + response_bytes)
                        continue
                        # print(f"客户端 {address} 完成图像发送")
                        # print(f"当前会话目录: {current_session_dir}")
                        # print(f"图像信息数组内容: {image_info_array}")
                        #
                        # # 将图像信息数组保存到文本文件
                        # array_file_path = f"{current_session_dir}/image_info_array.txt"
                        # try:
                        #     with open(array_file_path, 'w') as f:
                        #         # 写入真正的三维数组格式
                        #         f.write("# 三维数组格式：[图像编号][高度][宽度]\n")
                        #         f.write("# 每个图像表示为一个二维数组，包含行数据\n")
                        #         f.write("# 每行包含宽度个像素值\n\n")
                        #
                        #         # 构建三维数组
                        #         f.write("[\n")
                        #         with image_info_array_lock:
                        #             for i, (info, image_array) in enumerate(zip(image_info_array, image_data_list)):
                        #                 # [序号, 高度, 宽度]
                        #                 index, height, width = info
                        #                 f.write(f"    # 第{i + 1}个图像 ({height}x{width})\n")
                        #                 f.write(f"    [\n")
                        #                 # 写入实际的图像数据
                        #                 for h in range(height):
                        #                     f.write("        [")
                        #                     # 写入每行的像素值
                        #                     for w in range(width):
                        #                         f.write(f"{image_array[h][w]}")
                        #                         if w < width - 1:
                        #                             f.write(", ")
                        #                     f.write("]")
                        #                     if h < height - 1:
                        #                         f.write(",\n")
                        #                     else:
                        #                         f.write("\n")
                        #                 f.write("    ]")
                        #                 if i < len(image_info_array) - 1:
                        #                     f.write(",\n")
                        #                 else:
                        #                     f.write("\n")
                        #         f.write("]\n")
                        #     print(f"图像信息数组已保存到: {array_file_path}")
                        # except Exception as e:
                        #     print(f"保存图像信息数组时出错: {e}")
                        #     import traceback
                        #     traceback.print_exc()
                        #
                        # response = "服务器已接收所有图像数据"
                        # # 发送响应消息
                        # response_bytes = response.encode('utf-8')
                        # response_type = b'\x01'  # 文本类型
                        # response_length = struct.pack('!I', len(response_bytes))
                        # client_socket.send(response_type + response_length + response_bytes)
                        # continue

                    # 处理以"编号"开头的消息
                    if message.startswith("编号"):
                        #print(f"收到来自 {address} 的编号信息: {message}")
                        # 将编号信息回传给客户端
                        response = f"服务器已收到编号信息: {message}"
                        response_bytes = response.encode('utf-8')
                        response_type = b'\x01'  # 文本类型
                        response_length = struct.pack('!I', len(response_bytes))
                        client_socket.send(response_type + response_length + response_bytes)
                        continue

                    # 回显普通消息给客户端
                    response = f"服务器收到: {message}"
                    response_bytes = response.encode('utf-8')
                    response_type = b'\x01'  # 文本类型
                    response_length = struct.pack('!I', len(response_bytes))
                    client_socket.send(response_type + response_length + response_bytes)
                except UnicodeDecodeError:
                    print(f"处理来自 {address} 的文本消息时出错")
            elif data_type == 0x02:  # 图像数据
                data = recvall(client_socket, data_length)
                if not data:
                    break

                image_count += 1

                # 在新线程中处理图像保存，避免阻塞主处理循环
                image_thread = threading.Thread(
                    target=save_image_data,
                    args=(current_session_dir, image_count, data, client_socket, address, image_info_array_lock,
                          image_info_array, image_data_list)
                )
                image_thread.daemon = True  # 设置为守护线程
                image_thread.start()
    except Exception as e:
        print(f"处理客户端 {address} 时出错: {e}")
    finally:
        client_socket.close()
        print(f"客户端 {address} 已断开连接")


def start_server(host='0.0.0.0', port=9999):
    """
    启动服务器
    """
    # 创建socket对象
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    # 设置SO_REUSEADDR选项，允许重用地址
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

    # 绑定地址和端口
    server.bind((host, port))

    # 开始监听，最多允许5个连接排队
    server.listen(5)
    print(f"服务器启动成功，正在监听 {host}:{port}")

    try:
        while True:
            # 等待客户端连接
            client_socket, address = server.accept()

            # 为每个客户端创建一个新线程
            client_thread = threading.Thread(
                target=handle_client,
                args=(client_socket, address)
            )
            client_thread.start()
    except KeyboardInterrupt:
        print("\n服务器正在关闭...")
    finally:
        server.close()


if __name__ == "__main__":
    start_server()
