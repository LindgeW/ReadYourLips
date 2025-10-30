# -*- coding: utf-8 -*-
"""
一个简单的Python客户端，用于通过摄像头或视频文件，
将视频发送到唇语识别服务器进行测试。

使用前请确保已安装所需库:
pip install opencv-python numpy

如何运行:

1. 确保 server.py 正在运行。
2. 修改下方的 HOST 变量，填入服务器的IP地址。

模式一：实时摄像头测试
- 直接运行脚本: python simple_client.py
- 在弹出的窗口中:
  - 按下 's' 键开始/停止录制。
  - 按下 'q' 键退出程序。

模式二：发送视频文件
- 使用 --file 或 -f 参数指定视频文件路径:
  python simple_client.py --file /path/to/your/video.mp4
"""
import cv2
import socket
import struct
import time
import numpy as np
import argparse
import os

# --- 配置 ---
HOST = '127.0.0.1'  # 服务器的IP地址 (如果服务器在同一台电脑上，请使用 '127.0.0.1')
PORT = 9999         # 服务器的端口号
TARGET_SIZE = (88, 88) # 服务器期望的图像尺寸

# --- 全局变量 ---
client_socket = None
is_recording = False # 仅用于交互模式

def connect_to_server():
    """连接到服务器"""
    global client_socket
    try:
        client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        print(f"正在连接到服务器 {HOST}:{PORT}...")
        client_socket.connect((HOST, PORT))
        print("成功连接到服务器。")
        return True
    except Exception as e:
        print(f"连接服务器失败: {e}")
        return False

def send_message(sock, message, msg_type=0x01):
    """发送文本消息"""
    try:
        msg_bytes = message.encode('utf-8')
        packet = struct.pack('!B', msg_type) + struct.pack('!I', len(msg_bytes)) + msg_bytes
        sock.sendall(packet)
    except Exception as e:
        print(f"发送消息失败: {e}")

def send_image(sock, frame):
    """预处理单帧图像并发送"""
    try:
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        resized_frame = cv2.resize(gray_frame, TARGET_SIZE)
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 90]
        result, img_encoded = cv2.imencode('.jpg', resized_frame, encode_param)
        if not result:
            print("图像编码失败")
            return
        img_bytes = img_encoded.tobytes()
        packet = struct.pack('!B', 0x02) + struct.pack('!I', len(img_bytes)) + img_bytes
        sock.sendall(packet)
    except Exception as e:
        print(f"发送图像失败: {e}")

def receive_message(sock):
    """接收并解析来自服务器的消息"""
    try:
        msg_type_buf = sock.recv(1)
        if not msg_type_buf: return None
        msg_type = struct.unpack('!B', msg_type_buf)[0]

        msg_len_buf = sock.recv(4)
        if not msg_len_buf: return None
        msg_len = struct.unpack('!I', msg_len_buf)[0]

        msg_data = b''
        while len(msg_data) < msg_len:
            packet = sock.recv(msg_len - len(msg_data))
            if not packet: return None
            msg_data += packet

        return msg_data.decode('utf-8') if msg_type == 0x01 else f"收到未知类型的消息: {msg_type}"
    except Exception as e:
        print(f"接收消息失败: {e}")
        return None

def stream_from_file(cap):
    """从视频文件读取并发送流"""
    print("\n>>> 开始从文件发送视频帧...")
    send_message(client_socket, "开始录制")
    response = receive_message(client_socket)
    print(f"服务器响应: {response}")
    if response is None: return

    # 尝试遵循视频原始帧率
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = 1 / fps if fps > 0 else 1 / 25  # 默认25fps

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break  # 视频结束
        send_image(client_socket, frame)
        frame_count += 1
        time.sleep(frame_interval)

    print(f">>> 文件发送完毕，共发送 {frame_count} 帧。正在通知服务器...")
    send_message(client_socket, "发送完成")

    print("正在等待服务器返回识别结果...")
    result = receive_message(client_socket)
    print("\n--- 识别结果 ---")
    print(result)
    print("-----------------\n")

def stream_from_webcam(cap):
    """从摄像头进行交互式视频流传输"""
    global is_recording
    print("\n--- 实时摄像头模式 ---")
    print("在视频窗口中按下 's' 键开始/停止录制。")
    print("在视频窗口中按下 'q' 键退出程序。")
    print("-----------------------\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("无法从摄像头读取帧")
            break

        display_frame = frame.copy()
        status_text = "Recording..." if is_recording else "Press 's' to start"
        color = (0, 0, 255) if is_recording else (0, 255, 0)
        cv2.putText(display_frame, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        cv2.imshow('Camera Feed', display_frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            if is_recording: # 如果退出时仍在录制，先停止
                send_message(client_socket, "发送完成")
                receive_message(client_socket) # 接收最后的响应
            break
        elif key == ord('s'):
            is_recording = not is_recording
            if is_recording:
                print("\n>>> 开始录制...")
                send_message(client_socket, "开始录制")
                response = receive_message(client_socket)
                print(f"服务器响应: {response}")
            else:
                print(">>> 停止录制，正在通知服务器...")
                send_message(client_socket, "发送完成")
                print("正在等待服务器返回识别结果...")
                result = receive_message(client_socket)
                print("\n--- 识别结果 ---")
                print(result)
                print("-----------------\n")

        if is_recording:
            send_image(client_socket, frame)
            time.sleep(1/25) # 限制发送速率为25fps

def main():
    """主函数，解析参数并选择输入模式"""
    parser = argparse.ArgumentParser(description="客户端，用于向唇语识别服务器发送视频。")
    parser.add_argument('-f', '--file', type=str, help='要发送的视频文件的路径。如果未提供，则使用实时摄像头。')
    args = parser.parse_args()

    if not connect_to_server():
        return

    cap = None
    if args.file:
        if not os.path.exists(args.file):
            print(f"错误: 文件 '{args.file}' 不存在。")
            return
        cap = cv2.VideoCapture(args.file)
        if not cap.isOpened():
            print(f"错误: 无法打开视频文件 '{args.file}'。")
            return
        stream_from_file(cap)
    else:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("错误: 无法打开摄像头。")
            return
        stream_from_webcam(cap)

    # 清理资源
    if cap:
        cap.release()
    cv2.destroyAllWindows()
    if client_socket:
        client_socket.close()
        print("与服务器的连接已关闭。")

if __name__ == '__main__':
    main()
