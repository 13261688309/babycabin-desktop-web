# BabyCabin 电脑端网页

本地 Python 服务与浏览器页面，包括 ESP32 取图、识别、提醒、云台软件预演及鸿蒙配对网关。网页前端与后端均在 [companion](companion/)；无需连接设备即可预览界面。

## 启动

在仓库根目录用 PowerShell 执行：

    py -3 -m venv .venv
    .\.venv\Scripts\python.exe -m pip install -r companion/requirements.txt
    .\.venv\Scripts\python.exe -m companion.gimbal_console --port 8766

在浏览器打开 http://127.0.0.1:8766/ 。默认本机软件预演，不打开串口，不发出实物动作。

接入自己的相机时，给启动命令添加 --source 和真实局域网拍摄地址；如需同时启用已纳入的模型，再添加 --emotion-weights models/baby-state.pt 或 --face-weights models/face_detection_yunet_2023mar.onnx。Camera/USB 当前状态以界面实际回执为准。网页需要同机 Python 服务，GitHub 不会托管这个硬件后端。

手机联动需要同一受信局域网、显式启用网关和本机生成的 TLS 证书及私钥；用 --help 查看 --mobile-bind、--mobile-cert、--mobile-key 参数，然后访问桌面配对页 /mobile。不要把服务端口转发到公网。

本地软件测试（不连接硬件）：

    .\.venv\Scripts\python.exe -m unittest discover -s tests

[项目及结题材料](https://github.com/13261688309/babycabin-project) · [鸿蒙 APP](https://github.com/13261688309/babycabin-harmony-app)
