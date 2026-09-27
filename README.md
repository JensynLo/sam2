# SAM 2 图像分割(推理版)

本仓库 fork 自 Meta 的 [SAM 2](https://github.com/facebookresearch/sam2),去掉了训练代码和原 Web demo,只保留推理部分,并提供两个开箱即用的入口:

- **`infer.py`**:命令行,输入图片 + 点/框提示,输出分割掩码
- **`app.py`**:网页服务,拖入图片、点击或画框即可实时预览掩码,并保存到指定路径;可通过 Tailscale 在其他设备上使用

![SAM 2 architecture](assets/model_diagram.png?raw=true)

## 重构说明

基于上游 [facebookresearch/sam2](https://github.com/facebookresearch/sam2) 的 `2b90b9f`(2024-12,SAM 2.1)。目标是得到一个只做推理、开箱即用的版本:**模型结构和推理计算与上游完全一致,可直接使用官方权重**,改动集中在删除训练相关代码和新增端到端入口。

### 删除

| 内容 | 说明 |
| :--- | :--- |
| `training/`、`sam2/configs/sam2.1_training/` | 训练 / 微调代码(Trainer、数据集、损失、优化器)和训练配置 |
| `demo/`、`backend.Dockerfile`、原 `docker-compose.yaml` | 原视频标注 demo(React 前端 + GraphQL 后端)及其 Docker 配置,由 `app.py` 和新的 `Dockerfile` 取代 |
| `sam2/modeling/sam2_utils.py` 中的 `sample_box_points`、`get_next_point` 等 | 只在训练时用于模拟用户点击 |
| Hiera 骨干网络的 `weights_path` 参数及 `iopath` 依赖 | 只在训练时加载 ImageNet 预训练权重 |
| `setup.py` 中的 `interactive-demo` 依赖和 `dev` 里的训练依赖 | fvcore、tensorboard、submitit、pycocotools 等 |

### 修改

- **`sam2/build_sam.py`**:模型直接在目标设备(GPU)上构建,权重以 mmap 方式读取,不再先在 CPU 上建一份模型、再把整个权重文件读进内存。以 large 模型为例,加载后进程内存 1.89 GB → 1.20 GB,启动峰值 2.76 GB → 2.03 GB;推理结果与修改前逐像素一致。
- **`setup.py`**:新增 `web` 可选依赖(Flask);`dev` 只保留格式化工具。
- **CI 格式检查**(`.github/workflows/check_fmt.yml`)覆盖新增的 `infer.py`、`app.py`。

### 新增

| 文件 | 说明 |
| :--- | :--- |
| `infer.py` | 命令行推理;同时提供 `Segmenter`、`load_image`、`render` 等公共逻辑,供 `app.py` 和其他代码复用 |
| `app.py` | 网页服务后端(Flask) |
| `web/index.html` | 网页前端(单文件,无需构建) |
| `Dockerfile`、`docker-compose.yaml` | 网页服务的 GPU 容器(替代上游 demo 的 Docker 配置) |

几点设计:

- **只用 GPU**:默认设备为 `cuda`,CUDA 不可用时直接报错,不会悄悄退回 CPU;确实要用 CPU 需显式传 `--device cpu`。
- **每张图片只编码一次**:上传时计算图像特征,之后每次改动点/框只运行轻量的提示编码器和掩码解码器,交互几乎是实时的;服务端缓存最近 8 张图片。
- **预览与保存一致**:每次预测带版本号,服务端只保留最新版本的结果;保存 / 导出时校验版本,当前提示的预测未完成或失败时不允许保存,避免把旧结果当成新结果保存。连续上传时,晚到的旧图片不会覆盖新选择的图片。
- **严格的参数校验**:非法的点、框、标签或请求体返回 400,而不是 500 或被静默截断。
- **可部署在反向代理后**:前端使用相对路径,可以挂在 `tailscale serve --set-path /sam2` 等子路径下。

### 保留

`sam2/` 包的其余部分(图像 / 视频预测器、自动掩码生成、模型配置)、`notebooks/` 示例、`tools/vos_inference.py`、`sav_dataset/` 评测工具保持不变,视频分割等上游功能仍可按原方式使用。`RELEASE_NOTES.md` 是上游的历史记录,其中指向 `training/`、`demo/` 的链接已失效。

## 安装

需要 `python>=3.10`、`torch>=2.5.1`、`torchvision>=0.20.1`。建议使用独立的 conda 环境;如果已有装好 CUDA 版 PyTorch 的环境,可以直接克隆,免去重新下载 PyTorch:

```bash
conda create -n sam2 --clone <已有的 PyTorch 环境>   # 或新建环境后按 https://pytorch.org 安装 PyTorch
conda activate sam2
SAM2_BUILD_CUDA=0 pip install --no-build-isolation -e ".[web]"
```

- `--no-build-isolation` 让安装直接使用环境里已有的 PyTorch,否则 pip 会为了构建再下载一份。
- `SAM2_BUILD_CUDA=0` 跳过可选的 CUDA 扩展(用于掩码补洞后处理,对结果影响很小)。如果装有与 PyTorch 版本匹配的 `nvcc`,可以去掉它来编译该扩展;编译失败会被忽略,不影响使用。常见问题见 [`INSTALL.md`](./INSTALL.md)。
- RTX 50 系列(Blackwell)显卡需要 CUDA 12.8 及以上版本的 PyTorch(如 `torch 2.7+` 的 `cu128` 版本)。

## 下载权重

至少下载一个 SAM 2.1 权重到 `checkpoints/`。`infer.py` 和 `app.py` 默认使用其中精度最高的那个,也可以用 `--model` 指定。

```bash
cd checkpoints
wget https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt  # 推荐
# 或者一次下载全部 4 个:./download_ckpts.sh
cd ..
```

| `--model`   | 权重文件 | 参数量 (M) | 速度 (FPS) | SA-V test (J&F) |
| :---------- | :------- | :--------: | :--------: | :-------------: |
| `large`     | [sam2.1_hiera_large.pt](https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt) | 224.4 | 39.5 | 79.5 |
| `base_plus` | [sam2.1_hiera_base_plus.pt](https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_base_plus.pt) | 80.8 | 64.1 | 78.2 |
| `small`     | [sam2.1_hiera_small.pt](https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_small.pt) | 46 | 84.8 | 76.6 |
| `tiny`      | [sam2.1_hiera_tiny.pt](https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt) | 38.9 | 91.2 | 76.5 |

速度在 A100 上测得(`torch 2.5.1, cuda 12.4`)。

## 命令行:`infer.py`

```bash
# 一个前景点
python infer.py notebooks/images/truck.jpg --point 500,375

# 前景点 + 背景点(第三个数是标签:1 前景,0 背景,默认 1)
python infer.py notebooks/images/truck.jpg --point 500,375 --point 1125,625,0

# 框,也可以再加点修正
python infer.py notebooks/images/truck.jpg --box 425,600,700,875 --point 575,750,0
```

坐标都是原图像素坐标 `(x, y)`,原点在左上角。结果默认保存到 `outputs/`:

| `--save` 取值 | 输出文件 | 内容 |
| :------------ | :------- | :--- |
| `mask`        | `<图片名>_mask.png`    | 黑白掩码(前景 255,背景 0) |
| `overlay`     | `<图片名>_overlay.png` | 原图叠加半透明蓝色掩码 |
| `cutout`      | `<图片名>_cutout.png`  | 抠图,掩码外透明(RGBA) |

其他参数:`--output DIR` 输出目录,`--save mask,overlay,cutout` 选择输出(默认 `mask,overlay`),`--model` / `--checkpoint` 选择模型,`--device` 选择设备(默认 `cuda`;CUDA 不可用时直接报错,不会自动退回 CPU,确实要用 CPU 请显式传 `--device cpu`)。完整说明见 `python infer.py -h`。

只给一个点时提示有歧义(比如点在车窗上,可能指车窗也可能指整辆车),程序会让模型输出 3 个候选,再选质量分最高的一个。结果不理想时,可以加背景点排除多余区域,或者直接用框。

## 网页服务:`app.py`

```bash
python app.py        # 然后在浏览器打开 http://127.0.0.1:7860
```

1. 把图片拖进页面(也可以点击选择,或 Ctrl+V 粘贴)
2. **左键**点击添加前景点,**右键**(或 Shift+左键)添加背景点,**按住拖动**画框;每次修改后都会重新分割并显示结果
3. `Ctrl+Z` 撤销,`Esc` 清空
4. 选择保存类型(掩码 / 叠加图 / 抠图),然后:
   - **保存到该路径**:保存到运行 `app.py` 的机器上。路径可以是文件,也可以是目录(以 `/` 结尾或已存在的目录,文件名自动生成);相对路径以 `--output-dir`(默认 `outputs/`)为基准。目标文件已存在时会先询问是否覆盖。
   - **另存为…**:下载到浏览器所在的机器。Chrome / Edge 会弹出系统对话框让你选择保存位置,其他浏览器保存到默认下载目录。

参数:`--host`(默认 `127.0.0.1`,`tailscale` 表示只监听 Tailscale 地址)、`--port`(默认 `7860`)、`--output-dir`,以及和 `infer.py` 相同的 `--model` / `--checkpoint` / `--device`。

> 网页可以把文件写到服务器的任意路径,默认只监听本机。如果用 `--host 0.0.0.0` 对外开放,请只在可信网络中使用。

### 通过 Tailscale 在其他设备上访问

**方式一:直接监听 tailnet 地址**(最简单)

```bash
python app.py --host tailscale
```

`--host tailscale` 会自动取本机的 Tailscale IP 并只监听该地址:tailnet 内的设备可以访问,局域网和其他网络访问不到。启动时会打印访问地址,例如 `tailnet URL: http://jensynpc.<tailnet>.ts.net:7860`,在 MacBook 等设备上打开即可。

**方式二:`tailscale serve` 反向代理**(HTTPS)

`app.py` 保持默认只监听本机,由 Tailscale 转发并提供 HTTPS 证书:

```bash
python app.py                                   # 监听 127.0.0.1:7860
tailscale serve --bg 7860                       # https://<机器名>.<tailnet>.ts.net/
# 或挂在子路径下,与其他服务共用一个域名:
tailscale serve --bg --set-path /sam2 7860      # https://<机器名>.<tailnet>.ts.net/sam2
tailscale serve status                          # 查看当前转发
tailscale serve reset                           # 取消所有转发
```

HTTPS 下,Chrome / Edge 的「另存为…」可以弹出系统对话框选择保存位置;通过普通 `http://` 远程访问时,浏览器出于安全限制不提供该对话框,文件会存到默认下载目录。

注意:

- **「保存到该路径」写入的是运行 `app.py` 的机器**,「另存为…」才是下载到你正在用的设备上。
- 如果客户端开着 Clash 等代理,请把 `100.64.0.0/10` 和 `*.ts.net` 加入直连 / 绕过列表,否则请求会被发给代理而无法访问。
- 不要用 `tailscale funnel` 把它公开到互联网,网页可以在服务器上写文件。

## Docker

镜像基于 `pytorch/pytorch:2.11.0-cuda12.8-cudnn9-runtime`,只包含网页服务;权重不打进镜像,运行时挂载。

**前提**:安装 Docker 和 NVIDIA Container Toolkit,让容器能使用 GPU。

- Windows + WSL2:安装 [Docker Desktop](https://docs.docker.com/desktop/features/wsl/) 并在设置中为该 WSL 发行版开启 WSL integration,GPU 支持开箱即用。
- Linux(或在 WSL 内直接安装 Docker Engine):安装 [Docker Engine](https://docs.docker.com/engine/install/ubuntu/) 和 [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html),然后执行 `sudo nvidia-ctk runtime configure --runtime=docker && sudo systemctl restart docker`。

可以用 `docker run --rm --gpus all nvidia/cuda:12.8.0-base-ubuntu22.04 nvidia-smi` 检查容器能否看到 GPU。

**启动**

```bash
mkdir -p outputs                                   # 先建好,否则 Docker 会以 root 身份创建,容器内无法写入
docker compose up -d --build                       # 构建并在后台启动,打开 http://127.0.0.1:7860
docker compose logs -f                             # 查看日志(加载模型约需几十秒)
docker compose down                                # 停止
```

通过环境变量调整监听地址和端口:

```bash
SAM2_BIND=100.79.92.107 docker compose up -d       # 只监听本机的 Tailscale IP(换成 `tailscale ip -4` 的输出)
SAM2_PORT=8080 docker compose up -d                # 换端口
```

也可以保持默认的 `127.0.0.1`,在宿主机上用 `tailscale serve --bg 7860` 转发(见上一节)。要指定模型,在 `docker-compose.yaml` 中设置 `command: ["--model", "large"]`。

不使用 compose 时:

```bash
docker build -t sam2-web .
docker run -d --name sam2 --gpus all -p 127.0.0.1:7860:7860 \
  -v "$PWD/checkpoints:/app/checkpoints:ro" -v "$PWD/outputs:/app/outputs" \
  --user "$(id -u):$(id -g)" sam2-web
```

注意:

- 容器内**只有 `/app/outputs` 映射到宿主机的 `outputs/`**。在网页上「保存到该路径」时请保存到 `/app/outputs` 下(默认路径就是这里),写到容器内其他位置的文件会随容器删除而丢失。「另存为…」下载到浏览器所在设备,不受影响。
- 容器内没有 GPU 时服务会直接报错退出,不会退回 CPU。
- 如果拉取镜像或安装依赖需要代理,可在 Docker Desktop 的设置中配置代理,或构建时传入 `--build-arg HTTP_PROXY=... --build-arg HTTPS_PROXY=...`。

## 在代码中调用

```python
from infer import load_image, render, Segmenter

segmenter = Segmenter()                   # 可选参数:model="large", device="cuda"
image = load_image("notebooks/images/truck.jpg")
segmenter.set_image(image)                # 每张图片只需计算一次特征
mask, score = segmenter.predict(points=[(500, 375, 1)], box=None)
render(image, mask, "cutout").save("truck_cutout.png")
```

更底层的接口(`SAM2ImagePredictor`、`SAM2AutomaticMaskGenerator`、视频用的 `SAM2VideoPredictor`)见 `sam2/` 包和 [`notebooks/`](./notebooks) 中的示例。

## 目录结构

```
infer.py             命令行推理 + 公共的分割/渲染逻辑
app.py               网页服务(Flask)
web/index.html       网页前端
Dockerfile           网页服务镜像(docker-compose.yaml 为启动配置)
sam2/                SAM 2 模型与预测器
checkpoints/         权重(下载脚本 download_ckpts.sh)
notebooks/           官方示例 notebook
tools/               视频目标分割(VOS)批量推理脚本
sav_dataset/         SA-V 数据集说明与评测工具
```

## License

SAM 2 的模型权重和代码基于 [Apache 2.0](./LICENSE) 许可。GPU 连通域算法改编自 [`cc_torch`](https://github.com/zsef123/Connected_components_PyTorch)(许可见 [`LICENSE_cctorch`](./LICENSE_cctorch))。

## 引用 SAM 2

```bibtex
@article{ravi2024sam2,
  title={SAM 2: Segment Anything in Images and Videos},
  author={Ravi, Nikhila and Gabeur, Valentin and Hu, Yuan-Ting and Hu, Ronghang and Ryali, Chaitanya and Ma, Tengyu and Khedr, Haitham and R{\"a}dle, Roman and Rolland, Chloe and Gustafson, Laura and Mintun, Eric and Pan, Junting and Alwala, Kalyan Vasudev and Carion, Nicolas and Wu, Chao-Yuan and Girshick, Ross and Doll{\'a}r, Piotr and Feichtenhofer, Christoph},
  journal={arXiv preprint arXiv:2408.00714},
  url={https://arxiv.org/abs/2408.00714},
  year={2024}
}
```
