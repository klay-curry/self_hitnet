import os
import shutil
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm

# ================= 配置路径 =================
# 假设脚本运行在 code/ 目录下
ROOT = Path(".")
DATA_ROOT = ROOT / "assert" / "data"

# 题目目录
Q1_DIR = ROOT / "assert" / "问题一：可视化展示图像"
Q2_DIR = ROOT / "assert" / "问题二：可视化展示图像"
Q3_DIR = ROOT / "assert" / "问题三：可视化展示图像"

# 数据集源路径配置 (Img Dir, GT Dir)
# 注意：Camouflage-people 的 GT 通常是 png，CAMO 也是
SOURCE_DATASETS = [
    {
        "name": "CAMO",
        "img_dirs": [DATA_ROOT / "CAMO" / "CAMO-D" / "train", DATA_ROOT / "CAMO" / "CAMO-D" / "test"],
        "gt_dir": DATA_ROOT / "CAMO" / "CAMO-D" / "gt"
    },
    {
        "name": "People",
        "img_dirs": [DATA_ROOT / "Camouflage-people" / "CamouflageData" / "img"],
        "gt_dir": DATA_ROOT / "Camouflage-people" / "CamouflageData" / "gt"
    }
]

# ================= 核心函数 =================

def get_image_hash(img_path):
    """
    计算图片的感知哈希（简化版）。
    读取图片 -> 缩小到 16x16 -> 转灰度 -> 扁平化 -> 转字节串
    """
    # 必须要以依旧能读取的方式打开，防止中文路径报错
    img = cv2.imdecode(np.fromfile(str(img_path), dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        return None
    img = cv2.resize(img, (16, 16), interpolation=cv2.INTER_AREA)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img.tobytes()

def build_database():
    """遍历所有源数据集，建立 Hash -> GT_Path 的映射"""
    db = {}
    print(">>> 正在索引源数据集 (CAMO / NC4K / People)...")

    for dataset in SOURCE_DATASETS:
        name = dataset["name"]
        gt_dir = dataset["gt_dir"]

        # 收集该数据集下的所有图片
        src_images = []
        for img_dir in dataset["img_dirs"]:
            if img_dir.exists():
                src_images.extend(list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png")))

        for p in tqdm(src_images, desc=f"Indexing {name}"):
            h = get_image_hash(p)
            if h:
                # 尝试寻找对应的 GT
                # 策略：先找同名 png，再找同名 jpg
                gt_path = gt_dir / f"{p.stem}.png"
                if not gt_path.exists():
                    gt_path = gt_dir / f"{p.stem}.jpg"

                if gt_path.exists():
                    db[h] = gt_path
                # else:
                #     # 某些数据集可能GT命名不规则，暂忽略
                #     pass
    return db

def match_and_copy(source_dir, db, task_name):
    """通用匹配函数：针对 Q1 和 Q3"""
    if not source_dir.exists():
        print(f"跳过 {task_name}: 目录不存在")
        return

    target_gt_dir = source_dir / "gt"
    target_gt_dir.mkdir(exist_ok=True)

    images = sorted(list(source_dir.glob("*.jpg")) + list(source_dir.glob("*.png")))
    print(f"\n>>> 开始匹配 {task_name} ({len(images)} 张)...")

    count = 0
    for img_path in tqdm(images, desc=f"Matching {task_name}"):
        h = get_image_hash(img_path)

        if h in db:
            src_gt = db[h]
            # 目标文件名：保持与题目图片同名，但后缀改为 .png
            dst_name = img_path.with_suffix(".png").name
            dst_path = target_gt_dir / dst_name

            shutil.copy(src_gt, dst_path)
            count += 1
        else:
            print(f"Warning: 未找到 GT -> {img_path.name}")

    print(f"[{task_name}] 匹配完成: {count}/{len(images)}")

def process_q2_resize():
    """
    处理问题二：不通过哈希匹配，而是直接寻找问题一的 GT 并缩放
    逻辑：Question_2_01_xxx.jpg -> 对应 Question_1_01.png
    """
    if not Q2_DIR.exists() or not Q1_DIR.exists():
        print("跳过问题二: 目录不完整")
        return

    q1_gt_dir = Q1_DIR / "gt"
    if not q1_gt_dir.exists():
        print("错误: 问题一的 GT 目录不存在，请先成功运行问题一的匹配！")
        return

    target_gt_dir = Q2_DIR / "gt"
    target_gt_dir.mkdir(exist_ok=True)

    images = sorted(list(Q2_DIR.glob("*.jpg")) + list(Q2_DIR.glob("*.png")))
    print(f"\n>>> 开始处理问题二 (多尺度缩放) ({len(images)} 张)...")

    count = 0
    for img_path in tqdm(images, desc="Resizing GT for Q2"):
        # 1. 解析文件名找对应的问题一 ID
        # 文件名示例: Question_2_01_1024.jpg
        # 我们需要提取 "01" 部分
        parts = img_path.stem.split('_') # ['Question', '2', '01', '1024']
        if len(parts) < 3:
            continue

        img_id = parts[2] # '01'

        # 2. 找到对应的问题一 GT
        q1_gt_name = f"Question_1_{img_id}.png"
        q1_gt_path = q1_gt_dir / q1_gt_name

        if not q1_gt_path.exists():
            # print(f"Warning: 问题一 GT 不存在 ({q1_gt_name})，无法生成问题二 GT")
            continue

        # 3. 读取问题二原图获取目标尺寸
        target_img = cv2.imread(str(img_path))
        if target_img is None:
            continue
        h, w = target_img.shape[:2]

        # 4. 读取源 GT 并缩放
        src_gt = cv2.imread(str(q1_gt_path), cv2.IMREAD_GRAYSCALE)
        if src_gt is None:
            continue

        # 使用最近邻插值 (INTER_NEAREST) 保持二值性质，防止边缘模糊
        resized_gt = cv2.resize(src_gt, (w, h), interpolation=cv2.INTER_NEAREST)

        # 5. 保存
        dst_name = img_path.with_suffix(".png").name
        cv2.imwrite(str(target_gt_dir / dst_name), resized_gt)
        count += 1

    print(f"[问题二] 处理完成: {count}/{len(images)}")

# ================= 主程序 =================

def main():
    # 1. 建立索引
    db = build_database()
    print(f"索引构建完毕，共包含 {len(db)} 个唯一指纹。")

    # 2. 匹配问题一 (基础)
    match_and_copy(Q1_DIR, db, "问题一")

    # 3. 匹配问题三 (People/NC4K/CAMO 混合)
    match_and_copy(Q3_DIR, db, "问题三")

    # 4. 处理问题二 (基于问题一的 GT 进行缩放)
    # 这一步依赖于步骤 2 的成功执行
    process_q2_resize()

    print("\n所有任务 GT 准备完毕！")

if __name__ == "__main__":
    main()