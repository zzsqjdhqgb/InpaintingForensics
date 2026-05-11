#!/usr/bin/env python3
"""
IID‑Net 图像篡改检测 – Keras 3 (JAX 后端) 完整实现
包含：数据解压与预处理、模型定义、训练流程
"""

import os
import shutil
import tarfile
import zipfile
import random
import numpy as np
import cv2
from glob import glob

os.environ["KERAS_BACKEND"] = "jax"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.8"   # 最多占用 80% 显存
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"  # 按需分配而非预分配全部

import keras
from keras import layers, Model, ops, constraints
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, CSVLogger
from keras.utils import PyDataset


# =================== 配置参数 ===================
# 三个原始压缩包（放在项目根目录）
DRESDEN_ARCHIVE = "IIDNet_TrainData_GC_Dresden.tar.gz"
PLACES_ARCHIVE  = "IIDNet_TrainData_GC_Places.tar.gz"
DIVERSE_ZIP     = "DiverseInpaintingDataset.zip"

# 数据输出目录（所有解压数据与列表均在此，可加入 .gitignore）
DATA_DIR = "./data"
CACHE_DIR = os.path.join(DATA_DIR, "cache")
TRAIN_TXT = os.path.join(DATA_DIR, "train.txt")
VAL_TXT   = os.path.join(DATA_DIR, "val.txt")
TEST_DIR  = os.path.join(DATA_DIR, "test_images")

# 训练超参数
NUM_TRAIN = 48000
NUM_VAL   = 1000
BATCH_SIZE = 1          # 根据显存调整，JAX 会自动分配
EPOCHS = 1000
INIT_LR = 1e-4
# ================================================


# =================== 1. 数据准备 ===================
def extract_tar(archive_path, extract_to):
    """解压 .tar.gz 到指定目录，已有非空则跳过"""
    basename = os.path.splitext(os.path.splitext(os.path.basename(archive_path))[0])[0]
    target = os.path.join(extract_to, basename)
    if os.path.isdir(target) and os.listdir(target):
        print(f"[Data] ✅ 已存在：{target}")
    else:
        print(f"[Data] 📦 解压 tar.gz: {archive_path} → {target}")
        os.makedirs(target, exist_ok=True)
        with tarfile.open(archive_path, "r:gz") as tar:
            tar.extractall(target)
    return target


def extract_zip(archive_path, extract_to):
    """解压 .zip 到指定目录，已有非空则跳过"""
    basename = os.path.splitext(os.path.basename(archive_path))[0]
    target = os.path.join(extract_to, basename)
    if os.path.isdir(target) and os.listdir(target):
        print(f"[Data] ✅ 已存在：{target}")
    else:
        print(f"[Data] 📦 解压 zip: {archive_path} → {target}")
        os.makedirs(target, exist_ok=True)
        with zipfile.ZipFile(archive_path, 'r') as zf:
            zf.extractall(target)
    return target


def collect_pairs(root_dir):
    """递归收集 (img, mask) 对，mask 文件名为 *_mask.jpg"""
    pairs = []
    for mask_path in glob(os.path.join(root_dir, "**", "*_mask.jpg"), recursive=True):
        base, ext = os.path.splitext(mask_path)
        if base.endswith("_mask"):
            img_path = base[:-5] + ext
        else:
            continue
        if os.path.exists(img_path):
            pairs.append((os.path.abspath(img_path), os.path.abspath(mask_path)))
    return pairs


def prepare_dataset():
    """解压数据并生成训练/验证列表及测试图片"""
    print("=" * 60)
    print("[Data] 开始准备数据集 ...")
    os.makedirs(DATA_DIR, exist_ok=True)

    # 解压三个压缩包
    dresden_dir = extract_tar(DRESDEN_ARCHIVE, CACHE_DIR)
    places_dir  = extract_tar(PLACES_ARCHIVE, CACHE_DIR)
    diverse_dir = extract_zip(DIVERSE_ZIP, CACHE_DIR)

    # 收集训练图像对
    print("[Data] 🔍 扫描训练图像对 ...")
    dresden_pairs = collect_pairs(dresden_dir)
    places_pairs  = collect_pairs(places_dir)
    all_pairs = dresden_pairs + places_pairs
    print(f"   Dresden: {len(dresden_pairs)} 对")
    print(f"   Places : {len(places_pairs)} 对")
    print(f"   总计   : {len(all_pairs)} 对")

    random.seed(42)
    random.shuffle(all_pairs)

    # 划分训练集与验证集
    if len(all_pairs) < NUM_TRAIN + NUM_VAL:
        n_train = min(NUM_TRAIN, len(all_pairs))
        n_val = len(all_pairs) - n_train
    else:
        n_train = NUM_TRAIN
        n_val = NUM_VAL

    train_pairs = all_pairs[:n_train]
    val_pairs   = all_pairs[n_train:n_train + n_val]

    # 保存为 txt 文件（每行：图片路径 掩码路径）
    def save_txt(pairs, filepath):
        with open(filepath, 'w') as f:
            for img, mask in pairs:
                f.write(f"{img} {mask}\n")

    save_txt(train_pairs, TRAIN_TXT)
    save_txt(val_pairs, VAL_TXT)
    print(f"[Data] 💾 {TRAIN_TXT} ({len(train_pairs)} 对)")
    print(f"[Data] 💾 {VAL_TXT} ({len(val_pairs)} 对)")

    # 准备测试图片（Diverse Inpainting 中的非 mask png）
    print("[Data] 🖼️  准备测试图片 ...")
    if os.path.exists(TEST_DIR):
        shutil.rmtree(TEST_DIR)
    os.makedirs(TEST_DIR)

    test_images = []
    for f in glob(os.path.join(diverse_dir, "**", "*.png"), recursive=True):
        if not f.endswith("_mask.png"):
            test_images.append(f)
    test_images.sort(key=os.path.basename)

    for img_path in test_images:
        shutil.copy2(img_path, os.path.join(TEST_DIR, os.path.basename(img_path)))

    print(f"[Data] ✅ 复制 {len(test_images)} 张测试图片到 {TEST_DIR}/")
    print("=" * 60)


# =================== 2. 自定义约束与层 ===================
class BayarConstraint(constraints.Constraint):
    """Bayar 卷积权重约束：中心为 -1，其余权重归一化"""
    def __call__(self, w):
        w = w * 10000.0
        center = 2
        # 将中心置0
        mask = ops.numpy.where(
            (ops.arange(5)[:, None, None, None] == center) &
            (ops.arange(5)[None, :, None, None] == center),
            0.0, 1.0
        )
        w = w * mask
        w_sum = ops.sum(w, axis=[0, 1], keepdims=True)
        w_sum = ops.where(w_sum == 0, 1, w_sum)
        w = w / w_sum
        # 恢复中心为 -1
        w = ops.numpy.where(
            (ops.arange(5)[:, None, None, None] == center) &
            (ops.arange(5)[None, :, None, None] == center),
            -1.0, w
        )
        return w


class PFFiltersConv(layers.Layer):
    """3→9 固定高通滤波卷积 (不可训练)"""
    def __init__(self, **kwargs):
        super().__init__(trainable=False, **kwargs)

    def build(self, input_shape):
        pf1 = np.array([[0, 0, 0], [0, -1, 0], [0, 1, 0]], dtype='float32')
        pf2 = np.array([[0, 0, 0], [0, -1, 1], [0, 0, 0]], dtype='float32')
        pf3 = np.array([[0, 0, 0], [0, -1, 0], [0, 0, 1]], dtype='float32')
        pf_list = [pf1, pf2, pf3, pf1, pf2, pf3, pf1, pf2, pf3]

        # kernel 形状: (3, 3, 3, 9)  -> H, W, in_channels, out_channels
        kernel = np.zeros((3, 3, 3, 9), dtype='float32')
        for j in range(9):        # 输出通道
            for i in range(3):    # 输入通道
                kernel[:, :, i, j] = pf_list[j]

        self.kernel = self.add_weight(
            shape=(3, 3, 3, 9),
            initializer='zeros',
            trainable=False
        )
        self.kernel.assign(kernel)
        self.built = True

    def call(self, x):
        return keras.ops.conv(x, self.kernel, strides=1, padding='same')

class CustomizedConv(layers.Layer):
    """固定高斯核深度可分离卷积 (5×5)"""
    def __init__(self, channels=256, **kwargs):
        super().__init__(trainable=False, **kwargs)
        self.channels = channels

    def build(self, input_shape):
        kernel = np.array([[0.03598, 0.03735, 0.03997, 0.03713, 0.03579],
                           [0.03682, 0.03954, 0.04446, 0.03933, 0.03673],
                           [0.03864, 0.04242, 0.07146, 0.04239, 0.03859],
                           [0.03679, 0.03936, 0.04443, 0.03950, 0.03679],
                           [0.03590, 0.03720, 0.04003, 0.03738, 0.03601]], dtype='float32')
        kernel = kernel.reshape(5, 5, 1, 1)
        kernel = np.tile(kernel, (1, 1, self.channels, 1))
        self.dw_kernel = self.add_weight(shape=(5, 5, self.channels, 1), initializer='zeros', trainable=False)
        self.dw_kernel.assign(kernel)

    def call(self, x):
        return keras.ops.depthwise_conv(x, self.dw_kernel, strides=1, padding='same')


class MedianFilter2D(layers.Layer):
    """3×3 中值滤波（反射填充）"""
    def call(self, x):
        x_pad = keras.ops.pad(x, [[0,0], [1,1], [1,1], [0,0]], mode='reflect')
        patches = keras.ops.image.extract_patches(x_pad, size=3, strides=1, padding='valid')
        C = x.shape[-1]
        patches = keras.ops.reshape(patches, (-1, x.shape[1], x.shape[2], 9, C))
        return keras.ops.median(patches, axis=3)


class SeparableConv2d(layers.Layer):
    """深度可分离卷积：dw(same) + bn + pw(1x1)"""
    def __init__(self, filters, kernel_size=3, strides=1, dilation_rate=1, **kwargs):
        super().__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.dilation_rate = dilation_rate

    def build(self, input_shape):
        self.depthwise = layers.DepthwiseConv2D(
            self.kernel_size, strides=self.strides,
            dilation_rate=self.dilation_rate, padding='same', use_bias=False)
        self.bn = layers.BatchNormalization()
        self.pointwise = layers.Conv2D(self.filters, 1, use_bias=False)

    def call(self, x, training=False):
        x = self.depthwise(x)
        x = self.bn(x, training=training)
        x = self.pointwise(x)
        return x


class SepConv(layers.Layer):
    """带 ReLU 的分离卷积（BN 无 affine）"""
    def __init__(self, filters, kernel_size=3, strides=1, dilation_rate=1, **kwargs):
        super().__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.dilation_rate = dilation_rate

    def build(self, input_shape):
        self.relu = layers.ReLU()
        self.depthwise = layers.DepthwiseConv2D(
            self.kernel_size, strides=self.strides,
            dilation_rate=self.dilation_rate, padding='same', use_bias=False)
        self.pointwise = layers.Conv2D(self.filters, 1, use_bias=False)
        self.bn = layers.BatchNormalization(scale=False, center=False)

    def call(self, x, training=False):
        x = self.relu(x)
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x, training=training)
        return x


class Identity(layers.Layer):
    def call(self, x):
        return x


operation_candidates = {
    '00': lambda f_in, f_out, stride, dilation: SeparableConv2d(f_out, 3, stride, dilation),
    '01': lambda f_in, f_out, stride, dilation: SepConv(f_out, 3, stride, 1),
    '02': lambda f_in, f_out, stride, dilation: SepConv(f_out, 5, stride, 2),
    '03': lambda f_in, f_out, stride, dilation: Identity(),
}


class Block(layers.Layer):
    """提取块，含 skip connection 和 genotype 序列"""
    def __init__(self, planes, reps=3, stride=1, dilation=1,
                 start_with_relu=True, grow_first=True, genotype=None, **kwargs):
        super().__init__(**kwargs)
        self.planes = planes
        self.reps = reps
        self.stride = stride
        self.dilation = dilation
        self.start_with_relu = start_with_relu
        self.grow_first = grow_first
        self.genotype = genotype if genotype else ['03', '03', '03']

    def build(self, input_shape):
        inplanes = input_shape[-1]
        if self.planes != inplanes or self.stride != 1:
            self.skip_conv = layers.Conv2D(self.planes, 1, strides=self.stride, use_bias=False)
            self.skip_bn   = layers.BatchNormalization()
        else:
            self.skip_conv = None

        ops_list = []
        filters = inplanes
        if self.grow_first:
            ops_list.append(layers.ReLU())
            ops_list.append(SeparableConv2d(self.planes, 3, 1, self.dilation))
            ops_list.append(layers.BatchNormalization())
            filters = self.planes

        for i in range(self.reps - 1):
            ops_list.append(layers.ReLU())
            ops_list.append(operation_candidates[self.genotype[i]](filters, filters, 1, self.dilation))
            ops_list.append(layers.BatchNormalization())

        if not self.grow_first:
            ops_list.append(layers.ReLU())
            ops_list.append(SeparableConv2d(self.planes, 3, 1, self.dilation))
            ops_list.append(layers.BatchNormalization())

        ops_list.append(layers.ReLU())
        ops_list.append(operation_candidates[self.genotype[2]](filters, filters, self.stride, 1))
        ops_list.append(layers.BatchNormalization())

        if not self.start_with_relu:
            ops_list = ops_list[1:]

        self.ops_list = ops_list

    def call(self, x, training=False):
        residual = x
        if self.skip_conv is not None:
            residual = self.skip_conv(residual)
            residual = self.skip_bn(residual, training=training)

        for layer in self.ops_list:
            if isinstance(layer, layers.BatchNormalization):
                x = layer(x, training=training)
            else:
                x = layer(x)
        return x + residual


class GlobalLocalAttention(layers.Layer):
    """全局‑局部注意力，输出通道数 = 输入通道数 × 3"""
    def __init__(self, channels=256, **kwargs):
        super().__init__(**kwargs)
        self.channels = channels

    def build(self, input_shape):
        self.local_conv = CustomizedConv(channels=self.channels)
        self.built = True

    def call(self, x, training=False):
        B, H, W, C = x.shape
        F_local = self.local_conv(x)

        # 全局注意力
        former = ops.reshape(x, (B, H * W, C))          # (B, N, C), N = H*W
        num = ops.einsum('bik,bjk->bij', former, former)
        norm = ops.einsum('bij,bij->bi', former, former)
        den = ops.sqrt(ops.einsum('bi,bj->bij', norm, norm)) + 1e-8
        cosine = num / den

        _, indexes = ops.top_k(cosine, k=15)            # (B, N, 15)，返回整数索引

        dy_T = 15
        rtn = ops.copy(former)
        N = H * W
        batch_offset = ops.arange(B) * N                # (B,)
        batch_offset = ops.expand_dims(batch_offset, axis=-1)   # (B, 1)

        for t in range(1, dy_T):
            neighbor_idx = indexes[:, :, t]             # (B, N)
            neighbor_idx = ops.cast(neighbor_idx, 'int32')  # 确保整数类型
            # 转换为展平后的全局索引
            flat_idx = neighbor_idx + batch_offset      # (B, N)
            flat_idx = ops.reshape(flat_idx, (-1,))     # (B*N,)
            former_flat = ops.reshape(former, (B * N, C))
            neighbor_feat_flat = ops.take(former_flat, flat_idx, axis=0)  # (B*N, C)
            neighbor_feat = ops.reshape(neighbor_feat_flat, (B, N, C))
            rtn = rtn + neighbor_feat

        rtn = rtn / float(dy_T)
        F_global = ops.reshape(rtn, (B, H, W, C))

        return ops.concatenate([x, F_global, F_local], axis=-1)   # 256 -> 768
    
# =================== 3. IID‑Net 模型 ===================
class IIDNet(Model):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Enhancement Block
        self.normal_conv = layers.Conv2D(3, 5, padding='same', use_bias=False, name='normal_conv')
        self.pf_conv = PFFiltersConv(name='pf_conv')
        self.bayar_conv = layers.Conv2D(3, 5, padding='same', use_bias=False,
                                        kernel_constraint=BayarConstraint(), name='bayar_conv')
        self.enhance_conv1 = layers.Conv2D(32, 3, strides=2, padding='same', use_bias=False)
        self.enhance_bn1 = layers.BatchNormalization()
        self.enhance_relu = layers.ReLU()
        self.enhance_conv2 = layers.Conv2D(64, 3, padding='same', use_bias=False)
        self.enhance_bn2 = layers.BatchNormalization()

        # Extraction Block (10 cells)
        self.cell1  = Block(128, 3, stride=2, dilation=1, start_with_relu=False, grow_first=True, genotype=['01','03','00'])
        self.cell2  = Block(256, 3, stride=2, dilation=1, start_with_relu=True,  grow_first=True, genotype=['02','00','00'])
        self.cell3  = Block(256, 3, stride=1, dilation=1, start_with_relu=True,  grow_first=True, genotype=['03','02','00'])
        self.cell4  = Block(256, 3, stride=1, dilation=2, start_with_relu=True,  grow_first=True, genotype=['01','00','01'])
        self.cell5  = Block(256, 3, stride=1, dilation=2, start_with_relu=True,  grow_first=True, genotype=['00','02','00'])
        self.cell6  = Block(256, 3, stride=1, dilation=2, start_with_relu=True,  grow_first=True, genotype=['00','01','00'])
        self.cell7  = Block(256, 3, stride=1, dilation=2, start_with_relu=True,  grow_first=True, genotype=['02','03','02'])
        self.cell8  = Block(256, 3, stride=1, dilation=2, start_with_relu=True,  grow_first=True, genotype=['03','03','00'])
        self.cell9  = Block(256, 3, stride=1, dilation=2, start_with_relu=True,  grow_first=True, genotype=['02','02','00'])
        self.cell10 = Block(256, 3, stride=1, dilation=2, start_with_relu=True,  grow_first=True, genotype=['00','01','03'])

        # Decision Block
        self.att = GlobalLocalAttention()
        self.upsample1 = layers.UpSampling2D(size=2)
        self.dec_conv1 = layers.Conv2D(256, 3, padding='same')
        self.dec_bn1   = layers.BatchNormalization()
        self.dec_relu1 = layers.ReLU()
        self.dec_conv2 = layers.Conv2D(256, 3, padding='same')
        self.dec_bn2   = layers.BatchNormalization()
        self.dec_relu2 = layers.ReLU()

        self.upsample2 = layers.UpSampling2D(size=2)
        self.dec_conv3 = layers.Conv2D(256, 3, padding='same')
        self.dec_bn3   = layers.BatchNormalization()
        self.dec_relu3 = layers.ReLU()
        self.dec_conv4 = layers.Conv2D(256, 3, padding='same')
        self.dec_bn4   = layers.BatchNormalization()
        self.dec_relu4 = layers.ReLU()

        self.upsample3 = layers.UpSampling2D(size=2)
        self.dec_conv5 = layers.Conv2D(1, 3, padding='same')
        self.median = MedianFilter2D()
        self.final_sigmoid = layers.Activation('sigmoid')

    def call(self, inputs, training=False):
        x = inputs
        B, H, W, C = x.shape

        # Enhancement
        normal_x = self.normal_conv(x)
        pf_x = self.pf_conv(x)
        bayar_x = self.bayar_conv(x)
        x = ops.concatenate([normal_x, bayar_x, pf_x], axis=-1)  # 3+3+9=15
        x = self.enhance_conv1(x)
        x = self.enhance_bn1(x, training=training)
        x = self.enhance_relu(x)
        x = self.enhance_conv2(x)
        x = self.enhance_bn2(x, training=training)
        x = self.enhance_relu(x)

        # Extraction
        x = self.cell1(x, training=training)
        x = self.cell2(x, training=training)
        x = self.cell3(x, training=training)
        x = self.cell4(x, training=training)
        x = self.cell5(x, training=training)
        x = self.cell6(x, training=training)
        x = self.cell7(x, training=training)
        x = self.cell8(x, training=training)
        x = self.cell9(x, training=training)
        x = self.cell10(x, training=training)

        # Decision
        x = self.att(x, training=training)  # 256 -> 768
        x = self.upsample1(x)
        x = self.dec_conv1(x)
        x = self.dec_bn1(x, training=training)
        x = self.dec_relu1(x)
        x = self.dec_conv2(x)
        x = self.dec_bn2(x, training=training)
        x = self.dec_relu2(x)

        x = self.upsample2(x)
        x = self.dec_conv3(x)
        x = self.dec_bn3(x, training=training)
        x = self.dec_relu3(x)
        x = self.dec_conv4(x)
        x = self.dec_bn4(x, training=training)
        x = self.dec_relu4(x)

        x = self.upsample3(x)
        x = self.dec_conv5(x)
        x = self.median(x)
        x = ops.image.resize(x, (H, W))
        x = self.final_sigmoid(x)
        return x


# =================== 4. 损失函数 ===================
def joint_focal_bce_loss(y_true, y_pred):
    eps = keras.config.epsilon()
    y_pred = ops.clip(y_pred, eps, 1 - eps)

    alpha = 0.25
    gamma = 2.0
    alpha_t = ops.where(y_true == 1, alpha, 1 - alpha)
    pt = ops.where(y_true == 1, y_pred, 1 - y_pred)

    bce = - (y_true * ops.log(y_pred) + (1 - y_true) * ops.log(1 - y_pred))
    focal_loss = alpha_t * ops.power(1 - pt, gamma) * bce
    bce_loss = bce

    return ops.mean(focal_loss) + ops.mean(bce_loss)


# =================== 5. 数据集 ===================
class IIDPyDataset(PyDataset):
    def __init__(self, file_list, batch_size=2, image_size=(256, 256), shuffle=True, choice='train', **kwargs):
        super().__init__(**kwargs)
        self.file_list = file_list
        self.batch_size = batch_size
        self.image_size = image_size          # 目标高、宽
        self.shuffle = shuffle
        self.choice = choice

        with open(file_list, 'r') as f:
            lines = f.read().strip().split('\n')
        self.data = [line.split() for line in lines if line.strip() != '']
        if self.shuffle:
            random.shuffle(self.data)

    def __len__(self):
        return int(np.ceil(len(self.data) / self.batch_size))

    def __getitem__(self, idx):
        batch_data = self.data[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_imgs, batch_masks = [], []
        H, W = self.image_size

        for item in batch_data:
            if self.choice != 'test':
                img_path, mask_path = item
            else:
                img_path = item[0]
                mask_path = None

            # 读取图像
            img = cv2.imread(img_path).astype('float32') / 255.0
            if img is None:
                raise FileNotFoundError(f"Image not found: {img_path}")

            # 读取 mask（若无则生成全零掩码）
            if mask_path:
                mask = cv2.imread(mask_path, 0).astype('float32') / 255.0
                mask = np.expand_dims(mask, axis=-1)
            else:
                mask = np.zeros((img.shape[0], img.shape[1], 1), dtype='float32')

            # 训练集数据增强
            if self.choice == 'train':
                if random.random() < 0.5:
                    img = cv2.flip(img, 0)
                    mask = cv2.flip(mask, 0)
                if random.random() < 0.5:
                    img = cv2.flip(img, 1)
                    mask = cv2.flip(mask, 1)

            # 统一缩放到固定尺寸（防止 batch 内尺寸不一致）
            img = cv2.resize(img, (W, H), interpolation=cv2.INTER_LINEAR)
            mask = cv2.resize(mask, (W, H), interpolation=cv2.INTER_NEAREST)

            # 归一化到 [-1, 1]
            img = (img - 0.5) / 0.5
            batch_imgs.append(img)
            batch_masks.append(mask)

        batch_imgs = np.stack(batch_imgs, axis=0)
        batch_masks = np.stack(batch_masks, axis=0)
        return batch_imgs, batch_masks

# =================== 6. 训练主流程 ===================
def train():
    # 定义图像尺寸，必须与数据集的 resize 尺寸一致
    image_size = (256, 256)

    # 若数据列表不存在，则自动准备数据集
    if not (os.path.exists(TRAIN_TXT) and os.path.exists(VAL_TXT)):
        prepare_dataset()

    train_dataset = IIDPyDataset(TRAIN_TXT, batch_size=BATCH_SIZE,
                                 image_size=image_size, shuffle=True, choice='train')
    val_dataset   = IIDPyDataset(VAL_TXT, batch_size=1,
                                 image_size=image_size, shuffle=False, choice='val')

    model = IIDNet()

    # 显式构建模型（触发变量创建）
    dummy_input = np.zeros((1, image_size[0], image_size[1], 3), dtype='float32')
    model(dummy_input)
    model.summary()

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=INIT_LR, beta_1=0.9, beta_2=0.999),
        loss=joint_focal_bce_loss,
        metrics=[keras.metrics.AUC(from_logits=False, name='auc')]
    )

    callbacks = [
        ReduceLROnPlateau(monitor='val_auc', factor=0.5, patience=10, mode='max', min_lr=1e-7),
        ModelCheckpoint('best_model.keras', monitor='val_auc', save_best_only=True, mode='max'),
        CSVLogger('training_log.csv')
    ]

    model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=EPOCHS,
        callbacks=callbacks,
        verbose=2
    )

    model.save('final_model.keras')


if __name__ == '__main__':
    train()