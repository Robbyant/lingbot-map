# LingBot-MAP 完整算子分析报告（含实测 profiling）

> 日期：2026-09-04 · 环境：RTX 5070 Ti Laptop (12GB)，`lingbot-map` conda 环境，PyTorch 2.9.1
> 数据来源：源码逐文件精读 + `torch.profiler` 实测（10 帧合成输入，8 scale + 2 streaming，bf16 + SDPA 兜底）
> 配套脚本：`profile_operators.py`（同目录，可复现）；原始数据：`operator_profile.txt` / `operator_profile.csv`

---

## 0. TL;DR（先看结论）

| 项 | 结论 |
|---|---|
| 模型在干嘛 | 三层 Transformer 接力（DINOv2 24 → 帧内 24 → 跨帧 24），逐帧流式重建 3D |
| **实测最慢的算子** | `aten::cat`（KV 缓存拼接），占 **74% GPU 时间** |
| 真正矩阵乘 | `addmm` self CUDA 仅 **1.8%**（GPU 太快），但 CPU 启动开销占 66% |
| 卷积 | `cudnn_convolution` 占 **15.5%**（patch embed + DPT 头，fp32） |
| **核心洞察** | FLOP 分布（90% 矩阵乘）**≠** 实测时间分布（74% 内存拷贝）|

> ⚠️ 本报告最重要的收获：**之前"90% 算力在矩阵乘"是 FLOP 口径（正确，指算术量）；但真实跑起来，慢在 KV 缓存的 `torch.cat` 反复整块拷贝上，而不是矩阵乘本身。** 这解释了为什么 demo 在 SDPA 兜底 + 无 `--compile` 下要跑 77 分钟。

---

## 1. 方法与口径：时间、频次是怎么来的

### 1.1 两个"频次"
- **静态频次**：代码里"每种算子有几个实例、一轮跑几次"——**直接数代码**即可（§3）。
- **运行频次/时间**：真跑一次，每个算子被调用几次、各花多久——**必须 `torch.profiler`**（§4）。

### 1.2 两个"时间"
- **self time（自身时间）**：算子自己内部花的 GPU 时间，不含它调用的子算子。看"谁真的重"用它。
- **total time（总时间）**：算子 + 子算子一起。看"谁把链拖慢"用它。

### 1.3 工具
`torch.profiler.profile(activities=[CPU, CUDA], record_shapes=True)` 包住一次前向，再 `key_averages().table(sort_by="self_cuda_time_total" / "count")` 输出。复现：`python profile_operators.py`。

---

## 2. 完整运行图（模块级：输入 → 模块 → 算子 → 输出）

```
输入：S 张图，每张 [3, 294, 518]（courthouse 真实尺寸，→ 21×37=777 patch）
│
├─ 模块① PatchEmbed         算子：Conv2d(3→1024, 14×14, stride14)
│    [3,294,518] → [1024,21,37] → flatten/transpose → [777,1024]
│
├─ 模块② 拼特殊 token        算子：torch.cat（无算力）
│    [777,1024] → [783,1024]（camera×1 + register×4 + scale×1）
│
├─ 模块③ DINOv2 主干 24 层   每层 Block = LN → 注意力(QKV/proj/SDPA) → 残差 → LN → MLP(fc1/GELU/fc2) → 残差
│    形状不变 [783,1024]，每层内部做 16 头自注意力（帧内）
│
├─ 模块④ frame 块 24 层      同上结构（帧内自注意力，带 RoPE）
│
├─ 模块⑤ global 块 24 层     结构同，但注意力 = 当前帧 vs 历史 KV 缓存（因果），
│    核心算子：QKV/proj Linear、SDPA、**torch.cat（KV 缓存拼接/逐出）**、RoPE
│
├─ 模块⑥ 抽 4 层特征 selected_idx=[4,11,17,23] → 给三个头
│    ├─ Camera 头   算子：LN、Linear(9→2048, 2048→6144/8192)、16头自注意力×4层×4迭代、GELU/SiLU
│    │   输出 [S, 9]（平移3 + 四元数4 + 视场角2）
│    ├─ Depth 头(DPT) 算子：1×1/3×3 Conv、反卷积、bilinear 上采样、ReLU、exp
│    │   输出 [S, 1, 294, 518]
│    └─ Point 头(DPT) 同 Depth 骨架（默认 enable_point=False，demo 未开）
│
└─ 模块⑦ 后处理             算子：位姿解码（矩阵求逆）、深度反投影（逐像素 einsum）
    输出：外参/内参 + 3D 点云 → GLB
```

**一张图记住**：`卷积切块 → 72 层"矩阵乘开会/消化" → 三个头解压 → 几何反投影`；每一步都是你会的 C++ 循环（三重循环矩阵乘 / 滑窗卷积 / 单层循环逐元素）。

---

## 3. 算子清单 + 静态频次 + 规模

> 静态频次 = 一轮前向中该算子类型的实例数（数代码）。规模按单帧/单层典型值。

| # | 算子（代码层）| 底层 aten 算子 | 代码位置 | 静态实例数（每轮）| 典型张量规模 | 规模量级 |
|---|---|---|---|---|---|---|
| 1 | 切块卷积 | cudnn_convolution | patch_embed.py:62 | 1/帧 | 3→1024, 14×14 | ~0.5 GFLOP/帧 |
| 2 | QKV 线性 | addmm/mm | attention.py:57 | 72（每 Block 1）| [783,1024]×[1024,3072] | 4.9 GFLOP |
| 3 | 输出投影 | addmm | attention.py:61 | 72 | [783,1024]×[1024,1024] | 1.6 GFLOP |
| 4 | 注意力矩阵乘 | _efficient_attention_forward | attention.py:76 | 72（帧内 24 轻 + 跨帧 24 重）| 帧内 783²；跨帧 783×56400 | 帧内 2.5 G / 跨帧 181 G |
| 5 | MLP 放大 fc1 | addmm | mlp.py:29 | 72 | [783,1024]×[1024,4096] | 6.6 GFLOP |
| 6 | MLP 缩回 fc2 | addmm | mlp.py:31 | 72 | [783,4096]×[4096,1024] | 6.6 GFLOP |
| 7 | GELU | gelu | mlp.py:30 | 72 | 逐元素 [783,4096] | <0.1 G |
| 8 | LayerNorm | native_layer_norm | block.py:50/67 | 72×2=144 + 头 | 逐行 [783,1024] | <0.1 G |
| 9 | 残差相加 | add | block.py:99 | 72×2=144 | 逐元素 | <0.1 G |
| 10 | RoPE 旋转 | mul/neg/sin/cos | rope.py:157 | 24 frame + 24 global | 逐元素 [16,783,64] | <0.1 G |
| 11 | KV 缓存拼接/逐出 | **cat** / copy_ / contiguous | attention.py:635-729 | 24 层×每帧多次 | 整块缓存拷贝 | **时间瓶颈** |
| 12 | softmax | (SDPA 内部) | attention.py | 72 | 逐行 | <0.1 G |
| 13 | DPT 1×1/3×3 卷积 | cudnn_convolution | dpt_head.py | 每头 ~30 | 网格→全分辨率 | ~0.1 G/帧 |
| 14 | 反卷积 | cudnn_convolution_transpose | dpt_head.py | 每头 2 | k4s4/k2s2 | 小 |
| 15 | bilinear 上采样 | upsample_bilinear2d | dpt_head.py | 每头 5 | →294×518 | 带宽密集 |
| 16 | 位姿激活 exp/relu 等 | 逐元素 | head_act.py | 头 | 逐元素 | 忽略 |
| 17 | 相机头宽线性 | addmm | camera_head.py | 4层×4迭代 | 2048→6144/8192 | 小 |

**静态口径汇总**：全模型约 **72 个 Block**；每 Block 4 个大 Linear（qkv/proj/fc1/fc2）→ 约 **290 个 Linear**；注意力 72 处；LayerNorm ~150 处；逐元素/逐行算子（GELU/RoPE/softmax/add）遍布全程，数量最大但单个最便宜。

---

## 4. 实测：算子时间与频次（torch.profiler，10 帧）

> 口径：self CUDA time = 该算子自身 GPU 时间；count = 本次 10 帧运行中的调用次数。数据已从 `operator_profile.csv` 提取整理。

### 4.1 按 GPU 时间排序（谁最慢）

| 算子 | self CUDA 时间 | 占比 | 调用次数 | 是什么 |
|---|---|---|---|---|
| **aten::cat** | **22.01 s** | **74.0%** | 862 | **KV 缓存拼接（整块拷贝）** |
| （cat 内部 copy kernel，2 个）| 11.45s + 10.45s | — | 15 + 30 | cat 的底层内存搬运 |
| aten::cudnn_convolution | 4.61 s | 15.5% | 93 | patch embed + DPT 头卷积（fp32）|
| （conv 内部 cutlass kernel）| 3.70 s | — | 44 | 卷积底层 |
| aten::add | 0.71 s | 2.4% | 1062 | 残差相加 |
| **aten::addmm** | **0.53 s** | **1.8%** | 1103 | **矩阵乘（Linear 主体）** |
| aten::_efficient_attention_forward | 0.51 s | 1.7% | 264 | 注意力 SDPA |
| （gemm 内部 cutlass bf16）| 0.41 s | — | 504 | 矩阵乘底层 |
| convertTensor 格式转换 | 0.39 s | 1.3% | 46 | bf16↔fp32 转换 |
| aten::upsample_bilinear2d | 0.35 s | 1.2% | 15 | DPT 上采样 |
| aten::native_layer_norm | 0.23 s | 0.8% | 858 | LayerNorm |
| aten::copy_ | 0.20 s | 0.7% | 3537 | 拷贝 |
| aten::mul | 0.15 s | 0.5% | 1779 | 逐元素乘（RoPE 等）|
| aten::clamp_min_ | 0.10 s | 0.3% | 45 | ReLU/激活 |
| aten::sin / cos | 0.04 s each | 0.1% | 30 each | RoPE 三角 |
| aten::gelu | 0.03 s | 0.1% | 276 | GELU |

### 4.2 按调用次数排序（谁最频繁）

| 算子 | 调用次数 | 是什么 |
|---|---|---|
| cudaLaunchKernel | 10887 | GPU kernel 启动（总）|
| aten::as_strided / view / reshape | 9039 / 6650 / 2787 | **形状变换（零算力，纯搬指针）** |
| aten::empty / empty_strided | 4858 / 3174 | 分配内存 |
| aten::copy_ | 3537 | 拷贝 |
| aten::to / _to_copy | 3009 / 2928 | 类型/设备转换 |
| aten::transpose / slice | 2427 / 1982 | 转置/切片 |
| aten::mul | 1779 | 逐元素乘 |
| aten::layer_norm / linear | 1581 / 1536 | LayerNorm / Linear |
| aten::addmm | 1103 | 矩阵乘 |
| aten::add | 1062 | 残差 |
| aten::cat | 862 | KV 拼接 |
| aten::native_layer_norm | 858 | LayerNorm |
| aten::gelu | 276 | GELU |
| aten::scaled_dot_product_attention | 408 | 注意力 |

> **频次的关键洞察**：调用次数最多的不是矩阵乘，而是 **`view`/`reshape`/`transpose`/`slice`（合计 1.7 万+ 次）和 `copy_`/`to`（6000+ 次）**——这些是"搬数据/改形状"的零算力操作，但它们在 GPU 上触发大量内存搬运和 kernel 启动，是隐形的带宽/调度负担。

---

## 5. 静态估算 vs 实测（对照与修正）

| 口径 | 结论 | 谁对 |
|---|---|---|
| FLOP 分布 | 90% 是矩阵乘（global 注意力 + MLP）| 对（算术量口径）|
| **实测时间分布** | **74% 是 KV 缓存 `cat` 拷贝，矩阵乘仅 ~4%** | **本次实测，修正了"矩阵乘最慢"的直觉** |

### 为什么差这么大？三条原因（都重要）

1. **GPU 算 bf16 矩阵乘极快**（Tensor Core），所以矩阵乘在墙上钟里占比反而不大；瓶颈转移到了内存。
2. **SDPA 兜底路径的 `torch.cat` 是 O(n²)**：每来一帧，`SDPAAttention.forward` 里 `torch.cat((旧缓存, 新帧), dim=2)` 会把**整块历史 KV 重新拷一遍**。帧越多拷贝越狠 → 这就是 demo 慢的直接原因，也是 FlashInfer"分页缓存（原地 append、不拷贝）"存在的理由。
3. **CPU 启动开销巨大**：`aten::linear` 的 CPU total 31s（104%）、`addmm` self CPU 19.6s——1500+ 个小 Linear 的 kernel 启动把 CPU 打满，GPU 大部分时间在等 CPU。这是 `--compile`（torch.compile 融合算子、减少启动）能提速的原因。

> 结论：**LingBot-MAP 在"SDPA + 无 compile"下的真实瓶颈是"KV 缓存内存拷贝 + 算子启动开销"，不是矩阵乘算术。** 这两点，恰好就是 FPGA/加速最该先解决的方向——用分页/流式 KV 缓存 + 算子融合。

---

## 6. 对 FPGA / 部署的启示（按实测更新）

1. **别只盯着矩阵乘引擎**。实测 74% 时间在 KV 缓存 `cat` 拷贝——硬件设计第一优先是**"KV 缓存如何分页、如何原地 append、如何流式进出片外 DDR"**，而不是堆 DSP 算矩阵乘。
2. **卷积是第二大头（15.5%）**，且 DPT 头在 fp32 跑——量化时头部也要单独处理（int8 卷积），否则它是次瓶颈。
3. **大量"形状变换/拷贝"（1.7 万次 view/reshape/transpose + 6000 次 copy）** 在 GPU 上是带宽杀手，在 FPGA 上意味着"数据搬移网络"要和计算引擎同等重视。
4. **算子融合**（把 LN+残差+激活融成一个 kernel，像 `--compile` 做的那样）能砍掉海量 kernel 启动和中间 tensor 读写，对 FPGA 同样关键。
5. 量化（INT8）仍是最直接省 4 倍内存/带宽的手段，但**对象要覆盖 KV 缓存**，不只是权重。

---

## 7. 复现方法

```bash
# 环境 lingbot-map、项目根目录
python profile_operators.py
```

产出：`operator_profile.txt`（两张排序表）+ `operator_profile.csv`（逐算子 count/self cuda time）。想换真实图片/更多帧，改脚本里的 `S` 和 `images` 来源即可。

---

*报告完。静态部分基于源码精读，动态部分基于 10 帧实测；286 帧全量会因 KV 缓存增长使 `cat` 占比更高（O(n²)），故 74% 是保守下限。*
