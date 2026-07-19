# photo-with-depth

把一张普通照片变成**分层的立体照片(LDI)**:估计深度 → 按深度/物体分层 → 每层抠成可移动贴图 → 网页里做视差(鼠标 / 手机陀螺仪)。

## 流水线(一步一个文件)

```
照片
 └─ step_1_crop_frame.py      ① 裁掉画框/白边(局部方差找内容边界)
     └─ step_2_build_depth_map.py  ② Depth Anything V2 估深度 → 0~1 深度图
         └─ step_3_build_regions.py ③ 按深度分层(valley:峰做中心、谷做分界)
             └─ objects.py           SAM2 物体分割 + 归层 + 抠平整整体
                 └─ build_background.py 背景补全(填充算法,见下)
                     └─ build_sprites.py  生成 LDI 分层贴图(sprite_00..NN.png)
                         └─ index.html    网页视差编辑器
```

- **step_3_build_regions.py** 把分层拆成清晰小步:`prepare_depth → choose_levels/valley_levels → assign → split → describe`,默认 `method="valley"`。
- **objects.py**:`build_sam_valley_regions`(默认路径)= valley 分层 + SAM 物体 ≥90% 归层 + 抠出平整大整体。另有 `build_layer_groups` / `build_hybrid_regions` 两种模式。
- **regions.py**:底层工具(1D k-means、肘部选层、命名、归一化)+ `save_scene`。

## 填充算法(build_background.py)

移开的物体/层留下的洞怎么补,`--method`:

| 方法 | 特点 |
|---|---|
| **harmonic** | Laplace 膜,邻居白更白/黑更黑,平滑无裂缝(build_sprites 默认) |
| **pushpull** | 金字塔,快但深处偏平均 |
| bleed / mode | 最近邻 / 众数(会条纹 / 面片) |
| lama | AI 补全(需 simple-lama-inpainting) |
| telea | 经典快速模糊 |

## 入口(命令行)

```bash
# 轻量:裁边 + 深度 + valley 分层(不跑 SAM,快)
python generate_depth_photo.py --input examples/wall-street.png

# 完整:SAM + valley + 归层 + 抠整体 + 背景 + LDI 贴图(慢,首次跑 SAM)
python build_objects.py                 # 默认:同层合并
python build_objects.py --per-object    # 每个物体一块
python build_objects.py --layer-groups  # 旧的层组模式

# 只重建分区(用现成深度)/ 只重做背景 / 只重做贴图
python build_scene.py
python build_background.py --method harmonic
python build_sprites.py

# 视差预览 GIF
python build_preview.py --layered
```

## 网页编辑器(index.html)

```bash
python -m http.server 8000      # 打开 http://localhost:8000
```

- 鼠标移动 / **Auto-sway** / **Phone tilt**(手机陀螺仪,需 https)驱动视差。
- 选层 → **深度偏移(re-rank)/ Ignore(静止)/ Hide(隐藏)**。
- 顶部下拉切换案例;手机上控制面板可收起。
- 案例数据在 `outputs/cases/<name>/`(`scene.json` + `sprites/` + 深度/标签图)。

## 调参 notebook

`notebooks/depth_classifier_lab.ipynb`:交互式调分层/边界(valley 参数、SAM 归层阈值、抠整体阈值),满意后发布到 `outputs/`。

## 目录

```
step_1/2/3_*.py, objects.py, regions.py   核心流水线
build_objects/scene/background/sprites/preview.py, generate_depth_photo.py  入口
index.html                                网页编辑器
notebooks/  tests/  examples/             lab / 单测 / 示例图
outputs/                                  生成产物(gitignored)
```

依赖:`torch` `transformers` `opencv-python` `numpy` `Pillow`(可选 `simple-lama-inpainting`)。
