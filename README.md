# AV2

测试性质的 AV2 编码器：架构参考 AOM-AV1 和 SVT-AV1，以 AV2 技术规范为准。项目目标是在相同 VMAF V1 下，相比 SVT-AV1 preset 6 自动关键帧编码，耗时不超过 3 倍、体积为 75–80%；目前尚未达成。

M6.11 将快速单帧和低延迟 P 帧合入同一个 `SvtAv2EncApp`。六张 1080p／2160p 代表帧在 VMAF V1 93–97 整带的等权体积为 AOM p0 的 **0.9622×**，同时约束 V1 和 NEG 时为 **1.1680×**。单线程耗时平均 **0.985×**，最慢一组 **1.279×**。两张二维动画的局部边缘误差仍超过对照门槛。

1080p 二维动画 383 帧全片在 V1 ≥93 的最佳测点为 **1,703,292 B**，V1 93.796、NEG 91.178。分别按 V1 和 NEG 匹配 AOM p0 后，体积比为 **3.528×** 和 **3.771×**。同一程序的两种编码命令、组合消融和测试范围见阶段报告。[M6.11 报告与低延迟视频命令](docs/M6.11-Intra-Inter-Fusion.md) · [逐组数据](benchmarks/m89-fusion.json)

## 构建和单帧编码

需要 CMake、Ninja 和 C11 编译器；Windows 使用 MSVC 开发环境。以下命令启用融合版使用的搜索、速度和量化配置。

```sh
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DSVT_AV2_STILL_PROFILE=ON -DSVT_AV2_SPEED_PROFILE=ON -DSVT_AV2_EXPERIMENTAL_DCT64_MATCH=ON -DSVT_AV2_EXPERIMENTAL_STILL_UV_RD=ON -DSVT_AV2_BUILD_SHARED=OFF -DSVT_AV2_WARNINGS_AS_ERRORS=ON
cmake --build build
ctest --test-dir build --output-on-failure
```

发布测速使用 MSVC x64 Release，并在配置时追加 `"-DCMAKE_C_FLAGS_RELEASE=/O2 /Ob2 /DNDEBUG /arch:AVX2"`。

单帧脚本需要 Python 3.11 或更高版本，输入为单张 1920×1080 或 3840×2160 的 10-bit 4:2:0 Y4M。可显式指定 `--content-type animation` 或 `real`；自动识别需额外安装 [分类依赖](tools/public-classifier/requirements.txt)，并提供本地 CLIP ViT-B/32 权重及 FFmpeg。示例：

```sh
python tools/single_frame.py --fusion-gdf --input frame.y4m --output frame.obu --encoder build/bin/SvtAv2EncApp --qp 99 --content-type animation
```

Windows 下程序名为 `SvtAv2EncApp.exe`。`SVT_AV2_EXPERIMENTAL_SATD_ANGLES` 和 `SVT_AV2_EXPERIMENTAL_FINE16_PAIR` 默认关闭，实验配置不能视为完整质量验收。

旧批测脚本可能引用已清理的历史合同和素材；重跑前需按当前环境调整路径和输入。仓库中的合成测试素材用于自动测试，私人测试素材和构建产物不纳入版本控制。

详见 [项目说明](project.md)、[M0–M5](docs/M0-M5.md)、[M6 帧内](docs/M6-Intra.md)、[M6 帧间](docs/M6-Inter.md) 和 [测试规则](tests/Test-Rule.md)。历史构建与验收记录已归并至三份阶段文档。

## 许可证

项目自有代码采用 BSD-3-Clause-Clear。AVM 移植部分的上游软件与专利许可随源码提供。[LICENSE](LICENSE) · [第三方许可与来源](THIRD_PARTY_NOTICES.md)
