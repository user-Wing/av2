# SVT-AV2 项目

SVT-AV2 是测试性质的 AV2 编码器，以 AOM-AV1、SVT-AV1 为架构参考，以 AV2 技术规范为准。最终目标是在相同 VMAF V1 下，相对 SVT-AV1 preset 6 自动关键帧编码，编码耗时不超过 3 倍，体积为其 75–80%。该目标尚未验收。

当前版本 M6.11 在同一原生编码器中提供快速单帧配置和低延迟 P 帧配置。单帧公开入口为 `tools/single_frame.py --fusion-gdf`，帧间采用双参考、按运动变化刷新参考帧、量化矩阵、关键帧 GDF 和分层量化。六组单帧平均等 V1 体积为 AOM p0 的 0.9622×。1080p 二维动画 383 帧在 V1 ≥93 的最佳测点为 AOM 的 3.528×，视频 2× 目标仍有差距。[融合测试与实测配置](docs/M6.11-Intra-Inter-Fusion.md)

项目自有代码按 BSD-3-Clause-Clear 发布。移植部分保留相应的上游许可。[项目许可证](LICENSE) · [第三方说明](THIRD_PARTY_NOTICES.md)

[第一阶段](docs/M0-M5.md) · [测试规则](tests/Test-Rule.md)
