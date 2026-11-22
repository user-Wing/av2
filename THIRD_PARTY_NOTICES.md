# 第三方许可与来源

项目自有代码采用 BSD-3-Clause-Clear。移植的代码与数据保留其版权声明和上游条款。[项目许可证](LICENSE)

- AVM 的 GDF、CDEF、PC-Wiener、量化矩阵及相关表使用 Alliance for Open Media 的 BSD 3-Clause Clear License 和 Alliance for Open Media Patent License 1.0。发布包附有上游两份许可全文。QM4、QM8、QM12 和帧间 GDF 表已与 AVM 提交 `1cfda4fa4879ef2da411b02fe2f51597daab3451` 和 `01803db6afa5a0616702a3814fc5934ab9e99adf` 逐项核对。[软件许可证](Source/Lib/Restoration/AVM_LICENSE) · [专利许可证](Source/Lib/Restoration/AVM_PATENTS) · [AVM 源码](https://aomedia.googlesource.com/avm/)
- 规范派生表的来源见相应源文件或生成脚本。[AV2 v1.0.0 规范](https://av2.aomedia.org/v1.0.0/) · [IST 表生成脚本](tools/gen_intra_ist_tables.py)
- 可选的自动内容分类使用 OpenAI CLIP。手动选择 `animation` 或 `real` 可直接运行单帧编码。CLIP 及其依赖的许可见分类器说明。[分类器第三方声明](tools/public-classifier/THIRD_PARTY_NOTICES.txt) · [CLIP 许可证](tools/public-classifier/LICENSE.CLIP.txt)
