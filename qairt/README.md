# qairt/ — Z-Image Turbo on Snapdragon HTP (QAIRT) 开发资料

本目录是 `qairt-dev` 分支的转换方法与排查记录，**不参与 app 构建**。
模型权重、ONNX、DLC、context `.bin`、QAIRT SDK 均**不在仓库内**（体积与再分发限制）。

| 路径 | 内容 |
|---|---|
| `docs/QNN_CONVERSION_GUIDE.md` | 可复用的 QNN 转换/量化流程，入口 **§47 路线图** |
| `docs/HANDOVER_2026-08-13_ZIMAGE_MVP.md` | 排查全过程（叙事 + 证据） |
| `docs/DELIVERY_FROZEN.md` | 已交付版本的冻结快照（sha256） |
| `MAINLINE.md` | 线索台账（未关闭条目全文） |
| `CLAUDE.md` | 项目约束（每条对应一次真实事故） |
| `scripts/` | 转换、验证、测量脚本与实验方案 `EXP_PLAN_*.md` |
| `archive/` | 已过时的交接文档快照，仅供追溯 |

⚠️ 文档与脚本里的 `D:\...` 路径是原开发机的布局，照做时按你的环境替换。

## 构建 app 需要的本机配置（写进仓库根目录的 `local.properties`，该文件不入库）

```properties
sdk.dir=<Android SDK>
qnn.sdk.dir=<QAIRT SDK 根目录，本项目用 2.48.0.260626>
spm.protoc.exe=<宿主可执行的 protoc 35.1>
```

也可以用 Gradle 属性（`-Pqnn.sdk.dir=...`）或环境变量 `QNN_SDK_ROOT` / `SPM_PROTOC_EXECUTABLE`。
缺任何一项时，CMake 会直接报错并提示怎么配置。
protoc 从 https://github.com/protocolbuffers/protobuf/releases 下载对应构建系统的 35.1 版本。
