# MindYOLO 贡献指南

## 贡献者许可协议

首次向 MindYOLO 社区提交代码前，需签署 CLA。

个人贡献者请参考 [ICLA 在线文档](https://www.mindspore.cn/icla) 了解详细信息。

## 入门指南

- 在 [Github](https://github.com/mindspore-lab/mindyolo) 上 Fork 代码库。
- 阅读 [README.md](../index.md)。

## 贡献流程

### 代码风格

请遵循此风格，以便 MindYOLO 易于审查、维护和开发。

- 编码指南

MindYOLO 社区使用 [Python PEP 8 编码风格](https://pep8.org/) 建议的 *Python* 编码风格和 [Google C++ 编码指南](http://google.github.io/styleguide/cppguide.html) 建议的 *C++* 编码风格。 [CppLint](https://github.com/cpplint/cpplint)、[CppCheck](http://cppcheck.sourceforge.net)、[CMakeLint](https://github.com/cmake-lint/cmake-lint)、[CodeSpell](https://github.com/codespell-project/codespell)、[Lizard](http://www.lizard.ws)、[ShellCheck](https://github.com/koalaman/shellcheck) 和 [PyLint](https://pylint.org) 用于检查代码格式，建议在 IDE 中安装这些插件。

- 单元测试指南

MindYOLO 社区使用 [pytest](http://www.pytest.org/en/latest/) 建议的 *Python* 单元测试风格和 [Googletest Primer](https://github.com/google/googletest/blob/master/docs/primer.md) 建议的 *C++* 单元测试风格。测试用例的设计意图应该通过其注释名称来体现。

- 重构指南

我们鼓励开发人员重构我们的代码以消除 [代码异味](https://en.wikipedia.org/wiki/Code_smell)。所有代码都应符合编码风格和测试风格的需求，重构代码也不例外。[Lizard](http://www.lizard.ws) 对 nloc（无注释的代码行数）的阈值为 100，对 cnc（循环复杂度数）的阈值为 20，当您收到 *Lizard* 警告时，您必须重构要合并的代码。

- 文档指南

我们使用 *MarkdownLint* 检查 markdown 文档的格式。MindYOLO CI 根据默认配置修改了以下规则。

- MD007（无序列表缩进）：**indent**参数设置为**4**，表示无序列表中的所有内容都需要使用四个空格进行缩进。
- MD009（行末空格）：**br_spaces**参数设置为**2**，表示行末可以有0个或2个空格。
- MD029（有序列表的序号）：**style**参数设置为**ordered**，表示有序列表的序号按升序排列。

具体请参见[RULES](https://github.com/markdownlint/markdownlint/blob/master/docs/RULES.md)。

### Fork-Pull开发模式

- Fork MindYOLO仓库

在向MindYOLO项目提交代码之前，请确保该项目已经fork到你自己的仓库。这意味着MindYOLO 仓库和你自己的仓库之间会并行开发，所以要小心避免两者不一致。

- 克隆远程仓库

如果要将代码下载到本地机器，`git` 是最好的方式：

```shell
# 对于 GitHub
git clone https://github.com/{insert_your_forked_repo}/mindyolo.git
git remote add upper https://github.com/mindspore-lab/mindyolo.git
```

- 本地开发代码

为避免多个分支之间不一致，`SUGGESTED` 建议签出到新分支：

```shell
git checkout -b {new_branch_name} origin/master
```

以 master 分支为例，MindYOLO 可能会根据需要创建版本分支和下游开发分支，请先修复上游的 bug。
然后你可以任意更改代码。

- 将代码推送到远程仓库

更新代码后，应以正式方式推送更新：

```shell
git add .
git status # 检查更新状态
git commit -m "您的提交标题"
git commit -s --amend #添加提交的具体描述
git push origin {new_branch_name}
```

- 将请求拉取到 MindYOLO 仓库

最后一步，您需要将新分支与 MindYOLO `master` 分支进行比较。完成拉取请求后，Jenkins CI 将自动设置为构建测试。您的拉取请求应尽快合并到上游主分支中，以降低合并风险。

### 报告问题

为项目做出贡献的一种好方法是在遇到问题时发送详细报告。我们始终欣赏写得好、详尽的错误报告，并会为此感谢您！

报告问题时，请参考以下格式：

- 您使用的是哪个版本的环境（MindSpore、os、python、MindYOLO 等）？
- 这是错误报告还是功能请求？
- 什么类型的问题，请添加标签以在问题仪表板上突出显示它。
- 发生了什么？
- 您期望发生什么？
- 如何重现它？（尽可能简短和准确）
- 给审阅者的特别说明？

**问题咨询：**

- **如果您发现一个未关闭的问题，而这正是您要解决的问题，** 请在该问题上发表一些评论，告诉其他人您将负责该问题。
- **如果问题打开了一段时间，** 建议贡献者在解决该问题之前进行预检查。
- **如果您解决了自己报告的问题，** 也需要在关闭该问题之前通知其他人。
- **如果您希望问题尽快得到回复，** 请尝试为其添加标签，您可以在 [标签列表](https://gitee.com/mindspore/community/blob/master/sigs/dx/docs/labels.md) 上找到各种标签

### 提出 PR

- 在 [GitHub](https://github.com/mindspore-lab/mindyolo/issues) 上以 *issue* 形式提出您的想法

- 如果是需要大量设计细节的新功能，还应提交设计提案。

- 在问题讨论和设计提案审查中达成共识后，完成分叉仓库的开发并提交 PR。

- 任何 PR 都必须收到来自批准者的 **2+ LGTM** 才能被允许。请注意，批准者不得在自己的 PR 上添加 *LGTM*。

- PR 经过充分讨论后，将根据讨论结果进行合并、放弃或拒绝。

**PR 建议：**

- 应避免任何不相关的更改。
- 确保您的提交历史记录有序。
- 始终让您的分支与主分支保持一致。
- 对于错误修复 PR，请确保所有相关问题都已链接。