# BabelByte

AI 驱动的内容订阅系统，支持 Reddit 和 Twitter，每日通过邮件发送 AI 整理后的中文内容摘要。

## 功能特性

- **多平台支持**: Reddit (RSS) 和 Twitter (API v2)
- **AI 智能处理**: 通过 Claude Code CLI 生成中文摘要、分类和重要性评分
- **邮件推送**: 每日定时发送精美的 HTML 格式摘要邮件
- **命令行管理**: 简洁的 CLI 工具管理订阅和配置
- **定时调度**: 自动抓取内容和发送摘要

## 安装

### 环境要求

- Python 3.10+
- Claude Code CLI (已安装并登录)

### 安装步骤

```bash
# 克隆项目
git clone <repo-url>
cd babelbyte

# 创建虚拟环境 (推荐使用 conda)
conda create -n babelbyte python=3.10
conda activate babelbyte

# 安装依赖
pip install -e .
```

## 配置

### 1. 创建配置文件

```bash
cp .env.example .env
```

### 2. 编辑 `.env` 文件

```bash
# AI 提供商配置 (claude / openai / auto)
AI_PROVIDER=auto

# Twitter API v2 (可选，免费版 1,500条/月)
# 申请地址: https://developer.twitter.com
TWITTER_BEARER_TOKEN=your_twitter_bearer_token

# 邮件配置 (139邮箱示例)
SMTP_HOST=smtp.139.com
SMTP_PORT=465
SMTP_USER=your_email@139.com
SMTP_PASSWORD=your_smtp_authorization_code
EMAIL_FROM=your_email@139.com
EMAIL_TO=recipient@example.com

# 调度设置
DIGEST_SEND_TIME=08:00
FETCH_INTERVAL_HOURS=6
```

### 3. AI 提供商配置

支持两种 AI 提供商，可通过 `AI_PROVIDER` 环境变量切换：

#### Claude Code CLI (默认)

确保已安装 Claude Code CLI 并完成登录:

```bash
# 验证安装
claude --version
```

#### OpenAI API

如果使用 OpenAI，需要安装额外依赖并配置 API Key:

```bash
# 安装 OpenAI 支持
pip install -e ".[openai]"

# 在 .env 中配置
OPENAI_API_KEY=your_openai_api_key
OPENAI_MODEL=gpt-4o-mini  # 或 gpt-4o, gpt-3.5-turbo 等
AI_PROVIDER=openai
```

## 使用方法

### 订阅管理

```bash
# 订阅 Reddit 子版块
babelbyte subscribe reddit MachineLearning
babelbyte subscribe reddit Python

# 订阅 Reddit 用户
babelbyte subscribe reddit --user spez

# 订阅 Twitter 用户
babelbyte subscribe twitter elonmusk

# 查看订阅列表
babelbyte list

# 取消订阅
babelbyte unsubscribe reddit MachineLearning
babelbyte unsubscribe twitter elonmusk
```

### 内容抓取

```bash
# 手动抓取所有订阅源
babelbyte fetch
```

### 生成摘要

```bash
# 预览摘要 (不发送邮件)
babelbyte digest --dry-run

# 生成并发送摘要邮件
babelbyte digest

# 自定义参数
babelbyte digest --min-importance 7 --max-items 20
```

### 配置查看

```bash
# 查看当前配置
babelbyte config

# 发送测试邮件
babelbyte test-email
```

### 运行调度器

```bash
# 启动后台调度服务
babelbyte run
```

### 测试模式

使用 `--mock` 参数可在无 API 配置时测试:

```bash
babelbyte --mock fetch
babelbyte --mock digest --dry-run
```

## 数据流

```
订阅配置 → 内容抓取 → 存入数据库 → Claude AI 处理 → 邮件发送
           (定时)     (SQLite)    (摘要/分类/评分)   (每日定时)
```

## 项目结构

```
babelbyte/
├── pyproject.toml              # 依赖管理
├── .env.example                # 环境变量模板
├── main.py                     # 主入口
├── config/
│   ├── settings.py             # 配置加载
│   └── subscriptions.yaml      # 订阅列表模板
├── src/
│   ├── fetchers/               # 内容抓取
│   │   ├── base.py             # 基类
│   │   ├── reddit.py           # Reddit RSS
│   │   └── twitter.py          # Twitter API v2
│   ├── processors/             # AI 处理
│   │   ├── claude_cli.py       # Claude CLI 封装
│   │   └── digest_processor.py # 摘要生成
│   ├── delivery/
│   │   └── email_sender.py     # 邮件发送
│   ├── storage/
│   │   ├── models.py           # 数据模型
│   │   └── database.py         # SQLite 操作
│   ├── scheduler/
│   │   └── jobs.py             # 定时任务
│   └── cli/
│       └── commands.py         # CLI 命令
├── templates/
│   └── email_digest.html       # 邮件模板
├── data/                       # SQLite 数据库
└── logs/                       # 日志文件
```

## API 配置说明

### Reddit

无需 API Key，使用原生 RSS:
- Subreddit: `https://www.reddit.com/r/{name}/.rss`
- 用户: `https://www.reddit.com/user/{name}/.rss`

### Twitter API v2

免费版限额: 1,500 条推文/月

1. 访问 https://developer.twitter.com
2. 创建项目和应用
3. 获取 Bearer Token
4. 配置到 `.env` 文件

### 邮箱 SMTP

以 139 邮箱为例:
1. 登录 139 邮箱
2. 设置 → POP3/SMTP 服务 → 开启
3. 获取授权码
4. 配置到 `.env` 文件

## 命令参考

| 命令 | 说明 |
|------|------|
| `babelbyte subscribe reddit <name>` | 订阅子版块 |
| `babelbyte subscribe reddit -u <name>` | 订阅 Reddit 用户 |
| `babelbyte subscribe twitter <name>` | 订阅 Twitter 用户 |
| `babelbyte unsubscribe <source> <name>` | 取消订阅 |
| `babelbyte list` | 查看订阅列表 |
| `babelbyte list -a` | 查看所有订阅 (含已禁用) |
| `babelbyte fetch` | 抓取内容 |
| `babelbyte digest` | 生成并发送摘要 |
| `babelbyte digest --dry-run` | 预览摘要 |
| `babelbyte config` | 查看配置 |
| `babelbyte test-email` | 发送测试邮件 |
| `babelbyte run` | 启动调度器 |
| `babelbyte --mock <cmd>` | 使用模拟数据 |

## 许可证

MIT License
