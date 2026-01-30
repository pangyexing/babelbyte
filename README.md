# BabelByte

AI 驱动的**个人情报产品**，从 Reddit 和 Twitter 抓取内容，通过 AI 处理后生成结构化情报简报。支持事件追踪、主题雷达、知识库检索和周期性复盘报告。

## 核心能力

| 能力 | 说明 | 状态 |
|------|------|------|
| **每日 Digest** | 结构化情报简报：摘要 + 关键点 + 影响评估 + 行动项 | ✅ |
| **事件流** | 同一事件多篇合并，追踪进展而非堆叠 | ✅ |
| **主题雷达** | 按主题聚合，追踪趋势变化 | ✅ |
| **知识库** | 全文搜索、按日期浏览、状态管理 | ✅ |
| **行动清单** | 从高价值内容自动提取可执行项 | ✅ |
| **周报/月报** | 周期性复盘，知识增长可见 | ✅ |

## 功能特性

- **多平台支持**: Reddit (RSS) 和 Twitter (API v2)
- **AI 智能处理**: 通过 Claude/Codex CLI 生成结构化分析
- **结构化输出**: 摘要、关键点、短期/长期影响、行动项
- **事件聚类**: 相关内容自动合并成事件流
- **主题追踪**: 定义关键词，持续追踪主题动态
- **全文检索**: 基于 FTS5 的快速搜索
- **周期报告**: 自动生成周报和月报
- **邮件推送**: 每日定时发送精美的 HTML 格式摘要邮件

## 安装

### 环境要求

- Python 3.10+
- Claude Code CLI 或 Codex CLI (已安装并登录)

### 安装步骤

```bash
# 克隆项目
git clone <repo-url>
cd babelbyte

# 创建虚拟环境 (推荐)
conda create -n babelbyte python=3.10
conda activate babelbyte

# 安装依赖
pip install -e .

# 开发环境
pip install -e ".[dev]"
```

## 配置

### 1. 创建配置文件

```bash
cp .env.example .env
```

### 2. 编辑 `.env` 文件

```bash
# AI 提供商配置 (claude / codex / auto)
AI_PROVIDER=auto

# Twitter API v2 (可选，免费版 1,500条/月)
TWITTER_BEARER_TOKEN=your_twitter_bearer_token

# 邮件配置
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

## 使用方法

### 订阅管理

```bash
# 订阅 Reddit 子版块
babelbyte subscribe reddit MachineLearning

# 订阅 Twitter 用户
babelbyte subscribe twitter elonmusk

# 查看订阅列表
babelbyte list

# 取消订阅
babelbyte unsubscribe reddit MachineLearning
```

### 内容抓取与处理

```bash
# 手动抓取所有订阅源
babelbyte fetch

# 预览摘要 (不发送邮件)
babelbyte digest --dry-run

# 生成并发送摘要邮件
babelbyte digest

# 运行事件聚类
babelbyte cluster
```

### 知识库检索 (Phase 4)

```bash
# 全文搜索
babelbyte search "GPT"
babelbyte search "AI融资" --category AI --from 2024-01-01 --min-importance 7

# 按日期浏览
babelbyte browse                      # 今天
babelbyte browse --date yesterday     # 昨天
babelbyte browse --date 2024-01-15 --category AI

# 查看内容详情
babelbyte item 123
babelbyte item 123 --open   # 在浏览器打开原文

# 标记状态
babelbyte mark 123 saved    # 收藏
babelbyte mark 123 read     # 已读
babelbyte mark 123 flagged  # 标记

# 统计信息
babelbyte stats

# 重建搜索索引
babelbyte rebuild-index
```

### 事件追踪 (Phase 2)

```bash
# 查看最近事件
babelbyte events
babelbyte events --days 3 --category AI

# 查看事件详情
babelbyte event 123

# 运行事件聚类
babelbyte cluster --limit 100
```

### 主题雷达 (Phase 3)

```bash
# 列出所有主题
babelbyte topics

# 添加主题
babelbyte topic add "AI应用" --keywords "GPT,ChatGPT,AI,人工智能"
babelbyte topic add "OpenAI" --keywords "OpenAI,Sam Altman" --description "OpenAI 公司动态"

# 查看主题详情
babelbyte topic show "AI应用"

# 删除主题
babelbyte topic delete "AI应用" --yes
```

### 行动清单 (Phase 5)

```bash
# 查看待办行动
babelbyte actions
babelbyte actions --status pending --priority 高

# 完成行动
babelbyte action 123 done

# 忽略行动
babelbyte action 123 dismissed
```

### 周报/月报 (Phase 6)

```bash
# 生成周报
babelbyte report week
babelbyte report week --weeks-ago 1   # 上周

# 生成月报
babelbyte report month
babelbyte report month --months-ago 1 # 上月
```

### 其他命令

```bash
# 查看当前配置
babelbyte config

# 发送测试邮件
babelbyte test-email

# 启动调度器 (后台服务)
babelbyte run

# 测试模式 (无需 API)
babelbyte --mock fetch
babelbyte --mock digest --dry-run
```

## 架构

```
CLI (Click) → Scheduler (APScheduler)
                    ↓
    ┌───────────────┼───────────────┐
    ↓               ↓               ↓
Fetchers      Event Stream    Topic Radar
(Reddit/Twitter)    ↓               ↓
    ↓          Clustering      Matching
    └──────→ SQLite + FTS5 ←───────┘
                    ↓
            AI Processors
         (Claude/Codex CLI)
                    ↓
         ┌─────────┼─────────┐
         ↓         ↓         ↓
    Digest    Actions    Reports
    Email     Extract    Weekly/Monthly
```

### 数据流

```
1. 订阅配置 → 内容抓取 → 存入数据库
2. AI 处理: 摘要 + 关键点 + 影响评估 + 行动项
3. 事件聚类: 相关内容合并成事件流
4. 主题匹配: 内容关联到已定义主题
5. 行动提取: 从高价值内容生成待办
6. 邮件发送: 结构化 Digest
7. 周期报告: 周报/月报复盘
```

## 项目结构

```
babelbyte/
├── pyproject.toml              # 依赖管理
├── .env.example                # 环境变量模板
├── main.py                     # 主入口
├── config/
│   └── settings.py             # 配置加载
├── src/
│   ├── fetchers/               # 内容抓取
│   │   ├── base.py
│   │   ├── reddit.py           # Reddit RSS
│   │   └── twitter.py          # Twitter API v2
│   ├── processors/             # AI 处理
│   │   ├── base.py             # 基类 + 增强结果
│   │   ├── claude_cli.py       # Claude CLI
│   │   ├── openai_cli.py       # Codex CLI
│   │   ├── digest_processor.py # 摘要 + 行动提取
│   │   ├── event_stream.py     # 事件聚类 (Phase 2)
│   │   └── rule_classifier.py  # 规则预分类
│   ├── analytics/              # 分析模块
│   │   ├── topic_radar.py      # 主题雷达 (Phase 3)
│   │   └── reports.py          # 周报/月报 (Phase 6)
│   ├── delivery/
│   │   └── email_sender.py     # 邮件发送
│   ├── storage/
│   │   ├── models.py           # 数据模型 (含所有 Phase)
│   │   └── database.py         # SQLite + FTS5
│   ├── scheduler/
│   │   └── jobs.py             # 定时任务
│   └── cli/
│       └── commands.py         # CLI 命令
├── templates/
│   └── email_digest.html       # 增强版邮件模板
├── data/                       # SQLite 数据库
└── logs/                       # 日志文件
```

## 数据模型

### 内容项 (ContentItem)

```python
# 基础字段
title, content, url, author, published_at

# AI 处理结果 (Phase 1 增强)
summary           # 50字摘要
category          # AI/编程/产品/技术/创业/科学/商业/其他
importance_score  # 1-10
one_liner         # 一句话结论
key_points        # 关键点: [{type, value, impact}]
impact_assessment # 影响评估: {short_term, long_term, certainty}
actionable_items  # 行动项: [{type, description, priority}]

# 状态管理 (Phase 4)
state             # unread/read/saved/archived/flagged
```

### 事件聚类 (Phase 2)

```
event_clusters    # 事件定义
event_members     # 事件-内容关联
event_timeline    # 事件时间线
```

### 主题雷达 (Phase 3)

```
topics            # 主题定义 + 关键词
content_topics    # 内容-主题关联
topic_snapshots   # 主题快照 (周期性)
```

### 行动清单 (Phase 5)

```
action_items      # 行动项: type/description/priority/status
triggers          # 触发器 (自动化规则)
```

## 命令参考

| 命令 | 说明 |
|------|------|
| **订阅管理** | |
| `subscribe reddit/twitter <name>` | 订阅 |
| `unsubscribe <source> <name>` | 取消订阅 |
| `list [-a]` | 查看订阅 |
| **内容处理** | |
| `fetch` | 抓取内容 |
| `digest [--dry-run]` | 生成摘要 |
| `cluster` | 事件聚类 |
| **知识库** | |
| `search <query>` | 全文搜索 |
| `browse [--date]` | 按日期浏览 |
| `item <id> [--open]` | 查看详情 |
| `mark <id> <state>` | 标记状态 |
| `stats` | 统计信息 |
| **事件** | |
| `events [--days]` | 事件列表 |
| `event <id>` | 事件详情 |
| **主题** | |
| `topics` | 主题列表 |
| `topic add/show/delete` | 主题管理 |
| **行动** | |
| `actions [--status]` | 行动列表 |
| `action <id> done/dismissed` | 更新状态 |
| **报告** | |
| `report week [--weeks-ago]` | 周报 |
| `report month [--months-ago]` | 月报 |
| **其他** | |
| `config` | 查看配置 |
| `test-email` | 测试邮件 |
| `run` | 启动调度器 |
| `--mock <cmd>` | 测试模式 |

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

## 许可证

MIT License
