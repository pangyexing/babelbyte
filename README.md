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
| **Token 追踪** | AI 调用统计、成本估算、缓存命中率 | ✅ |
| **数据校验** | 12 项完整性检查，自动修复 | ✅ |

## 功能特性

- **多平台支持**: Reddit、Twitter、Hacker News、通用 RSS/Atom
- **AI 智能处理**: 通过 Claude/Codex CLI 生成结构化分析
- **结构化输出**: 摘要、关键点、短期/长期影响、行动项
- **语义聚类**: 混合相似度 (40% 规则 + 60% Embedding) 智能合并
- **自动主题发现**: 基于实体频率、关键词共现、趋势突增自动发现主题
- **主题追踪**: 定义关键词，持续追踪主题动态
- **全文检索**: 基于 FTS5 的快速搜索
- **周期报告**: 自动生成周报和月报
- **邮件推送**: 每日定时发送精美的 HTML 格式摘要邮件
- **Token 优化**: 规则预分类、模型分层、批量处理，节省 AI 成本
- **持久缓存**: 事件确认结果缓存，避免重复 AI 调用
- **数据校验**: 12 项完整性检查，支持自动修复

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

# Twitter API (TwitterAPI.io)
TWITTERAPI_IO_KEY=your_key

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

# AI 缓存配置
AI_CACHE_ENABLED=true
AI_CACHE_TTL=86400            # 24 小时

# 聚类配置
CLUSTER_CACHE_TTL=60          # 聚类缓存秒数
CLUSTER_RETRY_HOURS=24        # 失败重试间隔

# Embedding 配置 (可选)
EMBEDDING_PROVIDER=sentence-transformers  # 或 openai
EMBEDDING_RULE_WEIGHT=0.4     # 规则相似度权重
EMBEDDING_SEMANTIC_WEIGHT=0.6 # 语义相似度权重

# Token 优化配置
RULE_ONLY_ENABLED=true        # 规则预处理 (高置信度跳过 AI)
RULE_ONLY_MAX_CONTENT=1000    # 规则处理最大内容长度
RULE_ONLY_MIN_BOOST=1         # 最小关键词增强
SKIP_ENABLED=true             # 跳过垃圾/招聘等低价值内容
MINIMAL_PROMPT_ENABLED=true   # 低重要性内容使用简化提示
MINIMAL_PROMPT_THRESHOLD=3    # 简化提示阈值
```

## 使用方法

### 订阅管理

```bash
# 订阅 Reddit 子版块
bb subscribe reddit MachineLearning

# 订阅 Twitter 用户
bb subscribe twitter elonmusk

# 订阅 Hacker News (支持 front/new/best/ask/show)
bb subscribe hackernews tech --type front
bb subscribe hackernews ai --type best

# 订阅通用 RSS/Atom Feed
bb subscribe rss "TechCrunch" --url "https://techcrunch.com/feed/"
bb subscribe rss "OpenAI Blog" --url "https://openai.com/blog/rss.xml"

# 查看订阅列表
bb list

# 取消订阅
bb unsubscribe reddit MachineLearning
bb unsubscribe hackernews tech
bb unsubscribe rss "TechCrunch"
```

### 每日使用 (推荐)

```bash
# 一键运行完整流程: fetch → process → embeddings → topics → cluster → digest
bb daily

# 预览不发送邮件
bb daily --dry-run

# 跳过抓取 (已有内容时)
bb daily --skip-fetch
```

**流程说明:**
```
bb daily
  ├── 1. fetch            # 抓取所有订阅源
  ├── 2. embeddings       # 本地 Embedding (免费，启用语义去重)
  ├── 3. process_content  # AI 处理 (跳过相似内容，节省 Token)
  ├── 4. topic discovery  # 主题自动发现 (免费)
  ├── 5. clustering       # 事件聚类 (消耗 LLM Token)
  └── 6. digest + email   # 生成并发送邮件
```

### 内容抓取与处理 (分步执行)

```bash
# 手动抓取所有订阅源
bb fetch

# 预览摘要 (不发送邮件)
bb digest --dry-run

# 生成并发送摘要邮件
bb digest

# 运行事件聚类
bb cluster
```

### 知识库检索 (Phase 4)

```bash
# 全文搜索
bb search "GPT"
bb search "AI融资" --category AI --from 2024-01-01 --min-importance 7

# 按日期浏览
bb browse                      # 今天
bb browse --date yesterday     # 昨天
bb browse --date 2024-01-15 --category AI

# 查看内容详情
bb item 123
bb item 123 --open   # 在浏览器打开原文

# 标记状态
bb mark 123 saved    # 收藏
bb mark 123 read     # 已读
bb mark 123 flagged  # 标记

# 统计信息
bb stats

# 重建搜索索引
bb rebuild-index
```

### 事件追踪 (Phase 2)

```bash
# 查看最近事件
bb events
bb events --days 3 --category AI

# 查看事件详情
bb event 123

# 运行事件聚类
bb cluster --limit 100
```

### 主题雷达 (Phase 3)

```bash
# 列出所有主题
bb topics

# 添加主题
bb topic add "AI应用" --keywords "GPT,ChatGPT,AI,人工智能"
bb topic add "OpenAI" --keywords "OpenAI,Sam Altman" --description "OpenAI 公司动态"

# 查看主题详情
bb topic show "AI应用"

# 删除主题
bb topic delete "AI应用" --yes

# 自动发现主题 (基于实体频率、关键词共现、趋势突增)
bb topic discover --days 14 --min-frequency 5

# 查看主题建议
bb topic suggestions

# 交互式审核建议 (接受/拒绝/合并)
bb topic review
```

### 行动清单 (Phase 5)

```bash
# 查看待办行动
bb actions
bb actions --status pending --priority 高

# 完成行动
bb action 123 done

# 忽略行动
bb action 123 dismissed
```

### 周报/月报 (Phase 6)

```bash
# 生成周报
bb report week
bb report week --weeks-ago 1   # 上周

# 生成月报
bb report month
bb report month --months-ago 1 # 上月
```

### 诊断与优化

```bash
# 数据完整性校验
bb validate                  # 运行 12 项检查
bb validate --fix            # 自动修复问题
bb validate --verbose        # 显示详细信息

# Token 使用统计
bb token-stats               # 查看调用统计和成本估算
bb token-stats --reset       # 重置统计数据

# 缓存管理
bb cache-stats               # 查看缓存指标
bb cache-stats --cleanup     # 清理过期缓存
```

### 并行处理

```bash
# 并行事件聚类 (默认 4 线程)
bb cluster --parallel
bb cluster --parallel --workers 8

# 并行生成摘要
bb digest --parallel
```

### 语义 Embedding

```bash
# 计算内容 Embedding
bb embeddings compute --limit 1000

# 查看 Embedding 统计
bb embeddings stats

# 重建聚类中心向量
bb embeddings rebuild-centroids
```

### 其他命令

```bash
# 查看当前配置
bb config

# 发送测试邮件
bb test-email

# 启动调度器 (后台服务)
bb run

# 测试模式 (无需 API)
bb --mock fetch
bb --mock digest --dry-run
```

## 架构

```
                        CLI (Click)
                            ↓
                   Scheduler (APScheduler)
                            ↓
┌───────────────────────────┴───────────────────────────┐
↓                                                       ↓
Fetchers                                         Rule Classifier
(Reddit/Twitter/HN/RSS)                          (40+ 域名预分类)
↓                                                       ↓
└───────────────→ SQLite + FTS5 ←───────────────────────┘
                            ↓
                      Embeddings ←─── (本地免费，启用语义去重)
                 (SentenceTransformers)
                            ↓
                    AI Processors ←─── Token Tracker
                 (Claude/Codex CLI)     (8类调用追踪)
                 (跳过相似内容:92%阈值)
                            ↓
        ┌───────────────────┼───────────────────┐
        ↓                   ↓                   ↓
  Event Stream         Topic Radar        Topic Discovery
  (混合聚类:            (关键词匹配         (实体频率/
   40%规则+60%语义       趋势检测)           趋势突增)
   +AI确认)
        ↓                   ↓                   ↓
        └─────────┬─────────┴─────────┬─────────┘
                  ↓                   ↓
    ┌─────────────┼─────────────┬─────┴─────┐
    ↓             ↓             ↓           ↓
 Digest       Actions       Reports    Validation
 Email        Extract      Week/Month  (12项检查)
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
│   ├── settings.py             # 配置加载
│   └── ai_models.yaml          # AI 模型分层配置
├── src/
│   ├── fetchers/               # 内容抓取
│   │   ├── base.py
│   │   ├── reddit.py           # Reddit RSS + 增量抓取
│   │   ├── twitter.py          # Twitter (TwitterAPI.io)
│   │   ├── hackernews.py       # Hacker News (5 种 Feed)
│   │   └── rss.py              # 通用 RSS/Atom
│   ├── processors/             # AI 处理
│   │   ├── base.py             # 基类 + 增强结果
│   │   ├── claude_cli.py       # Claude CLI
│   │   ├── openai_cli.py       # Codex CLI
│   │   ├── digest_processor.py # 摘要 + 行动提取
│   │   ├── event_stream.py     # 事件聚类 + 混合相似度
│   │   ├── embeddings.py       # Embedding 提供者 + 管理器
│   │   └── rule_classifier.py  # 规则预分类 (40+ 域名)
│   ├── analytics/              # 分析模块
│   │   ├── topic_radar.py      # 主题雷达
│   │   ├── topic_discovery.py  # 自动主题发现
│   │   ├── reports.py          # 周报/月报
│   │   └── token_tracker.py    # Token 使用追踪
│   ├── optimization/           # 性能优化
│   │   ├── cache_optimizer.py  # 缓存指标与清理
│   │   └── dedup_optimizer.py  # 重复检测
│   ├── validation/             # 数据校验
│   │   ├── data_validator.py   # 12 项完整性检查
│   │   └── diagnostic_queries.py # SQL 诊断查询
│   ├── delivery/
│   │   └── email_sender.py     # 邮件发送
│   ├── storage/
│   │   ├── models.py           # 数据模型
│   │   └── database.py         # SQLite + FTS5
│   ├── scheduler/
│   │   └── jobs.py             # 定时任务
│   └── cli/
│       └── commands.py         # CLI 命令 (20+)
├── templates/
│   └── email_digest.html       # 增强版邮件模板
├── tests/                      # 测试用例 (3400+ 行)
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
| `subscribe reddit/twitter <name>` | 订阅 Reddit/Twitter |
| `subscribe hackernews <name> --type` | 订阅 HN (front/new/best/ask/show) |
| `subscribe rss <name> --url` | 订阅 RSS/Atom Feed |
| `unsubscribe <source> <name>` | 取消订阅 |
| `list [-a]` | 查看订阅 |
| **内容处理** | |
| `daily [--dry-run] [--skip-fetch]` | 一键运行完整流程 (推荐) |
| `fetch` | 抓取内容 |
| `digest [--dry-run] [--parallel]` | 生成摘要 |
| `cluster [--parallel] [--workers]` | 事件聚类 |
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
| `topic discover [--days]` | 自动发现主题 |
| `topic suggestions` | 查看主题建议 |
| `topic review` | 交互式审核建议 |
| **行动** | |
| `actions [--status]` | 行动列表 |
| `action <id> done/dismissed` | 更新状态 |
| **报告** | |
| `report week [--weeks-ago]` | 周报 |
| `report month [--months-ago]` | 月报 |
| **诊断优化** | |
| `validate [--fix]` | 数据完整性校验 |
| `token-stats [--reset]` | Token 使用统计 |
| `cache-stats [--cleanup]` | 缓存管理 |
| **Embedding** | |
| `embeddings compute [--limit]` | 计算内容 Embedding |
| `embeddings stats` | Embedding 统计 |
| `embeddings rebuild-centroids` | 重建聚类中心向量 |
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

### Twitter (TwitterAPI.io)

使用 TwitterAPI.io 第三方服务获取推文:

1. 访问 https://twitterapi.io 注册账号
2. 获取 API Key
3. 配置 `TWITTERAPI_IO_KEY` 到 `.env` 文件

### 邮箱 SMTP

以 139 邮箱为例:
1. 登录 139 邮箱
2. 设置 → POP3/SMTP 服务 → 开启
3. 获取授权码
4. 配置到 `.env` 文件

## 许可证

MIT License
