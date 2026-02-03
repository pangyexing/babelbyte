# BabelByte AI Processing 深度分析

## 概述

本文档详细分析 `bb daily --skip-fetch` 中 Step 3/5 AI Processing 的预过滤和 Model Tier 选择逻辑。

---

## Phase 1: 预过滤 (Zero Token Cost)

### 1.1 `should_skip_ai_processing()` - 垃圾内容跳过

**文件**: `src/processors/rule_classifier.py:365-503`

#### 配置
- **启用开关**: `SKIP_ENABLED` (默认: true)

#### 跳过条件优先级

| 优先级 | 条件 | 触发规则 | 说明 |
|--------|------|----------|------|
| 1 | 空内容 | `len(content) < 1` | 直接跳过 |
| 2 | 超短内容 | `len(content) < 100 AND len(title) < 50` | 同时满足 |
| 3 | Twitter短推 | `source=twitter AND len < 200` | 有高价值信号例外 |
| 4 | Reddit链接帖 | `len < 100 AND "[link]" in content` | 纯链接无价值 |
| 5 | 转推 | `title.startswith("rt @") AND len < 500` | 平均重要性仅4.8 |
| 6 | 艺术帖 | 匹配艺术模式 | 与技术无关 |
| 7 | 招聘帖 | 匹配10+招聘模式 | 不属于内容范围 |
| 8 | 垃圾/推广 | 匹配4类垃圾模式 | 低质量 |
| 9 | 跨发帖 | "cross-posted"/"xpost" | 重复内容 |
| 10 | 机器人 | 3种bot标记 | 自动生成 |
| 11 | 社交请求 | "follow me" + `len < 200` | 互动请求 |
| 12 | 链接聚合 | `url_count >= 5 AND len < 500` | 链接列表 |
| 13 | 重复词 | `unique_ratio < 0.3 AND words >= 20` | 低信息密度 |

#### Twitter 高价值信号例外
即使 < 200 字符，包含以下信号仍处理：
```python
high_value_signals = [
    r"\b(announce|launch|release|发布|上线)\b",
    r"\b(GPT|Claude|Gemini|LLM|AI model)\b",
    r"\$\d+[BMK]?\b",  # 融资金额
    r"\b(breaking|重大|突破)\b",
]
```

---

### 1.2 `try_rule_only_processing()` - 规则分类已知域名

**文件**: `src/processors/rule_classifier.py:600-856`

#### 配置
| 配置项 | 环境变量 | 默认值 |
|--------|----------|--------|
| 启用 | `RULE_ONLY_ENABLED` | true |
| 最大内容长度 | `RULE_ONLY_MAX_CONTENT` | 1000 |
| 最小关键字boost | `RULE_ONLY_MIN_BOOST` | 1 |

#### 处理流程
```
ContentItem
  ↓
[1] 检查启用 → 否 → return None
  ↓
[2] 检查字段 (url, title) → 缺失 → return None
  ↓
[3] 内容长度 > 1000 → return None (需要AI深度分析)
  ↓
[4] 提取域名 → 失败 → return None
  ↓
[5] 域名在 DOMAINS_REQUIRE_AI? → 是 → return None
  ↓
[6] 域名在 HIGH_CONFIDENCE_DOMAINS? → 否 → return None
  ↓
[7] 验证关键字信号 → 无信号 → return None
  ↓
[8] keyword_boost >= 1? → 否 → return None
  ↓
[9] 调整重要性: min(10, base + boost)
  ↓
[10] 生成规则摘要、one_liner、key_points
  ↓
return ProcessingResult (完整,无需AI)
```

#### 高信度域名示例 (HIGH_CONFIDENCE_DOMAINS)
```python
# AI公司 (importance=7-8)
openai.com, anthropic.com, deepmind.com, huggingface.co

# 科学期刊 (importance=7-8)
arxiv.org, nature.com, science.org

# 编程/框架 (importance=6-7)
github.com, pytorch.org, rust-lang.org

# 科技媒体 (importance=5-6)
techcrunch.com, arstechnica.com

# 中文来源 (importance=5-7)
jiqizhixin.com(AI,7), 36kr.com(创业,6)
```

#### 禁止规则处理的域名 (DOMAINS_REQUIRE_AI)
```python
{"news.ycombinator.com", "hnrss.org"}  # HN内容多样化
```

---

### 1.3 `find_similar_processed_item()` - Embedding语义去重

**文件**: `src/optimization/dedup_optimizer.py:228-323`

#### 配置
| 配置项 | 默认值 | 说明 |
|--------|--------|------|
| `EMBEDDING_ENABLED` | true | 启用去重 |
| 相似度阈值 | 0.85 | 85%以上复用 |
| 搜索窗口 | 7天 | 仅搜索近期 |
| 搜索上限 | 500项 | 避免性能问题 |

#### 处理流程
```
ContentItem (未处理)
  ↓
[1] EMBEDDING_ENABLED? → false → return None
  ↓
[2] 获取当前项embedding → 不存在 → return None
  ↓
[3] 查询近7天已处理项 (有embedding, limit=500)
  ↓
[4] 遍历计算cosine_similarity
  ↓
[5] 标准化: (similarity + 1) / 2  # [-1,1] → [0,1]
  ↓
[6] 最高分 >= 0.85? → 是 → return SimilarItemResult
  ↓
return None
```

#### 相似度计算
```python
cosine = dot(vec_a, vec_b) / (norm_a * norm_b)  # [-1, 1]
normalized = (cosine + 1) / 2                    # [0, 1]

# 示例:
# cosine=0.7 → normalized=0.85 ✓ 通过
# cosine=0.5 → normalized=0.75 ✗ 不通过
```

---

## Phase 2: AI处理

> **重要**: Two-stage 和 Model Tier 是**互斥**的：
> - 启用 Two-stage (Ollama) → 跳过 Model Tier，直接进入 two-stage 流程
> - 未启用 Two-stage → 使用 Model Tier 选择 Heavy/Light 模型
>
> 见 `digest_processor.py:403-405`:
> ```python
> if settings.ai.get_provider() == "ollama" and settings.ollama.two_stage_enabled:
>     return self._process_items_two_stage(items, progress_callback)  # 直接返回
> ```

### Model Tier选择 (仅非 Two-stage 模式)

### 2.1 `estimate_importance()` - 重要性估计

**文件**: `src/processors/rule_classifier.py:299-362`

#### 返回结构
```python
@dataclass
class ImportanceEstimate:
    score: int        # 1-10
    confidence: float # 0-1
    reason: str       # "domain:xxx", "weak_keyword:xxx", etc.
```

#### 四层估计逻辑

| 层级 | 优先级 | 判断类型 | 置信度 | 说明 |
|------|--------|----------|--------|------|
| 1 | 最高 | 域名匹配 | 0.85 | 直接返回 |
| 2 | 高 | 强关键词模式 | 0.70-0.80 | 直接返回 |
| 3 | 中 | 弱关键词模式 | 0.45-0.55 | 累积boost |
| 4 | 低 | 来源质量 | 0.50-0.60 | 兜底 |

#### 模式示例
```python
# 强模式 (立即返回)
high_value_patterns = [
    (r'(OpenAI|Anthropic)\s+(announces?|launches?)', boost=3, conf=0.8),
    (r'\$\d+[BM]\b', boost=2, conf=0.75),  # 融资
]

# 弱模式 (累积)
weak_patterns = [
    (r'\b(GPT|Claude|Gemini|LLM)\b', boost=1, conf=0.55),
    (r'\b(launch|announce|release)\b', boost=1, conf=0.45),
]

# 来源置信度 (兜底)
source_confidence = {
    "reddit": 0.5,
    "twitter": 0.5,
    "hackernews": 0.6,
}
```

---

### 2.2 Model Tier 配置

**文件**: `config/settings.py:204-291`

| 环境变量 | 默认值 | 说明 |
|----------|--------|------|
| `MODEL_TIER_ENABLED` | true | 启用分层 |
| `MODEL_TIER_HEAVY_THRESHOLD` | 7 | 分数>=7用重型 |
| `MODEL_TIER_CONFIDENCE_CUTOFF` | 0.5 | 置信度<0.5用重型 |
| `CLAUDE_MODEL_HEAVY` | sonnet | 重型模型 |
| `CLAUDE_MODEL_LIGHT` | haiku | 轻型模型 |
| `QUALITY_UNKNOWN_DOMAIN_USE_HEAVY` | true | 未知域名用重型 |
| `QUALITY_REPROCESS_HIGH_SCORE` | true | 轻型高分重新处理 |
| `QUALITY_REPROCESS_THRESHOLD` | 7 | 重新处理阈值 |

---

### 2.3 Heavy/Light 模型选择决策

**文件**: `src/processors/digest_processor.py:412-431`

#### 决策树
```
estimate_importance(item) → (score, confidence, reason)
  │
  ├─ IF score >= 7
  │  └─> HEAVY MODEL ✓ (高价值必须用强模型)
  │
  ├─ IF confidence < 0.5
  │  └─> HEAVY MODEL ✓ (低置信保守策略)
  │
  ├─ IF reason == "unknown" AND unknown_domain_use_heavy
  │  └─> HEAVY MODEL ✓ (未知域名保护质量)
  │
  └─ ELSE
     └─> LIGHT MODEL (低分数+高置信)
```

#### 场景示例
| 场景 | score | confidence | reason | 模型选择 |
|------|-------|------------|--------|----------|
| 高分 | 8 | 0.3 | domain:openai.com | HEAVY ✓ |
| 中分低置信 | 6 | 0.3 | weak_keyword | HEAVY ✓ |
| 中分高置信 | 5 | 0.85 | domain:github.com | LIGHT |
| 未知域名 | 5 | 0.3 | unknown | HEAVY ✓ |
| HN来源 | 5 | 0.6 | source:hackernews | LIGHT |

---

### 2.4 Quality Guard - 重新处理机制

#### 触发条件
- `QUALITY_REPROCESS_HIGH_SCORE=true`
- 轻型模型返回 `importance_score >= 7`

#### 工作流
```
[1] 处理 Light Items (LIGHT MODEL)
  ↓
[2] 检查每个结果的 importance_score
  ↓
[3] IF score >= 7 → 添加到 reprocess_items
  ↓
[4] 合并: heavy_items.extend(reprocess_items)
  ↓
[5] 处理所有 Heavy Items (HEAVY MODEL)
```

**逻辑**: 轻型模型本应给出低分，如果意外返回高分，用重型模型重新确认。

---

## 完整处理流水线

```
ContentItem (未处理)
    │
    ├─[FILTER 1] should_skip_ai_processing()
    │  └─ 垃圾/招聘/跨发 → 跳过 (importance=1)
    │
    ├─[FILTER 2] try_rule_only_processing()
    │  └─ 高信度域名+关键字 → 返回结果 (无AI)
    │
    ├─[DEDUP] find_similar_processed_item()
    │  └─ 相似度>=0.85 → 复用结果 (无AI)
    │
    └─[AI PROCESS] 需要AI处理
       │
       ├─ estimate_importance() → (score, confidence)
       │
       ├─ 模型选择:
       │  ├─ score>=7 OR confidence<0.5 → HEAVY
       │  └─ 其他 → LIGHT
       │
       └─ Quality Guard:
          └─ LIGHT返回score>=7 → 用HEAVY重新处理
```

---

## Token 优化效果

| 阶段 | Token消耗 | 预期节省 |
|------|-----------|----------|
| should_skip | 0 | 10-15% |
| try_rule_only | 0 | 15-25% |
| find_similar | 0 | 10-20% |
| Model Tier | 减少 | 20-30% |
| **总计** | - | **55-70%** |

---

## 配置建议

### 追求质量 (保守)
```bash
MODEL_TIER_HEAVY_THRESHOLD=6
MODEL_TIER_CONFIDENCE_CUTOFF=0.6
QUALITY_UNKNOWN_DOMAIN_USE_HEAVY=true
```

### 平衡质量和成本 (推荐)
```bash
MODEL_TIER_HEAVY_THRESHOLD=7
MODEL_TIER_CONFIDENCE_CUTOFF=0.5
QUALITY_UNKNOWN_DOMAIN_USE_HEAVY=true
```

### 追求成本优化 (激进)
```bash
MODEL_TIER_HEAVY_THRESHOLD=8
MODEL_TIER_CONFIDENCE_CUTOFF=0.4
QUALITY_UNKNOWN_DOMAIN_USE_HEAVY=false
```
