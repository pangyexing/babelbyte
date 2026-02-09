"""Bulletin generator for daily news video briefings.

Generates unified news bulletins from event clusters using AI to create
cohesive, broadcast-ready content.
"""

import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

from config.settings import get_settings
from src.analytics.token_tracker import AICallType, record_ai_call
from src.storage.models import ContentItem, EventCluster

logger = logging.getLogger(__name__)


@dataclass
class DouyinContent:
    """Content for a single Douyin short video (15-30s)."""

    hook: str  # 3-second hook (8-15 chars)
    headline: str  # Title (8-15 chars)
    summary: str  # Core fact (40-50 chars)
    impact: str  # Impact analysis (20-35 chars)
    action: str  # Actionable advice (15-25 chars)
    cta: str  # Engagement CTA (10-15 chars)
    image_prompt: str  # FLUX illustration prompt (English)
    segment_scripts: list[str]  # 5-segment TTS scripts
    hashtags: list[str] = field(default_factory=list)  # 5 topic hashtags


@dataclass
class BulletinItem:
    """A single item in the news bulletin."""

    cluster: EventCluster
    members: list[ContentItem]
    headline: str  # AI-generated short headline (20 chars max)
    summary: str  # AI-generated summary (50 chars max)
    one_liner: str = ""  # One sentence conclusion (30 chars max)
    impact: str = ""  # Short/long term impact combined (40 chars max)
    actions: list[str] = field(default_factory=list)  # Actionable items (max 2, 25 chars each)
    order: int = 0  # Display order in bulletin
    image_prompt: str = ""  # LLM-generated English prompt for FLUX illustration


@dataclass
class BulletinResult:
    """Result of bulletin generation."""

    items: list[BulletinItem]
    date: datetime
    total_events: int
    script: str = ""  # Combined TTS script for entire bulletin
    segment_scripts: list[str] = field(default_factory=list)  # Per-slide scripts
    success: bool = True
    error: Optional[str] = None


class BulletinGenerator:
    """Generate news bulletins from event clusters.

    Uses AI to organize event clusters into a cohesive news briefing format
    with headlines, summaries, and key facts for each event.
    """

    # Prompt for bulletin generation - returns JSON array
    BULLETIN_PROMPT = """你是一位马三立风格的科技新闻主播——表面一本正经，骨子里全是冷幽默。
将以下{count}个热点事件整理成简洁的快报。

事件列表：
{events_json}

要求：
1. 为每个事件生成（必须使用中文）：
   - headline: 中文吸睛标题（8-15个中文字，可以带点戏谑但不失信息量）
   - summary: 核心摘要（40-50字，正经讲事实，但措辞可以有巧思）
   - one_liner: 一句话总结（15-25字，像马三立抖包袱一样点睛）
   - impact: 影响分析（20-35字，如"短期利好XX，长期推动XX"）
   - actions: 具体行动建议（最多2条，每条15-25字，必须是完整句子）
2. 创新/创业类排最前面，其余按报道数量和重要性排序（返回的顺序就是播报顺序）
3. 使用口语化表达，适合朗读，像跟人聊天一样自然
4. 突出数字、时间、人物等关键信息
5. actions必须具体可操作，如"下载体验ACE-Step 1.5音乐生成"，不要写"关注XX的"这种半截话
6. 【重要】每个字段都必须包含明确主语（公司名/产品名/人物名），禁止省略主体
   - 正确：headline="OpenAI发布GPT-5" summary="OpenAI正式发布GPT-5，性能提升3倍"
   - 错误：headline="发布新模型" summary="正式发布新一代大模型，性能大幅提升"

返回JSON数组格式：
[
  {{"id": 事件ID, "headline": "标题", "summary": "摘要",
    "one_liner": "一句话", "impact": "影响", "actions": ["建议1"]}},
  ...
]

仅返回JSON，无其他文字："""

    # Prompt for deduplication - merges fields into concise broadcast scripts
    DEDUP_SCRIPT_PROMPT = """将以下新闻事件的各字段整合成一段简洁的口播文本，去除重复信息。
风格：马三立式冷幽默——看似正经地陈述事实，但用词和节奏本身就带喜感。

事件列表：
{events_json}

要求：
1. 每个事件生成一段30-50字的播报文本
2. 去除重复信息（如标题和摘要说的是同一件事，只保留一次）
3. 保留关键数字、人名、公司名
4. 语气像跟人唠家常，一本正经但不动声色地藏着幽默
5. 每句话必须包含明确主语，不要省略"谁做了什么"中的"谁"
6. 返回JSON数组：[{{"id": 事件ID, "script": "播报文本"}}]

仅返回JSON："""

    # Prompt for script generation - combines all bulletins into one script
    SCRIPT_PROMPT = """将以下新闻快报内容整合成一段完整的口播脚本（60-90秒朗读时长）。
你是巴别情报站的主播，风格参考马三立——表面一本正经，实际句句藏着冷幽默，
不动声色地抖包袱，让人听完才反应过来哪里好笑。

快报内容：
{bulletins}

要求：
1. 开头用一句话点明主题，带一点马三立式的闲聊感（如：各位好，巴别情报站，今天科技圈又热闹了，跟您说几件事）
2. 每条新闻之间用自然的聊天式过渡（如：话说回来/您猜怎么着/这边还没完呢）
3. 结尾轻松收束（如：今天就这些，巴别情报站，回见了您呐）
4. 核心是"正经地说不正经的话"——事实准确，但措辞、节奏、用词让人会心一笑
5. 不要使用emoji
6. 总字数300-400字
7. 每句话必须包含明确主语（公司名/产品名/人物名），不要省略主体
8. 幽默要克制，冷幽默为主，绝不尬笑不硬搞笑，宁可正经也不要刻意搞笑

直接返回脚本文本："""

    # Prompt for Douyin single-event content generation
    DOUYIN_CONTENT_PROMPT = """你是抖音科技号运营专家。为以下单条新闻事件生成15-30秒短视频的全部内容。

事件信息：
{event_json}

要求：
1. hook: 3秒钩子文案（8-15个中文字），使用疑问/冲突/震惊/悬念角度，直入主题
2. headline: 标题（8-15个中文字）
3. summary: 核心事实（40-50字，正经讲事实）
4. impact: 影响分析（20-35字，如"短期利好XX，长期推动XX"）
5. action: 行动建议（15-25字，具体可操作的完整句子）
6. cta: 评论引导问题（10-15字），让观众想在评论区表达看法
7. image_prompt: 英文配图描述（30-50单词），用于AI图像生成，漫画插画风格
8. segment_scripts: 5段TTS脚本数组，分别对应：
   - [0] hook播报（8-15字，直接抛出悬念/问题）
   - [1] 核心事实播报（40-50字）
   - [2] 影响分析播报（20-35字）
   - [3] 行动建议播报（15-25字）
   - [4] CTA引导（15-20字，如"你觉得呢？评论区聊聊"）
9. hashtags: 5个相关话题标签（混合热门+垂直，不带#号）

【重要】每个字段都必须包含明确主语（公司名/产品名/人物名），禁止省略主体。
image_prompt中不要包含任何文字、字母、数字。

返回JSON格式：
{{"hook": "...", "headline": "...", "summary": "...", "impact": "...",
  "action": "...", "cta": "...", "image_prompt": "...",
  "segment_scripts": ["...", "...", "...", "...", "..."],
  "hashtags": ["...", "...", "...", "...", "..."]}}

仅返回JSON，无其他文字："""

    # Prompt for generating English image prompts for FLUX illustration
    IMAGE_PROMPT_PROMPT = (
        "为以下{count}个新闻事件各生成一段英文配图描述(image prompt)，"
        "用于AI图像生成模型生成漫画风格配图。\n\n"
        "事件列表：\n{events_json}\n\n"
        "要求：\n"
        "1. 每个描述30-50个英文单词\n"
        "2. 漫画插画风格：bold ink outlines, flat vibrant colors, "
        "pop art, clean composition\n"
        "3. 描述具体的视觉元素和场景，必须与新闻内容直接相关\n"
        "4. 不要在画面中包含任何文字、字母、数字\n"
        "5. 每个事件的配图必须有明显差异，不要重复使用相同的视觉元素\n"
        "6. 指定与分类匹配的色调（AI=蓝青色, 科技=蓝银色, 金融=金琥珀色, "
        "创业=青绿色, 创新=绿青色, 技术=紫蓝色）\n\n"
        '返回JSON数组：[{{"id": 事件ID, "prompt": "英文描述"}}]\n\n'
        "仅返回JSON，无其他文字："
    )

    def __init__(self, processor=None):
        """Initialize BulletinGenerator.

        Args:
            processor: Optional AI processor instance. If None, created on demand.
        """
        self._processor = processor
        self._settings = None

    @property
    def settings(self):
        """Lazy load settings."""
        if self._settings is None:
            self._settings = get_settings()
        return self._settings

    def _get_processor(self):
        """Get or create AI processor."""
        if self._processor is None:
            from src.processors.digest_processor import get_ai_processor

            self._processor = get_ai_processor()
        return self._processor

    @staticmethod
    def _extract_headline(cluster: EventCluster, rep: ContentItem = None) -> str:
        """Extract a meaningful Chinese headline from cluster/member data.

        Tries one_liner first, then summary, then event_title as last resort.
        Truncates at sentence boundaries when possible.
        """
        # Prefer one_liner (usually a concise Chinese conclusion)
        if rep and rep.one_liner:
            text = rep.one_liner.strip()
            if len(text) <= 20:
                return text
            # Truncate at punctuation boundary
            for sep in ("，", "。", "；", "、", ","):
                idx = text.find(sep, 8)
                if 8 <= idx <= 20:
                    return text[:idx]
            return text[:20]

        # Fall back to summary first sentence
        if rep and rep.summary:
            text = rep.summary.strip()
            for sep in ("。", "，", "；"):
                idx = text.find(sep)
                if 0 < idx <= 20:
                    return text[:idx]
            if len(text) <= 20:
                return text
            return text[:20]

        # Last resort: event_title (may be English)
        title = cluster.event_title or ""
        if len(title) <= 20:
            return title
        # Try to break at word boundary for English
        truncated = title[:20]
        last_space = truncated.rfind(" ")
        if last_space > 10:
            return truncated[:last_space]
        return truncated

    def _format_events_for_prompt(
        self,
        clusters: list[EventCluster],
        members_dict: dict[int, list[ContentItem]],
    ) -> str:
        """Format event clusters as JSON for the AI prompt.

        Args:
            clusters: List of event clusters.
            members_dict: Dict mapping cluster_id to list of members.

        Returns:
            JSON string of events.
        """
        events = []
        for cluster in clusters:
            members = members_dict.get(cluster.id, [])
            if not members:
                continue

            # Get representative member (highest importance)
            rep = max(members, key=lambda m: m.importance_score or 0)

            event = {
                "id": cluster.id,
                "title": cluster.event_title,
                "category": cluster.category,
                "article_count": cluster.article_count,
                "summary": rep.summary[:300] if rep.summary else "",
                "one_liner": rep.one_liner[:100] if rep.one_liner else "",
                "importance": rep.importance_score or 5,
            }

            # Add key points if available
            if rep.key_points:
                try:
                    kp_data = json.loads(rep.key_points)
                    event["key_points"] = [
                        kp.get("value", "")[:60] for kp in kp_data[:3] if kp.get("value")
                    ]
                except json.JSONDecodeError:
                    pass

            # Add impact assessment for better context
            if rep.impact_assessment:
                try:
                    impact_data = json.loads(rep.impact_assessment)
                    event["impact_info"] = {
                        "short_term": impact_data.get("short_term", "")[:80],
                        "long_term": impact_data.get("long_term", "")[:80],
                    }
                except json.JSONDecodeError:
                    pass

            # Add actionable items for better suggestions
            if rep.actionable_items:
                try:
                    action_data = json.loads(rep.actionable_items)
                    event["original_actions"] = [
                        a.get("description", "")[:60]
                        for a in action_data[:3]
                        if a.get("description")
                    ]
                except json.JSONDecodeError:
                    pass

            events.append(event)

        return json.dumps(events, ensure_ascii=False, indent=2)

    def generate_bulletin(
        self,
        clusters: list[EventCluster],
        members_dict: dict[int, list[ContentItem]],
    ) -> BulletinResult:
        """Generate a news bulletin from event clusters.

        Args:
            clusters: List of today's event clusters.
            members_dict: Dict mapping cluster_id to list of member items.

        Returns:
            BulletinResult with bulletin items and combined script.
        """
        if not clusters:
            return BulletinResult(
                items=[],
                date=datetime.now(),
                total_events=0,
                success=False,
                error="No event clusters provided",
            )

        # Build prompt
        events_json = self._format_events_for_prompt(clusters, members_dict)
        prompt = self.BULLETIN_PROMPT.format(
            count=len(clusters),
            events_json=events_json,
        )

        # Call AI
        processor = self._get_processor()
        start_time = time.time()

        if hasattr(processor, "_call_api"):
            success, response, error = processor._call_api(
                prompt,
                model=processor.model,
                max_tokens=1024,
                disable_thinking=True,
            )
            duration_ms = int((time.time() - start_time) * 1000)

            record_ai_call(
                call_type=AICallType.DIGEST_GENERATE,
                cached=False,
                input_chars=len(prompt),
                output_chars=len(response) if success else 0,
                duration_ms=duration_ms,
                success=success,
                error=error[:100] if error else None,
            )

            if not success:
                logger.warning(f"Bulletin generation failed: {error}")
                return self._fallback_bulletin(clusters, members_dict, error)

            # Parse response
            items = self._parse_bulletin_response(response, clusters, members_dict)
        else:
            # Fallback for non-API processors
            items = self._create_fallback_items(clusters, members_dict)

        if not items:
            return self._fallback_bulletin(clusters, members_dict, "Failed to parse AI response")

        # Generate combined script and segment scripts
        script = self._generate_combined_script(items)
        segment_scripts = self._generate_segment_scripts(items)

        # Generate LLM image prompts (Ollama still on GPU, before FLUX phase)
        self._generate_image_prompts(items)

        return BulletinResult(
            items=items,
            date=datetime.now(),
            total_events=len(items),
            script=script,
            segment_scripts=segment_scripts,
            success=True,
        )

    def _parse_bulletin_response(
        self,
        response: str,
        clusters: list[EventCluster],
        members_dict: dict[int, list[ContentItem]],
    ) -> list[BulletinItem]:
        """Parse AI response into BulletinItems.

        Args:
            response: AI response JSON string.
            clusters: Original clusters for reference.
            members_dict: Cluster members for reference.

        Returns:
            List of BulletinItems.
        """
        try:
            # Clean response
            response = response.strip()
            if response.startswith("```"):
                lines = response.split("\n")
                response = "\n".join(line for line in lines if not line.strip().startswith("```"))

            # Find JSON array
            json_start = response.find("[")
            json_end = response.rfind("]") + 1
            if json_start != -1 and json_end > json_start:
                response = response[json_start:json_end]

            data = json.loads(response)
            if not isinstance(data, list):
                return []

            # Build cluster lookup
            cluster_map = {c.id: c for c in clusters}

            items = []
            for i, item_data in enumerate(data):
                cluster_id = item_data.get("id")
                if cluster_id not in cluster_map:
                    continue

                cluster = cluster_map[cluster_id]
                members = members_dict.get(cluster_id, [])

                # Build fallback headline from representative member
                rep = max(members, key=lambda m: m.importance_score or 0) if members else None
                fallback_headline = self._extract_headline(cluster, rep)

                raw_actions = item_data.get("actions", [])[:2]
                actions = [a[:25] for a in raw_actions if isinstance(a, str) and a]

                items.append(
                    BulletinItem(
                        cluster=cluster,
                        members=members,
                        headline=item_data.get("headline") or fallback_headline,
                        summary=item_data.get("summary", "")[:60],
                        one_liner=item_data.get("one_liner", "")[:30],
                        impact=item_data.get("impact", "")[:40],
                        actions=actions,
                        order=i,
                    )
                )

            return items

        except (json.JSONDecodeError, TypeError, KeyError) as e:
            logger.warning(f"Failed to parse bulletin response: {e}")
            return []

    def _create_fallback_items(
        self,
        clusters: list[EventCluster],
        members_dict: dict[int, list[ContentItem]],
    ) -> list[BulletinItem]:
        """Create bulletin items without AI (rule-based fallback).

        Args:
            clusters: Event clusters to process.
            members_dict: Cluster members.

        Returns:
            List of BulletinItems.
        """
        items = []

        # Get max importance for each cluster from members
        cluster_importance = {}
        for cluster in clusters:
            members = members_dict.get(cluster.id, [])
            if members:
                max_importance = max(m.importance_score or 0 for m in members)
                cluster_importance[cluster.id] = max_importance
            else:
                cluster_importance[cluster.id] = 0

        # Sort: 创业/创新 first, then others by article_count and importance
        def sort_key(c):
            is_priority = c.category in ("创业", "创新")
            importance = cluster_importance.get(c.id, 0)
            return (0 if is_priority else 1, -c.article_count, -importance)

        sorted_clusters = sorted(clusters, key=sort_key)

        for i, cluster in enumerate(sorted_clusters):
            members = members_dict.get(cluster.id, [])
            if not members:
                continue

            # Get representative
            rep = max(members, key=lambda m: m.importance_score or 0)

            # Extract headline from one_liner or summary (not raw event_title)
            headline = self._extract_headline(cluster, rep)

            # Extract one_liner
            one_liner = rep.one_liner[:30] if rep.one_liner else ""

            # Extract impact from impact_assessment
            impact = ""
            if rep.impact_assessment:
                try:
                    impact_data = json.loads(rep.impact_assessment)
                    short_term = impact_data.get("short_term", "")
                    long_term = impact_data.get("long_term", "")
                    if short_term and long_term:
                        impact = f"短期{short_term[:15]}，长期{long_term[:15]}"
                    elif short_term:
                        impact = short_term[:40]
                    elif long_term:
                        impact = long_term[:40]
                except json.JSONDecodeError:
                    pass

            # Extract actions from actionable_items
            actions = []
            if rep.actionable_items:
                try:
                    action_data = json.loads(rep.actionable_items)
                    actions = [
                        a.get("description", "")[:25]
                        for a in action_data[:2]
                        if a.get("description")
                    ]
                except json.JSONDecodeError:
                    pass

            items.append(
                BulletinItem(
                    cluster=cluster,
                    members=members,
                    headline=headline,
                    summary=rep.summary[:60] if rep.summary else "",
                    one_liner=one_liner,
                    impact=impact,
                    actions=actions,
                    order=i,
                )
            )

        return items

    def _fallback_bulletin(
        self,
        clusters: list[EventCluster],
        members_dict: dict[int, list[ContentItem]],
        error: str,
    ) -> BulletinResult:
        """Create fallback bulletin when AI fails.

        Args:
            clusters: Event clusters.
            members_dict: Cluster members.
            error: Error message.

        Returns:
            BulletinResult with fallback items.
        """
        items = self._create_fallback_items(clusters, members_dict)
        script = self._generate_fallback_script(items)
        segment_scripts = self._generate_segment_scripts(items)

        return BulletinResult(
            items=items,
            date=datetime.now(),
            total_events=len(items),
            script=script,
            segment_scripts=segment_scripts,
            success=len(items) > 0,
            error=error if not items else None,
        )

    def _generate_combined_script(self, items: list[BulletinItem]) -> str:
        """Generate a combined TTS script for all bulletin items using AI.

        Args:
            items: List of bulletin items.

        Returns:
            Combined script text.
        """
        if not items:
            return ""

        # Format bulletins for prompt
        bulletins_text = []
        for i, item in enumerate(items, 1):
            text = f"{i}. {item.headline}\n摘要：{item.summary}"
            if item.one_liner:
                text += f"\n结论：{item.one_liner}"
            if item.impact:
                text += f"\n影响：{item.impact}"
            if item.actions:
                text += "\n建议：" + "；".join(item.actions)
            bulletins_text.append(text)

        prompt = self.SCRIPT_PROMPT.format(bulletins="\n\n".join(bulletins_text))

        # Call AI
        processor = self._get_processor()
        start_time = time.time()

        if hasattr(processor, "_call_api"):
            success, response, error = processor._call_api(
                prompt,
                model=processor.model,
                max_tokens=512,
                disable_thinking=True,
            )
            duration_ms = int((time.time() - start_time) * 1000)

            record_ai_call(
                call_type=AICallType.DIGEST_GENERATE,
                cached=False,
                input_chars=len(prompt),
                output_chars=len(response) if success else 0,
                duration_ms=duration_ms,
                success=success,
                error=error[:100] if error else None,
            )

            if success:
                # Clean up response
                script = response.strip()
                if script.startswith("```"):
                    lines = script.split("\n")
                    script = "\n".join(line for line in lines if not line.strip().startswith("```"))
                return script.strip()

        # Fallback to simple script
        return self._generate_fallback_script(items)

    def _generate_fallback_script(self, items: list[BulletinItem]) -> str:
        """Generate a simple script without AI.

        Args:
            items: List of bulletin items.

        Returns:
            Simple script text.
        """
        if not items:
            return "暂无巴别情报站。"

        parts = [f"欢迎收看巴别情报站，今天有{len(items)}件动态需要关注。"]

        transitions = ["首先，", "另外，", "此外，", "值得关注的是，", "最后，"]

        for i, item in enumerate(items):
            transition = transitions[min(i, len(transitions) - 1)]
            parts.append(f"{transition}{item.headline}。{item.summary}")

            # Add one-liner if available
            if item.one_liner:
                parts.append(item.one_liner)

            # Add impact if available
            if item.impact:
                parts.append(f"影响：{item.impact}")

        parts.append("感谢收看巴别情报站，保持关注，保持思考。")

        return "。".join(parts)

    def _generate_deduped_scripts(self, items: list[BulletinItem]) -> dict[int, str]:
        """Use AI to generate deduplicated broadcast scripts.

        Combines headline, summary, one_liner, impact, and actions into a single
        concise script that avoids repeating the same information.

        Args:
            items: List of bulletin items.

        Returns:
            Dict mapping cluster_id to deduplicated script text.
        """
        if not items:
            return {}

        # Prepare events data for the prompt
        events = []
        for item in items:
            events.append(
                {
                    "id": item.cluster.id,
                    "headline": item.headline,
                    "summary": item.summary,
                    "one_liner": item.one_liner,
                    "impact": item.impact,
                    "actions": item.actions,
                }
            )

        prompt = self.DEDUP_SCRIPT_PROMPT.format(
            events_json=json.dumps(events, ensure_ascii=False, indent=2)
        )

        processor = self._get_processor()
        start_time = time.time()

        if hasattr(processor, "_call_api"):
            success, response, error = processor._call_api(
                prompt,
                model=processor.model,
                max_tokens=1024,
                disable_thinking=False,  # Enable thinking for better deduplication
            )
            duration_ms = int((time.time() - start_time) * 1000)

            record_ai_call(
                call_type=AICallType.DIGEST_GENERATE,
                cached=False,
                input_chars=len(prompt),
                output_chars=len(response) if success else 0,
                duration_ms=duration_ms,
                success=success,
                error=error[:100] if error else None,
            )

            if success:
                try:
                    # Clean response
                    response = response.strip()
                    if response.startswith("```"):
                        lines = response.split("\n")
                        response = "\n".join(
                            line for line in lines if not line.strip().startswith("```")
                        )

                    # Find JSON array
                    json_start = response.find("[")
                    json_end = response.rfind("]") + 1
                    if json_start != -1 and json_end > json_start:
                        response = response[json_start:json_end]

                    data = json.loads(response)
                    if isinstance(data, list):
                        return {
                            item.get("id"): item.get("script", "")
                            for item in data
                            if item.get("id") and item.get("script")
                        }
                except (json.JSONDecodeError, TypeError, KeyError) as e:
                    logger.warning(f"Failed to parse dedup response: {e}")

        return {}

    def _generate_segment_scripts(self, items: list[BulletinItem]) -> list[str]:
        """Generate per-slide scripts for precise audio sync.

        Generates scripts matching the slide structure:
        1. Opening slide
        2. N event cards (using AI-deduplicated scripts)
        3. Summary slide (if N > 1)
        4. Closing slide

        Args:
            items: List of bulletin items.

        Returns:
            List of script segments, one per slide.
        """
        if not items:
            return ["暂无巴别情报站。", "感谢收看。"]

        # Generate deduplicated scripts using AI
        deduped = self._generate_deduped_scripts(items)

        segments = []
        n_items = len(items)

        # 1. Opening slide script
        opening = f"欢迎收看巴别情报站，今天有{n_items}件动态需要关注。"
        segments.append(opening)

        # 2. Event card scripts - use deduped scripts if available
        transitions = ["首先，", "接下来，", "另外，", "此外，", "最后，"]
        for i, item in enumerate(items):
            transition = transitions[min(i, len(transitions) - 1)]
            script = deduped.get(item.cluster.id)

            if script:
                # Use AI-generated deduplicated script
                segments.append(f"{transition}{script}")
            else:
                # Fallback: use summary only to avoid repetition
                segments.append(f"{transition}{item.summary}")

        # 3. Summary slide script (only if more than 1 item)
        if n_items > 1:
            headlines = "、".join(item.headline for item in items[:3])
            if n_items > 3:
                headlines += f"等{n_items}件大事"
            summary = f"以上就是今日要点回顾：{headlines}。"
            segments.append(summary)

        # 4. Closing slide script
        closing = "感谢收看巴别情报站，保持关注，保持思考。"
        segments.append(closing)

        return segments

    def _generate_image_prompts(self, items: list[BulletinItem]) -> None:
        """Generate English image prompts for all bulletin items using LLM.

        Writes prompts directly into each item's image_prompt field.
        Fails silently — items keep empty image_prompt (template fallback).

        Args:
            items: List of bulletin items to generate prompts for.
        """
        if not items:
            return

        # Build events JSON for the prompt
        events = []
        for item in items:
            events.append(
                {
                    "id": item.cluster.id,
                    "headline": item.headline,
                    "summary": item.summary,
                    "category": item.cluster.category or "资讯",
                }
            )

        prompt = self.IMAGE_PROMPT_PROMPT.format(
            count=len(items),
            events_json=json.dumps(events, ensure_ascii=False, indent=2),
        )

        processor = self._get_processor()
        start_time = time.time()

        if not hasattr(processor, "_call_api"):
            logger.debug("Processor has no _call_api, skipping image prompts")
            return

        success, response, error = processor._call_api(
            prompt,
            model=processor.model,
            max_tokens=1024,
            disable_thinking=True,
        )
        duration_ms = int((time.time() - start_time) * 1000)

        record_ai_call(
            call_type=AICallType.DIGEST_GENERATE,
            cached=False,
            input_chars=len(prompt),
            output_chars=len(response) if success else 0,
            duration_ms=duration_ms,
            success=success,
            error=error[:100] if error else None,
        )

        if not success:
            logger.warning(f"Image prompt generation failed: {error}")
            return

        try:
            # Clean response
            response = response.strip()
            if response.startswith("```"):
                lines = response.split("\n")
                response = "\n".join(line for line in lines if not line.strip().startswith("```"))

            # Find JSON array
            json_start = response.find("[")
            json_end = response.rfind("]") + 1
            if json_start == -1 or json_end <= json_start:
                logger.warning("No JSON array found in image prompt response")
                return

            data = json.loads(response[json_start:json_end])
            if not isinstance(data, list):
                return

            # Build lookup: cluster_id -> prompt
            prompt_map = {
                entry.get("id"): entry.get("prompt", "")
                for entry in data
                if entry.get("id") and entry.get("prompt")
            }

            # Write prompts into items
            for item in items:
                prompt_text = prompt_map.get(item.cluster.id, "")
                if prompt_text:
                    item.image_prompt = prompt_text
                    logger.debug(
                        f"Image prompt for cluster {item.cluster.id}: " f"{prompt_text[:60]}..."
                    )

            logger.info(f"Generated {len(prompt_map)} image prompts " f"for {len(items)} items")

        except (json.JSONDecodeError, TypeError, KeyError) as e:
            logger.warning(f"Failed to parse image prompt response: {e}")

    def generate_douyin_content(
        self,
        cluster: "EventCluster",
        members: list["ContentItem"],
    ) -> Optional[DouyinContent]:
        """Generate Douyin short video content for a single event.

        Produces hook, headline, summary, impact, action, CTA, image prompt,
        5-segment TTS scripts, and hashtags in one Ollama call.

        Args:
            cluster: Event cluster.
            members: Member content items.

        Returns:
            DouyinContent or None on failure.
        """
        if not members:
            return None

        rep = max(members, key=lambda m: m.importance_score or 0)

        event = {
            "title": cluster.event_title,
            "category": cluster.category or "资讯",
            "article_count": cluster.article_count,
            "summary": rep.summary[:400] if rep.summary else "",
            "one_liner": rep.one_liner[:100] if rep.one_liner else "",
            "importance": rep.importance_score or 5,
        }

        if rep.key_points:
            try:
                kp_data = json.loads(rep.key_points)
                event["key_points"] = [
                    kp.get("value", "")[:60]
                    for kp in kp_data[:3]
                    if kp.get("value")
                ]
            except json.JSONDecodeError:
                pass

        if rep.impact_assessment:
            try:
                impact_data = json.loads(rep.impact_assessment)
                event["impact_info"] = {
                    "short_term": impact_data.get("short_term", "")[:80],
                    "long_term": impact_data.get("long_term", "")[:80],
                }
            except json.JSONDecodeError:
                pass

        prompt = self.DOUYIN_CONTENT_PROMPT.format(
            event_json=json.dumps(event, ensure_ascii=False, indent=2),
        )

        processor = self._get_processor()
        start_time = time.time()

        if not hasattr(processor, "_call_api"):
            logger.warning("Processor has no _call_api, using fallback")
            return self._fallback_douyin_content(cluster, rep)

        success, response, error = processor._call_api(
            prompt,
            model=processor.model,
            max_tokens=1024,
            disable_thinking=True,
        )
        duration_ms = int((time.time() - start_time) * 1000)

        record_ai_call(
            call_type=AICallType.DIGEST_GENERATE,
            cached=False,
            input_chars=len(prompt),
            output_chars=len(response) if success else 0,
            duration_ms=duration_ms,
            success=success,
            error=error[:100] if error else None,
        )

        if not success:
            logger.warning(f"Douyin content generation failed: {error}")
            return self._fallback_douyin_content(cluster, rep)

        try:
            response = response.strip()
            if response.startswith("```"):
                lines = response.split("\n")
                response = "\n".join(
                    line for line in lines if not line.strip().startswith("```")
                )

            json_start = response.find("{")
            json_end = response.rfind("}") + 1
            if json_start == -1 or json_end <= json_start:
                return self._fallback_douyin_content(cluster, rep)

            data = json.loads(response[json_start:json_end])

            scripts = data.get("segment_scripts", [])
            if len(scripts) < 5:
                # Pad with defaults
                defaults = [
                    data.get("hook", ""),
                    data.get("summary", ""),
                    data.get("impact", ""),
                    data.get("action", ""),
                    data.get("cta", "你觉得呢？评论区聊聊"),
                ]
                scripts = (scripts + defaults)[:5]

            return DouyinContent(
                hook=data.get("hook", "")[:15],
                headline=data.get("headline", "")[:15],
                summary=data.get("summary", "")[:60],
                impact=data.get("impact", "")[:40],
                action=data.get("action", "")[:30],
                cta=data.get("cta", "你觉得呢？")[:20],
                image_prompt=data.get("image_prompt", ""),
                segment_scripts=scripts,
                hashtags=data.get("hashtags", [])[:5],
            )

        except (json.JSONDecodeError, TypeError, KeyError) as e:
            logger.warning(f"Failed to parse douyin content response: {e}")
            return self._fallback_douyin_content(cluster, rep)

    def _fallback_douyin_content(
        self,
        cluster: "EventCluster",
        rep: "ContentItem",
    ) -> DouyinContent:
        """Create fallback Douyin content without AI."""
        headline = self._extract_headline(cluster, rep)
        summary = rep.summary[:60] if rep.summary else headline
        one_liner = rep.one_liner[:30] if rep.one_liner else ""

        impact = ""
        if rep.impact_assessment:
            try:
                impact_data = json.loads(rep.impact_assessment)
                short_term = impact_data.get("short_term", "")
                if short_term:
                    impact = short_term[:40]
            except json.JSONDecodeError:
                pass

        hook = headline[:15]
        cta = "你觉得呢？评论区聊聊"
        action = one_liner[:25] if one_liner else "关注后续发展"

        return DouyinContent(
            hook=hook,
            headline=headline,
            summary=summary,
            impact=impact or "值得持续关注",
            action=action,
            cta=cta,
            image_prompt="",
            segment_scripts=[
                hook,
                summary,
                impact or "值得持续关注",
                action,
                cta,
            ],
            hashtags=[cluster.category or "科技", "AI", "科技前沿", "每日科技", "巴别情报站"],
        )
