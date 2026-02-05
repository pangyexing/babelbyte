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
class BulletinItem:
    """A single item in the news bulletin."""

    cluster: EventCluster
    members: list[ContentItem]
    headline: str  # AI-generated short headline (10 chars max)
    summary: str  # AI-generated summary (50 chars max)
    one_liner: str = ""  # One sentence conclusion
    impact: str = ""  # Short/long term impact combined
    actions: list[str] = field(default_factory=list)  # Actionable items (max 2)
    order: int = 0  # Display order in bulletin


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
    BULLETIN_PROMPT = """你是新闻快报主播，将以下{count}个热点事件整理成简洁的快报。

事件列表：
{events_json}

要求：
1. 为每个事件生成：
   - headline: 10字以内的吸睛标题
   - summary: 50字内的核心摘要
   - one_liner: 一句话总结（20字以内）
   - impact: 影响分析（30字以内，如"短期利好XX，长期推动XX"）
   - actions: 行动建议（最多2条，每条15字以内）
2. 按重要性排序（返回的顺序就是播报顺序）
3. 使用口语化表达，适合朗读
4. 突出数字、时间、人物等关键信息

返回JSON数组格式：
[
  {{"id": 事件ID, "headline": "标题", "summary": "摘要",
    "one_liner": "一句话", "impact": "影响", "actions": ["建议1"]}},
  ...
]

仅返回JSON，无其他文字："""

    # Prompt for deduplication - merges fields into concise broadcast scripts
    DEDUP_SCRIPT_PROMPT = """将以下新闻事件的各字段整合成一段简洁的口播文本，去除重复信息。

事件列表：
{events_json}

要求：
1. 每个事件生成一段30-50字的播报文本
2. 去除重复信息（如标题和摘要说的是同一件事，只保留一次）
3. 保留关键数字、人名、公司名
4. 语气自然口语化
5. 返回JSON数组：[{{"id": 事件ID, "script": "播报文本"}}]

仅返回JSON："""

    # Prompt for script generation - combines all bulletins into one script
    SCRIPT_PROMPT = """将以下新闻快报内容整合成一段完整的口播脚本（60-90秒朗读时长）。

快报内容：
{bulletins}

要求：
1. 开头用一句话点明主题（如：欢迎收看巴别情报站，今天有N件大事）
2. 每条新闻之间用过渡语连接（如：另外/此外/值得关注的是）
3. 结尾有行动号召
4. 语气自然、口语化
5. 不要使用emoji
6. 总字数300-400字

直接返回脚本文本："""

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
                "summary": rep.summary[:150] if rep.summary else "",
                "one_liner": rep.one_liner[:50] if rep.one_liner else "",
                "importance": rep.importance_score or 5,
            }

            # Add key points if available
            if rep.key_points:
                try:
                    kp_data = json.loads(rep.key_points)
                    event["key_points"] = [
                        kp.get("value", "")[:30] for kp in kp_data[:3] if kp.get("value")
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

                items.append(
                    BulletinItem(
                        cluster=cluster,
                        members=members,
                        headline=item_data.get("headline", cluster.event_title[:10]),
                        summary=item_data.get("summary", "")[:50],
                        one_liner=item_data.get("one_liner", "")[:20],
                        impact=item_data.get("impact", "")[:30],
                        actions=item_data.get("actions", [])[:2],
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

        # Sort: 创业/创新 first, then others by importance and article_count
        def sort_key(c):
            is_priority = c.category in ("创业", "创新")
            importance = cluster_importance.get(c.id, 0)
            return (0 if is_priority else 1, -importance, -c.article_count)

        sorted_clusters = sorted(clusters, key=sort_key)

        for i, cluster in enumerate(sorted_clusters):
            members = members_dict.get(cluster.id, [])
            if not members:
                continue

            # Get representative
            rep = max(members, key=lambda m: m.importance_score or 0)

            # Extract one_liner
            one_liner = rep.one_liner[:20] if rep.one_liner else ""

            # Extract impact from impact_assessment
            impact = ""
            if rep.impact_assessment:
                try:
                    impact_data = json.loads(rep.impact_assessment)
                    short_term = impact_data.get("short_term", "")
                    if short_term:
                        impact = short_term[:30]
                except json.JSONDecodeError:
                    pass

            # Extract actions from actionable_items
            actions = []
            if rep.actionable_items:
                try:
                    action_data = json.loads(rep.actionable_items)
                    actions = [
                        a.get("description", "")[:15]
                        for a in action_data[:2]
                        if a.get("description")
                    ]
                except json.JSONDecodeError:
                    pass

            items.append(
                BulletinItem(
                    cluster=cluster,
                    members=members,
                    headline=cluster.event_title[:10],
                    summary=rep.summary[:50] if rep.summary else "",
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

        parts = [f"欢迎收看巴别情报站，今天有{len(items)}件大事值得关注。"]

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

        parts.append("以上就是今天的巴别情报站，关注我们获取更多资讯。")

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
            events.append({
                "id": item.cluster.id,
                "headline": item.headline,
                "summary": item.summary,
                "one_liner": item.one_liner,
                "impact": item.impact,
                "actions": item.actions,
            })

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
        opening = f"欢迎收看巴别情报站，今天有{n_items}件大事值得关注。"
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
        closing = "感谢收看巴别情报站，关注我们获取更多资讯。"
        segments.append(closing)

        return segments
