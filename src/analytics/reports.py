"""Report generation for weekly and monthly summaries."""

import json
import logging
import subprocess
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional

from config.settings import get_settings
from src.storage.database import SyncDatabase
from src.storage.models import ActionStatus, ContentItem, EventCluster, Topic, TopicSnapshot

logger = logging.getLogger(__name__)


@dataclass
class TopEventSummary:
    """Summary of a top event in the report."""

    title: str
    category: str
    article_count: int
    summary: str
    trend: str = "stable"


@dataclass
class TopicChange:
    """Change summary for a topic."""

    name: str
    this_period: int
    last_period: int
    change_percent: float
    trend: str
    summary: str = ""


@dataclass
class ActionReview:
    """Review of action items."""

    total_created: int
    completed: int
    pending: int
    dismissed: int
    completion_rate: float


@dataclass
class ReportData:
    """Data structure for a report."""

    period_type: str  # 'week' or 'month'
    period_start: str
    period_end: str
    generated_at: datetime = field(default_factory=datetime.now)

    # Content summary
    total_items: int = 0
    high_importance_items: int = 0
    items_by_category: dict[str, int] = field(default_factory=dict)

    # Top events
    top_events: list[TopEventSummary] = field(default_factory=list)

    # Topic changes
    topic_changes: list[TopicChange] = field(default_factory=list)

    # Trends and signals
    trending_keywords: list[str] = field(default_factory=list)
    new_entities: list[str] = field(default_factory=list)

    # Action review
    action_review: Optional[ActionReview] = None


class ReportGenerator:
    """
    Generates weekly and monthly reports from accumulated data.

    Report structure:
    1. Summary statistics
    2. Top 10 events (by importance and article count)
    3. Topic changes (comparison with previous period)
    4. Trends and signals (rising keywords, new entities)
    5. Action review (completion rate, pending items)
    """

    def __init__(self, db: SyncDatabase, use_mock: bool = False):
        self.db = db
        self.use_mock = use_mock
        self.settings = get_settings()

    def generate_weekly_report(self, weeks_ago: int = 0) -> ReportData:
        """
        Generate a weekly report.

        Args:
            weeks_ago: Generate report for N weeks ago (0 = current week)

        Returns:
            Report data structure
        """
        # Calculate period
        today = datetime.now()
        week_start = today - timedelta(days=today.weekday() + 7 * weeks_ago)
        week_start = week_start.replace(hour=0, minute=0, second=0, microsecond=0)
        week_end = week_start + timedelta(days=6, hours=23, minutes=59, seconds=59)

        return self._generate_report(
            period_type="week",
            period_start=week_start,
            period_end=week_end,
        )

    def generate_monthly_report(self, months_ago: int = 0) -> ReportData:
        """
        Generate a monthly report.

        Args:
            months_ago: Generate report for N months ago (0 = current month)

        Returns:
            Report data structure
        """
        # Calculate period
        today = datetime.now()
        month = today.month - months_ago
        year = today.year
        while month <= 0:
            month += 12
            year -= 1

        month_start = datetime(year, month, 1)
        if month == 12:
            month_end = datetime(year + 1, 1, 1) - timedelta(seconds=1)
        else:
            month_end = datetime(year, month + 1, 1) - timedelta(seconds=1)

        return self._generate_report(
            period_type="month",
            period_start=month_start,
            period_end=month_end,
        )

    def _generate_report(
        self, period_type: str, period_start: datetime, period_end: datetime
    ) -> ReportData:
        """Generate report for a specific period."""
        report = ReportData(
            period_type=period_type,
            period_start=period_start.strftime("%Y-%m-%d"),
            period_end=period_end.strftime("%Y-%m-%d"),
        )

        # Get content for the period
        period_days = (period_end - period_start).days + 1
        content = self._get_period_content(period_start, period_end)

        # Summary statistics
        report.total_items = len(content)
        report.high_importance_items = len([c for c in content if c.importance_score and c.importance_score >= 7])

        # Items by category
        for item in content:
            cat = item.category or "未分类"
            report.items_by_category[cat] = report.items_by_category.get(cat, 0) + 1

        # Top events
        report.top_events = self._get_top_events(period_start, period_end)

        # Topic changes
        report.topic_changes = self._get_topic_changes(period_days)

        # Trends
        report.trending_keywords = self._extract_trending_keywords(content)
        report.new_entities = self._find_new_entities(content, period_start)

        # Action review
        report.action_review = self._review_actions(period_start, period_end)

        return report

    def _get_period_content(
        self, start: datetime, end: datetime
    ) -> list[ContentItem]:
        """Get content items for a specific period."""
        # Use browse_by_date for each day in the period (simplified approach)
        all_content = []
        current = start
        while current <= end:
            date_str = current.strftime("%Y-%m-%d")
            daily = self.db.browse_by_date(date_str, limit=200)
            all_content.extend(daily)
            current += timedelta(days=1)

        return all_content

    def _get_top_events(
        self, start: datetime, end: datetime, limit: int = 10
    ) -> list[TopEventSummary]:
        """Get top events for the period."""
        # Get recent event clusters
        period_days = (end - start).days + 1
        clusters = self.db.get_recent_event_clusters(days=period_days, limit=30)

        # Filter to period and sort by article count + importance
        period_clusters = []
        for cluster in clusters:
            if start <= cluster.last_updated_at <= end:
                period_clusters.append(cluster)

        # Sort by article count
        period_clusters.sort(key=lambda c: c.article_count, reverse=True)

        # Generate summaries for top events
        top_events = []
        for cluster in period_clusters[:limit]:
            summary = self._generate_event_summary(cluster)
            top_events.append(TopEventSummary(
                title=cluster.event_title,
                category=cluster.category,
                article_count=cluster.article_count,
                summary=summary,
            ))

        return top_events

    def _generate_event_summary(self, cluster: EventCluster) -> str:
        """Generate a brief summary for an event."""
        if self.use_mock:
            return f"事件「{cluster.event_title}」共有{cluster.article_count}篇报道"

        # Get event members for context
        members = self.db.get_event_members(cluster.id)
        if not members:
            return f"事件「{cluster.event_title}」共有{cluster.article_count}篇报道"

        try:
            summaries = "\n".join([f"- {m.summary or m.title}" for m in members[:3]])
            prompt = f"""用30字总结这个事件：

事件：{cluster.event_title}
报道：
{summaries}"""

            cli_path = self.settings.claude.cli_path
            result = subprocess.run(
                [cli_path, "-p", prompt, "--output-format", "text"],
                capture_output=True,
                text=True,
                timeout=20,
            )

            if result.returncode == 0:
                return result.stdout.strip()[:100]

        except Exception as e:
            logger.warning(f"Event summary generation failed: {e}")

        return f"事件「{cluster.event_title}」共有{cluster.article_count}篇报道"

    def _get_topic_changes(self, period_days: int) -> list[TopicChange]:
        """Get topic changes compared to previous period."""
        topics = self.db.list_topics()
        changes = []

        for topic in topics:
            # Get recent snapshots
            snapshots = self.db.get_topic_snapshots(topic.id, limit=2)

            if len(snapshots) >= 2:
                current = snapshots[0]
                previous = snapshots[1]

                current_metrics = json.loads(current.metrics) if current.metrics else {}
                previous_metrics = json.loads(previous.metrics) if previous.metrics else {}

                this_count = current_metrics.get("article_count", 0)
                last_count = previous_metrics.get("article_count", 0)

                if last_count > 0:
                    change_pct = ((this_count - last_count) / last_count) * 100
                else:
                    change_pct = 100.0 if this_count > 0 else 0.0

                changes.append(TopicChange(
                    name=topic.name,
                    this_period=this_count,
                    last_period=last_count,
                    change_percent=round(change_pct, 1),
                    trend=current.trend,
                    summary=current.summary,
                ))

        # Sort by change magnitude
        changes.sort(key=lambda x: abs(x.change_percent), reverse=True)
        return changes

    def _extract_trending_keywords(self, content: list[ContentItem]) -> list[str]:
        """Extract trending keywords from content."""
        import re
        from collections import Counter

        # Count word occurrences
        word_counts: Counter = Counter()

        for item in content:
            text = f"{item.title} {item.summary or ''}"
            # Extract words (Chinese and English)
            words = re.findall(r"[\u4e00-\u9fff]{2,}|[a-zA-Z]{3,}", text)
            word_counts.update(words)

        # Filter common words
        stopwords = {"的", "是", "在", "了", "和", "与", "有", "这", "一个", "可以", "the", "and", "for", "with"}

        trending = [
            word for word, count in word_counts.most_common(20)
            if word.lower() not in stopwords and count >= 3
        ]

        return trending[:10]

    def _find_new_entities(
        self, content: list[ContentItem], period_start: datetime
    ) -> list[str]:
        """Find entities that appear in this period but not before."""
        import re

        # Extract entities from current period
        current_entities: set[str] = set()
        entity_patterns = [
            r"\b(OpenAI|Google|Microsoft|Meta|Apple|Amazon|Tesla|Anthropic|Nvidia)\b",
            r"\b(GPT-\d+|Claude|Gemini|Llama|ChatGPT|DALL-E|Midjourney)\b",
        ]

        for item in content:
            text = f"{item.title} {item.summary or ''}"
            for pattern in entity_patterns:
                matches = re.findall(pattern, text, re.IGNORECASE)
                current_entities.update(m.lower() for m in matches)

        # For simplicity, just return entities with high frequency
        # (full implementation would compare with previous period)
        return list(current_entities)[:10]

    def _review_actions(
        self, start: datetime, end: datetime
    ) -> ActionReview:
        """Review action items for the period."""
        # Get all action items
        all_actions = self.db.get_action_items(limit=500)

        # Filter to period
        period_actions = [
            a for a in all_actions
            if start <= a.created_at <= end
        ]

        if not period_actions:
            return ActionReview(
                total_created=0,
                completed=0,
                pending=0,
                dismissed=0,
                completion_rate=0.0,
            )

        completed = len([a for a in period_actions if a.status == ActionStatus.DONE])
        pending = len([a for a in period_actions if a.status == ActionStatus.PENDING])
        dismissed = len([a for a in period_actions if a.status == ActionStatus.DISMISSED])
        total = len(period_actions)

        completion_rate = (completed / total * 100) if total > 0 else 0.0

        return ActionReview(
            total_created=total,
            completed=completed,
            pending=pending,
            dismissed=dismissed,
            completion_rate=round(completion_rate, 1),
        )

    def format_report_text(self, report: ReportData) -> str:
        """Format report as readable text."""
        lines = []
        period_name = "周报" if report.period_type == "week" else "月报"

        lines.append("=" * 60)
        lines.append(f"BabelByte {period_name}")
        lines.append(f"时间范围: {report.period_start} 至 {report.period_end}")
        lines.append(f"生成时间: {report.generated_at.strftime('%Y-%m-%d %H:%M')}")
        lines.append("=" * 60)

        # Summary
        lines.append(f"\n## 总览")
        lines.append(f"- 共收录 {report.total_items} 条内容")
        lines.append(f"- 其中高重要性 ({'\u2265'}7分) {report.high_importance_items} 条")

        # By category
        lines.append(f"\n## 分类分布")
        for cat, count in sorted(report.items_by_category.items(), key=lambda x: x[1], reverse=True):
            lines.append(f"- {cat}: {count} 条")

        # Top events
        if report.top_events:
            lines.append(f"\n## 本期重要事件 (Top {len(report.top_events)})")
            for i, event in enumerate(report.top_events, 1):
                trend_icon = {"up": "↑", "down": "↓", "stable": "→"}.get(event.trend, "→")
                lines.append(f"\n{i}. [{event.category}] {event.title} {trend_icon}")
                lines.append(f"   报道数: {event.article_count}")
                lines.append(f"   摘要: {event.summary}")

        # Topic changes
        if report.topic_changes:
            lines.append(f"\n## 主题动态")
            for tc in report.topic_changes[:5]:
                trend_icon = {"up": "↑", "down": "↓", "stable": "→"}.get(tc.trend, "→")
                change_str = f"+{tc.change_percent}%" if tc.change_percent > 0 else f"{tc.change_percent}%"
                lines.append(f"- {tc.name}: {tc.this_period}篇 ({change_str}) {trend_icon}")
                if tc.summary:
                    lines.append(f"  {tc.summary[:50]}...")

        # Trending keywords
        if report.trending_keywords:
            lines.append(f"\n## 热门关键词")
            lines.append(", ".join(report.trending_keywords))

        # Action review
        if report.action_review:
            ar = report.action_review
            lines.append(f"\n## 行动项回顾")
            lines.append(f"- 新增: {ar.total_created}")
            lines.append(f"- 已完成: {ar.completed}")
            lines.append(f"- 待处理: {ar.pending}")
            lines.append(f"- 已忽略: {ar.dismissed}")
            lines.append(f"- 完成率: {ar.completion_rate}%")

        lines.append("\n" + "=" * 60)
        return "\n".join(lines)


def generate_weekly_report_cli(db: SyncDatabase, use_mock: bool = False) -> str:
    """Generate and format weekly report for CLI output."""
    generator = ReportGenerator(db=db, use_mock=use_mock)
    report = generator.generate_weekly_report()
    return generator.format_report_text(report)


def generate_monthly_report_cli(db: SyncDatabase, use_mock: bool = False) -> str:
    """Generate and format monthly report for CLI output."""
    generator = ReportGenerator(db=db, use_mock=use_mock)
    report = generator.generate_monthly_report()
    return generator.format_report_text(report)
