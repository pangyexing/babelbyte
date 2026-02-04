"""Email delivery module for BabelByte."""

import logging
import smtplib
import ssl
from dataclasses import dataclass
from datetime import datetime
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path
from typing import Optional

from jinja2 import Environment, FileSystemLoader

from config.settings import get_settings, PROJECT_ROOT
from src.processors.digest_processor import DigestResult

logger = logging.getLogger(__name__)


@dataclass
class EmailResult:
    """Result of email sending operation."""

    success: bool
    message: str
    sent_at: Optional[datetime] = None


class EmailSender:
    """SMTP email sender for digest delivery."""

    def __init__(
        self,
        host: Optional[str] = None,
        port: Optional[int] = None,
        user: Optional[str] = None,
        password: Optional[str] = None,
        from_addr: Optional[str] = None,
        to_addr: Optional[str] = None,
    ):
        settings = get_settings().email
        self.host = host or settings.host
        self.port = port or settings.port
        self.user = user or settings.user
        self.password = password or settings.password
        self.from_addr = from_addr or settings.from_addr
        self.to_addr = to_addr or settings.to_addr

        # Setup Jinja2 template environment
        templates_dir = PROJECT_ROOT / "templates"
        self.jinja_env = Environment(
            loader=FileSystemLoader(templates_dir),
            autoescape=True,
        )

    def is_configured(self) -> bool:
        """Check if email is properly configured."""
        return all([
            self.host,
            self.port,
            self.user,
            self.password,
            self.from_addr,
            self.to_addr,
        ])

    def send_digest(
        self,
        digest: DigestResult,
        to_addr: Optional[str] = None,
        subject: Optional[str] = None,
    ) -> EmailResult:
        """
        Send a digest email.

        Args:
            digest: The digest to send.
            to_addr: Override recipient address.
            subject: Override email subject.

        Returns:
            EmailResult indicating success or failure.
        """
        if not self.is_configured():
            return EmailResult(
                success=False,
                message="Email not configured. Please set SMTP settings in .env file.",
            )

        recipient = to_addr or self.to_addr
        email_subject = subject or f"🌐 BabelByte 每日摘要 - {datetime.now().strftime('%Y-%m-%d')}"

        try:
            # Render the email template
            html_content = self._render_digest_html(digest)

            # Create the email message
            msg = MIMEMultipart("alternative")
            msg["Subject"] = email_subject
            msg["From"] = self.from_addr
            msg["To"] = recipient

            # Create plain text version
            text_content = self._render_digest_text(digest)
            msg.attach(MIMEText(text_content, "plain", "utf-8"))
            msg.attach(MIMEText(html_content, "html", "utf-8"))

            # Send the email
            self._send_email(msg, recipient)

            logger.info(f"Digest email sent to {recipient}")
            return EmailResult(
                success=True,
                message=f"Email sent successfully to {recipient}",
                sent_at=datetime.now(),
            )

        except smtplib.SMTPAuthenticationError as e:
            logger.error(f"SMTP authentication failed: {e}")
            return EmailResult(
                success=False,
                message="SMTP authentication failed. Please check your credentials.",
            )
        except smtplib.SMTPException as e:
            logger.error(f"SMTP error: {e}")
            return EmailResult(
                success=False,
                message=f"SMTP error: {str(e)}",
            )
        except ssl.SSLError as e:
            logger.error(f"SSL error: {e}")
            return EmailResult(
                success=False,
                message=f"SSL connection error: {str(e)}",
            )
        except OSError as e:
            logger.error(f"Network error: {e}")
            return EmailResult(
                success=False,
                message=f"Network error: {str(e)}",
            )

    def _send_email(self, msg: MIMEMultipart, recipient: str, max_retries: int = 3) -> None:
        """Send the email via SMTP with timeout and retry.

        Args:
            msg: The email message to send.
            recipient: The recipient email address.
            max_retries: Maximum number of retry attempts (default: 3).
        """
        import time

        # Create SSL context
        context = ssl.create_default_context()
        timeout = 60  # 60 seconds timeout for connection

        last_error = None
        for attempt in range(max_retries):
            try:
                if attempt > 0:
                    logger.info(f"Retrying email send (attempt {attempt + 1}/{max_retries})...")
                    time.sleep(2)  # Wait before retry

                if self.port == 465:
                    # SSL connection with timeout
                    with smtplib.SMTP_SSL(
                        self.host, self.port, context=context, timeout=timeout
                    ) as server:
                        server.login(self.user, self.password)
                        server.sendmail(self.from_addr, recipient, msg.as_string())
                else:
                    # TLS connection with timeout
                    with smtplib.SMTP(self.host, self.port, timeout=timeout) as server:
                        server.starttls(context=context)
                        server.login(self.user, self.password)
                        server.sendmail(self.from_addr, recipient, msg.as_string())

                return  # Success, exit the retry loop

            except (smtplib.SMTPException, ssl.SSLError, OSError, TimeoutError) as e:
                last_error = e
                logger.warning(f"Email send attempt {attempt + 1} failed: {e}")

        # All retries failed
        raise last_error

    def _render_digest_html(self, digest: DigestResult) -> str:
        """Render the digest as HTML."""
        template = self.jinja_env.get_template("email_digest.html")
        return template.render(
            date=digest.generated_at.strftime("%Y年%m月%d日"),
            total_items=digest.total_items,
            event_count=len(digest.events),
            individual_count=len(digest.regular_items),
            paper_count=len(digest.papers),
            category_count=len(digest.by_category),
            items=digest.items,
            events=digest.events,
            items_by_category=digest.by_category,
            events_by_category=digest.events_by_category,
            # Section 2: 独立事件 (non-paper items)
            regular_items_by_category=digest.regular_items_by_category,
            # Section 3: 论文 (papers)
            papers_by_category=digest.papers_by_category,
            # Legacy (kept for compatibility)
            individual_by_category=digest.items_by_category,
        )

    def _render_digest_text(self, digest: DigestResult) -> str:
        """Render the digest as plain text."""
        lines = []
        lines.append("=" * 50)
        lines.append("BabelByte 每日摘要")
        lines.append(f"日期: {digest.generated_at.strftime('%Y年%m月%d日')}")
        lines.append(f"共 {digest.total_items} 条内容")
        if digest.events:
            lines.append(f"  - {len(digest.events)} 个事件")
        if digest.regular_items:
            lines.append(f"  - {len(digest.regular_items)} 条独立事件")
        if digest.papers:
            lines.append(f"  - {len(digest.papers)} 篇论文")
        lines.append("=" * 50)

        if not digest.items and not digest.events:
            lines.append("\n今日暂无新内容")
            return "\n".join(lines)

        # Section 1: Event Clusters
        if digest.events_by_category:
            lines.append("\n" + "=" * 20 + " 事件聚合 " + "=" * 20)
            for category, category_events in sorted(digest.events_by_category.items()):
                lines.append(f"\n【{category}】({len(category_events)}个事件)")
                lines.append("-" * 30)
                for item in category_events:
                    lines.append(f"\n[事件] ★ [{item.importance_score}/10] {item.event_title}")
                    lines.append(f"   {item.summary}")
                    lines.append(f"   来源: {item.source_display}")
                    lines.append("   相关报道:")
                    for member in item.members[:3]:
                        lines.append(f"     - {member.title[:40]}...")
                    if len(item.members) > 3:
                        lines.append(f"     ...还有 {len(item.members) - 3} 篇")

        # Section 2: Regular Items (non-paper)
        if digest.regular_items_by_category:
            lines.append("\n" + "=" * 20 + " 独立事件 " + "=" * 20)
            for category, category_items in sorted(digest.regular_items_by_category.items()):
                lines.append(f"\n【{category}】({len(category_items)}条)")
                lines.append("-" * 30)
                for item in category_items:
                    title = item.content_item.title[:50]
                    lines.append(f"\n★ [{item.importance_score}/10] {title}")
                    lines.append(f"   {item.summary}")
                    lines.append(f"   来源: {item.source_display} | 作者: {item.content_item.author}")
                    lines.append(f"   链接: {item.content_item.url}")

        # Section 3: Papers (RSS)
        if digest.papers_by_category:
            lines.append("\n" + "=" * 22 + " 论文 " + "=" * 22)
            for category, category_items in sorted(digest.papers_by_category.items()):
                lines.append(f"\n【{category}】({len(category_items)}篇)")
                lines.append("-" * 30)
                for item in category_items:
                    title = item.content_item.title[:50]
                    lines.append(f"\n[论文] ★ [{item.importance_score}/10] {title}")
                    lines.append(f"   {item.summary}")
                    lines.append(f"   作者: {item.content_item.author}")
                    lines.append(f"   链接: {item.content_item.url}")

        lines.append("\n" + "=" * 50)
        lines.append("由 BabelByte 自动生成")
        return "\n".join(lines)

    def send_test_email(self, to_addr: Optional[str] = None) -> EmailResult:
        """Send a test email to verify configuration."""
        if not self.is_configured():
            return EmailResult(
                success=False,
                message="Email not configured. Please set SMTP settings in .env file.",
            )

        recipient = to_addr or self.to_addr

        try:
            msg = MIMEMultipart("alternative")
            msg["Subject"] = "🌐 BabelByte 测试邮件"
            msg["From"] = self.from_addr
            msg["To"] = recipient

            text_content = """
BabelByte 测试邮件

恭喜！您的邮件配置已成功。

这是一封测试邮件，用于验证 SMTP 设置是否正确。

---
BabelByte - AI 内容订阅系统
"""

            html_content = """
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
</head>
<body style="font-family: Arial, sans-serif; padding: 20px;">
    <h1 style="color: #4a90d9;">🌐 BabelByte 测试邮件</h1>
    <p>恭喜！您的邮件配置已成功。</p>
    <p>这是一封测试邮件，用于验证 SMTP 设置是否正确。</p>
    <hr>
    <p style="color: #888; font-size: 12px;">BabelByte - AI 内容订阅系统</p>
</body>
</html>
"""

            msg.attach(MIMEText(text_content, "plain", "utf-8"))
            msg.attach(MIMEText(html_content, "html", "utf-8"))

            self._send_email(msg, recipient)

            return EmailResult(
                success=True,
                message=f"Test email sent to {recipient}",
                sent_at=datetime.now(),
            )

        except (smtplib.SMTPException, ssl.SSLError, OSError) as e:
            return EmailResult(
                success=False,
                message=f"Failed to send test email: {str(e)}",
            )
