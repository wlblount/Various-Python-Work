#!/usr/bin/env python3
"""
Substack Financial Article Scanner

Scans Substack publications for financial articles and extracts:
- Stock symbols mentioned in posts
- Whether the post is free or paid
- Links to the posts

No external dependencies required - uses only Python standard library.
"""

import re
import html
import argparse
import xml.etree.ElementTree as ET
from typing import Optional
from dataclasses import dataclass, field
from urllib.request import urlopen, Request
from urllib.error import URLError, HTTPError
from datetime import datetime, timedelta
from email.utils import parsedate_to_datetime
import ssl
import json
import os


# Common stock ticker pattern: $AAPL or standalone like AAPL, TSLA
TICKER_PATTERN = re.compile(r'\$([A-Z]{1,5})\b|\b([A-Z]{2,5})\b')

# Common words to exclude from ticker detection
EXCLUDED_WORDS = {
    'THE', 'AND', 'FOR', 'ARE', 'BUT', 'NOT', 'YOU', 'ALL', 'CAN', 'HAD',
    'HER', 'WAS', 'ONE', 'OUR', 'OUT', 'HAS', 'HIS', 'HOW', 'ITS', 'MAY',
    'NEW', 'NOW', 'OLD', 'SEE', 'WAY', 'WHO', 'BOY', 'DID', 'GET', 'HIM',
    'LET', 'PUT', 'SAY', 'SHE', 'TOO', 'USE', 'CEO', 'CFO', 'COO', 'IPO',
    'GDP', 'CPI', 'FED', 'SEC', 'ETF', 'NYSE', 'API', 'USA', 'USD', 'EUR',
    'GBP', 'JPY', 'PDF', 'RSS', 'URL', 'CEO', 'THIS', 'THAT', 'WITH',
    'HAVE', 'FROM', 'THEY', 'BEEN', 'HAVE', 'WERE', 'SAID', 'EACH',
    'WHICH', 'THEIR', 'WILL', 'OTHER', 'ABOUT', 'MANY', 'THEN', 'THEM',
    'THESE', 'SOME', 'WOULD', 'MAKE', 'LIKE', 'INTO', 'YEAR', 'YOUR',
    'GOOD', 'COULD', 'THAN', 'FIRST', 'BEEN', 'CALL', 'AFTER', 'BACK',
    'ONLY', 'OVER', 'SUCH', 'ALSO', 'LAST', 'MORE', 'MOST', 'VERY',
    'JUST', 'WHERE', 'MUCH', 'BEFORE', 'RIGHT', 'WHILE', 'STILL',
    'READ', 'FREE', 'PAID', 'POST', 'LINK', 'VIEW', 'CLICK', 'HERE',
    'MORE', 'LESS', 'HIGH', 'LOW', 'BUY', 'SELL', 'HOLD', 'LONG', 'SHORT',
    'BULL', 'BEAR', 'RISK', 'FUND', 'CASH', 'DEBT', 'BOND', 'RATE', 'BANK',
    'LOAN', 'STOCK', 'SHARE', 'PRICE', 'VALUE', 'MARKET', 'TRADE', 'NEWS',
    'DATA', 'WEEK', 'TODAY', 'DAILY', 'WEEKLY', 'MONTHLY', 'ANNUAL', 'YOY',
    'QOQ', 'MOM', 'GAAP', 'NON', 'TTM', 'FY', 'EPS', 'PE', 'PS', 'PB',
    'ROE', 'ROA', 'EBITDA', 'EBIT', 'DCF', 'WACC', 'CAGR', 'NAV', 'AUM',
    'OTC', 'SPAC', 'REIT', 'MLP', 'ADR', 'GDR', 'ESOP', 'RSU', 'AI',
    'ML', 'EV', 'MEMO', 'NOTE', 'NOTES', 'EDIT', 'UPDATE', 'VIA', 'ICYMI',
    'RSS', 'HTML', 'HTTP', 'HTTPS', 'WWW', 'COM', 'ORG', 'NET', 'IO'
}

# Well-known stock tickers to prioritize
KNOWN_TICKERS = {
    'AAPL', 'MSFT', 'GOOGL', 'GOOG', 'AMZN', 'META', 'TSLA', 'NVDA', 'AMD',
    'INTC', 'NFLX', 'DIS', 'BA', 'GE', 'F', 'GM', 'T', 'VZ', 'JPM', 'BAC',
    'WFC', 'C', 'GS', 'MS', 'V', 'MA', 'PYPL', 'SQ', 'COIN', 'HOOD',
    'WMT', 'TGT', 'COST', 'HD', 'LOW', 'NKE', 'SBUX', 'MCD', 'KO', 'PEP',
    'PG', 'JNJ', 'PFE', 'MRNA', 'UNH', 'CVS', 'ABBV', 'MRK', 'LLY', 'BMY',
    'XOM', 'CVX', 'COP', 'SLB', 'OXY', 'BP', 'SHEL', 'VALE', 'RIO', 'BHP',
    'SPY', 'QQQ', 'IWM', 'DIA', 'VTI', 'VOO', 'ARKK', 'XLF', 'XLE', 'XLK',
    'BTC', 'ETH', 'MSTR', 'PLTR', 'SNOW', 'CRM', 'ORCL', 'IBM', 'CSCO',
    'UBER', 'LYFT', 'ABNB', 'DASH', 'RBLX', 'SNAP', 'PINS', 'TWTR', 'SHOP',
    'ZM', 'DOCU', 'CRWD', 'ZS', 'NET', 'DDOG', 'MDB', 'OKTA', 'TWLO',
    'SOFI', 'UPST', 'AFRM', 'NU', 'OPEN', 'RIVN', 'LCID', 'NIO', 'XPEV', 'LI',
    'ARM', 'SMCI', 'AVGO', 'MU', 'QCOM', 'TXN', 'MRVL', 'KLAC', 'LRCX', 'ASML'
}

# Config file path (same directory as script)
CONFIG_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'my_substacks.txt')

# Initial publications to populate config file on first run
INITIAL_PUBLICATIONS = [
    'thegeneralist',
    'netinterest',
    'marketsentiment',
    'compoundingquality',
    'investoramnesia',
    'capitalflows',
    'thediff',
    'notboring',
    'kyla',
    'noahpinion',
    'mattstoller',
    'platformer',
    'dirtycapitalism',
    'alluvialcapital',
    'valuedegen',
    'ragingbullinvestments',
    'specialsituations',
    'marginofsanity',
    'fenixvanlangerode',
    'colubeat',
    'scavengersledger',
    'valuedontlie',
    'kairosresearch',
    'theatomicmoat',
    'bearstone',
    'stockanalysiscompilation',
    '310value',
    'edelweisscapital',
    'epbresearch',
    'klementoninvesting',
    'marketjiujitsu',
    'moontowerweekly',
    'multibaggermonitor',
    'pernasresearch',
    'philoinvestor',
    'prometheusresearch',
    'edgealchemy',
    'pennyonthedollar',
    'unreasonableasymmetric',
    'journalofavalueinvestor',
    'lakecornelia',
    'businessmodelmastery',
    'behindthebalancesheet',
    'clarkstreetvalue',
    'specialsituationinvest',
]


def get_config_path() -> str:
    """Get the path to the config file."""
    return CONFIG_FILE


def load_publications() -> list:
    """Load publications from config file. Creates file with initial list if it doesn't exist."""
    config_path = get_config_path()

    # If config doesn't exist, create it with initial publications
    if not os.path.exists(config_path):
        save_publications(INITIAL_PUBLICATIONS)
        print(f"Created config file: {config_path}")
        return INITIAL_PUBLICATIONS.copy()

    # Load from file
    publications = []
    with open(config_path, 'r') as f:
        for line in f:
            line = line.strip()
            # Skip empty lines and comments
            if line and not line.startswith('#'):
                publications.append(line)

    return publications


def save_publications(publications: list) -> None:
    """Save publications to config file."""
    config_path = get_config_path()
    with open(config_path, 'w') as f:
        f.write("# Substack publications to scan (one per line)\n")
        f.write("# Lines starting with # are comments\n\n")
        for pub in sorted(set(publications)):  # Remove duplicates and sort
            f.write(f"{pub}\n")


def add_publication(name: str) -> bool:
    """Add a publication to the config file. Returns True if added, False if already exists."""
    name = name.lower().strip()
    publications = load_publications()

    if name in publications:
        return False

    publications.append(name)
    save_publications(publications)
    return True


def remove_publication(name: str) -> bool:
    """Remove a publication from the config file. Returns True if removed, False if not found."""
    name = name.lower().strip()
    publications = load_publications()

    if name not in publications:
        return False

    publications.remove(name)
    save_publications(publications)
    return True


@dataclass
class Article:
    """Represents a scanned article."""
    title: str
    link: str
    publication: str
    is_free: bool
    stock_symbols: list = field(default_factory=list)
    summary: str = ""
    published: str = ""
    published_dt: Optional[datetime] = None


def check_page_for_paid_status(url: str) -> Optional[bool]:
    """
    Fetch the actual article page and check for PAID/FREE indicator.
    Returns True if paid, False if free, None if can't determine.
    """
    if not url:
        return None

    content = fetch_url(url, timeout=10)
    if not content:
        return None

    try:
        html_text = content.decode('utf-8', errors='ignore').lower()

        # Look for Substack's paid indicator patterns in the page
        # Pattern: "· paid" or ">paid<" or "class="paid"" near the date
        if '· paid' in html_text or '>paid<' in html_text:
            return True
        if '· free' in html_text or '>free<' in html_text:
            return False

        # Check meta tags
        if 'content="paid"' in html_text:
            return True
        if 'content="free"' in html_text:
            return False

        # Check for paywall elements
        if 'paywall' in html_text and 'class="paywall"' in html_text:
            return True

    except Exception:
        pass

    return None


def parse_published_date(date_str: str) -> Optional[datetime]:
    """Parse RSS publication date string into datetime object."""
    if not date_str:
        return None

    # Try RFC 2822 format (common in RSS feeds): "Thu, 22 Jan 2026 15:41:21 GMT"
    try:
        return parsedate_to_datetime(date_str)
    except (ValueError, TypeError):
        pass

    # Try ISO 8601 format (common in Atom feeds): "2026-01-22T15:41:21Z"
    iso_formats = [
        "%Y-%m-%dT%H:%M:%SZ",
        "%Y-%m-%dT%H:%M:%S%z",
        "%Y-%m-%dT%H:%M:%S.%fZ",
        "%Y-%m-%d",
    ]
    for fmt in iso_formats:
        try:
            return datetime.strptime(date_str, fmt)
        except ValueError:
            continue

    return None


def clean_html(text: str) -> str:
    """Remove HTML tags and decode entities."""
    if not text:
        return ""
    # Remove HTML tags
    clean = re.sub(r'<[^>]+>', ' ', text)
    # Decode HTML entities
    clean = html.unescape(clean)
    # Normalize whitespace
    clean = ' '.join(clean.split())
    return clean


def extract_stock_symbols(text: str) -> list:
    """Extract stock ticker symbols from text."""
    text = clean_html(text)
    symbols = set()

    # Find all potential tickers
    matches = TICKER_PATTERN.findall(text)

    for match in matches:
        # match is a tuple from the two groups in our pattern
        ticker = match[0] if match[0] else match[1]

        if not ticker:
            continue

        # Prioritize known tickers
        if ticker in KNOWN_TICKERS:
            symbols.add(ticker)
        # Include if preceded by $ (from first group)
        elif match[0]:
            symbols.add(ticker)
        # Skip uncertain matches to reduce noise

    return sorted(list(symbols))


def is_article_free(content: str, title: str = "") -> bool:
    """
    Determine if an article is free or paywalled based on content.

    Note: RSS feeds often only include preview/teaser content, so we default
    to PAID/UNKNOWN unless we have strong evidence the full article is free.
    """
    if not content:
        return False

    clean_content = clean_html(content)
    content_lower = clean_content.lower()
    title_lower = title.lower() if title else ""

    # Check for explicit FREE indicators
    free_indicators = [
        'this post is free',
        'free post',
        'available to all readers',
        'available to everyone',
        'public post'
    ]

    for indicator in free_indicators:
        if indicator in content_lower:
            return True

    # Check for paywall/paid indicators
    paywall_indicators = [
        'subscribe to read',
        'for paid subscribers',
        'paid subscribers only',
        'unlock this post',
        'become a paid subscriber',
        'this post is for paid subscribers',
        'subscribe to continue reading',
        'rest of this post is for paid subscribers',
        'upgrade to paid',
        'this post is for paying subscribers',
        'keep reading with a',
        'subscription',
        'subscribe to',
        'paid post',
        'premium post',
        'members only',
        'subscriber-only',
        'subscribers only'
    ]

    for indicator in paywall_indicators:
        if indicator in content_lower:
            return False

    # Check title for paid indicators
    paid_title_indicators = ['paid', 'premium', 'subscriber', 'members']
    for indicator in paid_title_indicators:
        if indicator in title_lower:
            return False

    # RSS feeds typically contain truncated content for paid posts
    # Very long content (>3000 chars) is more likely to be free/full article
    # But we default to UNKNOWN (shown as PAID) to be conservative
    if len(clean_content) > 3000:
        return True

    # Default: assume paid/unknown since RSS often only has teasers
    return False


def fetch_url(url: str, timeout: int = 15) -> Optional[bytes]:
    """Fetch content from a URL."""
    try:
        # Create SSL context that doesn't verify (for compatibility)
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE

        request = Request(
            url,
            headers={
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
        )
        with urlopen(request, timeout=timeout, context=ctx) as response:
            return response.read()
    except (URLError, HTTPError, TimeoutError) as e:
        return None


def parse_rss_feed(xml_content: bytes) -> list:
    """Parse RSS feed XML and return list of entries."""
    entries = []

    try:
        root = ET.fromstring(xml_content)

        # Handle both RSS and Atom feeds
        # RSS format
        for item in root.findall('.//item'):
            entry = {}
            title = item.find('title')
            link = item.find('link')
            description = item.find('description')
            pubDate = item.find('pubDate')
            content = item.find('{http://purl.org/rss/1.0/modules/content/}encoded')

            # Check for Substack-specific enclosure (podcasts/paid indicator)
            enclosure = item.find('enclosure')

            # Check all child elements for any "paid" or "access" indicators
            is_paid_from_feed = False
            for child in item:
                tag_lower = child.tag.lower()
                text_lower = (child.text or '').lower()
                if 'paid' in tag_lower or 'paid' in text_lower:
                    is_paid_from_feed = True
                if 'access' in tag_lower and 'paid' in text_lower:
                    is_paid_from_feed = True

            entry['title'] = title.text if title is not None else 'Untitled'
            entry['link'] = link.text if link is not None else ''
            entry['summary'] = description.text if description is not None else ''
            entry['published'] = pubDate.text if pubDate is not None else ''
            entry['content'] = content.text if content is not None else entry['summary']
            entry['is_paid_from_feed'] = is_paid_from_feed

            entries.append(entry)

        # Atom format (if no RSS items found)
        if not entries:
            ns = {'atom': 'http://www.w3.org/2005/Atom'}
            for item in root.findall('.//atom:entry', ns):
                entry = {}
                title = item.find('atom:title', ns)
                link = item.find('atom:link', ns)
                summary = item.find('atom:summary', ns)
                content = item.find('atom:content', ns)
                published = item.find('atom:published', ns)

                entry['title'] = title.text if title is not None else 'Untitled'
                entry['link'] = link.get('href') if link is not None else ''
                entry['summary'] = summary.text if summary is not None else ''
                entry['published'] = published.text if published is not None else ''
                entry['content'] = content.text if content is not None else entry['summary']

                entries.append(entry)

    except ET.ParseError:
        return []

    return entries


def fetch_substack_feed(publication: str) -> Optional[list]:
    """Fetch and parse a Substack RSS feed."""
    url = f"https://{publication}.substack.com/feed"

    content = fetch_url(url)
    if not content:
        return None

    return parse_rss_feed(content)


def scan_publication(publication: str, max_articles: int = 10, verify_access: bool = False) -> list:
    """Scan a single Substack publication for financial articles.

    Args:
        publication: Substack publication name
        max_articles: Maximum articles to fetch
        verify_access: If True, fetch each article page to verify paid/free status (slower)
    """
    articles = []

    entries = fetch_substack_feed(publication)
    if not entries:
        return articles

    for entry in entries[:max_articles]:
        title = entry.get('title', 'Untitled')
        link = entry.get('link', '')
        published = entry.get('published', '')
        content = entry.get('content', '') or entry.get('summary', '')
        is_paid_from_feed = entry.get('is_paid_from_feed', False)

        # Extract stock symbols from title and content
        full_text = f"{title} {content}"
        symbols = extract_stock_symbols(full_text)

        # Determine if free - check multiple sources
        free = None

        # First check RSS feed indicator
        if is_paid_from_feed:
            free = False

        # If --verify-access, check the actual page (most accurate)
        if verify_access and free is None:
            page_paid = check_page_for_paid_status(link)
            if page_paid is not None:
                free = not page_paid

        # Fall back to content analysis
        if free is None:
            free = is_article_free(content, title)

        # Parse publication date
        published_dt = parse_published_date(published)

        article = Article(
            title=title,
            link=link,
            publication=publication,
            is_free=free,
            stock_symbols=symbols,
            summary=clean_html(content)[:200] + "..." if content else "",
            published=published,
            published_dt=published_dt
        )

        articles.append(article)

    return articles


def scan_substacks(publications: list = None,
                   max_articles_per_pub: int = 20,
                   only_with_tickers: bool = False,
                   only_free: bool = False,
                   start_date: datetime = None,
                   verify_access: bool = False,
                   verbose: bool = True) -> list:
    """
    Scan multiple Substack publications for financial articles.

    Args:
        publications: List of publication names to scan (uses defaults if None)
        max_articles_per_pub: Maximum articles to fetch per publication
        only_with_tickers: Only return articles that mention stock tickers
        only_free: Only return free articles
        start_date: Only return articles published on or after this date
        verify_access: Fetch each article page to verify paid/free status (slower)
        verbose: Print progress information

    Returns:
        List of Article objects
    """
    if publications is None:
        publications = load_publications()

    all_articles = []

    for pub in publications:
        if verbose:
            print(f"Scanning {pub}.substack.com...")

        articles = scan_publication(pub, max_articles_per_pub, verify_access)

        if verbose and articles:
            print(f"  Found {len(articles)} articles")
        elif verbose:
            print(f"  No articles found or feed unavailable")

        all_articles.extend(articles)

    # Apply filters
    if start_date:
        # Make start_date timezone-naive for comparison if needed
        start_date_naive = start_date.replace(tzinfo=None) if start_date.tzinfo else start_date
        filtered = []
        for a in all_articles:
            if a.published_dt:
                # Make article date timezone-naive for comparison
                article_date = a.published_dt.replace(tzinfo=None) if a.published_dt.tzinfo else a.published_dt
                if article_date >= start_date_naive:
                    filtered.append(a)
            # If no date could be parsed, exclude it (can't verify it's recent)
            # else: skip
        all_articles = filtered

    if only_with_tickers:
        all_articles = [a for a in all_articles if a.stock_symbols]

    if only_free:
        all_articles = [a for a in all_articles if a.is_free]

    return all_articles


def format_article(article: Article, show_summary: bool = False) -> str:
    """Format an article for display."""
    status = "FREE" if article.is_free else "PAID"
    symbols = ", ".join(f"${s}" for s in article.stock_symbols) if article.stock_symbols else "None"

    output = f"""
{'='*70}
Title: {article.title}
Publication: {article.publication}.substack.com
Status: [{status}]
Tickers: {symbols}
Link: {article.link}
Published: {article.published}"""

    if show_summary and article.summary:
        output += f"\nSummary: {article.summary}"

    return output


def export_json(articles: list, filename: str):
    """Export articles to JSON file."""
    data = []
    for a in articles:
        data.append({
            'title': a.title,
            'publication': a.publication,
            'link': a.link,
            'is_free': a.is_free,
            'stock_symbols': a.stock_symbols,
            'published': a.published
        })

    with open(filename, 'w') as f:
        json.dump(data, f, indent=2)

    print(f"Exported {len(articles)} articles to {filename}")


def main():
    parser = argparse.ArgumentParser(
        description='Scan Substack for financial articles and extract stock tickers'
    )
    parser.add_argument(
        '-p', '--publications',
        nargs='+',
        help='Specific Substack publications to scan (e.g., thegeneralist notboring)'
    )
    parser.add_argument(
        '-n', '--num-articles',
        type=int,
        default=20,
        help='Number of articles to fetch per publication before date filter (default: 20)'
    )
    parser.add_argument(
        '--tickers-only',
        action='store_true',
        help='Only show articles that mention stock tickers'
    )
    parser.add_argument(
        '--free-only',
        action='store_true',
        help='Only show free articles'
    )
    parser.add_argument(
        '--verify-access',
        action='store_true',
        help='Fetch each article page to verify paid/free status (slower but more accurate)'
    )
    parser.add_argument(
        '-q', '--quiet',
        action='store_true',
        help='Suppress progress output'
    )
    parser.add_argument(
        '--summary',
        action='store_true',
        help='Show article summaries'
    )
    parser.add_argument(
        '--export-json',
        metavar='FILE',
        help='Export results to JSON file'
    )
    parser.add_argument(
        '--list-publications',
        action='store_true',
        help='List all publications from config file and exit'
    )
    parser.add_argument(
        '--add-pub',
        metavar='NAME',
        help='Add a publication to your config file (e.g., --add-pub newsubstack)'
    )
    parser.add_argument(
        '--remove-pub',
        metavar='NAME',
        help='Remove a publication from your config file'
    )
    parser.add_argument(
        '-d', '--days',
        type=int,
        default=5,
        help='Only show articles from the last N days (default: 5)'
    )
    parser.add_argument(
        '--start-date',
        metavar='YYYY-MM-DD',
        help='Only show articles on or after this date (overrides --days)'
    )

    args = parser.parse_args()

    if args.list_publications:
        pubs = load_publications()
        print(f"Substack publications ({len(pubs)} total):")
        print(f"Config file: {get_config_path()}")
        print("-" * 40)
        for pub in sorted(pubs):
            print(f"  {pub}.substack.com")
        return

    if args.add_pub:
        name = args.add_pub.lower().strip()
        if add_publication(name):
            print(f"Added: {name}.substack.com")
            print(f"Config file: {get_config_path()}")
        else:
            print(f"Already exists: {name}.substack.com")
        return

    if args.remove_pub:
        name = args.remove_pub.lower().strip()
        if remove_publication(name):
            print(f"Removed: {name}.substack.com")
            print(f"Config file: {get_config_path()}")
        else:
            print(f"Not found: {name}.substack.com")
        return

    # Determine start date
    if args.start_date:
        try:
            start_date = datetime.strptime(args.start_date, "%Y-%m-%d")
        except ValueError:
            print(f"Error: Invalid date format '{args.start_date}'. Use YYYY-MM-DD.")
            return
    else:
        start_date = datetime.now() - timedelta(days=args.days)

    print("=" * 70)
    print("SUBSTACK FINANCIAL ARTICLE SCANNER")
    print("=" * 70)
    print(f"Showing articles from: {start_date.strftime('%Y-%m-%d')} onwards")
    print("=" * 70)

    articles = scan_substacks(
        publications=args.publications,
        max_articles_per_pub=args.num_articles,
        only_with_tickers=args.tickers_only,
        only_free=args.free_only,
        start_date=start_date,
        verify_access=args.verify_access,
        verbose=not args.quiet
    )

    print(f"\n{'='*70}")
    print(f"RESULTS: Found {len(articles)} articles")
    print(f"{'='*70}")

    for article in articles:
        print(format_article(article, show_summary=args.summary))

    # Summary of tickers found
    all_tickers = {}
    for article in articles:
        for ticker in article.stock_symbols:
            if ticker not in all_tickers:
                all_tickers[ticker] = []
            all_tickers[ticker].append(article.title[:40])

    if all_tickers:
        print(f"\n{'='*70}")
        print("TICKER SUMMARY")
        print(f"{'='*70}")
        for ticker, titles in sorted(all_tickers.items()):
            print(f"\n${ticker} - mentioned in {len(titles)} article(s):")
            for title in titles:
                print(f"  - {title}...")

    # Export if requested
    if args.export_json:
        export_json(articles, args.export_json)


if __name__ == '__main__':
    main()
