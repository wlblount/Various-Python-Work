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
import ssl
import json


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

# Popular financial Substacks to scan
DEFAULT_FINANCIAL_SUBSTACKS = [
    # Tech/Finance/Economics
    'thegeneralist',          # The Generalist - tech/finance
    'netinterest',            # Net Interest - financial services
    'marketsentiment',        # Market Sentiment
    'compoundingquality',     # Compounding Quality
    'investoramnesia',        # Investor Amnesia
    'capitalflows',           # Capital Flows
    'thediff',                # The Diff - Byrne Hobart
    'notboring',              # Not Boring - Packy McCormick
    'kyla',                   # Kyla Scanlon
    'noahpinion',             # Noah Smith - economics
    'mattstoller',            # Matt Stoller - Big/Monopoly
    'platformer',             # Tech/Business
    'dirtycapitalism',        # Dirty Capitalism
    # Value Investing / Stock Analysis
    'alluvialcapital',        # Exploring with Alluvial Capital
    'valuedegen',             # Value Degen's Substack
    'ragingbullinvestments',  # Raging Bull Investments
    'specialsituations',      # Triple S Special Situations Investing
    'marginofsanity',         # Margin of Sanity
    'fenixvanlangerode',      # Fenix Vanlangerode
    'colubeat',               # Colubeat Investment Desk
    'scavengersledger',       # Scavenger's Ledger
    'valuedontlie',           # Value Don't Lie
    'kairosresearch',         # Kairos Research
    'theatomicmoat',          # The Atomic Moat
    'bearstone',              # Bearstone
    'stockanalysiscompilation',  # Stock Analysis Compilation
    '310value',               # 310 Value's Newsletter
    # Research / Macro / More Value Investing
    'edelweisscapital',       # Edelweiss Capital Research
    'epbresearch',            # EPB Research
    'klementoninvesting',     # Klement on Investing
    'marketjiujitsu',         # Market Jiujitsu
    'moontowerweekly',        # Moontower Weekly
    'multibaggermonitor',     # Multibagger Monitor
    'pernasresearch',         # Pernas Research
    'philoinvestor',          # Philoinvestor
    'prometheusresearch',     # Prometheus Research: Macro Mechanics
    'edgealchemy',            # Edge Alchemy
    'pennyonthedollar',       # Penny on the Dollar
    'unreasonableasymmetric', # UnreasonableAsymmetric's Substack
    'journalofavalueinvestor',  # The Journal of a Value Investor
    'lakecornelia',           # Lake Cornelia Commentary
    'businessmodelmastery',   # Business Model Mastery
    'behindthebalancesheet',  # Behind the Balance Sheet
    'clarkstreetvalue',       # Clark Street Value
    'specialsituationinvest', # Special Situation Investments
]


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

            entry['title'] = title.text if title is not None else 'Untitled'
            entry['link'] = link.text if link is not None else ''
            entry['summary'] = description.text if description is not None else ''
            entry['published'] = pubDate.text if pubDate is not None else ''
            entry['content'] = content.text if content is not None else entry['summary']

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


def scan_publication(publication: str, max_articles: int = 10) -> list:
    """Scan a single Substack publication for financial articles."""
    articles = []

    entries = fetch_substack_feed(publication)
    if not entries:
        return articles

    for entry in entries[:max_articles]:
        title = entry.get('title', 'Untitled')
        link = entry.get('link', '')
        published = entry.get('published', '')
        content = entry.get('content', '') or entry.get('summary', '')

        # Extract stock symbols from title and content
        full_text = f"{title} {content}"
        symbols = extract_stock_symbols(full_text)

        # Determine if free
        free = is_article_free(content, title)

        article = Article(
            title=title,
            link=link,
            publication=publication,
            is_free=free,
            stock_symbols=symbols,
            summary=clean_html(content)[:200] + "..." if content else "",
            published=published
        )

        articles.append(article)

    return articles


def scan_substacks(publications: list = None,
                   max_articles_per_pub: int = 10,
                   only_with_tickers: bool = False,
                   only_free: bool = False,
                   verbose: bool = True) -> list:
    """
    Scan multiple Substack publications for financial articles.

    Args:
        publications: List of publication names to scan (uses defaults if None)
        max_articles_per_pub: Maximum articles to fetch per publication
        only_with_tickers: Only return articles that mention stock tickers
        only_free: Only return free articles
        verbose: Print progress information

    Returns:
        List of Article objects
    """
    if publications is None:
        publications = DEFAULT_FINANCIAL_SUBSTACKS

    all_articles = []

    for pub in publications:
        if verbose:
            print(f"Scanning {pub}.substack.com...")

        articles = scan_publication(pub, max_articles_per_pub)

        if verbose and articles:
            print(f"  Found {len(articles)} articles")
        elif verbose:
            print(f"  No articles found or feed unavailable")

        all_articles.extend(articles)

    # Apply filters
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
        default=5,
        help='Number of articles to fetch per publication (default: 5)'
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
        help='List default financial publications and exit'
    )

    args = parser.parse_args()

    if args.list_publications:
        print("Default financial Substack publications:")
        print("-" * 40)
        for pub in DEFAULT_FINANCIAL_SUBSTACKS:
            print(f"  {pub}.substack.com")
        return

    print("=" * 70)
    print("SUBSTACK FINANCIAL ARTICLE SCANNER")
    print("=" * 70)

    articles = scan_substacks(
        publications=args.publications,
        max_articles_per_pub=args.num_articles,
        only_with_tickers=args.tickers_only,
        only_free=args.free_only,
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
