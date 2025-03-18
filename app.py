import os
import sys
import json
import csv
import subprocess
import datetime
import pytz
import random
import feedparser
import tldextract

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.align import Align
from rich.live import Live

console = Console()

############################################################################
# KEY CAPTURE FOR ARROW / j / k
############################################################################

if sys.platform.startswith("win"):
    import msvcrt

    def get_single_key_raw():
        return msvcrt.getch()
else:
    import tty
    import termios

    def get_single_key_raw():
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            tty.setraw(fd)
            ch = os.read(fd, 1)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        return ch


def read_key_sequence():
    """
    Reads raw bytes and interprets arrow keys, j/k, space, q, enter, etc.
    Returns a short string like "up", "down", "left", "right", "j", "k",
    " ", "q", "enter", etc.
    """
    first = get_single_key_raw()
    if not first:
        return ""
    if first == b"\r":  # Windows Enter
        return "enter"
    if first == b"\n":  # Unix Enter
        return "enter"

    # Check for ESC-based arrow sequences
    if first == b"\x1b":  # ESC
        # Attempt to read next two bytes
        rest = os.read(sys.stdin.fileno(), 2) if sys.stdin.isatty() else b""
        if rest in (b"[A", b"OA"):  # Up
            return "up"
        elif rest in (b"[B", b"OB"):  # Down
            return "down"
        elif rest in (b"[C", b"OC"):  # Right
            return "right"
        elif rest in (b"[D", b"OD"):  # Left
            return "left"
        return ""
    else:
        ch = first.decode(errors="ignore")
        if ch == "j":
            return "down"
        elif ch == "k":
            return "up"
        elif ch == " ":
            return "space"
        elif ch == "q":
            return "q"
        return ch


############################################################################
# HELPER: INTERACTIVE MENU (NO FLICKER)
############################################################################


def interactive_menu(title: str, items: list[str]) -> int:
    """
    Displays an interactive menu using arrow keys (or j/k) to move,
    Enter/Space to select, and q to quit (returns -1).
    Returns the selected index (0-based), or -1 if user quits.
    """
    position = 0  # highlight index

    def make_table():
        # Build a Rich Table with highlight on 'position'
        table = Table(show_header=False)
        table.title = title
        for i, item in enumerate(items):
            if i == position:
                # Highlight row
                table.add_row(
                    f"[bold magenta]>[/bold magenta]", f"[reverse]{item}[/reverse]"
                )
            else:
                table.add_row(" ", item)
        return table

    with Live(auto_refresh=False, console=console) as live:
        while True:
            menu_table = make_table()
            live.update(Align.center(menu_table), refresh=True)
            key = read_key_sequence()

            if key in ("up"):
                position = max(0, position - 1)
            elif key in ("down"):
                position = min(len(items) - 1, position + 1)
            elif key in ("enter", "space"):
                return position
            elif key in ("q", "\x03"):  # q or Ctrl+C
                return -1
            # ignore other keys
            live.refresh()


############################################################################
# YOUR RSS FEED, CACHING, AND RATING LOGIC
############################################################################

# The rest is basically the same as before.
# We'll just remove or avoid console.clear() in the menus to reduce flicker.

CACHE_FILE = "headlines_cache.json"


def load_cache():
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, "r", encoding="utf-8") as f:
            try:
                return json.load(f)
            except json.JSONDecodeError:
                return []
    return []


def save_cache(cache):
    with open(CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(cache, f, indent=2)


def load_feeds_from_json(file_path="my_feeds.json"):
    if not os.path.isfile(file_path):
        console.print(Align.center(f"[red]{file_path} does not exist.[/red]"))
        return []
    with open(file_path, "r", encoding="utf-8") as f:
        try:
            data = json.load(f)
            feeds = data.get("feeds", [])
            return [feed for feed in feeds if feed.get("enabled", True)]
        except json.JSONDecodeError as e:
            console.print(
                Align.center(f"[red]Error decoding JSON from {file_path}: {e}[/red]")
            )
            return []


def get_articles_from_feed(url, category):
    feed = feedparser.parse(url)
    today = datetime.datetime.now(pytz.utc).date()
    ext = tldextract.extract(url)
    main_domain = f"{ext.domain}.{ext.suffix}"
    today_entries = []
    for entry in feed.entries:
        date_tuple = None
        if hasattr(entry, "published_parsed") and entry.published_parsed:
            date_tuple = entry.published_parsed
        elif hasattr(entry, "updated_parsed") and entry.updated_parsed:
            date_tuple = entry.updated_parsed
        elif hasattr(entry, "created_parsed") and entry.created_parsed:
            date_tuple = entry.created_parsed
        if date_tuple and len(date_tuple) >= 6:
            entry_date = datetime.datetime(*date_tuple[:6], tzinfo=pytz.utc).date()
            if entry_date == today:
                today_entries.append(
                    {
                        "title": entry.title,
                        "link": entry.link,
                        "published": datetime.datetime(
                            *date_tuple[:6], tzinfo=pytz.utc
                        ).isoformat(),
                        "category": category,
                        "source": main_domain,
                    }
                )
    return today_entries


def display_status_bar(target=10):
    counts = {str(i): 0 for i in range(6)}
    if os.path.exists("dataset.csv"):
        with open("dataset.csv", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    label_int = int(row["label"])
                    if 0 <= label_int <= 5:
                        counts[str(label_int)] += 1
                except ValueError:
                    pass

    table = Table(
        title="Rating Progress", show_header=True, header_style="bold magenta"
    )
    table.add_column("Rating", justify="center")
    table.add_column("Count", justify="center")
    table.add_column("Target", justify="center")
    table.add_column("Progress", justify="center")

    for i in range(6):
        count = counts[str(i)]
        progress_percent = min(count / target, 1.0)
        filled = int(progress_percent * 10)
        bar = "█" * filled + " " * (10 - filled)
        percentage = int(progress_percent * 100)
        table.add_row(str(i), str(count), str(target), f"[{bar}] {percentage}%")

    console.print()
    console.print(Align.center(table))
    console.print()


def create_or_update_dataset():
    feeds = load_feeds_from_json("my_feeds.json")
    if not feeds:
        console.print(
            Align.center(
                "[red]No feeds available. Please add some sources first.[/red]"
            )
        )
        return

    cache = load_cache()

    # fetch new articles
    for feed in feeds:
        console.print(
            Align.center(
                f"[bold cyan]Fetching articles from:[/bold cyan] {feed.get('name', '?')} ([green]{feed.get('url')}[/green])"
            )
        )
        new_articles = get_articles_from_feed(feed["url"], feed["category"])
        for article in new_articles:
            if not any(article["title"] == c.get("title") for c in cache):
                cache.append(article)

    # remove already-rated
    rated_titles = set()
    if os.path.exists("dataset.csv"):
        with open("dataset.csv", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                rated_titles.add(row["text"])
    cache = [a for a in cache if a.get("title") not in rated_titles]

    if not cache:
        console.print(Align.center("[green]No new headlines to rate.[/green]"))
        return

    random.shuffle(cache)

    dataset_file = "dataset.csv"
    with open(dataset_file, "a", newline="", encoding="utf-8") as f:
        fieldnames = ["text", "label"]
        writer = csv.DictWriter(f, fieldnames=fieldnames, quoting=csv.QUOTE_NONNUMERIC)
        if os.stat(dataset_file).st_size == 0:
            writer.writeheader()

        key_to_rating = {
            "0": 0,
            "1": 1,
            "2": 2,
            "3": 3,
            "4": 4,
            "5": 5,
            "a": 0,
            "A": 0,
            "s": 1,
            "S": 1,
            "d": 2,
            "D": 2,
            "f": 3,
            "F": 3,
            "g": 4,
            "G": 4,
            "h": 5,
            "H": 5,
        }
        TARGET_COUNT = 10
        idx = 0

        while idx < len(cache):
            article = cache[idx]
            headline = article.get("title")

            console.print()
            display_status_bar(TARGET_COUNT)

            panel = Panel(
                f"[bold yellow]{headline}[/bold yellow]\n[dim]Source:[/dim] {article.get('source')} | [dim]Category:[/dim] {article.get('category')}",
                title="New Headline",
                subtitle="[grey](Press 0-5 or a,s,d,f,g,h to rate, q to quit)[/grey]",
            )
            console.print(Align.center(panel))
            console.print()

            while True:
                key = read_key_sequence()
                if key.lower() == "q":
                    console.print(Align.center("[red]Quitting dataset creation.[/red]"))
                    save_cache(cache)
                    return
                if key in key_to_rating:
                    rating = key_to_rating[key]
                    writer.writerow({"text": headline, "label": rating})
                    console.print(
                        Align.center(f"[green]Rating '{rating}' saved.[/green]")
                    )
                    cache.pop(idx)
                    save_cache(cache)
                    break

        console.print(
            Align.center(
                "[bold green]All cached headlines have been rated.[/bold green]"
            )
        )


def train_model():
    console.print(Align.center("[bold cyan]Starting model training...[/bold cyan]"))
    result = subprocess.run(["python", "train.py"])
    if result.returncode == 0:
        console.print(Align.center("[green]Training completed successfully.[/green]"))
    else:
        console.print(Align.center("[red]Training encountered an error.[/red]"))


############################################################################
# INTERACTIVE TOGGLE
############################################################################


def interactive_toggle_sources(feeds):
    """
    Allows user to arrow/j/k through the feed list, press space to toggle,
    or q to quit.  No flicker, using Live.
    """
    if not feeds:
        console.print(Align.center("[red]No sources available.[/red]"))
        return

    position = 0

    def make_table():
        table = Table(show_header=True, header_style="bold magenta")
        table.title = "Toggle Source Status"
        table.add_column("No.", justify="right")
        table.add_column("Name", style="cyan")
        table.add_column("Status", style="yellow")
        for i, feed in enumerate(feeds):
            name = feed.get("name", "No Name")
            status = "Enabled" if feed.get("enabled", True) else "Disabled"
            if i == position:
                table.add_row(
                    f"[bold magenta]{i + 1}[/bold magenta]",
                    f"[reverse]{name}[/reverse]",
                    f"[reverse]{status}[/reverse]",
                )
            else:
                table.add_row(str(i + 1), name, status)
        return table

    with Live(auto_refresh=False, console=console) as live:
        while True:
            live.update(Align.center(make_table()), refresh=True)
            console.print(
                Align.center(
                    "[bold]Use ↑/↓ or j/k to move, space to toggle, q to quit[/bold]"
                ),
                justify="center",
            )
            key = read_key_sequence()
            if key in ("up"):
                position = max(0, position - 1)
            elif key in ("down"):
                position = min(len(feeds) - 1, position + 1)
            elif key in ("j"):
                position = min(len(feeds) - 1, position + 1)
            elif key in ("k"):
                position = max(0, position - 1)
            elif key == "space":
                current = feeds[position].get("enabled", True)
                feeds[position]["enabled"] = not current
            elif key.lower() == "q":
                break
            live.refresh()


############################################################################
# MANAGE SOURCES MENU
############################################################################


def import_curated_feeds(data):
    curated = [
        {
            "name": "TechCrunch",
            "url": "https://techcrunch.com/rss",
            "category": "Technology",
            "enabled": True,
        },
        {
            "name": "Wired",
            "url": "https://www.wired.com/feed",
            "category": "Technology",
            "enabled": True,
        },
        {
            "name": "The Verge",
            "url": "https://www.theverge.com/rss/index.xml",
            "category": "Technology",
            "enabled": True,
        },
        {
            "name": "Hacker News",
            "url": "https://hnrss.org/frontpage",
            "category": "Technology",
            "enabled": True,
        },
        {
            "name": "BBC News",
            "url": "http://feeds.bbci.co.uk/news/rss.xml",
            "category": "News",
            "enabled": True,
        },
        {
            "name": "Reuters",
            "url": "https://www.reutersagency.com/feed/",
            "category": "News",
            "enabled": True,
        },
    ]
    added = 0
    existing_urls = {feed["url"] for feed in data.get("feeds", [])}
    for feed in curated:
        if feed["url"] not in existing_urls:
            data.setdefault("feeds", []).append(feed)
            added += 1
    console.print(Align.center(f"[green]Imported {added} curated feeds.[/green]"))


def manage_sources():
    json_file = "my_feeds.json"
    if os.path.exists(json_file):
        with open(json_file, "r", encoding="utf-8") as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                console.print(Align.center("[red]Error decoding JSON file.[/red]"))
                return
    else:
        data = {"feeds": []}

    while True:
        items = [
            "List sources",
            "Add source",
            "Remove source",
            "Toggle source status (interactive)",
            "Import curated feeds",
            "Return to main menu",
        ]
        choice = interactive_menu("Sources Management", items)
        if choice == -1 or choice == 5:
            break

        feeds = data.setdefault("feeds", [])
        if choice == 0:  # List sources
            if not feeds:
                console.print(Align.center("[red]No sources available.[/red]"))
            else:
                table = Table(title="Current RSS Sources")
                table.add_column("No.", justify="right")
                table.add_column("Name", style="cyan")
                table.add_column("URL", style="green")
                table.add_column("Category", style="magenta")
                table.add_column("Status", style="yellow")
                for idx, feed in enumerate(feeds, start=1):
                    status = "Enabled" if feed.get("enabled", True) else "Disabled"
                    table.add_row(
                        str(idx),
                        feed.get("name", "No Name"),
                        feed.get("url"),
                        feed.get("category"),
                        status,
                    )
                console.print(Align.center(table))

        elif choice == 1:  # Add source
            console.print()
            name = console.input("[bold magenta]Enter source name:[/bold magenta] ")
            category = console.input("[bold magenta]Enter category:[/bold magenta] ")
            url = console.input("[bold magenta]Enter RSS feed URL:[/bold magenta] ")
            new_feed = {"name": name, "category": category, "url": url, "enabled": True}
            feeds.append(new_feed)
            console.print(Align.center("[green]Source added.[/green]"))

        elif choice == 2:  # Remove source
            if not feeds:
                console.print(Align.center("[red]No sources available.[/red]"))
            else:
                table = Table(title="Remove RSS Source")
                table.add_column("No.", justify="right")
                table.add_column("Name", style="cyan")
                table.add_column("URL", style="green")
                for idx, feed in enumerate(feeds, start=1):
                    table.add_row(
                        str(idx), feed.get("name", "No Name"), feed.get("url")
                    )
                console.print(Align.center(table))
                console.print()
                selection = console.input(
                    "[bold magenta]Enter the number of the source to remove (or 'q' to cancel):[/bold magenta] "
                )
                if selection.lower() == "q":
                    pass
                else:
                    try:
                        index = int(selection) - 1
                        if 0 <= index < len(feeds):
                            removed = feeds.pop(index)
                            console.print(
                                Align.center(
                                    f"[green]Removed source: {removed.get('name')}[/green]"
                                )
                            )
                        else:
                            console.print(Align.center("[red]Invalid selection.[/red]"))
                    except ValueError:
                        console.print(Align.center("[red]Invalid input.[/red]"))

        elif choice == 3:  # Toggle source status
            interactive_toggle_sources(feeds)

        elif choice == 4:  # Import curated
            import_curated_feeds(data)

        # Save changes
        with open(json_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        console.print(Align.center("[bold green]Sources updated.[/bold green]"))


############################################################################
# TEST MODEL
############################################################################


def test_model():
    console.print(Align.center("[bold cyan]Loading model for testing...[/bold cyan]"))
    try:
        import torch
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
    except ImportError:
        console.print(
            Align.center("[red]Make sure transformers and torch are installed.[/red]")
        )
        return

    def get_latest_model_path(results_dir="./results"):
        if not os.path.exists(results_dir):
            console.print(
                Align.center(
                    f"[red]Results directory '{results_dir}' does not exist.[/red]"
                )
            )
            return None
        checkpoints = [
            os.path.join(results_dir, d)
            for d in os.listdir(results_dir)
            if os.path.isdir(os.path.join(results_dir, d))
            and d.startswith("checkpoint-")
        ]
        if not checkpoints:
            console.print(
                Align.center("[red]No trained model checkpoints found.[/red]")
            )
            return None
        latest_checkpoint = max(checkpoints, key=os.path.getctime)
        console.print(
            Align.center(
                f"[bold green]Loading latest model from:[/bold green] {latest_checkpoint}"
            )
        )
        return latest_checkpoint

    model_path = get_latest_model_path()
    if not model_path:
        console.print(
            Align.center(
                "[red]No model available for testing. Please train the model first.[/red]"
            )
        )
        return

    tokenizer = AutoTokenizer.from_pretrained("answerdotai/ModernBERT-base")
    model = AutoModelForSequenceClassification.from_pretrained(model_path)

    def classify_headline(headline):
        inputs = tokenizer(
            headline,
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            max_length=128,
        )
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            probabilities = torch.softmax(logits, dim=1)
            predicted_class = torch.argmax(probabilities).item()
        label_map = {0: "slop", 1: "quality"}
        return {
            "headline": headline,
            "predicted": label_map.get(predicted_class, str(predicted_class)),
            "confidence": probabilities[0][predicted_class].item(),
        }

    while True:
        console.print()
        headline = console.input(
            "[bold magenta]Enter a headline to classify (or 'q' to quit):[/bold magenta] "
        )
        if headline.lower() == "q":
            break
        result = classify_headline(headline)
        panel = Panel(
            f"[bold yellow]{result['headline']}[/bold yellow]\nPrediction: [bold green]{result['predicted']}[/bold green]\nConfidence: {result['confidence']:.2f}",
            title="Classification Result",
        )
        console.print(Align.center(panel))


############################################################################
# MAIN MENU (ARROW/j/k, NO FLICKER)
############################################################################


def main():
    while True:
        items = [
            "Create or Update Dataset",
            "Train Model",
            "Add or remove sources",
            "Test Model",
            "Exit",
        ]
        choice = interactive_menu("Main Menu", items)
        if choice == -1 or choice == 4:
            console.print(Align.center("[bold red]Exiting TUI.[/bold red]"))
            break
        elif choice == 0:
            create_or_update_dataset()
        elif choice == 1:
            train_model()
        elif choice == 2:
            manage_sources()
        elif choice == 3:
            test_model()


if __name__ == "__main__":
    main()
