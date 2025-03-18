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
from time import sleep
from concurrent.futures import ThreadPoolExecutor

from rich.console import Console, Group
from rich.table import Table
from rich.panel import Panel
from rich.align import Align
from rich.live import Live
from rich.layout import Layout
from rich.prompt import Prompt

console = Console()

app_title = """
            ███████╗██╗      ██████╗ ██████╗ ██████╗ ███████╗██████╗ ████████╗
            ██╔════╝██║     ██╔═══██╗██╔══██╗██╔══██╗██╔════╝██╔══██╗╚══██╔══╝
            ███████╗██║     ██║   ██║██████╔╝██████╔╝█████╗  ██████╔╝   ██║   
            ╚════██║██║     ██║   ██║██╔═══╝ ██╔══██╗██╔══╝  ██╔══██╗   ██║   
            ███████║███████╗╚██████╔╝██║     ██████╔╝███████╗██║  ██║   ██║   
            ╚══════╝╚══════╝ ╚═════╝ ╚═╝     ╚═════╝ ╚══════╝╚═╝  ╚═╝   ╚═╝   
            """

############################################################################
# KEY CAPTURE (ARROW, j/k, space, q, enter)
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
    first = get_single_key_raw()
    if not first:
        return ""
    if first in (b"\r", b"\n"):
        return "enter"
    if first == b"\x1b":
        rest = os.read(sys.stdin.fileno(), 2) if sys.stdin.isatty() else b""
        if rest in (b"[A", b"OA"):
            return "up"
        elif rest in (b"[B", b"OB"):
            return "down"
        elif rest in (b"[C", b"OC"):
            return "right"
        elif rest in (b"[D", b"OD"):
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
# INTERACTIVE MENU (CENTERED)
############################################################################
def interactive_menu_in_layout(layout, live, title: str, items: list[str]) -> int:
    position = 0
    while True:
        table = Table(show_header=False, box=None, pad_edge=True)
        table.add_column("", justify="right", width=2)
        table.add_column("Menu", justify="left", style="bold")
        for i, item in enumerate(items):
            if i == position:
                table.add_row(
                    "[bold magenta]>[/bold magenta]", f"[reverse]{item}[/reverse]"
                )
            else:
                table.add_row("", item)
        centered_menu = Align.center(table, vertical="middle")
        body_panel = Panel(centered_menu, border_style="green", expand=True)
        layout["body"].update(body_panel)
        live.refresh()
        key = read_key_sequence()
        if key == "up":
            position = (position - 1) % len(items)
        elif key == "down":
            position = (position + 1) % len(items)
        elif key == "enter":
            return position
        elif key == "q":
            return -1


############################################################################
# HELPER: Wait for any key before returning to main menu
############################################################################
def wait_for_key(layout, live, prompt="Press any key to return to the main menu"):
    panel = Panel(Align.center(f"[bold cyan]{prompt}[/bold cyan]"), expand=True)
    layout["body"].update(panel)
    live.refresh()
    read_key_sequence()


############################################################################
# RSS FEED, CACHING, AND RATING LOGIC
############################################################################
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
        return []
    with open(file_path, "r", encoding="utf-8") as f:
        try:
            data = json.load(f)
            feeds = data.get("feeds", [])
            return [feed for feed in feeds if feed.get("enabled", True)]
        except json.JSONDecodeError:
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


def get_status_bar_text(target=10):
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
    table = Table(show_header=True, header_style="bold magenta")
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
    return table


############################################################################
# WRAPPER FUNCTIONS FOR SUB-SCREENS
############################################################################
def run_create_or_update_dataset(layout, live):
    feeds = load_feeds_from_json("my_feeds.json")
    if not feeds:
        layout["body"].update(
            Panel(
                Align.center(
                    "[red]No feeds available. Please add some sources first.[/red]"
                ),
                expand=True,
            )
        )
        live.refresh()
        wait_for_key(layout, live)
        return

    cache = load_cache()

    def fetch_articles(feed):
        msg = f"[bold cyan]Fetching articles from:[/bold cyan] {feed.get('name', '?')} ([green]{feed.get('url')}[/green])"
        layout["body"].update(
            Panel(Align.center(msg), expand=True, border_style="cyan")
        )
        live.refresh()
        return get_articles_from_feed(feed["url"], feed["category"])

    max_workers = min(len(feeds), 10)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_feed = {executor.submit(fetch_articles, feed): feed for feed in feeds}
        for future in future_to_feed:
            feed = future_to_feed[future]
            try:
                new_articles = future.result()
            except Exception:
                new_articles = []
            for article in new_articles:
                if not any(article["title"] == c.get("title") for c in cache):
                    cache.append(article)

    rated_titles = set()
    if os.path.exists("dataset.csv"):
        with open("dataset.csv", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                rated_titles.add(row["text"])
    cache = [a for a in cache if a.get("title") not in rated_titles]

    if not cache:
        layout["body"].update(
            Panel(Align.center("[green]No new headlines to rate.[/green]"), expand=True)
        )
        live.refresh()
        wait_for_key(layout, live)
        return

    random.shuffle(cache)
    dataset_file = "dataset.csv"
    with open(dataset_file, "a", newline="", encoding="utf-8") as f:
        fieldnames = ["text", "label"]
        writer = csv.DictWriter(f, fieldnames=fieldnames, quoting=csv.QUOTE_NONNUMERIC)
        if os.stat(dataset_file).st_size == 0:
            writer.writeheader()

        # Adjusted mapping: 0-1 become "slop", 2-5 become "quality"
        key_to_rating = {
            "0": "slop",
            "1": "meh",
            "2": "ok",
            "3": "not bad",
            "4": "good stuff",
            "5": "banger",
            "a": "slop",
            "A": "slop",
            "s": "meh",
            "S": "meh",
            "d": "ok",
            "D": "ok",
            "f": "not bad",
            "F": "not bad",
            "g": "good stuff",
            "G": "good stuff",
            "h": "banger",
            "H": "banger",
        }
        TARGET_COUNT = 10
        idx = 0
        while idx < len(cache):
            article = cache[idx]
            headline = article.get("title")
            headline_panel = Panel(
                f"[bold yellow]{headline}[/bold yellow]\n[dim]Source:[/dim] {article.get('source')} | [dim]Category:[/dim] {article.get('category')}",
                title="New Headline",
                subtitle="[grey](Press 0-5 or a,s,d,f,g,h to rate, q to quit)[/grey]",
                expand=True,
            )
            status_table = get_status_bar_text(TARGET_COUNT)
            rating_panel = Panel(
                Align.center(status_table),
                title="Rating Progress",
                border_style="magenta",
                expand=True,
            )
            sub_layout = Layout()
            sub_layout.split_column(
                Layout(name="headline", ratio=1),
                Layout(name="rating", size=12),
            )
            sub_layout["headline"].update(
                Align.center(headline_panel, vertical="middle")
            )
            sub_layout["rating"].update(Align.center(rating_panel))
            layout["body"].update(Panel(sub_layout, border_style="green", expand=True))
            live.refresh()
            key = read_key_sequence()
            if key.lower() == "q":
                layout["body"].update(
                    Panel(
                        Align.center("[red]Quitting dataset creation.[/red]"),
                        expand=True,
                    )
                )
                live.refresh()
                save_cache(cache)
                wait_for_key(layout, live)
                return
            if key in key_to_rating:
                rating = key_to_rating[key]
                writer.writerow({"text": headline, "label": rating})
                layout["body"].update(
                    Panel(
                        Align.center(f"[green]Rating '{rating}' saved.[/green]"),
                        expand=True,
                    )
                )
                live.refresh()
                cache.pop(idx)
                # save_cache(cache)
                # sleep(0.5)
                # -----------
                # Display rated headline alongside the rating
                # -----------
                # Build a new panel that includes the rating text
                rated_headline_panel = Panel(
                    f"[bold yellow]{headline}[/bold yellow]\n"
                    f"[dim]Source:[/dim] {article.get('source')} | [dim]Category:[/dim] {article.get('category')}\n\n"
                    f"[bold green]Rating: {rating}[/bold green]",
                    title="New Headline",
                    subtitle="[grey](Press 0-5 or a,s,d,f,g,h to rate, q to quit)[/grey]",
                    expand=True,
                )

                # Build the status table panel
                status_table = get_status_bar_text(TARGET_COUNT)
                rating_panel = Panel(
                    Align.center(status_table),
                    title="Rating Progress",
                    border_style="magenta",
                    expand=True,
                )

                # Create a sub-layout that displays both the new "rated" headline and the rating panel
                sub_layout = Layout()
                sub_layout.split_column(
                    Layout(name="headline", ratio=1),
                    Layout(name="rating", size=12),
                )
                sub_layout["headline"].update(
                    Align.center(rated_headline_panel, vertical="middle")
                )
                sub_layout["rating"].update(Align.center(rating_panel))
                layout["body"].update(
                    Panel(sub_layout, border_style="green", expand=True)
                )
                live.refresh()

                # Save the updated cache and pause briefly so the user can see the rating
                save_cache(cache)
                sleep(1.0)  # 1 second pause to show the rating
            else:
                continue
    layout["body"].update(
        Panel(
            Align.center(
                "[bold green]All cached headlines have been rated.[/bold green]"
            ),
            expand=True,
        )
    )
    live.refresh()
    wait_for_key(layout, live)


def run_train_model(layout, live):
    layout["body"].update(
        Panel(
            Align.center("[bold cyan]Starting model training...[/bold cyan]"),
            expand=True,
        )
    )
    live.refresh()
    cmd = ["python", "train.py"]
    process = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
    )
    lines = []
    max_lines = 20
    while True:
        line = process.stdout.readline()
        if not line and process.poll() is not None:
            break
        if line:
            line = line.rstrip("\n")
            lines.append(line)
            if len(lines) > max_lines:
                lines.pop(0)
            log_text = "\n".join(lines)
            block = Align.center(log_text, vertical="middle")
            layout["body"].update(Panel(block, border_style="green", expand=True))
            live.refresh()
    retcode = process.poll()
    if retcode == 0:
        msg = "[green]Training completed successfully.[/green]"
    else:
        msg = "[red]Training encountered an error.[/red]"
    layout["body"].update(Panel(Align.center(msg), expand=True))
    live.refresh()
    wait_for_key(layout, live)


def run_manage_sources(layout, live):
    json_file = "my_feeds.json"
    if os.path.exists(json_file):
        try:
            with open(json_file, "r", encoding="utf-8") as f:
                data = json.load(f)
        except json.JSONDecodeError:
            data = {"feeds": []}
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
        choice = interactive_menu_in_layout(layout, live, "Sources Management", items)
        if choice == -1 or choice == 5:
            break
        feeds = data.setdefault("feeds", [])
        if choice == 0:
            if not feeds:
                panel = Panel(
                    Align.center("[red]No sources available.[/red]"), expand=True
                )
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
                panel = Panel(table, expand=True)
            layout["body"].update(panel)
            live.refresh()
            wait_for_key(layout, live)
        elif choice == 1:
            layout["body"].update(
                Panel(
                    Align.center(
                        "[bold magenta]Enter source name in the terminal prompt.[/bold magenta]"
                    ),
                    expand=True,
                )
            )
            live.refresh()
            name = console.input("[bold magenta]Enter source name:[/bold magenta] ")
            category = console.input("[bold magenta]Enter category:[/bold magenta] ")
            url = console.input("[bold magenta]Enter RSS feed URL:[/bold magenta] ")
            new_feed = {"name": name, "category": category, "url": url, "enabled": True}
            feeds.append(new_feed)
            layout["body"].update(
                Panel(Align.center("[green]Source added.[/green]"), expand=True)
            )
            live.refresh()
            wait_for_key(layout, live)
        elif choice == 2:
            if not feeds:
                layout["body"].update(
                    Panel(Align.center("[red]No sources available.[/red]"), expand=True)
                )
                live.refresh()
                wait_for_key(layout, live)
            else:
                table = Table(title="Remove RSS Source")
                table.add_column("No.", justify="right")
                table.add_column("Name", style="cyan")
                table.add_column("URL", style="green")
                for idx, feed in enumerate(feeds, start=1):
                    table.add_row(
                        str(idx), feed.get("name", "No Name"), feed.get("url")
                    )
                layout["body"].update(Panel(table, expand=True))
                live.refresh()
                selection = console.input(
                    "[bold magenta]Enter the number of the source to remove (or 'q' to cancel):[/bold magenta] "
                )
                if selection.lower() != "q":
                    try:
                        index = int(selection) - 1
                        if 0 <= index < len(feeds):
                            removed = feeds.pop(index)
                            layout["body"].update(
                                Panel(
                                    Align.center(
                                        f"[green]Removed source: {removed.get('name')}[/green]"
                                    ),
                                    expand=True,
                                )
                            )
                        else:
                            layout["body"].update(
                                Panel(
                                    Align.center("[red]Invalid selection.[/red]"),
                                    expand=True,
                                )
                            )
                    except ValueError:
                        layout["body"].update(
                            Panel(
                                Align.center("[red]Invalid input.[/red]"), expand=True
                            )
                        )
                    live.refresh()
                    wait_for_key(layout, live)
        elif choice == 3:
            run_toggle_sources(feeds, layout, live)
        elif choice == 4:
            run_import_curated_feeds(data, layout, live)
        with open(json_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        layout["body"].update(
            Panel(
                Align.center("[bold green]Sources updated.[/bold green]"), expand=True
            )
        )
        live.refresh()
        wait_for_key(layout, live)


def run_toggle_sources(feeds, layout, live):
    if not feeds:
        layout["body"].update(
            Panel(Align.center("[red]No sources available.[/red]"), expand=True)
        )
        live.refresh()
        wait_for_key(layout, live)
        return
    position = 0
    while True:
        table = Table(show_header=True, header_style="bold magenta")
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
        layout["body"].update(Panel(table, border_style="green", expand=True))
        live.refresh()
        key = read_key_sequence()
        if key in ("up", "k"):
            position = max(0, position - 1)
        elif key in ("down", "j"):
            position = min(len(feeds) - 1, position + 1)
        elif key == "space":
            feeds[position]["enabled"] = not feeds[position].get("enabled", True)
        elif key.lower() == "q":
            break


def run_import_curated_feeds(data, layout, live):
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
    layout["body"].update(
        Panel(
            Align.center(f"[green]Imported {added} curated feeds.[/green]"), expand=True
        )
    )
    live.refresh()
    wait_for_key(layout, live)


def classify_headline(headline, tokenizer, model):
    inputs = tokenizer(
        headline,
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=128,
    )
    import torch

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probabilities = torch.softmax(logits, dim=1)
        predicted_class = torch.argmax(probabilities).item()
    # Adjusted mapping: 0-1 => "slop", 2-5 => "quality"
    label_map = {
        0: "slop",
        1: "meh",
        2: "ok",
        3: "not bad",
        4: "good stuff",
        5: "banger",
    }
    return {
        "headline": headline,
        "predicted": label_map.get(predicted_class, str(predicted_class)),
        "confidence": probabilities[0][predicted_class].item(),
    }


def get_latest_model_path(results_dir="./results"):
    if not os.path.exists(results_dir):
        return None
    checkpoints = [
        os.path.join(results_dir, d)
        for d in os.listdir(results_dir)
        if os.path.isdir(os.path.join(results_dir, d)) and d.startswith("checkpoint-")
    ]
    if not checkpoints:
        return None
    latest_checkpoint = max(checkpoints, key=os.path.getctime)
    return latest_checkpoint


def run_test_model(layout, live):
    layout["body"].update(
        Panel(
            Align.center("[bold cyan]Loading model for testing...[/bold cyan]"),
            expand=True,
        )
    )
    live.refresh()
    try:
        import torch
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
    except ImportError:
        layout["body"].update(
            Panel(
                Align.center(
                    "[red]Make sure transformers and torch are installed.[/red]"
                ),
                expand=True,
            )
        )
        live.refresh()
        wait_for_key(layout, live)
        return
    model_path = get_latest_model_path()
    if not model_path:
        layout["body"].update(
            Panel(
                Align.center("[red]No trained model checkpoints found.[/red]"),
                expand=True,
            )
        )
        live.refresh()
        wait_for_key(layout, live)
        return
    tokenizer = AutoTokenizer.from_pretrained("answerdotai/ModernBERT-base")
    from transformers import AutoModelForSequenceClassification

    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    results_list = []
    max_results = 10
    while True:
        lines = [
            f"[bold yellow]{r['headline']}[/bold yellow] => [bold green]{r['predicted']}[/bold green] ({r['confidence']:.2f})"
            for r in results_list
        ]
        results_text = "\n".join(lines) if lines else "[dim]No predictions yet.[/dim]"
        sub_layout = Layout()
        sub_layout.split_column(
            Layout(name="results", ratio=2),
            Layout(name="prompt", size=5),
        )
        sub_layout["results"].update(
            Panel(
                Align.center(results_text, vertical="top"),
                title="Classification History",
                border_style="blue",
                expand=True,
            )
        )
        # Use Prompt.ask so the user input appears inside the Live panel.
        sub_layout["prompt"].update(
            Panel(
                Align.center(
                    "[bold magenta]Enter a headline to classify (or 'q' to quit):[/bold magenta]"
                ),
                expand=True,
            )
        )
        layout["body"].update(Panel(sub_layout, border_style="green", expand=True))
        live.refresh()
        headline = Prompt.ask("")
        if headline.lower() == "q":
            break
        result = classify_headline(headline, tokenizer, model)
        results_list.append(result)
        if len(results_list) > max_results:
            results_list.pop(0)
    wait_for_key(layout, live, prompt="Press any key to return to the main menu...")


############################################################################
# MAIN APPLICATION (Global Layout Always Present)
############################################################################
def main():
    layout = Layout()
    layout.split(
        Layout(name="header", size=10),
        Layout(name="body"),
        Layout(name="footer", size=3),
    )
    layout["header"].update(
        Panel(Align.center(app_title), border_style="cyan", expand=True)
    )
    layout["footer"].update(
        Panel(
            Align.center(
                "[bold]Use arrow keys (or j/k) to navigate; Enter to select; q to quit an option[/bold]"
            ),
            expand=True,
        )
    )
    main_menu_items = [
        "Create or Update Dataset",
        "Train Model",
        "Add or remove sources",
        "Test Model",
        "Exit",
    ]
    with Live(layout, console=console, auto_refresh=False, screen=True) as live:
        while True:
            choice = interactive_menu_in_layout(
                layout, live, "Main Menu", main_menu_items
            )
            if choice == -1 or choice == 4:
                layout["body"].update(
                    Panel(
                        Align.center("[bold red]Exiting slopbert.[/bold red]"),
                        expand=True,
                    )
                )
                live.refresh()
                sleep(1)
                break
            elif choice == 0:
                run_create_or_update_dataset(layout, live)
            elif choice == 1:
                run_train_model(layout, live)
            elif choice == 2:
                run_manage_sources(layout, live)
            elif choice == 3:
                run_test_model(layout, live)
            layout["body"].update(
                Panel(
                    Align.center("[bold cyan]Returning to Main Menu...[/bold cyan]"),
                    expand=True,
                )
            )
            live.refresh()
            sleep(1)


if __name__ == "__main__":
    main()
