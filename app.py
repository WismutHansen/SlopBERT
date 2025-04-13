# app.py
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
from concurrent.futures import (
    ThreadPoolExecutor,
    as_completed,
    TimeoutError,
)  # Added as_completed, TimeoutError
from ast import literal_eval  # For parsing training output

from rich.console import Console, Group
from rich.table import Table
from rich.panel import Panel
from rich.align import Align
from rich.live import Live
from rich.layout import Layout
from rich.prompt import Prompt

# --- Added Imports for LoRA/Torch ---
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

try:
    from peft import PeftModel

    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False
# --- End Added Imports ---

console = Console()

# --- Configuration Constants ---
BASE_MODEL_NAME = "answerdotai/ModernBERT-base"
FULL_RESULTS_DIR = "./results"
LORA_RESULTS_DIR = "./results_lora"  # Directory for LoRA checkpoints/adapters
NUM_LABELS = 6  # 0-5
MAX_LENGTH = 128
# Adjusted label map for 0-5 scale
LABEL_MAP = {
    0: "slop",
    1: "meh",
    2: "ok",
    3: "not bad",
    4: "good stuff",
    5: "banger",
}
CACHE_FILE = "headlines_cache.json"
FEEDS_FILE = "my_feeds.json"
DATASET_FILE = "dataset.csv"
TARGET_RATING_COUNT = 50  # Target count per rating for status bar
FETCH_TIMEOUT_SECONDS = 25  # Timeout for fetching each RSS feed

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
    import tty, termios

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
        # Check if stdin is a TTY before attempting to read more (fixes issues in some environments)
        rest = os.read(sys.stdin.fileno(), 2) if sys.stdin.isatty() else b""
        if rest == b"[A":
            return "up"
        elif rest == b"[B":
            return "down"
        elif rest == b"[C":
            return "right"
        elif rest == b"[D":
            return "left"
        # Handle alternative escape sequences if necessary (e.g., from different terminals)
        # elif rest == b'OA': return "up" # Example for some terminals
        # elif rest == b'OB': return "down"
        return ""  # Unrecognized escape sequence
    else:
        try:
            ch = first.decode(errors="ignore")
            if ch == "j":
                return "down"
            elif ch == "k":
                return "up"
            elif ch == " ":
                return "space"
            elif ch == "q":
                return "q"
            return ch  # Return other characters if needed, otherwise ignore
        except UnicodeDecodeError:
            return ""  # Ignore bytes that cannot be decoded


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
                    "[bold magenta]>[/bold magenta]",
                    f"[reverse blue]{item}[/reverse blue]",
                )  # Use different highlight
            else:
                table.add_row("", item)
        centered_menu = Align.center(table, vertical="middle")
        body_panel = Panel(
            centered_menu, border_style="blue", expand=True, title=title
        )  # Use different border
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
            return -1  # Allow quitting the menu itself


############################################################################
# HELPER: Wait for any key before returning to main menu
############################################################################
def wait_for_key(layout, live, prompt="Press any key to return to the main menu"):
    panel = Panel(
        Align.center(f"[bold cyan]{prompt}[/bold cyan]"),
        expand=True,
        border_style="dim",
    )
    layout["body"].update(panel)
    live.refresh()
    read_key_sequence()  # Wait for any key press


############################################################################
# RSS FEED, CACHING, AND DATASET HELPERS
############################################################################


def load_cache():
    """Loads headlines cache from CACHE_FILE."""
    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            console.print(f"[red]Error loading cache file {CACHE_FILE}: {e}[/red]")
            return []
    return []


def save_cache(cache):
    """Saves headlines cache to CACHE_FILE."""
    try:
        with open(CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump(cache, f, indent=2)
    except IOError as e:
        console.print(f"[red]Error saving cache file {CACHE_FILE}: {e}[/red]")


def load_feeds_from_json(file_path=FEEDS_FILE):
    """Loads enabled RSS feeds configuration from JSON file."""
    if not os.path.isfile(file_path):
        console.print(f"[yellow]Feeds file '{file_path}' not found.[/yellow]")
        return []
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            feeds = data.get("feeds", [])
            # Filter for enabled feeds and ensure essential keys exist
            enabled_feeds = [
                feed
                for feed in feeds
                if feed.get("enabled", True)
                and feed.get("url")
                and feed.get("category")
            ]
            if len(enabled_feeds) < len(feeds):
                console.print(
                    f"[dim]Filtered out {len(feeds) - len(enabled_feeds)} disabled or incomplete feeds.[/dim]"
                )
            return enabled_feeds
    except (json.JSONDecodeError, IOError) as e:
        console.print(f"[red]Error loading feeds file {file_path}: {e}[/red]")
        return []


def get_articles_from_feed(url, category):
    """Fetches and parses articles from a single RSS feed URL for today."""
    articles_today = []
    try:
        # Use User-Agent to avoid potential blocks
        feed_data = feedparser.parse(url, agent="SlopBERT/1.0")

        if feed_data.bozo:
            # Log bozo errors (parsing issues) but don't necessarily stop
            bozo_msg = feed_data.bozo_exception
            # console.print(f"[yellow]Feedparser warning for {url}: {bozo_msg}[/yellow]")
            pass  # Continue processing entries even if there are warnings

        today = datetime.datetime.now(pytz.utc).date()
        ext = tldextract.extract(url)
        source_domain = (
            f"{ext.domain}.{ext.suffix}" if ext.domain and ext.suffix else url
        )  # Handle cases where extraction fails

        for entry in feed_data.entries:
            title = entry.get("title", "").strip()
            link = entry.get("link", "")

            # Skip entries without titles
            if not title:
                continue

            # Determine publication date (handle various fields)
            published_dt = None
            date_keys = ["published_parsed", "updated_parsed", "created_parsed"]
            for key in date_keys:
                parsed_time = entry.get(key)
                if parsed_time:
                    try:
                        # Ensure it's a valid time tuple before creating datetime
                        if len(parsed_time) >= 6:
                            dt_naive = datetime.datetime(*parsed_time[:6])
                            # Assume UTC if no timezone info, otherwise make aware
                            published_dt = (
                                pytz.utc.localize(dt_naive)
                                if dt_naive.tzinfo is None
                                else dt_naive.astimezone(pytz.utc)
                            )
                            break  # Use the first valid date found
                    except (ValueError, TypeError):
                        continue  # Ignore invalid date tuples

            # If a valid date was found and it's today, add the article
            if published_dt and published_dt.date() == today:
                articles_today.append(
                    {
                        "title": title,
                        "link": link,
                        "published": published_dt.isoformat(),
                        "category": category,
                        "source": source_domain,
                    }
                )

    except Exception as e:
        # Log errors during feed fetching/parsing for specific URL
        console.print(f"[red]Error processing feed {url}: {e}[/red]")

    return articles_today


def get_status_bar_text(target=TARGET_RATING_COUNT):
    """Generates a Rich Table summarizing rating progress."""
    counts = {str(i): 0 for i in range(NUM_LABELS)}
    total_count = 0
    if os.path.exists(DATASET_FILE):
        try:
            with open(DATASET_FILE, newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    try:
                        label_int = int(row["label"])
                        if 0 <= label_int < NUM_LABELS:
                            counts[str(label_int)] += 1
                            total_count += 1
                    except (ValueError, KeyError, TypeError):
                        pass  # Ignore rows with invalid/missing labels
        except (IOError, csv.Error) as e:
            console.print(f"[red]Error reading dataset for status: {e}[/red]")

    table = Table(
        show_header=True,
        header_style="bold magenta",
        title=f"Rating Counts (Target per rating: {target})",
        box=None,
    )
    table.add_column("Rating", justify="right", style="cyan", width=12)
    table.add_column("Count", justify="center", style="yellow", width=6)
    table.add_column("Progress", justify="left")

    for i in range(NUM_LABELS):
        count = counts[str(i)]
        progress_percent = min(count / target, 1.0) if target > 0 else 0
        filled_width = int(progress_percent * 15)  # Width of the progress bar
        bar = (
            "[green]"
            + "█" * filled_width
            + "[/green]"
            + "[dim]"
            + "░" * (15 - filled_width)
            + "[/dim]"
        )
        percentage = int(progress_percent * 100)
        label_name = LABEL_MAP.get(i, f"Label {i}")
        table.add_row(f"{label_name} ({i})", str(count), f"{bar} {percentage}%")

    # Add a separator and total row
    table.add_row("─" * 12, "─" * 6, "─" * 25)
    table.add_row("[bold]Total[/bold]", f"[bold]{total_count}[/bold]", "")
    return table


############################################################################
# DATASET CREATION (with improved fetching)
############################################################################
def run_create_or_update_dataset(layout, live):
    feeds = load_feeds_from_json(FEEDS_FILE)
    if not feeds:
        layout["body"].update(
            Panel(
                Align.center(
                    "[red]No enabled feeds found in 'my_feeds.json'. Add sources first.[/red]"
                ),
                expand=True,
            )
        )
        live.refresh()
        wait_for_key(layout, live)
        return

    cache = load_cache()

    # --- Modified Fetching Logic ---
    layout["body"].update(
        Panel(
            Align.center("[cyan]Initiating feed fetching...[/cyan]"),
            border_style="cyan",
            expand=True,
        )
    )
    live.refresh()
    sleep(0.5)  # Brief pause

    def fetch_articles_thread_wrapper(feed):
        # Wrapper to handle return value for the thread pool
        return feed, get_articles_from_feed(feed["url"], feed["category"])

    new_articles_list = []
    fetch_status = {}  # Store status per feed (using name as key)
    total_feeds = len(feeds)
    completed_count = 0
    failed_count = 0
    newly_fetched_count = 0

    max_workers = min(total_feeds, 12)  # Slightly increase max workers
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_feed = {
            executor.submit(fetch_articles_thread_wrapper, feed): feed for feed in feeds
        }

        for future in as_completed(future_to_feed):
            original_feed_info = future_to_feed[future]  # Get the original dict
            name = original_feed_info.get(
                "name", original_feed_info.get("url", "Unknown Feed")
            )
            url = original_feed_info.get("url", "Unknown URL")
            completed_count += 1

            # --- Update UI Progress ---
            status_lines = [
                f"[bold]Fetching feed {completed_count}/{total_feeds}:[/] [cyan]{name}[/cyan]"
            ]
            # Display status of already processed feeds, most recent first
            processed_feeds_display = list(fetch_status.items())[
                -5:
            ]  # Show last 5 statuses
            for feed_name_disp, status_disp in reversed(processed_feeds_display):
                status_lines.append(f"  - {feed_name_disp}: {status_disp}")
            if len(fetch_status) > 5:
                status_lines.append("  [dim]...[/dim]")

            remaining = total_feeds - completed_count
            if remaining > 0:
                status_lines.append(f"\n[dim]{remaining} feeds pending...[/dim]")

            progress_panel = Panel(
                Align.left("\n".join(status_lines)),
                title="Fetching Progress",
                border_style="cyan",
                expand=True,
            )
            layout["body"].update(progress_panel)
            live.refresh()
            # --- End UI Update ---

            try:
                # Get result with timeout
                _, articles = future.result(
                    timeout=FETCH_TIMEOUT_SECONDS
                )  # Unpack result
                new_articles_list.extend(articles)
                fetch_status[name] = f"[green]OK ({len(articles)} new)[/green]"
                newly_fetched_count += len(articles)
            except TimeoutError:
                failed_count += 1
                fetch_status[name] = "[yellow]TIMEOUT[/yellow]"
            except Exception as e:
                failed_count += 1
                fetch_status[name] = f"[red]ERROR[/red]"
                # Log error discreetly
                console.print(f"[dim red]Fetch error for {name}: {e}[/dim red]")

    # --- Display Final Fetch Summary ---
    summary_lines = ["[bold blue]Fetch Summary:[/bold blue]"]
    summary_lines.append(f"Total Feeds Processed: {total_feeds}")
    summary_lines.append(f"Successful: {total_feeds - failed_count}")
    summary_lines.append(f"Failed/Timeout: {failed_count}")
    summary_lines.append(f"Total New Articles Found: {newly_fetched_count}")
    summary_lines.append("-" * 40)
    # Display status for all feeds in the summary
    for feed_name_sum, status_sum in fetch_status.items():
        summary_lines.append(f"- {feed_name_sum}: {status_sum}")

    summary_panel = Panel(
        Align.left("\n".join(summary_lines)),
        title="Fetching Complete",
        border_style="blue",
        expand=True,
    )
    layout["body"].update(summary_panel)
    live.refresh()
    wait_for_key(layout, live, prompt="Press any key to continue to rating...")
    # --- End Fetching Logic Modification ---

    # --- Update Cache & Filter ---
    existing_titles_in_cache = {c.get("title") for c in cache if c.get("title")}
    unique_new_articles = [
        a
        for a in new_articles_list
        if a.get("title") and a["title"] not in existing_titles_in_cache
    ]
    cache.extend(unique_new_articles)
    save_cache(cache)  # Save cache immediately after fetching
    console.print(
        f"[green]Added {len(unique_new_articles)} unique headlines to cache.[/green]"
    )

    # Filter out already rated headlines from the dataset file
    rated_titles_in_dataset = set()
    if os.path.exists(DATASET_FILE):
        try:
            with open(DATASET_FILE, newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if "text" in row and row["text"]:
                        rated_titles_in_dataset.add(row["text"])
        except (IOError, csv.Error) as e:
            console.print(
                f"[red]Error reading dataset {DATASET_FILE} for filtering: {e}[/red]"
            )

    # Keep only articles from cache that are NOT in the dataset file
    headlines_to_rate = [
        a for a in cache if a.get("title") and a["title"] not in rated_titles_in_dataset
    ]

    if not headlines_to_rate:
        layout["body"].update(
            Panel(
                Align.center(
                    "[green]No new headlines left to rate (already in dataset or cache).[/green]"
                ),
                expand=True,
            )
        )
        live.refresh()
        wait_for_key(layout, live)
        return

    random.shuffle(headlines_to_rate)  # Shuffle the list to be rated

    # --- Rating Logic ---
    try:
        # Open dataset in append mode
        with open(DATASET_FILE, "a", newline="", encoding="utf-8") as f_dataset:
            fieldnames = ["text", "label"]
            writer = csv.DictWriter(
                f_dataset, fieldnames=fieldnames, quoting=csv.QUOTE_NONNUMERIC
            )
            # Check if file is empty to write header (robustly)
            f_dataset.seek(0, os.SEEK_END)  # Go to end of file
            if f_dataset.tell() == 0:  # Check if file size is 0
                writer.writeheader()

            # Key mapping for 0-5 ratings
            key_to_rating = {str(i): i for i in range(NUM_LABELS)}
            key_to_rating.update({"a": 0, "s": 1, "d": 2, "f": 3, "g": 4, "h": 5})

            current_index = 0
            while current_index < len(headlines_to_rate):
                article = headlines_to_rate[current_index]
                headline = article.get("title", "").strip()
                source = article.get("source", "Unknown")
                category = article.get("category", "Unknown")

                # Skip if headline is empty after stripping
                if not headline:
                    current_index += 1
                    continue

                # Headline Panel
                headline_text = (
                    f"[bold yellow]{headline}[/bold yellow]\n"
                    f"[dim]Source:[/dim] {source} | [dim]Category:[/dim] {category}"
                )
                headline_panel = Panel(
                    Align.center(headline_text, vertical="middle"),
                    title=f"Rating Headline {current_index + 1} of {len(headlines_to_rate)}",
                    subtitle="Rate ([cyan]0-5[/cyan] or [cyan]a-h[/cyan]), [red]q[/red]=quit",
                    expand=True,
                    border_style="yellow",
                )

                # Status Bar Panel
                status_table = get_status_bar_text(TARGET_RATING_COUNT)
                rating_panel = Panel(
                    Align.center(status_table),
                    title="Rating Progress",
                    border_style="magenta",
                    expand=True,
                )

                # Combined Layout
                sub_layout = Layout()
                sub_layout.split_column(
                    Layout(headline_panel, name="headline", ratio=1),
                    Layout(
                        rating_panel, name="rating", size=10 + NUM_LABELS
                    ),  # Adjust size
                )
                layout["body"].update(
                    Panel(sub_layout, border_style="green", expand=True)
                )
                live.refresh()

                # Get Rating Input
                key = read_key_sequence().lower()

                if key == "q":
                    layout["body"].update(
                        Panel(
                            Align.center("[red]Quitting dataset rating.[/red]"),
                            expand=True,
                        )
                    )
                    live.refresh()
                    wait_for_key(layout, live)
                    return  # Exit function

                if key in key_to_rating:
                    rating = key_to_rating[key]
                    writer.writerow({"text": headline, "label": rating})
                    f_dataset.flush()  # Ensure data is written to disk

                    # Remove the rated item from the *original cache* as well to prevent re-rating later if dataset is cleared
                    # Find the corresponding item in the main cache list and remove it
                    for i, cache_item in enumerate(cache):
                        if cache_item.get("title") == headline:
                            cache.pop(i)
                            save_cache(cache)  # Update the cache file
                            break

                    rated_label_name = LABEL_MAP.get(rating, "Unknown")

                    # Show rated confirmation
                    rated_headline_text = (
                        f"{headline_text}\n\n"
                        f"[bold green]Rated as: {rated_label_name} ({rating})[/bold green]"
                    )
                    headline_panel_rated = Panel(
                        Align.center(rated_headline_text, vertical="middle"),
                        title=f"Headline {current_index + 1} of {len(headlines_to_rate)}",
                        subtitle="Rate ([cyan]0-5[/cyan] or [cyan]a-h[/cyan]), [red]q[/red]=quit",
                        expand=True,
                        border_style="green",  # Change border
                    )
                    status_table_updated = get_status_bar_text(TARGET_RATING_COUNT)
                    rating_panel_updated = Panel(
                        Align.center(status_table_updated),
                        title="Rating Progress",
                        border_style="magenta",
                        expand=True,
                    )
                    sub_layout["headline"].update(headline_panel_rated)
                    sub_layout["rating"].update(rating_panel_updated)
                    layout["body"].update(
                        Panel(sub_layout, border_style="green", expand=True)
                    )
                    live.refresh()
                    sleep(0.7)  # Brief pause

                    current_index += 1  # Move to the next headline
                else:
                    # Invalid key, just loop again for the same headline
                    continue

            # End of rating loop
            layout["body"].update(
                Panel(
                    Align.center(
                        "[bold green]All new headlines rated and saved.[/bold green]"
                    ),
                    expand=True,
                )
            )
            live.refresh()
            wait_for_key(layout, live)

    except IOError as e:
        layout["body"].update(
            Panel(
                Align.center(
                    f"[red]Error writing to dataset file {DATASET_FILE}: {e}[/red]"
                ),
                expand=True,
            )
        )
        live.refresh()
        wait_for_key(layout, live)
    except Exception as e:
        console.print_exception()  # Print full traceback for unexpected errors
        layout["body"].update(
            Panel(
                Align.center(
                    f"[red]An unexpected error occurred during rating: {e}[/red]"
                ),
                expand=True,
            )
        )
        live.refresh()
        wait_for_key(layout, live)


############################################################################
# TRAINING WRAPPERS (Shared Logic, Full Fine-tune, LoRA)
############################################################################
def _run_training_script(layout, live, script_name, title):
    """Helper function to run a training script and display output."""
    layout["body"].update(
        Panel(
            Align.center(f"[bold cyan]{title}...[/bold cyan]"),
            expand=True,
            border_style="cyan",
        )
    )
    live.refresh()

    # Check if script exists
    if not os.path.isfile(script_name):
        layout["body"].update(
            Panel(
                Align.center(
                    f"[red]Error: Training script '{script_name}' not found.[/red]"
                ),
                expand=True,
            )
        )
        live.refresh()
        wait_for_key(layout, live)
        return

    # Check if dataset exists and has data
    if (
        not os.path.exists(DATASET_FILE) or os.path.getsize(DATASET_FILE) <= 50
    ):  # Check size > header
        layout["body"].update(
            Panel(
                Align.center(
                    f"[red]Dataset file '{DATASET_FILE}' is empty or missing. Cannot train.[/red]"
                ),
                expand=True,
            )
        )
        live.refresh()
        wait_for_key(layout, live)
        return

    cmd = [sys.executable, script_name]  # Use sys.executable for consistency
    process = None
    try:
        # Start the subprocess
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,  # Redirect stderr to stdout
            text=True,
            encoding="utf-8",
            errors="replace",  # Handle potential encoding errors in output
            bufsize=1,  # Line-buffered output
        )
    except Exception as e:
        layout["body"].update(
            Panel(
                Align.center(f"[red]Error starting training process: {e}[/red]"),
                expand=True,
            )
        )
        live.refresh()
        wait_for_key(layout, live)
        return

    train_metrics = {}
    eval_metrics = {}
    log_lines = []
    max_log_lines = 18  # Display more log lines

    # --- Live Update Loop ---
    while True:
        if process.stdout:
            line = process.stdout.readline()
        else:  # Should not happen with Popen setup, but safety check
            line = None

        # Check if process has ended
        if not line and process.poll() is not None:
            break  # Exit loop if process terminated and no more output

        if line:
            line = line.strip()
            if not line:
                continue  # Skip empty lines

            # Simple log appending
            log_lines.append(line)
            if len(log_lines) > max_log_lines:
                log_lines.pop(0)

            # --- Try to parse metrics ---
            # Heuristic: looks like a Python dict literal and contains expected keys
            if line.startswith("{") and line.endswith("}"):
                try:
                    parsed = literal_eval(line)
                    if isinstance(parsed, dict):
                        # Check for evaluation metrics (more specific keys)
                        if "eval_loss" in parsed and "eval_runtime" in parsed:
                            eval_metrics = parsed
                        # Check for training metrics (look for loss and epoch/step)
                        elif (
                            "loss" in parsed
                            and ("epoch" in parsed or "step" in parsed)
                            and "learning_rate" in parsed
                        ):
                            train_metrics = parsed
                except (ValueError, SyntaxError):
                    pass  # Ignore lines that are not valid dicts
                except Exception as e:
                    # Log unexpected parsing errors, but don't crash
                    console.print(
                        f"[dim red]Error parsing metrics line: {e} -> {line}[/dim red]"
                    )

            # --- Build Display Panels (inside loop for live update) ---
            # Training Metrics Panel
            if train_metrics:
                train_table = Table(box=None, show_header=False, padding=(0, 1))
                train_table.add_column("Metric", style="deep_sky_blue1", width=15)
                train_table.add_column("Value", style="white")
                for k, v in train_metrics.items():
                    train_table.add_row(
                        str(k), f"{v:.4f}" if isinstance(v, float) else str(v)
                    )
                train_panel = Panel(
                    Align.center(train_table),
                    border_style="deep_sky_blue1",
                    title="Training Step",
                )
            else:
                train_panel = Panel(
                    Align.center("[dim]Waiting...[/dim]"),
                    border_style="dim",
                    title="Training Step",
                )

            # Evaluation Metrics Panel
            if eval_metrics:
                eval_table = Table(box=None, show_header=False, padding=(0, 1))
                eval_table.add_column("Metric", style="green", width=15)
                eval_table.add_column("Value", style="white")
                for k, v in eval_metrics.items():
                    eval_table.add_row(
                        str(k), f"{v:.4f}" if isinstance(v, float) else str(v)
                    )
                eval_panel = Panel(
                    Align.center(eval_table),
                    border_style="green",
                    title="Evaluation Epoch",
                )
            else:
                eval_panel = Panel(
                    Align.center("[dim]Waiting...[/dim]"),
                    border_style="dim",
                    title="Evaluation Epoch",
                )

            # Logs Panel
            log_text = "\n".join(log_lines)
            # Use Align.left for better readability of logs
            log_panel = Panel(
                Align.left(log_text), border_style="magenta", title="Logs"
            )

            # --- Arrange Panels in Layout ---
            # Determine dynamic height for metrics panels
            metrics_height = max(8, len(train_metrics) + 3, len(eval_metrics) + 3)
            metrics_layout = Layout(name="metrics", size=metrics_height)
            metrics_layout.split_row(Layout(train_panel), Layout(eval_panel))

            combined_layout = Layout(name="combined")
            combined_layout.split_column(
                metrics_layout, Layout(log_panel)
            )  # Logs below metrics

            layout["body"].update(
                Panel(combined_layout, border_style="yellow", expand=True, title=title)
            )
            live.refresh()

        else:
            # Small sleep if no output line to avoid busy-waiting
            sleep(0.1)

    # --- Training Finished ---
    retcode = process.poll()  # Get final exit code
    final_msg = (
        "[bold green]Training completed successfully.[/bold green]"
        if retcode == 0
        else f"[bold red]Training failed (Exit Code: {retcode}).[/bold red]"
    )
    log_lines.append("-" * 20)
    log_lines.append(final_msg)
    if len(log_lines) > max_log_lines:
        log_lines.pop(0)  # Keep log length constrained

    # Rebuild panels one last time with final state
    # Final Training Metrics Panel (reuse logic from loop)
    if train_metrics:
        train_table_f = Table(box=None, show_header=False, padding=(0, 1))
        train_table_f.add_column("Metric", style="deep_sky_blue1", width=15)
        train_table_f.add_column("Value", style="white")
        for k, v in train_metrics.items():
            train_table_f.add_row(
                str(k), f"{v:.4f}" if isinstance(v, float) else str(v)
            )
        train_panel_f = Panel(
            Align.center(train_table_f),
            border_style="deep_sky_blue1",
            title="Last Training Step",
        )
    else:
        train_panel_f = Panel(
            Align.center("[dim]No final data[/dim]"),
            border_style="dim",
            title="Last Training Step",
        )

    # Final Evaluation Metrics Panel (reuse logic from loop)
    if eval_metrics:
        eval_table_f = Table(box=None, show_header=False, padding=(0, 1))
        eval_table_f.add_column("Metric", style="green", width=15)
        eval_table_f.add_column("Value", style="white")
        for k, v in eval_metrics.items():
            eval_table_f.add_row(str(k), f"{v:.4f}" if isinstance(v, float) else str(v))
        eval_panel_f = Panel(
            Align.center(eval_table_f),
            border_style="green",
            title="Last Evaluation Epoch",
        )
    else:
        eval_panel_f = Panel(
            Align.center("[dim]No final data[/dim]"),
            border_style="dim",
            title="Last Evaluation Epoch",
        )

    # Final Logs Panel
    log_text_f = "\n".join(log_lines)
    log_panel_f = Panel(
        Align.left(log_text_f), border_style="magenta", title="Final Logs"
    )

    # Final Layout Arrangement
    metrics_height_f = max(8, len(train_metrics) + 3, len(eval_metrics) + 3)
    metrics_layout_f = Layout(name="metrics_final", size=metrics_height_f)
    metrics_layout_f.split_row(Layout(train_panel_f), Layout(eval_panel_f))
    final_combined_layout = Layout(name="final_combined")
    final_combined_layout.split_column(metrics_layout_f, Layout(log_panel_f))

    layout["body"].update(
        Panel(
            final_combined_layout,
            border_style="yellow",
            expand=True,
            title=f"{title} - Complete",
        )
    )
    live.refresh()

    wait_for_key(layout, live)  # Wait before returning to main menu


def run_train_model(layout, live):
    """Runs the full fine-tuning script (train.py)."""
    _run_training_script(layout, live, "train.py", "Starting Full Model Training")


def run_train_lora_model(layout, live):
    """Runs the LoRA fine-tuning script (lora_train.py)."""
    _run_training_script(layout, live, "lora_train.py", "Starting LoRA Model Training")


############################################################################
# SOURCES MANAGEMENT (List, Add, Remove, Toggle, Import)
############################################################################
def run_manage_sources(layout, live):
    """Main menu for managing RSS feed sources."""
    json_file = FEEDS_FILE
    try:
        if os.path.exists(json_file):
            with open(json_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                # Ensure 'feeds' key exists
                if "feeds" not in data or not isinstance(data["feeds"], list):
                    data = {"feeds": []}
        else:
            data = {"feeds": []}  # Initialize if file doesn't exist
    except (json.JSONDecodeError, IOError) as e:
        console.print(
            f"[red]Error loading feeds file {json_file}: {e}. Starting with empty list.[/red]"
        )
        data = {"feeds": []}

    needs_saving = False  # Flag to track if changes were made

    while True:
        items = [
            "List Sources",
            "Add Source",
            "Remove Source",
            "Toggle Source Status",
            "Import Curated Feeds",
            "Return to Main Menu",
        ]
        choice = interactive_menu_in_layout(layout, live, "Sources Management", items)

        # Ensure 'feeds' list exists in data
        feeds = data.setdefault("feeds", [])

        if choice == -1 or choice == 5:  # Return to main menu
            break  # Exit the sources management loop

        elif choice == 0:  # List Sources
            _list_sources(feeds, layout, live)

        elif choice == 1:  # Add Source
            if _add_source(feeds, layout, live):
                needs_saving = True

        elif choice == 2:  # Remove Source
            if _remove_source(feeds, layout, live):
                needs_saving = True

        elif choice == 3:  # Toggle Status
            if _toggle_sources_status(feeds, layout, live):
                needs_saving = True

        elif choice == 4:  # Import Curated Feeds
            if _import_curated_feeds(data, layout, live):  # Pass entire data dict
                needs_saving = True

    # Save if changes were made
    if needs_saving:
        try:
            with open(json_file, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            layout["body"].update(
                Panel(
                    Align.center("[bold green]Sources file updated.[/bold green]"),
                    expand=True,
                )
            )
            live.refresh()
            sleep(1)
        except IOError as e:
            layout["body"].update(
                Panel(Align.center(f"[red]Error saving feeds: {e}[/red]"), expand=True)
            )
            live.refresh()
            wait_for_key(layout, live)


# --- Helper functions for run_manage_sources ---


def _list_sources(feeds, layout, live):
    """Displays the list of configured RSS feeds."""
    if not feeds:
        panel = Panel(
            Align.center("[yellow]No sources configured yet.[/yellow]"),
            expand=True,
            border_style="yellow",
        )
    else:
        table = Table(title="Current RSS Sources", expand=True, box=None)
        table.add_column("No.", justify="right", style="dim", width=4)
        table.add_column("Name", style="cyan", overflow="fold")
        table.add_column("URL", style="green", overflow="fold")
        table.add_column("Category", style="magenta", width=15)
        table.add_column("Status", style="yellow", width=10)
        for idx, feed in enumerate(feeds, start=1):
            status_text = (
                "[green]Enabled[/green]"
                if feed.get("enabled", True)
                else "[red]Disabled[/red]"
            )
            table.add_row(
                str(idx),
                feed.get("name", "[dim]N/A[/dim]"),
                feed.get("url", "[dim]N/A[/dim]"),
                feed.get("category", "[dim]N/A[/dim]"),
                status_text,
            )
        panel = Panel(table, expand=True, border_style="blue")
    layout["body"].update(panel)
    live.refresh()
    wait_for_key(layout, live, prompt="Press any key to return...")


def _add_source(feeds, layout, live):
    """Prompts user to add a new RSS source."""
    layout["body"].update(
        Panel(
            Align.center("[bold magenta]Enter new source details:[/bold magenta]"),
            expand=True,
        )
    )
    live.refresh()
    try:
        name = Prompt.ask("[cyan]Source Name[/cyan]", default="").strip()
        category = Prompt.ask("[cyan]Category[/cyan]", default="General").strip()
        url = Prompt.ask("[cyan]RSS Feed URL[/cyan]", default="").strip()

        if not name or not category or not url:
            layout["body"].update(
                Panel(
                    Align.center(
                        "[yellow]All fields are required. Source not added.[/yellow]"
                    ),
                    expand=True,
                )
            )
            live.refresh()
            sleep(1.5)
            return False

        # Basic URL validation (optional but recommended)
        if not url.startswith(("http://", "https://")):
            layout["body"].update(
                Panel(
                    Align.center(
                        "[red]Invalid URL format. Must start with http:// or https://[/red]"
                    ),
                    expand=True,
                )
            )
            live.refresh()
            sleep(2)
            return False

        new_feed = {"name": name, "category": category, "url": url, "enabled": True}
        feeds.append(new_feed)
        layout["body"].update(
            Panel(Align.center(f"[green]Source '{name}' added.[/green]"), expand=True)
        )
        live.refresh()
        sleep(1)
        return True
    except Exception as e:
        layout["body"].update(
            Panel(Align.center(f"[red]Error adding source: {e}[/red]"), expand=True)
        )
        live.refresh()
        sleep(2)
        return False


def _remove_source(feeds, layout, live):
    """Prompts user to remove an existing RSS source by number."""
    if not feeds:
        layout["body"].update(
            Panel(Align.center("[yellow]No sources to remove.[/yellow]"), expand=True)
        )
        live.refresh()
        wait_for_key(layout, live)
        return False

    table = Table(title="Select Source to Remove", box=None)
    table.add_column("No.", justify="right", style="dim", width=4)
    table.add_column("Name", style="cyan")
    table.add_column("URL", style="green", overflow="fold")
    for idx, feed in enumerate(feeds, start=1):
        table.add_row(str(idx), feed.get("name", "N/A"), feed.get("url", "N/A"))
    layout["body"].update(Panel(table, expand=True, border_style="red"))
    live.refresh()

    try:
        selection = Prompt.ask(
            "[bold magenta]Enter number to remove (or 'q' to cancel)[/bold magenta]",
            default="q",
        ).strip()
        if selection.lower() == "q":
            layout["body"].update(
                Panel(Align.center("[yellow]Removal cancelled.[/yellow]"), expand=True)
            )
            live.refresh()
            sleep(1)
            return False

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
            live.refresh()
            sleep(1)
            return True
        else:
            layout["body"].update(
                Panel(Align.center("[red]Invalid selection number.[/red]"), expand=True)
            )
            live.refresh()
            sleep(1.5)
            return False
    except ValueError:
        layout["body"].update(
            Panel(
                Align.center("[red]Invalid input. Please enter a number.[/red]"),
                expand=True,
            )
        )
        live.refresh()
        sleep(1.5)
        return False
    except Exception as e:
        layout["body"].update(
            Panel(Align.center(f"[red]Error removing source: {e}[/red]"), expand=True)
        )
        live.refresh()
        sleep(2)
        return False


def _toggle_sources_status(feeds, layout, live):
    """Allows interactive enabling/disabling of sources."""
    if not feeds:
        layout["body"].update(
            Panel(Align.center("[yellow]No sources to toggle.[/yellow]"), expand=True)
        )
        live.refresh()
        wait_for_key(layout, live)
        return False

    position = 0
    changed = False
    while True:
        table = Table(
            show_header=True,
            header_style="bold magenta",
            box=None,
            title="Toggle Source Status",
        )
        table.add_column("No.", justify="right", style="dim", width=4)
        table.add_column("Name", style="cyan", overflow="fold")
        table.add_column("Status", style="yellow", width=10)
        for i, feed in enumerate(feeds):
            name = feed.get("name", "N/A")
            is_enabled = feed.get("enabled", True)
            status_text = (
                "[green]Enabled[/green]" if is_enabled else "[red]Disabled[/red]"
            )
            row_style = "reverse blue" if i == position else ""
            table.add_row(str(i + 1), name, status_text, style=row_style)

        layout["body"].update(
            Panel(
                table,
                border_style="blue",
                expand=True,
                subtitle="Navigate (↑/↓/j/k), Toggle (Space), Save & Quit (q)",
            )
        )
        live.refresh()

        key = read_key_sequence()
        if key in ("up", "k"):
            position = (position - 1) % len(feeds)
        elif key in ("down", "j"):
            position = (position + 1) % len(feeds)
        elif key == "space":
            # Toggle the 'enabled' status for the selected feed
            current_status = feeds[position].get("enabled", True)
            feeds[position]["enabled"] = not current_status
            changed = True  # Mark that a change occurred
        elif key.lower() == "q":
            break  # Exit the toggle loop

    # Return whether any changes were made
    if changed:
        layout["body"].update(
            Panel(Align.center("[green]Status changes applied.[/green]"), expand=True)
        )
        live.refresh()
        sleep(1)
    else:
        layout["body"].update(
            Panel(Align.center("[yellow]No changes made.[/yellow]"), expand=True)
        )
        live.refresh()
        sleep(1)
    return changed


def _import_curated_feeds(data, layout, live):
    """Imports a predefined list of curated feeds if they don't already exist."""
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
            "name": "Reuters Top News",
            "url": "https://www.reutersagency.com/feed/?best-topics=top-news",
            "category": "News",
            "enabled": True,
        },  # More specific Reuters feed
        {
            "name": "Ars Technica",
            "url": "http://feeds.arstechnica.com/arstechnica/index/",
            "category": "Technology",
            "enabled": True,
        },
        {
            "name": "Nature Briefing",
            "url": "https://www.nature.com/briefing/rss",
            "category": "Science",
            "enabled": True,
        },
        {
            "name": "Variety",
            "url": "https://variety.com/feed/",
            "category": "Entertainment",
            "enabled": True,
        },
    ]
    added_count = 0
    # Ensure 'feeds' list exists in data
    feeds = data.setdefault("feeds", [])
    existing_urls = {
        feed.get("url") for feed in feeds if feed.get("url")
    }  # Set of existing URLs

    for feed_to_add in curated:
        if feed_to_add["url"] not in existing_urls:
            feeds.append(feed_to_add)
            added_count += 1

    if added_count > 0:
        layout["body"].update(
            Panel(
                Align.center(
                    f"[green]Imported {added_count} new curated feeds.[/green]"
                ),
                expand=True,
            )
        )
        live.refresh()
        sleep(1.5)
        return True
    else:
        layout["body"].update(
            Panel(
                Align.center(
                    "[yellow]No new curated feeds to import (already exist).[/yellow]"
                ),
                expand=True,
            )
        )
        live.refresh()
        sleep(1.5)
        return False


############################################################################
# MODEL TESTING (Helpers and Wrappers for Full & LoRA)
############################################################################


def get_latest_checkpoint_path(results_dir):
    """Finds the latest Trainer checkpoint or adapter directory."""
    if not os.path.exists(results_dir) or not os.path.isdir(results_dir):
        console.print(f"[dim]Results directory not found: {results_dir}[/dim]")
        return None

    # --- Prioritize Checkpoints ---
    checkpoint_dirs = [
        d
        for d in os.listdir(results_dir)
        if os.path.isdir(os.path.join(results_dir, d)) and d.startswith("checkpoint-")
    ]

    if checkpoint_dirs:
        try:
            # Sort by step number embedded in the name
            latest_checkpoint_name = max(
                checkpoint_dirs, key=lambda p: int(p.split("-")[-1])
            )
            latest_path = os.path.join(results_dir, latest_checkpoint_name)
            # Verify it's a valid checkpoint (e.g., contains pytorch_model.bin or similar)
            if os.path.exists(
                os.path.join(latest_path, "pytorch_model.bin")
            ) or os.path.exists(os.path.join(latest_path, "adapter_model.bin")):
                console.print(f"[dim]Found latest checkpoint: {latest_path}[/dim]")
                return latest_path
            else:
                console.print(
                    f"[yellow]Checkpoint dir found but seems invalid: {latest_path}[/yellow]"
                )
                # Continue searching other options if invalid checkpoint found
        except (ValueError, TypeError):
            # Fallback to modification time if step number parsing fails
            try:
                latest_checkpoint_name = max(
                    checkpoint_dirs,
                    key=lambda p: os.path.getmtime(os.path.join(results_dir, p)),
                )
                latest_path = os.path.join(results_dir, latest_checkpoint_name)
                if os.path.exists(
                    os.path.join(latest_path, "pytorch_model.bin")
                ) or os.path.exists(os.path.join(latest_path, "adapter_model.bin")):
                    console.print(
                        f"[dim]Found latest checkpoint by mtime: {latest_path}[/dim]"
                    )
                    return latest_path
                else:
                    console.print(
                        f"[yellow]Checkpoint dir found by mtime but seems invalid: {latest_path}[/yellow]"
                    )
            except Exception as e_mtime:
                console.print(
                    f"[red]Error finding checkpoint by mtime: {e_mtime}[/red]"
                )

    # --- Fallback: Check for 'final_adapter' or root dir adapter (for LoRA) ---
    final_adapter_path = os.path.join(results_dir, "final_adapter")
    if os.path.isdir(final_adapter_path) and os.path.exists(
        os.path.join(final_adapter_path, "adapter_config.json")
    ):
        console.print(f"[dim]Found final adapter directory: {final_adapter_path}[/dim]")
        return final_adapter_path

    # Check root results_dir for adapter files (if trainer.save_model() was used)
    if os.path.exists(os.path.join(results_dir, "adapter_config.json")):
        console.print(
            f"[dim]Found adapter files in root directory: {results_dir}[/dim]"
        )
        return results_dir  # Use the results dir itself

    console.print(
        f"[yellow]No valid checkpoints or adapters found in {results_dir}[/yellow]"
    )
    return None  # No valid path found


def classify_headline(headline, tokenizer, model, device):
    """Classifies a single headline using the provided model and tokenizer."""
    # Basic input validation
    if not headline or not isinstance(headline, str):
        return {"headline": headline, "predicted": "Invalid Input", "confidence": 0.0}

    try:
        inputs = tokenizer(
            headline,
            return_tensors="pt",
            truncation=True,
            padding="max_length",  # Ensure padding consistent with training
            max_length=MAX_LENGTH,
        )
        # Move inputs to the correct device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # Perform inference
        with torch.no_grad():  # Disable gradient calculation
            outputs = model(**inputs)
            logits = outputs.logits
            # Apply softmax to get probabilities
            probabilities = torch.softmax(logits, dim=-1)
            # Get the predicted class index and probability
            predicted_class_id = torch.argmax(probabilities, dim=-1).item()
            confidence = probabilities[0, predicted_class_id].item()

        predicted_label = LABEL_MAP.get(
            predicted_class_id, f"Raw ID: {predicted_class_id}"
        )

        return {
            "headline": headline,
            "predicted": predicted_label,
            "confidence": confidence,
        }
    except Exception as e:
        console.print(f"[red]Error during classification: {e}[/red]")
        return {
            "headline": headline,
            "predicted": "Classification Error",
            "confidence": 0.0,
        }


def _run_testing_loop(layout, live, tokenizer, model, device, model_type_name):
    """Runs the interactive headline classification loop."""
    results_list = []
    max_results = 12  # Show a bit more history

    while True:
        # --- Build results display ---
        table = Table(
            title=f"Classification History ({model_type_name})",
            box=None,
            padding=(0, 1),
            expand=True,
        )
        table.add_column(
            "Input Headline", style="yellow", overflow="fold", no_wrap=False
        )  # Allow wrapping
        table.add_column("Prediction", style="bold green", width=15)
        table.add_column("Conf.", style="cyan", width=6, justify="right")

        # Display results, newest first
        for r in reversed(results_list):
            conf_str = f"{r['confidence']:.2f}"
            table.add_row(r["headline"], r["predicted"], conf_str)

        if not results_list:
            results_panel = Panel(
                Align.center("[dim]Enter a headline below to classify.[/dim]"),
                border_style="blue",
                expand=True,
            )
        else:
            results_panel = Panel(table, border_style="blue", expand=True)

        # --- Prompt Area ---
        prompt_panel = Panel(
            Align.center(
                "[bold magenta]Enter headline to classify (or 'q' to quit):[/bold magenta]"
            ),
            height=3,
            border_style="magenta",
        )

        # Arrange panels
        sub_layout = Layout()
        sub_layout.split_column(
            Layout(results_panel, name="results"),  # Results take most space
            Layout(prompt_panel, name="prompt"),
        )
        layout["body"].update(
            Panel(
                sub_layout,
                border_style="green",
                expand=True,
                title=f"Test Model - {model_type_name}",
            )
        )
        live.refresh()

        # --- Get user input ---
        headline = Prompt.ask("Headline").strip()  # Use Rich Prompt

        if not headline:
            continue  # Ignore empty input
        if headline.lower() == "q":
            break  # Quit loop

        # --- Classify and update ---
        result = classify_headline(headline, tokenizer, model, device)
        results_list.append(result)
        if len(results_list) > max_results:
            results_list.pop(0)  # Trim history

    # Exited loop
    wait_for_key(layout, live, prompt="Press any key to return to main menu...")


def run_test_model(layout, live):
    """Loads and tests the latest full fine-tuned model."""
    layout["body"].update(
        Panel(
            Align.center("[cyan]Loading full fine-tuned model checkpoint...[/cyan]"),
            expand=True,
        )
    )
    live.refresh()

    model_path = get_latest_checkpoint_path(FULL_RESULTS_DIR)
    if not model_path:
        layout["body"].update(
            Panel(
                Align.center(
                    f"[red]No valid trained model checkpoint found in '{FULL_RESULTS_DIR}'[/red]"
                ),
                expand=True,
            )
        )
        live.refresh()
        wait_for_key(layout, live)
        return

    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        console.print(f"[dim]Using device: {device}[/dim]")

        # Tokenizer always loaded from base
        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)

        # Load model from the specific checkpoint path found
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
        model.to(device)
        model.eval()  # Set to evaluation mode

        layout["body"].update(
            Panel(
                Align.center(
                    f"[green]Full model loaded successfully from {os.path.basename(model_path)}.[/green]"
                ),
                expand=True,
            )
        )
        live.refresh()
        sleep(1.5)
        _run_testing_loop(
            layout, live, tokenizer, model, device, "Full Fine-tune"
        )  # Start interactive loop

    except Exception as e:
        console.print_exception()
        layout["body"].update(
            Panel(
                Align.center(f"[red]Error loading full model: {e}[/red]"), expand=True
            )
        )
        live.refresh()
        wait_for_key(layout, live)


def run_test_lora_model(layout, live):
    """Loads the base model and applies the latest LoRA adapters for testing."""
    if not PEFT_AVAILABLE:
        layout["body"].update(
            Panel(
                Align.center(
                    "[red]PEFT library not installed. Run 'pip install peft'.[/red]"
                ),
                expand=True,
            )
        )
        live.refresh()
        wait_for_key(layout, live)
        return

    layout["body"].update(
        Panel(
            Align.center("[cyan]Loading LoRA model (base + adapters)...[/cyan]"),
            expand=True,
        )
    )
    live.refresh()

    # Find the latest adapter path (could be checkpoint or final_adapter)
    adapter_path = get_latest_checkpoint_path(LORA_RESULTS_DIR)
    if not adapter_path:
        layout["body"].update(
            Panel(
                Align.center(
                    f"[red]No valid LoRA adapters found in '{LORA_RESULTS_DIR}'[/red]"
                ),
                expand=True,
            )
        )
        live.refresh()
        wait_for_key(layout, live)
        return

    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        console.print(f"[dim]Using device: {device}[/dim]")

        # 1. Load Base Tokenizer
        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)

        # 2. Load Base Model (ensure num_labels matches training)
        base_model = AutoModelForSequenceClassification.from_pretrained(
            BASE_MODEL_NAME,
            num_labels=NUM_LABELS,
            # ignore_mismatched_sizes=True # Add if loading adapters onto a base model with a different head size initially
        )

        # 3. Load Adapters onto Base Model using PeftModel
        # PeftModel will automatically load the adapter from the specified path (checkpoint or final folder)
        model = PeftModel.from_pretrained(base_model, adapter_path)

        # Optional: Merge LoRA weights into the base model for potentially faster inference.
        # This modifies the model in-place and means you can't easily unload the adapter later.
        # model = model.merge_and_unload()
        # console.print("[dim]LoRA weights merged into base model.[/dim]")

        model.to(device)
        model.eval()  # Set to evaluation mode

        layout["body"].update(
            Panel(
                Align.center(
                    f"[green]LoRA model loaded successfully using adapters from {os.path.basename(adapter_path)}.[/green]"
                ),
                expand=True,
            )
        )
        live.refresh()
        sleep(1.5)
        _run_testing_loop(
            layout, live, tokenizer, model, device, "LoRA"
        )  # Start interactive loop

    except Exception as e:
        console.print_exception()
        layout["body"].update(
            Panel(
                Align.center(f"[red]Error loading LoRA model: {e}[/red]"), expand=True
            )
        )
        live.refresh()
        wait_for_key(layout, live)


############################################################################
# MAIN APPLICATION LOOP
############################################################################
def main():
    # --- Initial Setup & Checks ---
    # Check/Create necessary directories
    os.makedirs(FULL_RESULTS_DIR, exist_ok=True)
    os.makedirs(LORA_RESULTS_DIR, exist_ok=True)

    # Check/Create dataset file
    if not os.path.exists(DATASET_FILE):
        console.print(
            f"[yellow]Dataset file '{DATASET_FILE}' not found. Creating empty file with header.[/yellow]"
        )
        try:
            with open(DATASET_FILE, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["text", "label"])
        except IOError as e:
            console.print(
                f"[bold red]Fatal Error:[/bold red] Could not create dataset file '{DATASET_FILE}': {e}"
            )
            sys.exit(1)

    # Check/Create feeds file
    if not os.path.exists(FEEDS_FILE):
        console.print(
            f"[yellow]Feeds file '{FEEDS_FILE}' not found. Creating empty file.[/yellow]"
        )
        try:
            with open(FEEDS_FILE, "w", encoding="utf-8") as f:
                json.dump({"feeds": []}, f, indent=2)
        except IOError as e:
            console.print(
                f"[red]Error creating feeds file '{FEEDS_FILE}': {e}[/red]"
            )  # Non-fatal

    # --- Setup Rich Layout ---
    layout = Layout()
    layout.split(
        Layout(
            Panel(Align.center(app_title), border_style="bold blue"),
            name="header",
            size=10,
        ),  # Bold border
        Layout(
            Panel(
                Align.center("[dim]Welcome! Select an option.[/dim]"),
                border_style="green",
            ),
            name="body",
        ),  # Initial body
        Layout(
            Panel(
                Align.center(
                    "[bold cyan]Navigate:[/bold cyan] ↑/↓/j/k   [bold cyan]Select:[/bold cyan] Enter   [bold cyan]Quit Option:[/bold cyan] q"
                ),
                border_style="dim",
            ),
            name="footer",
            size=3,
        ),
    )

    # --- Main Menu Items ---
    main_menu_items = [
        "1. Create/Update Dataset",  # 0
        "2. Train Model (Full)",  # 1
        "3. Train Model (LoRA)",  # 2
        "4. Test Model (Full)",  # 3
        "5. Test Model (LoRA)",  # 4
        "6. Manage Feed Sources",  # 5
        "7. Exit",  # 6
    ]

    # --- Live Display Loop ---
    with Live(
        layout,
        console=console,
        auto_refresh=False,
        screen=True,
        vertical_overflow="visible",
    ) as live:
        live.refresh()  # Initial draw
        while True:
            choice = interactive_menu_in_layout(
                layout, live, "SlopBERT Main Menu", main_menu_items
            )

            action_taken = True  # Flag to pause before redrawing menu
            if choice == -1 or choice == 6:  # Exit
                layout["body"].update(
                    Panel(
                        Align.center("[bold red]Exiting SlopBERT. Goodbye![/bold red]"),
                        expand=True,
                    )
                )
                live.refresh()
                sleep(1.5)
                break
            elif choice == 0:
                run_create_or_update_dataset(layout, live)
            elif choice == 1:
                run_train_model(layout, live)
            elif choice == 2:
                run_train_lora_model(layout, live)
            elif choice == 3:
                run_test_model(layout, live)
            elif choice == 4:
                run_test_lora_model(layout, live)
            elif choice == 5:
                run_manage_sources(layout, live)
            else:
                action_taken = False  # No valid action if choice is out of bounds

            # Pause briefly before redrawing main menu only if an action was taken
            # if action_taken:
            #    layout["body"].update(Panel(Align.center("[dim]Returning to main menu...[/dim]"), expand=True)); live.refresh(); sleep(0.3)


if __name__ == "__main__":
    main()
