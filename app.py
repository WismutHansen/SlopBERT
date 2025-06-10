# app.py
import os
import sys
import json
import csv
import subprocess
import shutil
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
)
from ast import literal_eval

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.align import Align
from rich.live import Live
from rich.layout import Layout
from rich.prompt import Prompt

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
)

try:
    from peft import PeftModel

    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False

console = Console()

# --- Configuration Constants ---
BASE_MODEL_NAME = "answerdotai/ModernBERT-base"
FULL_RESULTS_DIR = "./results"
LORA_RESULTS_DIR = "./results_lora"
RESULTS_QUANTIZED_INT8_DIR = "./results_quantized_int8"
RESULTS_QUANTIZED_FLOAT8_DIR = "./results_quantized_float8"
LORA_MERGED_TEMP_DIR = "./lora_merged_temp"
NUM_LABELS = 6
MAX_LENGTH = 128
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
TARGET_RATING_COUNT = 50
FETCH_TIMEOUT_SECONDS = 25

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
        rest = os.read(sys.stdin.fileno(), 2) if sys.stdin.isatty() else b""
        if rest == b"[A":
            return "up"
        elif rest == b"[B":
            return "down"
        return ""
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
            return ch
        except UnicodeDecodeError:
            return ""


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
                )
            else:
                table.add_row("", item)
        centered_menu = Align.center(table, vertical="middle")
        body_panel = Panel(centered_menu, border_style="blue", expand=True, title=title)
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
    panel = Panel(
        Align.center(f"[bold cyan]{prompt}[/bold cyan]"),
        expand=True,
        border_style="dim",
    )
    layout["body"].update(panel)
    live.refresh()
    read_key_sequence()


############################################################################
# RSS FEED, CACHING, AND DATASET HELPERS
############################################################################
def load_cache():
    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            console.print(f"[red]Error loading cache file {CACHE_FILE}: {e}[/red]")
            return []
    return []


def save_cache(cache):
    try:
        with open(CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump(cache, f, indent=2)
    except IOError as e:
        console.print(f"[red]Error saving cache file {CACHE_FILE}: {e}[/red]")


def load_feeds_from_json(file_path=FEEDS_FILE):
    if not os.path.isfile(file_path):
        console.print(f"[yellow]Feeds file '{file_path}' not found.[/yellow]")
        return []
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            feeds = data.get("feeds", [])
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
    articles_today = []
    try:
        feed_data = feedparser.parse(url, agent="SlopBERT/1.0")
        if feed_data.bozo:
            pass

        today = datetime.datetime.now(pytz.utc).date()
        ext = tldextract.extract(url)
        source_domain = (
            f"{ext.domain}.{ext.suffix}" if ext.domain and ext.suffix else url
        )

        for entry in feed_data.entries:
            title = entry.get("title", "").strip()
            if not title:
                continue

            published_dt = None
            date_keys = ["published_parsed", "updated_parsed", "created_parsed"]
            for key in date_keys:
                parsed_time = entry.get(key)
                if parsed_time:
                    try:
                        if len(parsed_time) >= 6:
                            dt_naive = datetime.datetime(*parsed_time[:6])
                            published_dt = (
                                pytz.utc.localize(dt_naive)
                                if dt_naive.tzinfo is None
                                else dt_naive.astimezone(pytz.utc)
                            )
                            break
                    except (ValueError, TypeError):
                        continue

            if published_dt and published_dt.date() == today:
                articles_today.append(
                    {
                        "title": title,
                        "link": entry.get("link", ""),
                        "published": published_dt.isoformat(),
                        "category": category,
                        "source": source_domain,
                    }
                )
    except Exception as e:
        console.print(f"[red]Error processing feed {url}: {e}[/red]")
    return articles_today


def get_status_bar_text(target=TARGET_RATING_COUNT):
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
                        pass
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
        filled_width = int(progress_percent * 15)
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

    table.add_row("─" * 12, "─" * 6, "─" * 25)
    table.add_row("[bold]Total[/bold]", f"[bold]{total_count}[/bold]", "")
    return table


############################################################################
# DATASET CREATION
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
    layout["body"].update(
        Panel(
            Align.center("[cyan]Initiating feed fetching...[/cyan]"),
            border_style="cyan",
            expand=True,
        )
    )
    live.refresh()
    sleep(0.5)

    def fetch_articles_thread_wrapper(feed):
        return feed, get_articles_from_feed(feed["url"], feed["category"])

    new_articles_list = []
    fetch_status = {}
    total_feeds = len(feeds)
    completed_count, failed_count, newly_fetched_count = 0, 0, 0
    max_workers = min(total_feeds, 12)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_feed = {
            executor.submit(fetch_articles_thread_wrapper, feed): feed for feed in feeds
        }
        for future in as_completed(future_to_feed):
            original_feed_info = future_to_feed[future]
            name = original_feed_info.get(
                "name", original_feed_info.get("url", "Unknown Feed")
            )
            completed_count += 1

            status_lines = [
                f"[bold]Fetching feed {completed_count}/{total_feeds}:[/] [cyan]{name}[/cyan]"
            ]
            processed_feeds_display = list(fetch_status.items())[-5:]
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

            try:
                _, articles = future.result(timeout=FETCH_TIMEOUT_SECONDS)
                new_articles_list.extend(articles)
                fetch_status[name] = f"[green]OK ({len(articles)} new)[/green]"
                newly_fetched_count += len(articles)
            except TimeoutError:
                failed_count += 1
                fetch_status[name] = "[yellow]TIMEOUT[/yellow]"
            except Exception as e:
                failed_count += 1
                fetch_status[name] = f"[red]ERROR[/red]"
                console.print(f"[dim red]Fetch error for {name}: {e}[/dim red]")

    summary_lines = [
        "[bold blue]Fetch Summary:[/bold blue]",
        f"Total Feeds Processed: {total_feeds}",
        f"Successful: {total_feeds - failed_count}",
        f"Failed/Timeout: {failed_count}",
        f"Total New Articles Found: {newly_fetched_count}",
        "-" * 40,
    ]
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

    existing_titles_in_cache = {c.get("title") for c in cache if c.get("title")}
    unique_new_articles = [
        a
        for a in new_articles_list
        if a.get("title") and a["title"] not in existing_titles_in_cache
    ]
    cache.extend(unique_new_articles)
    save_cache(cache)
    console.print(
        f"[green]Added {len(unique_new_articles)} unique headlines to cache.[/green]"
    )

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

    headlines_to_rate = [
        a for a in cache if a.get("title") and a["title"] not in rated_titles_in_dataset
    ]
    if not headlines_to_rate:
        layout["body"].update(
            Panel(
                Align.center("[green]No new headlines left to rate.[/green]"),
                expand=True,
            )
        )
        live.refresh()
        wait_for_key(layout, live)
        return

    random.shuffle(headlines_to_rate)

    try:
        with open(DATASET_FILE, "a", newline="", encoding="utf-8") as f_dataset:
            fieldnames = ["text", "label"]
            writer = csv.DictWriter(
                f_dataset, fieldnames=fieldnames, quoting=csv.QUOTE_NONNUMERIC
            )
            f_dataset.seek(0, os.SEEK_END)
            if f_dataset.tell() == 0:
                writer.writeheader()

            key_to_rating = {str(i): i for i in range(NUM_LABELS)}
            key_to_rating.update({"a": 0, "s": 1, "d": 2, "f": 3, "g": 4, "h": 5})
            current_index = 0
            while current_index < len(headlines_to_rate):
                article = headlines_to_rate[current_index]
                headline = article.get("title", "").strip()
                if not headline:
                    current_index += 1
                    continue

                headline_text = f"[bold yellow]{headline}[/bold yellow]\n[dim]Source:[/dim] {article.get('source', 'N/A')} | [dim]Category:[/dim] {article.get('category', 'N/A')}"
                headline_panel = Panel(
                    Align.center(headline_text, vertical="middle"),
                    title=f"Rating Headline {current_index + 1} of {len(headlines_to_rate)}",
                    subtitle="Rate ([cyan]0-5[/cyan] or [cyan]a-h[/cyan]), [red]q[/red]=quit",
                    expand=True,
                    border_style="yellow",
                )

                status_table = get_status_bar_text(TARGET_RATING_COUNT)
                rating_panel = Panel(
                    Align.center(status_table),
                    title="Rating Progress",
                    border_style="magenta",
                    expand=True,
                )

                sub_layout = Layout()
                sub_layout.split_column(
                    Layout(headline_panel, name="headline", ratio=1),
                    Layout(rating_panel, name="rating", size=10 + NUM_LABELS),
                )
                layout["body"].update(
                    Panel(sub_layout, border_style="green", expand=True)
                )
                live.refresh()

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
                    return

                if key in key_to_rating:
                    rating = key_to_rating[key]
                    writer.writerow({"text": headline, "label": rating})
                    f_dataset.flush()

                    for i, cache_item in enumerate(cache):
                        if cache_item.get("title") == headline:
                            cache.pop(i)
                            save_cache(cache)
                            break

                    current_index += 1
                else:
                    continue

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
        console.print_exception()
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
# TRAINING & QUANTIZATION WRAPPERS
############################################################################
def _run_training_script(layout, live, script_name, title):
    layout["body"].update(
        Panel(
            Align.center(f"[bold cyan]{title}...[/bold cyan]"),
            expand=True,
            border_style="cyan",
        )
    )
    live.refresh()

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

    if not os.path.exists(DATASET_FILE) or os.path.getsize(DATASET_FILE) <= 50:
        layout["body"].update(
            Panel(
                Align.center(
                    f"[red]Dataset file '{DATASET_FILE}' is empty or missing.[/red]"
                ),
                expand=True,
            )
        )
        live.refresh()
        wait_for_key(layout, live)
        return

    cmd = [sys.executable, script_name]
    process = None
    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
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

    train_metrics, eval_metrics, log_lines = {}, {}, []
    max_log_lines = 18

    while True:
        line = process.stdout.readline() if process.stdout else None
        if not line and process.poll() is not None:
            break
        if line:
            line = line.strip()
            if not line:
                continue
            log_lines.append(line)
            if len(log_lines) > max_log_lines:
                log_lines.pop(0)

            if line.startswith("{") and line.endswith("}"):
                try:
                    parsed = literal_eval(line)
                    if isinstance(parsed, dict):
                        if "eval_loss" in parsed:
                            eval_metrics = parsed
                        elif "loss" in parsed:
                            train_metrics = parsed
                except (ValueError, SyntaxError, Exception):
                    pass

            train_panel = Panel(
                Align.center("[dim]Waiting...[/dim]"),
                border_style="dim",
                title="Training Step",
            )
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

            eval_panel = Panel(
                Align.center("[dim]Waiting...[/dim]"),
                border_style="dim",
                title="Evaluation Epoch",
            )
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

            log_panel = Panel(
                Align.left("\n".join(log_lines)), border_style="magenta", title="Logs"
            )
            metrics_height = max(8, len(train_metrics) + 3, len(eval_metrics) + 3)
            metrics_layout = Layout(name="metrics", size=metrics_height)
            metrics_layout.split_row(Layout(train_panel), Layout(eval_panel))
            combined_layout = Layout(name="combined")
            combined_layout.split_column(metrics_layout, Layout(log_panel))
            layout["body"].update(
                Panel(combined_layout, border_style="yellow", expand=True, title=title)
            )
            live.refresh()
        else:
            sleep(0.1)

    wait_for_key(layout, live)


def run_train_model(layout, live):
    _run_training_script(layout, live, "train.py", "Starting Full Model Training")


def run_train_lora_model(layout, live):
    _run_training_script(layout, live, "lora_train.py", "Starting LoRA Model Training")


def _run_quantization_script(
    layout, live, script_name, model_path, output_dir, quant_level
):
    title = f"Quantizing to {quant_level.upper()}"
    layout["body"].update(
        Panel(
            Align.center(f"[bold cyan]{title}...[/bold cyan]"),
            expand=True,
            border_style="cyan",
        )
    )
    live.refresh()

    cmd = [
        sys.executable,
        script_name,
        "--model_path",
        model_path,
        "--output_dir",
        output_dir,
        "--quant_level",
        quant_level,
        "--dataset_path",
        DATASET_FILE,
    ]
    process = None
    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
    except Exception as e:
        layout["body"].update(
            Panel(
                Align.center(f"[red]Error starting quantization process: {e}[/red]"),
                expand=True,
            )
        )
        live.refresh()
        wait_for_key(layout, live)
        return

    log_lines, eval_metrics = [], {}
    max_log_lines = 20
    while True:
        line = process.stdout.readline() if process.stdout else None
        if not line and process.poll() is not None:
            break
        if line:
            line = line.strip()
            if not line:
                continue
            log_lines.append(line)
            if len(log_lines) > max_log_lines:
                log_lines.pop(0)

            if line.startswith("{") and line.endswith("}"):
                try:
                    parsed = literal_eval(line)
                    if isinstance(parsed, dict) and "eval_loss" in parsed:
                        eval_metrics = parsed
                except (ValueError, SyntaxError):
                    pass

            log_panel = Panel(
                Align.left("\n".join(log_lines)),
                border_style="magenta",
                title="Quantization Logs",
            )
            metrics_panel = Panel(
                Align.center("[dim]Waiting for evaluation...[/dim]"),
                border_style="dim",
                title="Evaluation Metrics",
            )
            if eval_metrics:
                eval_table = Table(box=None, show_header=False, padding=(0, 1))
                eval_table.add_column("Metric", style="green", width=20)
                eval_table.add_column("Value", style="white")
                for k, v in eval_metrics.items():
                    eval_table.add_row(
                        str(k), f"{v:.4f}" if isinstance(v, float) else str(v)
                    )
                metrics_panel = Panel(
                    Align.center(eval_table),
                    border_style="green",
                    title="Evaluation Metrics",
                )

            combined_layout = Layout(name="combined")
            combined_layout.split_column(
                Layout(log_panel), Layout(metrics_panel, size=10)
            )
            layout["body"].update(
                Panel(combined_layout, border_style="yellow", expand=True, title=title)
            )
            live.refresh()

    retcode = process.poll()
    final_msg = (
        "[bold green]Quantization & evaluation completed.[/bold green]"
        if retcode == 0
        else f"[bold red]Process failed (Exit Code: {retcode}).[/bold red]"
    )
    final_panel = Panel(
        Align.center(f"{final_msg}\n\nCheck logs for details."),
        expand=True,
        title="Complete",
    )
    layout["body"].update(final_panel)
    live.refresh()
    wait_for_key(layout, live)


def _merge_lora_model(layout, live) -> str | None:
    adapter_path = get_latest_checkpoint_path(LORA_RESULTS_DIR)
    if not adapter_path:
        layout["body"].update(
            Panel(
                Align.center(
                    f"[red]No valid LoRA adapter found in '{LORA_RESULTS_DIR}'[/red]"
                ),
                expand=True,
            )
        )
        live.refresh()
        wait_for_key(layout, live)
        return None

    layout["body"].update(
        Panel(
            Align.center("[cyan]Merging LoRA adapter with base model...[/cyan]"),
            expand=True,
        )
    )
    live.refresh()

    try:
        if os.path.exists(LORA_MERGED_TEMP_DIR):
            shutil.rmtree(LORA_MERGED_TEMP_DIR)
        os.makedirs(LORA_MERGED_TEMP_DIR)
        base_model = AutoModelForSequenceClassification.from_pretrained(
            BASE_MODEL_NAME, num_labels=NUM_LABELS
        )
        model = PeftModel.from_pretrained(base_model, adapter_path)
        model = model.merge_and_unload()
        model.save_pretrained(LORA_MERGED_TEMP_DIR)
        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)
        tokenizer.save_pretrained(LORA_MERGED_TEMP_DIR)
        layout["body"].update(
            Panel(
                Align.center(
                    f"[green]Model merged and saved to '{LORA_MERGED_TEMP_DIR}'[/green]"
                ),
                expand=True,
            )
        )
        live.refresh()
        sleep(1.5)
        return LORA_MERGED_TEMP_DIR
    except Exception as e:
        console.print_exception()
        layout["body"].update(
            Panel(
                Align.center(f"[red]Error merging LoRA model: {e}[/red]"), expand=True
            )
        )
        live.refresh()
        wait_for_key(layout, live)
        return None


def run_quantize_model(layout, live):
    model_type_choice = interactive_menu_in_layout(
        layout,
        live,
        "Quantize Which Model Type?",
        ["Full Fine-tuned Model", "LoRA Model (will be merged)", "Cancel"],
    )
    model_path, is_temp_model = None, False
    if model_type_choice == 0:
        model_path = get_latest_checkpoint_path(FULL_RESULTS_DIR)
        if not model_path:
            layout["body"].update(
                Panel(
                    Align.center(
                        f"[red]No valid full model found in '{FULL_RESULTS_DIR}'[/red]"
                    ),
                    expand=True,
                )
            )
            live.refresh()
            wait_for_key(layout, live)
            return
    elif model_type_choice == 1:
        if not PEFT_AVAILABLE:
            layout["body"].update(
                Panel(Align.center("[red]PEFT library not installed.[/red]"))
            )
            live.refresh()
            wait_for_key(layout, live)
            return
        model_path = _merge_lora_model(layout, live)
        if not model_path:
            return
        is_temp_model = True
    else:
        return

    quant_level_choice = interactive_menu_in_layout(
        layout,
        live,
        "Select Quantization Level (`quanto`)",
        ["INT8 (Good Balance)", "FLOAT8 (Faster, Newer GPUs/MPS)", "Cancel"],
    )
    quant_level, output_dir = None, None
    if quant_level_choice == 0:
        quant_level = "int8"
        output_dir = RESULTS_QUANTIZED_INT8_DIR
    elif quant_level_choice == 1:
        quant_level = "float8"
        output_dir = RESULTS_QUANTIZED_FLOAT8_DIR
    else:
        if is_temp_model and os.path.exists(model_path):
            shutil.rmtree(model_path)
        return

    _run_quantization_script(
        layout, live, "quantize.py", model_path, output_dir, quant_level
    )

    if is_temp_model and os.path.exists(model_path):
        shutil.rmtree(model_path)
        console.print(f"[dim]Cleaned up temporary directory: {model_path}[/dim]")


############################################################################
# SOURCES MANAGEMENT
############################################################################
def run_manage_sources(layout, live):
    json_file = FEEDS_FILE
    try:
        data = {"feeds": []}
        if os.path.exists(json_file):
            with open(json_file, "r", encoding="utf-8") as f:
                loaded_data = json.load(f)
                if "feeds" in loaded_data and isinstance(loaded_data["feeds"], list):
                    data = loaded_data
    except (json.JSONDecodeError, IOError) as e:
        console.print(
            f"[red]Error loading feeds file {json_file}: {e}. Starting with empty list.[/red]"
        )
        data = {"feeds": []}

    needs_saving = False
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
        feeds = data.setdefault("feeds", [])

        if choice == -1 or choice == 5:
            break
        elif choice == 0:
            _list_sources(feeds, layout, live)
        elif choice == 1 and _add_source(feeds, layout, live):
            needs_saving = True
        elif choice == 2 and _remove_source(feeds, layout, live):
            needs_saving = True
        elif choice == 3 and _toggle_sources_status(feeds, layout, live):
            needs_saving = True
        elif choice == 4 and _import_curated_feeds(data, layout, live):
            needs_saving = True

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


def _list_sources(feeds, layout, live):
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
        if not url.startswith(("http://", "https://")):
            layout["body"].update(
                Panel(Align.center("[red]Invalid URL format.[/red]"), expand=True)
            )
            live.refresh()
            sleep(2)
            return False
        feeds.append({"name": name, "category": category, "url": url, "enabled": True})
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
    if not feeds:
        layout["body"].update(
            Panel(Align.center("[yellow]No sources to toggle.[/yellow]"), expand=True)
        )
        live.refresh()
        wait_for_key(layout, live)
        return False
    position, changed = 0, False
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
            status_text = (
                "[green]Enabled[/green]"
                if feed.get("enabled", True)
                else "[red]Disabled[/red]"
            )
            table.add_row(
                str(i + 1),
                feed.get("name", "N/A"),
                status_text,
                style="reverse blue" if i == position else "",
            )
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
            feeds[position]["enabled"] = not feeds[position].get("enabled", True)
            changed = True
        elif key.lower() == "q":
            break
    return changed


def _import_curated_feeds(data, layout, live):
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
        },
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
    feeds = data.setdefault("feeds", [])
    existing_urls = {feed.get("url") for feed in feeds if feed.get("url")}
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
# MODEL TESTING
############################################################################
def get_latest_checkpoint_path(results_dir):
    if not os.path.exists(results_dir) or not os.path.isdir(results_dir):
        return None
    checkpoint_dirs = [
        d
        for d in os.listdir(results_dir)
        if os.path.isdir(os.path.join(results_dir, d)) and d.startswith("checkpoint-")
    ]
    if checkpoint_dirs:
        try:
            latest_checkpoint_name = max(
                checkpoint_dirs, key=lambda p: int(p.split("-")[-1])
            )
            return os.path.join(results_dir, latest_checkpoint_name)
        except (ValueError, TypeError):
            pass
    final_adapter_path = os.path.join(results_dir, "final_adapter")
    if os.path.isdir(final_adapter_path) and os.path.exists(
        os.path.join(final_adapter_path, "adapter_config.json")
    ):
        return final_adapter_path
    if os.path.exists(os.path.join(results_dir, "adapter_config.json")):
        return results_dir
    return None


def classify_headline(headline, tokenizer, model, device):
    if not headline or not isinstance(headline, str):
        return {"headline": headline, "predicted": "Invalid Input", "confidence": 0.0}
    try:
        inputs = tokenizer(
            headline,
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            max_length=MAX_LENGTH,
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
            probabilities = torch.softmax(outputs.logits, dim=-1)
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
    results_list = []
    max_results = 12
    while True:
        table = Table(
            title=f"Classification History ({model_type_name})",
            box=None,
            padding=(0, 1),
            expand=True,
        )
        table.add_column(
            "Input Headline", style="yellow", overflow="fold", no_wrap=False
        )
        table.add_column("Prediction", style="bold green", width=15)
        table.add_column("Conf.", style="cyan", width=6, justify="right")
        for r in reversed(results_list):
            table.add_row(r["headline"], r["predicted"], f"{r['confidence']:.2f}")
        results_panel = (
            Panel(table, border_style="blue", expand=True)
            if results_list
            else Panel(
                Align.center("[dim]Enter a headline below.[/dim]"),
                border_style="blue",
                expand=True,
            )
        )
        prompt_panel = Panel(
            Align.center(
                "[bold magenta]Enter headline to classify (or 'q' to quit):[/bold magenta]"
            ),
            height=3,
            border_style="magenta",
        )
        sub_layout = Layout()
        sub_layout.split_column(
            Layout(results_panel, name="results"), Layout(prompt_panel, name="prompt")
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
        headline = Prompt.ask("Headline").strip()
        if not headline:
            continue
        if headline.lower() == "q":
            break
        result = classify_headline(headline, tokenizer, model, device)
        results_list.append(result)
        if len(results_list) > max_results:
            results_list.pop(0)
    wait_for_key(layout, live, prompt="Press any key to return to main menu...")


def run_test_model(layout, live):
    layout["body"].update(
        Panel(
            Align.center("[cyan]Loading full fine-tuned model...[/cyan]"), expand=True
        )
    )
    live.refresh()
    model_path = get_latest_checkpoint_path(FULL_RESULTS_DIR)
    if not model_path:
        layout["body"].update(
            Panel(
                Align.center(
                    f"[red]No valid model found in '{FULL_RESULTS_DIR}'[/red]"
                ),
                expand=True,
            )
        )
        live.refresh()
        wait_for_key(layout, live)
        return
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)
        model = (
            AutoModelForSequenceClassification.from_pretrained(model_path)
            .to(device)
            .eval()
        )
        _run_testing_loop(layout, live, tokenizer, model, device, "Full Fine-tune")
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
    if not PEFT_AVAILABLE:
        layout["body"].update(
            Panel(Align.center("[red]PEFT library not installed.[/red]"), expand=True)
        )
        live.refresh()
        wait_for_key(layout, live)
        return
    layout["body"].update(
        Panel(Align.center("[cyan]Loading LoRA model...[/cyan]"), expand=True)
    )
    live.refresh()
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
        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)
        base_model = AutoModelForSequenceClassification.from_pretrained(
            BASE_MODEL_NAME, num_labels=NUM_LABELS
        )
        model = PeftModel.from_pretrained(base_model, adapter_path).to(device).eval()
        _run_testing_loop(layout, live, tokenizer, model, device, "LoRA")
    except Exception as e:
        console.print_exception()
        layout["body"].update(
            Panel(
                Align.center(f"[red]Error loading LoRA model: {e}[/red]"), expand=True
            )
        )
        live.refresh()
        wait_for_key(layout, live)


def run_test_quantized_model(layout, live):
    choice = interactive_menu_in_layout(
        layout,
        live,
        "Test Which Quantized Model?",
        ["INT8 Model", "FLOAT8 Model", "Cancel"],
    )
    model_dir, model_type_name = None, None
    if choice == 0:
        model_dir, model_type_name = RESULTS_QUANTIZED_INT8_DIR, "Quantized INT8"
    elif choice == 1:
        model_dir, model_type_name = RESULTS_QUANTIZED_FLOAT8_DIR, "Quantized FLOAT8"
    else:
        return

    if not os.path.exists(os.path.join(model_dir, "config.json")):
        layout["body"].update(
            Panel(Align.center(f"[red]No model found in '{model_dir}'[/red]"))
        )
        live.refresh()
        wait_for_key(layout, live)
        return

    layout["body"].update(
        Panel(Align.center(f"[cyan]Loading {model_type_name} model...[/cyan]"))
    )
    live.refresh()
    try:
        device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else ("mps" if torch.backends.mps.is_available() else "cpu")
        )
        console.print(f"[dim]Using device: {device}[/dim]")
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        model = (
            AutoModelForSequenceClassification.from_pretrained(
                model_dir, trust_remote_code=True
            )
            .to(device)
            .eval()
        )
        _run_testing_loop(layout, live, tokenizer, model, device, model_type_name)
    except Exception as e:
        console.print_exception()
        layout["body"].update(
            Panel(
                Align.center(
                    f"[red]Error loading quantized model: {e}[/red]\nCheck if 'quanto' is installed and `trust_remote_code` is set."
                ),
                expand=True,
            )
        )
        live.refresh()
        wait_for_key(layout, live)


def run_benchmark(layout, live):
    """Run comprehensive model benchmarking"""
    layout["body"].update(
        Panel(
            Align.center("[cyan]Starting comprehensive model benchmark...[/cyan]"),
            expand=True,
            border_style="cyan",
        )
    )
    live.refresh()
    
    if not os.path.isfile("benchmark.py"):
        layout["body"].update(
            Panel(
                Align.center(
                    "[red]Benchmark script 'benchmark.py' not found.[/red]"
                ),
                expand=True,
            )
        )
        live.refresh()
        wait_for_key(layout, live)
        return
    
    cmd = [sys.executable, "benchmark.py"]
    process = None
    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
    except Exception as e:
        layout["body"].update(
            Panel(
                Align.center(f"[red]Error starting benchmark: {e}[/red]"),
                expand=True,
            )
        )
        live.refresh()
        wait_for_key(layout, live)
        return
    
    log_lines = []
    max_log_lines = 20
    
    while True:
        line = process.stdout.readline() if process.stdout else None
        if not line and process.poll() is not None:
            break
        if line:
            line = line.strip()
            if line:
                log_lines.append(line)
                if len(log_lines) > max_log_lines:
                    log_lines.pop(0)
                
                log_panel = Panel(
                    Align.left("\n".join(log_lines)),
                    border_style="yellow",
                    title="Benchmark Progress",
                    expand=True
                )
                layout["body"].update(log_panel)
                live.refresh()
        else:
            sleep(0.1)
    
    retcode = process.poll()
    if retcode == 0:
        final_message = "[bold green]Benchmark completed successfully![/bold green]\n\nResults saved to ./benchmark_results/"
        
        # Offer to generate visualization report
        choice = interactive_menu_in_layout(
            layout,
            live,
            "Benchmark Complete",
            ["Generate Visualization Report", "Return to Main Menu"]
        )
        
        if choice == 0:
            layout["body"].update(
                Panel(
                    Align.center("[cyan]Generating visualization report...[/cyan]"),
                    expand=True,
                )
            )
            live.refresh()
            
            try:
                viz_cmd = [sys.executable, "benchmark_visualizer.py"]
                viz_process = subprocess.run(
                    viz_cmd,
                    capture_output=True,
                    text=True,
                    timeout=60
                )
                
                if viz_process.returncode == 0:
                    final_message = "[bold green]Benchmark and visualization completed![/bold green]\n\nCheck ./benchmark_reports/ for detailed charts and HTML report."
                else:
                    final_message = "[bold green]Benchmark completed![/bold green]\n\n[yellow]Visualization generation failed. Check benchmark_visualizer.py[/yellow]"
            except Exception as e:
                final_message = f"[bold green]Benchmark completed![/bold green]\n\n[yellow]Visualization error: {e}[/yellow]"
        
        layout["body"].update(
            Panel(
                Align.center(final_message),
                expand=True,
                border_style="green"
            )
        )
    else:
        layout["body"].update(
            Panel(
                Align.center(
                    f"[bold red]Benchmark failed (Exit Code: {retcode})[/bold red]\n\nCheck the logs above for details."
                ),
                expand=True,
                border_style="red"
            )
        )
    
    live.refresh()
    wait_for_key(layout, live)


############################################################################
# MAIN APPLICATION LOOP
############################################################################
def main():
    os.makedirs(FULL_RESULTS_DIR, exist_ok=True)
    os.makedirs(LORA_RESULTS_DIR, exist_ok=True)
    os.makedirs(LORA_MERGED_TEMP_DIR, exist_ok=True)
    os.makedirs(RESULTS_QUANTIZED_INT8_DIR, exist_ok=True)
    os.makedirs(RESULTS_QUANTIZED_FLOAT8_DIR, exist_ok=True)

    if not os.path.exists(DATASET_FILE):
        console.print(
            f"[yellow]Dataset file '{DATASET_FILE}' not found. Creating empty file.[/yellow]"
        )
        with open(DATASET_FILE, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(["text", "label"])

    if not os.path.exists(FEEDS_FILE):
        console.print(
            f"[yellow]Feeds file '{FEEDS_FILE}' not found. Creating empty file.[/yellow]"
        )
        with open(FEEDS_FILE, "w", encoding="utf-8") as f:
            json.dump({"feeds": []}, f, indent=2)

    layout = Layout()
    layout.split(
        Layout(
            Panel(Align.center(app_title), border_style="bold blue"),
            name="header",
            size=10,
        ),
        Layout(
            Panel(
                Align.center("[dim]Welcome! Select an option.[/dim]"),
                border_style="green",
            ),
            name="body",
        ),
        Layout(
            Panel(
                Align.center(
                    "[bold cyan]Navigate:[/bold cyan] ↑/↓/j/k   [bold cyan]Select:[/bold cyan] Enter   [bold cyan]Quit:[/bold cyan] q"
                ),
                border_style="dim",
            ),
            name="footer",
            size=3,
        ),
    )

    main_menu_items = [
        "1. Create/Update Dataset",
        "2. Train Model (Full)",
        "3. Train Model (LoRA)",
        "4. Quantize & Evaluate Model",
        "5. Test Model (Full)",
        "6. Test Model (LoRA)",
        "7. Test Quantized Model",
        "8. Run Model Benchmark",
        "9. Manage Feed Sources",
        "10. Exit",
    ]

    with Live(
        layout,
        console=console,
        auto_refresh=False,
        screen=True,
        vertical_overflow="visible",
    ) as live:
        live.refresh()
        while True:
            choice = interactive_menu_in_layout(
                layout, live, "SlopBERT Main Menu", main_menu_items
            )
            if choice == -1 or choice == 9:
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
                run_quantize_model(layout, live)
            elif choice == 4:
                run_test_model(layout, live)
            elif choice == 5:
                run_test_lora_model(layout, live)
            elif choice == 6:
                run_test_quantized_model(layout, live)
            elif choice == 7:
                run_benchmark(layout, live)
            elif choice == 8:
                run_manage_sources(layout, live)


if __name__ == "__main__":
    main()
