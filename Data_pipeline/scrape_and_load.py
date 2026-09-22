#!/usr/bin/env python3
"""
Zepto Data Pipeline Module 1
Scrape → Clean → Fixed-rate convert → Normalized SQLite → SQL + pandas
"""

import requests
from bs4 import BeautifulSoup
import pandas as pd
import sqlite3
from pathlib import Path
import re
import time

# ---------- Constants ----------
BASE_URL = "https://books.toscrape.com/"
GBP_TO_INR = 105.50          # REQUIRED fixed project baseline rate
DB_PATH = Path(__file__).parent / "books.db"
MIN_BOOKS = 60

RATING_MAP = {
    "One": 1, "Two": 2, "Three": 3, "Four": 4, "Five": 5
}

# Choose ≥3 categories that together easily exceed 60 books
CATEGORIES = [
    ("Travel", "catalogue/category/books/travel_2/index.html"),
    ("Mystery", "catalogue/category/books/mystery_3/index.html"),
    ("Historical Fiction", "catalogue/category/books/historical-fiction_4/index.html"),
    ("Fiction", "catalogue/category/books/fiction_10/index.html"),
    ("Nonfiction", "catalogue/category/books/nonfiction_13/index.html"),
    ("Fantasy", "catalogue/category/books/fantasy_19/index.html"),
]

# ---------- Helpers ----------
def get_soup(url: str) -> BeautifulSoup:
    resp = requests.get(url, timeout=15)
    resp.raise_for_status()
    return BeautifulSoup(resp.content, "lxml")

def parse_price(text: str) -> float | None:
    try:
        return float(re.sub(r"[^\d.]", "", text))
    except Exception:
        return None

def parse_rating(tag) -> int | None:
    classes = tag.get("class", [])
    for c in classes:
        if c in RATING_MAP:
            return RATING_MAP[c]
    return None

def parse_availability(text: str) -> bool | None:
    text = text.strip().lower()
    if "in stock" in text:
        return True
    if "out of stock" in text:
        return False
    return None

def scrape_category(category_name: str, relative_url: str) -> list[dict]:
    books = []
    url = BASE_URL + relative_url
    page = 1

    while True:
        print(f"  Scraping {category_name} page {page} …")
        soup = get_soup(url)
        articles = soup.select("article.product_pod")
        if not articles:
            break

        for art in articles:
            title = art.h3.a.get("title", "").strip()
            price_tag = art.select_one("p.price_color")
            rating_tag = art.select_one("p.star-rating")
            avail_tag = art.select_one("p.instock.availability")

            price_raw = price_tag.get_text(strip=True) if price_tag else ""
            rating_raw = " ".join(rating_tag.get("class", [])) if rating_tag else ""
            avail_raw = avail_tag.get_text(strip=True) if avail_tag else ""

            books.append({
                "title": title,
                "price_raw": price_raw,
                "rating_raw": rating_raw,
                "availability_raw": avail_raw,
                "category": category_name,
            })

        # next page?
        next_li = soup.select_one("li.next a")
        if next_li:
            next_href = next_li["href"]
            # build absolute next URL
            if relative_url.endswith("index.html"):
                base = relative_url.rsplit("/", 1)[0] + "/"
            else:
                base = relative_url.rsplit("/", 1)[0] + "/"
            url = BASE_URL + base + next_href
            page += 1
            time.sleep(0.3)          # polite
        else:
            break

    return books

# ---------- Main pipeline ----------
def main():
    print("=== 1. Scraping ===")
    all_raw = []
    for name, rel in CATEGORIES:
        all_raw.extend(scrape_category(name, rel))
        if len(all_raw) >= MIN_BOOKS:
            break

    print(f"Scraped {len(all_raw)} raw rows")

    # ---------- 2. Clean ----------
    print("=== 2. Cleaning ===")
    rows = []
    for r in all_raw:
        price_gbp = parse_price(r["price_raw"])
        rating = parse_rating(
            BeautifulSoup(f'<p class="{r["rating_raw"]}"></p>', "lxml").p
        ) if r["rating_raw"] else None
        in_stock = parse_availability(r["availability_raw"])

        # Decision: drop any row that fails critical parsing
        # (justified: site is clean; missing values are extremely rare;
        #  keeping incomplete rows would pollute the benchmark dataset)
        if price_gbp is None or rating is None or in_stock is None or not r["title"]:
            continue

        price_inr = round(price_gbp * GBP_TO_INR, 2)

        rows.append({
            "title": r["title"],
            "price_gbp": price_gbp,
            "price_inr": price_inr,
            "rating": rating,
            "in_stock": int(in_stock),   # store as INTEGER 0/1
            "category": r["category"],
        })

    df = pd.DataFrame(rows)
    print(f"Clean rows kept: {len(df)}")
    assert len(df) >= MIN_BOOKS, f"Need ≥{MIN_BOOKS} books, got {len(df)}"
    print(f"Categories present: {df['category'].nunique()}")

    # ---------- 3. SQLite schema + load ----------
    print("=== 3. Loading into SQLite ===")
    if DB_PATH.exists():
        DB_PATH.unlink()

    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    cur.executescript("""
    CREATE TABLE categories (
        category_id   INTEGER PRIMARY KEY,
        category_name TEXT UNIQUE NOT NULL
    );

    CREATE TABLE books (
        book_id      INTEGER PRIMARY KEY,
        title        TEXT NOT NULL,
        price_gbp    REAL NOT NULL,
        price_inr    REAL NOT NULL,
        rating       INTEGER NOT NULL CHECK(rating BETWEEN 1 AND 5),
        in_stock     INTEGER NOT NULL CHECK(in_stock IN (0,1)),
        category_id  INTEGER NOT NULL,
        FOREIGN KEY (category_id) REFERENCES categories(category_id)
    );
    """)

    # insert categories
    cats = sorted(df["category"].unique())
    cat_id_map = {}
    for i, name in enumerate(cats, 1):
        cur.execute("INSERT INTO categories (category_id, category_name) VALUES (?, ?)", (i, name))
        cat_id_map[name] = i

    # insert books
    for _, row in df.iterrows():
        cur.execute("""
            INSERT INTO books (title, price_gbp, price_inr, rating, in_stock, category_id)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            row["title"], row["price_gbp"], row["price_inr"],
            row["rating"], row["in_stock"], cat_id_map[row["category"]]
        ))

    conn.commit()
    print(f"Database written → {DB_PATH}")

    # ---------- 4. SQL queries (≥5, covering all required clauses + JOIN) ----------
    print("\n=== 4. SQL Queries ===")

    queries = {
        "Q1 – SELECT + WHERE (in-stock books under £20)": """
            SELECT title, price_gbp, rating
            FROM books
            WHERE in_stock = 1 AND price_gbp < 20
            LIMIT 10;
        """,
        "Q2 – ORDER BY + LIMIT (top 10 most expensive)": """
            SELECT title, price_gbp, price_inr
            FROM books
            ORDER BY price_gbp DESC
            LIMIT 10;
        """,
        "Q3 – DISTINCT categories": """
            SELECT DISTINCT category_name
            FROM categories
            ORDER BY category_name;
        """,
        "Q4 – BETWEEN + IN": """
            SELECT title, rating, price_gbp
            FROM books
            WHERE rating BETWEEN 4 AND 5
              AND category_id IN (SELECT category_id FROM categories
                                  WHERE category_name IN ('Fiction','Fantasy','Mystery'))
            ORDER BY rating DESC, price_gbp
            LIMIT 15;
        """,
        "Q5 – JOIN (10 highest-rated books per category – simplified top overall with category)": """
            SELECT b.title, b.rating, b.price_gbp, c.category_name
            FROM books b
            JOIN categories c ON b.category_id = c.category_id
            ORDER BY b.rating DESC, b.price_gbp ASC
            LIMIT 15;
        """,
        "Q6 – aggregate JOIN example": """
            SELECT c.category_name,
                   COUNT(*) AS book_count,
                   ROUND(AVG(b.price_gbp), 2) AS avg_price_gbp,
                   ROUND(AVG(b.rating), 2) AS avg_rating
            FROM books b
            JOIN categories c ON b.category_id = c.category_id
            GROUP BY c.category_name
            ORDER BY book_count DESC;
        """,
    }

    results = {}
    for name, sql in queries.items():
        print(f"\n--- {name} ---")
        print(sql.strip())
        res = pd.read_sql_query(sql, conn)
        print(res.to_string(index=False))
        results[name] = res

    # ---------- 5. pandas parity check for the JOIN ----------
    print("\n=== 5. pandas vs SQL JOIN parity ===")

    # SQL result
    sql_join = results["Q5 – JOIN (10 highest-rated books per category – simplified top overall with category)"]

    # pure pandas equivalent
    cats_df = pd.read_sql("SELECT * FROM categories", conn)
    books_df = pd.read_sql("SELECT * FROM books", conn)

    pandas_join = (
        books_df
        .merge(cats_df, on="category_id")
        .sort_values(["rating", "price_gbp"], ascending=[False, True])
        .loc[:, ["title", "rating", "price_gbp", "category_name"]]
        .head(15)
        .reset_index(drop=True)
    )

    print("\nSQL JOIN result (first 5 rows):")
    print(sql_join.head().to_string(index=False))
    print("\npandas merge result (first 5 rows):")
    print(pandas_join.head().to_string(index=False))

    # exact match check (after sorting/reset)
    sql_cmp = sql_join.reset_index(drop=True)
    assert sql_cmp.equals(pandas_join), "SQL and pandas results differ!"
    print("\n✓ SQL JOIN and pd.merge produce identical output")

    conn.close()
    print("\n=== Pipeline finished successfully ===")
    print(f"Total books: {len(df)}")
    print(f"Fixed conversion rate used: 1 GBP = {GBP_TO_INR} INR")

if __name__ == "__main__":
    main()