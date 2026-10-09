"""SQLite persistence. Every private query includes the authenticated owner."""
import json
import sqlite3
import time
import uuid
from contextlib import contextmanager


class Store:
    def __init__(self, path):
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        with self.connection() as db:
            db.executescript("""
            PRAGMA journal_mode=WAL;
            CREATE TABLE IF NOT EXISTS conversations(
              id TEXT PRIMARY KEY, owner TEXT NOT NULL, title TEXT NOT NULL,
              created REAL NOT NULL, updated REAL NOT NULL, sample INTEGER DEFAULT 0);
            CREATE INDEX IF NOT EXISTS owner_chats ON conversations(owner, updated);
            CREATE TABLE IF NOT EXISTS messages(
              id TEXT PRIMARY KEY, chat_id TEXT REFERENCES conversations(id) ON DELETE CASCADE,
              role TEXT NOT NULL, content TEXT NOT NULL, detail TEXT NOT NULL, created REAL NOT NULL);
            CREATE TABLE IF NOT EXISTS documents(
              id TEXT PRIMARY KEY, owner TEXT NOT NULL, name TEXT NOT NULL,
              content TEXT NOT NULL, created REAL NOT NULL);
            CREATE TABLE IF NOT EXISTS usage(
              owner TEXT NOT NULL, created REAL NOT NULL);
            CREATE INDEX IF NOT EXISTS usage_date ON usage(created);
            """)

    @contextmanager
    def connection(self):
        with sqlite3.connect(self.path, timeout=15) as db:
            db.row_factory = sqlite3.Row
            db.execute("PRAGMA foreign_keys=ON")
            yield db

    def create_chat(self, owner, title="New research", sample=False):
        now, identity = time.time(), uuid.uuid4().hex
        with self.connection() as db:
            db.execute("INSERT INTO conversations VALUES (?,?,?,?,?,?)", (identity, owner, title[:100], now, now, int(sample)))
        return self.chat(owner, identity)

    def chats(self, owner):
        with self.connection() as db:
            return [dict(row) for row in db.execute("SELECT id,title,created,updated,sample FROM conversations WHERE owner=? ORDER BY updated DESC LIMIT 200", (owner,))]

    def chat(self, owner, identity):
        with self.connection() as db:
            row = db.execute("SELECT id,title,created,updated,sample FROM conversations WHERE owner=? AND id=?", (owner, identity)).fetchone()
            if not row:
                raise KeyError(identity)
            messages = []
            for msg in db.execute("SELECT * FROM messages WHERE chat_id=? ORDER BY created, rowid", (identity,)):
                item = dict(msg)
                item["detail"] = json.loads(item["detail"])
                messages.append(item)
            return {**dict(row), "messages": messages}

    def add_message(self, owner, identity, role, content, detail=None):
        now, message_id = time.time(), uuid.uuid4().hex
        with self.connection() as db:
            db.execute("BEGIN IMMEDIATE")
            if not db.execute("SELECT 1 FROM conversations WHERE id=? AND owner=?", (identity, owner)).fetchone():
                raise KeyError(identity)
            db.execute("INSERT INTO messages VALUES (?,?,?,?,?,?)", (message_id, identity, role, content, json.dumps(detail or {}, ensure_ascii=False), now))
            db.execute("UPDATE conversations SET updated=? WHERE id=?", (now, identity))
            if role == "user":
                db.execute("UPDATE conversations SET title=? WHERE id=? AND title='New research'", (content[:85], identity))
        return message_id

    def delete_chat(self, owner, identity):
        with self.connection() as db:
            return bool(db.execute("DELETE FROM conversations WHERE owner=? AND id=?", (owner, identity)).rowcount)

    def add_document(self, owner, name, content):
        identity = uuid.uuid4().hex
        with self.connection() as db:
            db.execute("BEGIN IMMEDIATE")
            if db.execute("SELECT count(*) FROM documents WHERE owner=?", (owner,)).fetchone()[0] >= 30:
                raise ValueError("Your library is full (30 documents). Remove a document first.")
            db.execute("INSERT INTO documents VALUES (?,?,?,?,?)", (identity, owner, name[:160], content[:40000], time.time()))
        return identity

    def documents(self, owner):
        with self.connection() as db:
            return [dict(row) for row in db.execute("SELECT id,name,created,length(content) AS characters FROM documents WHERE owner=? ORDER BY created DESC", (owner,))]

    def document(self, owner, identity):
        with self.connection() as db:
            row = db.execute("SELECT * FROM documents WHERE owner=? AND id=?", (owner, identity)).fetchone()
            if not row:
                raise KeyError(identity)
            return dict(row)

    def delete_document(self, owner, identity):
        with self.connection() as db:
            return bool(db.execute("DELETE FROM documents WHERE owner=? AND id=?", (owner, identity)).rowcount)

    def reserve_usage(self, owner, per_user, total):
        now = time.time()
        with self.connection() as db:
            db.execute("BEGIN IMMEDIATE")
            db.execute("DELETE FROM usage WHERE created<?", (now - 86400,))
            user_count = db.execute("SELECT count(*) FROM usage WHERE owner=?", (owner,)).fetchone()[0]
            total_count = db.execute("SELECT count(*) FROM usage").fetchone()[0]
            recent = db.execute("SELECT count(*) FROM usage WHERE owner=? AND created>?", (owner, now - 60)).fetchone()[0]
            if user_count >= per_user or total_count >= total or recent >= 6:
                raise ValueError("Research request limit reached. Please try again later.")
            db.execute("INSERT INTO usage VALUES (?,?)", (owner, now))

    def delete_account_data(self, owner):
        with self.connection() as db:
            db.execute("DELETE FROM conversations WHERE owner=?", (owner,))
            db.execute("DELETE FROM documents WHERE owner=?", (owner,))
            # Keep non-content usage counters for 24h to prevent limit bypass.
