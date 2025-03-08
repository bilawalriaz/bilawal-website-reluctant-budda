import sqlite3
from datetime import datetime

class RequestLogger:
    def __init__(self, db_path='requests.db'):
        self.conn = sqlite3.connect(db_path)
        self._create_table()

    def _create_table(self):
        cursor = self.conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS requests (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                ip_address TEXT,
                user_agent TEXT,
                headers TEXT,
                query TEXT,
                response TEXT,
                generation_time REAL
            )
        ''')
        # Check for existing columns
        cursor.execute("PRAGMA table_info(requests)")
        columns = [column[1] for column in cursor.fetchall()]
        for column in ['user_agent', 'headers', 'generation_time']:
            if column not in columns:
                cursor.execute(f'ALTER TABLE requests ADD COLUMN {column} TEXT')
        self.conn.commit()

    def log_request(self, query, response, generation_time, client_ip, headers):
        user_agent = headers.get('user-agent', '')
        cursor = self.conn.cursor()
        cursor.execute('''
            INSERT INTO requests (timestamp, ip_address, user_agent, headers, query, response, generation_time)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (datetime.now().isoformat(), client_ip, user_agent, str(headers), query, response, generation_time))
        self.conn.commit()

    def close(self):
        self.conn.close()
