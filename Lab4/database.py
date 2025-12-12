"""
Модуль для работы с базой данных
"""

import sqlite3
import json
from typing import List, Dict, Set, Tuple, Any, Optional
import logging
from datetime import datetime
from utils import logger


class Database:
    """Класс для работы с базой данных поискового движка"""

    def __init__(self, db_name: str = 'search_engine.db'):
        self.db_name = db_name
        self.conn = None
        self.cursor = None
        self._initialize_database()

    def _initialize_database(self):
        """Инициализация базы данных и создание таблиц"""
        try:
            self.conn = sqlite3.connect(self.db_name)
            self.cursor = self.conn.cursor()

            # Таблица документов
            self.cursor.execute('''
                CREATE TABLE IF NOT EXISTS documents (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    url TEXT UNIQUE NOT NULL,
                    title TEXT,
                    content TEXT,
                    content_length INTEGER,
                    parsed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    indexed BOOLEAN DEFAULT FALSE
                )
            ''')

            # Таблица слов
            self.cursor.execute('''
                CREATE TABLE IF NOT EXISTS words (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    word TEXT UNIQUE NOT NULL,
                    frequency INTEGER DEFAULT 0
                )
            ''')

            # Таблица ссылок
            self.cursor.execute('''
                CREATE TABLE IF NOT EXISTS links (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    source_doc_id INTEGER NOT NULL,
                    target_url TEXT NOT NULL,
                    target_doc_id INTEGER,
                    FOREIGN KEY (source_doc_id) REFERENCES documents (id),
                    FOREIGN KEY (target_doc_id) REFERENCES documents (id),
                    UNIQUE(source_doc_id, target_url)
                )
            ''')

            # Таблица обратного индекса
            self.cursor.execute('''
                CREATE TABLE IF NOT EXISTS inverted_index (
                    word_id INTEGER NOT NULL,
                    doc_id INTEGER NOT NULL,
                    tf REAL DEFAULT 0.0,
                    positions TEXT,  -- JSON список позиций
                    PRIMARY KEY (word_id, doc_id),
                    FOREIGN KEY (word_id) REFERENCES words (id),
                    FOREIGN KEY (doc_id) REFERENCES documents (id)
                )
            ''')

            # Таблица PageRank
            self.cursor.execute('''
                CREATE TABLE IF NOT EXISTS pagerank (
                    doc_id INTEGER PRIMARY KEY,
                    pagerank REAL DEFAULT 1.0,
                    iteration INTEGER DEFAULT 0,
                    FOREIGN KEY (doc_id) REFERENCES documents (id)
                )
            ''')

            self.conn.commit()
            logger.info("Database initialized successfully")

        except sqlite3.Error as e:
            logger.error(f"Database initialization error: {e}")
            if self.conn:
                self.conn.rollback()
            raise

    def add_document(self, url: str, title: str, content: str) -> int:
        """Добавление документа в базу данных"""
        try:
            content_length = len(content)

            self.cursor.execute('''
                INSERT OR IGNORE INTO documents (url, title, content, content_length)
                VALUES (?, ?, ?, ?)
            ''', (url, title, content, content_length))

            # Получаем ID документа
            self.cursor.execute('SELECT id FROM documents WHERE url = ?', (url,))
            result = self.cursor.fetchone()

            if result:
                doc_id = result[0]
                # Обновляем title и content если документ уже существует
                self.cursor.execute('''
                    UPDATE documents 
                    SET title = ?, content = ?, content_length = ?
                    WHERE id = ?
                ''', (title, content, content_length, doc_id))
            else:
                self.cursor.execute('''
                    INSERT INTO documents (url, title, content, content_length)
                    VALUES (?, ?, ?, ?)
                ''', (url, title, content, content_length))
                doc_id = self.cursor.lastrowid

            self.conn.commit()
            logger.debug(f"Document added: {url} (ID: {doc_id})")
            return doc_id

        except sqlite3.Error as e:
            logger.error(f"Error adding document {url}: {e}")
            self.conn.rollback()
            return -1

    def get_document_id(self, url: str) -> Optional[int]:
        """Получение ID документа по URL"""
        try:
            self.cursor.execute('SELECT id FROM documents WHERE url = ?', (url,))
            result = self.cursor.fetchone()
            return result[0] if result else None
        except sqlite3.Error as e:
            logger.error(f"Error getting document ID for {url}: {e}")
            return None

    def add_word(self, word: str) -> int:
        """Добавление слова в базу данных"""
        try:
            # Пытаемся получить существующее слово
            self.cursor.execute('SELECT id FROM words WHERE word = ?', (word,))
            result = self.cursor.fetchone()

            if result:
                word_id = result[0]
                # Обновляем частоту
                self.cursor.execute('UPDATE words SET frequency = frequency + 1 WHERE id = ?', (word_id,))
            else:
                # Добавляем новое слово
                self.cursor.execute('INSERT INTO words (word, frequency) VALUES (?, 1)', (word,))
                word_id = self.cursor.lastrowid

            self.conn.commit()
            return word_id

        except sqlite3.Error as e:
            logger.error(f"Error adding word {word}: {e}")
            self.conn.rollback()
            return -1

    def add_link(self, source_doc_id: int, target_url: str, target_doc_id: Optional[int] = None):
        """Добавление ссылки между документами"""
        try:
            self.cursor.execute('''
                INSERT OR IGNORE INTO links (source_doc_id, target_url, target_doc_id)
                VALUES (?, ?, ?)
            ''', (source_doc_id, target_url, target_doc_id))

            self.conn.commit()

        except sqlite3.Error as e:
            logger.error(f"Error adding link from {source_doc_id} to {target_url}: {e}")
            self.conn.rollback()

    def add_inverted_index_entry(self, word_id: int, doc_id: int, tf: float, positions: List[int]):
        """Добавление записи в обратный индекс"""
        try:
            positions_json = json.dumps(positions)

            self.cursor.execute('''
                INSERT OR REPLACE INTO inverted_index (word_id, doc_id, tf, positions)
                VALUES (?, ?, ?, ?)
            ''', (word_id, doc_id, tf, positions_json))

            self.conn.commit()

        except sqlite3.Error as e:
            logger.error(f"Error adding inverted index entry: {e}")
            self.conn.rollback()

    def update_pagerank(self, doc_id: int, pagerank: float, iteration: int):
        """Обновление значения PageRank для документа"""
        try:
            self.cursor.execute('''
                INSERT OR REPLACE INTO pagerank (doc_id, pagerank, iteration)
                VALUES (?, ?, ?)
            ''', (doc_id, pagerank, iteration))

            self.conn.commit()

        except sqlite3.Error as e:
            logger.error(f"Error updating PageRank for document {doc_id}: {e}")
            self.conn.rollback()

    def get_all_documents(self) -> List[Tuple[int, str, str]]:
        """Получение всех документов"""
        try:
            self.cursor.execute('SELECT id, url, title FROM documents')
            return self.cursor.fetchall()
        except sqlite3.Error as e:
            logger.error(f"Error getting all documents: {e}")
            return []

    def get_document_content(self, doc_id: int) -> Optional[str]:
        """Получение содержимого документа по ID"""
        try:
            self.cursor.execute('SELECT content FROM documents WHERE id = ?', (doc_id,))
            result = self.cursor.fetchone()
            return result[0] if result else None
        except sqlite3.Error as e:
            logger.error(f"Error getting document content for ID {doc_id}: {e}")
            return None

    def get_document_url(self, doc_id: int) -> Optional[str]:
        """Получение URL документа по ID"""
        try:
            self.cursor.execute('SELECT url FROM documents WHERE id = ?', (doc_id,))
            result = self.cursor.fetchone()
            return result[0] if result else None
        except sqlite3.Error as e:
            logger.error(f"Error getting document URL for ID {doc_id}: {e}")
            return None

    def get_outgoing_links(self, doc_id: int) -> List[Tuple[int, str]]:
        """Получение исходящих ссылок документа"""
        try:
            self.cursor.execute('''
                SELECT target_doc_id, target_url 
                FROM links 
                WHERE source_doc_id = ?
            ''', (doc_id,))
            return self.cursor.fetchall()
        except sqlite3.Error as e:
            logger.error(f"Error getting outgoing links for document {doc_id}: {e}")
            return []

    def get_incoming_links(self, doc_id: int) -> List[int]:
        """Получение входящих ссылок документа"""
        try:
            self.cursor.execute('''
                SELECT source_doc_id 
                FROM links 
                WHERE target_doc_id = ?
            ''', (doc_id,))
            results = self.cursor.fetchall()
            return [row[0] for row in results] if results else []
        except sqlite3.Error as e:
            logger.error(f"Error getting incoming links for document {doc_id}: {e}")
            return []

    def get_documents_for_word(self, word: str) -> List[Tuple[int, float]]:
        """Получение документов, содержащих слово"""
        try:
            self.cursor.execute('''
                SELECT ii.doc_id, ii.tf
                FROM inverted_index ii
                JOIN words w ON ii.word_id = w.id
                WHERE w.word = ?
            ''', (word,))
            results = self.cursor.fetchall()

            # Преобразуем результаты: tf хранится как абсолютная частота
            formatted_results = []
            for doc_id, tf in results:
                formatted_results.append((doc_id, float(tf)))

            return formatted_results

        except sqlite3.Error as e:
            logger.error(f"Error getting documents for word {word}: {e}")
            return []

    def get_word_frequency(self, word: str) -> int:
        """Получение частоты слова"""
        try:
            self.cursor.execute('SELECT frequency FROM words WHERE word = ?', (word,))
            result = self.cursor.fetchone()
            return result[0] if result else 0
        except sqlite3.Error as e:
            logger.error(f"Error getting frequency for word {word}: {e}")
            return 0

    def get_total_documents(self) -> int:
        """Получение общего количества документов"""
        try:
            self.cursor.execute('SELECT COUNT(*) FROM documents')
            result = self.cursor.fetchone()
            return result[0] if result else 0
        except sqlite3.Error as e:
            logger.error(f"Error getting total documents count: {e}")
            return 0

    def get_pagerank(self, doc_id: int) -> float:
        """Получение значения PageRank для документа"""
        try:
            self.cursor.execute('SELECT pagerank FROM pagerank WHERE doc_id = ?', (doc_id,))
            result = self.cursor.fetchone()
            return result[0] if result else 1.0
        except sqlite3.Error as e:
            logger.error(f"Error getting PageRank for document {doc_id}: {e}")
            return 1.0

    def get_all_pageranks(self) -> Dict[int, float]:
        """Получение всех значений PageRank"""
        try:
            self.cursor.execute('SELECT doc_id, pagerank FROM pagerank')
            results = self.cursor.fetchall()
            return {doc_id: pagerank for doc_id, pagerank in results}
        except sqlite3.Error as e:
            logger.error(f"Error getting all PageRanks: {e}")
            return {}

    def get_document_info(self, doc_id: int) -> Optional[Tuple[str, str]]:
        """Получение информации о документе"""
        try:
            self.cursor.execute('SELECT url, title FROM documents WHERE id = ?', (doc_id,))
            result = self.cursor.fetchone()
            return result if result else None
        except sqlite3.Error as e:
            logger.error(f"Error getting document info for ID {doc_id}: {e}")
            return None

    def clear_database(self):
        """Очистка базы данных (для тестирования)"""
        try:
            tables = ['documents', 'words', 'links', 'inverted_index', 'pagerank']
            for table in tables:
                self.cursor.execute(f'DELETE FROM {table}')

            self.cursor.execute('DELETE FROM sqlite_sequence')
            self.conn.commit()
            logger.info("Database cleared successfully")

        except sqlite3.Error as e:
            logger.error(f"Error clearing database: {e}")
            self.conn.rollback()

    def close(self):
        """Закрытие соединения с базой данных"""
        if self.conn:
            self.conn.close()
            logger.info("Database connection closed")

    def __del__(self):
        """Деструктор"""
        self.close()