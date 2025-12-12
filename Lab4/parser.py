"""
Модуль парсинга веб-страниц
"""

import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse, urljoin
import time
from typing import List, Dict, Set, Tuple, Optional
import logging
from utils import normalize_url, extract_links, clean_text, tokenize, logger
from config import PARSER_CONFIG, STOP_WORDS
from database import Database


class WebParser:
    """Класс для парсинга веб-страниц"""

    def __init__(self, db: Database):
        self.db = db
        self.visited_urls: Set[str] = set()
        self.urls_to_visit: List[str] = []
        self.session = requests.Session()
        self.session.headers.update({'User-Agent': PARSER_CONFIG['user_agent']})
        self.max_pages = PARSER_CONFIG['max_pages']
        self.max_content_length = PARSER_CONFIG['max_content_length']

        logger.info("WebParser initialized")

    def add_start_urls(self, urls: List[str]):
        """Добавление начальных URL для парсинга"""
        for url in urls:
            normalized_url = normalize_url('', url)
            if normalized_url not in self.visited_urls:
                self.urls_to_visit.append(normalized_url)

        logger.info(f"Added {len(urls)} start URLs")

    def parse_page(self, url: str) -> Optional[Tuple[str, str, str, List[str]]]:
        """
        Парсинг одной страницы
        Возвращает: (title, content, список ссылок) или None при ошибке
        """
        try:
            logger.info(f"Parsing: {url}")

            # Загрузка страницы
            response = self.session.get(url, timeout=PARSER_CONFIG['timeout'])
            response.raise_for_status()

            # Проверка размера контента
            if len(response.content) > self.max_content_length:
                logger.warning(f"Content too large for {url}, skipping")
                return None

            # Проверка типа контента
            content_type = response.headers.get('content-type', '')
            if 'text/html' not in content_type:
                logger.warning(f"Non-HTML content for {url}: {content_type}")
                return None

            # Парсинг HTML
            soup = BeautifulSoup(response.content, 'html.parser')

            # Извлечение заголовка
            title_tag = soup.find('title')
            title = title_tag.get_text(strip=True) if title_tag else "No title"

            # Извлечение основного текста
            # Удаление скриптов и стилей
            for script in soup(["script", "style", "nav", "footer", "header"]):
                script.decompose()

            # Получение текста
            text = soup.get_text(separator=' ', strip=True)

            # Очистка текста
            content = clean_text(text)

            # Извлечение ссылок
            links = extract_links(response.text, url)

            logger.debug(f"Parsed: {url} - Title: {title[:50]}... - Links: {len(links)}")

            return title, content, links

        except requests.RequestException as e:
            logger.error(f"Error downloading {url}: {e}")
            return None
        except Exception as e:
            logger.error(f"Error parsing {url}: {e}")
            return None

    def index_document(self, url: str, title: str, content: str, links: List[str]):
        """Индексация документа в базе данных"""
        try:
            # Добавление документа
            doc_id = self.db.add_document(url, title, content)

            if doc_id == -1:
                logger.error(f"Failed to add document {url} to database")
                return

            # Токенизация контента
            tokens = tokenize(content, STOP_WORDS)

            if not tokens:
                logger.warning(f"No tokens found in document {url}")
                # Все равно добавляем ссылки
                for link in links:
                    target_doc_id = self.db.get_document_id(link)
                    self.db.add_link(doc_id, link, target_doc_id)
                return

            # Расчет TF и позиций
            from collections import Counter
            total_terms = len(tokens)
            term_counts = Counter(tokens)

            # Добавление слов и создание обратного индекса
            positions_cache = {}
            for i, token in enumerate(tokens):
                if token not in positions_cache:
                    positions_cache[token] = []
                positions_cache[token].append(i)

            # Обрабатываем каждое уникальное слово в документе
            for word, count in term_counts.items():
                # Добавление слова
                word_id = self.db.add_word(word)
                if word_id == -1:
                    continue

                # Расчет TF (сохраняем как count для более точного восстановления)
                # Вместо tf = count / total_terms, сохраняем частоту
                tf = count  # Сохраняем абсолютную частоту

                # Получение позиций
                positions = positions_cache.get(word, [])

                # Добавление в обратный индекс
                self.db.add_inverted_index_entry(word_id, doc_id, tf, positions)

            # Добавление ссылок
            for link in links:
                # Проверяем, есть ли целевой документ уже в базе
                target_doc_id = self.db.get_document_id(link)
                self.db.add_link(doc_id, link, target_doc_id)

            logger.info(f"Indexed: {url} (ID: {doc_id}, Words: {len(term_counts)}, Links: {len(links)})")

        except Exception as e:
            logger.error(f"Error indexing document {url}: {e}")

    def crawl(self, start_urls: List[str], max_pages: Optional[int] = None):
        """
        Основной метод краулинга
        """
        if max_pages:
            self.max_pages = max_pages

        # Добавление начальных URL
        self.add_start_urls(start_urls)

        pages_parsed = 0

        while self.urls_to_visit and pages_parsed < self.max_pages:
            # Берем следующий URL
            url = self.urls_to_visit.pop(0)

            # Пропускаем уже посещенные
            if url in self.visited_urls:
                continue

            # Парсинг страницы
            result = self.parse_page(url)
            if not result:
                self.visited_urls.add(url)
                continue

            title, content, links = result

            # Индексация документа
            self.index_document(url, title, content, links)

            # Добавление в посещенные
            self.visited_urls.add(url)
            pages_parsed += 1

            # Добавление новых ссылок в очередь
            for link in links:
                if (link not in self.visited_urls and
                        link not in self.urls_to_visit):
                    self.urls_to_visit.append(link)

            # Задержка для избежания блокировки
            time.sleep(0.5)

            # Логирование прогресса
            if pages_parsed % 10 == 0:
                logger.info(f"Progress: {pages_parsed} pages parsed, {len(self.urls_to_visit)} in queue")

        logger.info(f"Crawling completed. Total pages parsed: {pages_parsed}")

    def get_statistics(self) -> Dict[str, int]:
        """Получение статистики парсинга"""
        return {
            'visited_urls': len(self.visited_urls),
            'urls_to_visit': len(self.urls_to_visit),
            'total_documents': self.db.get_total_documents(),
        }