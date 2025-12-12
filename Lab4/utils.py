"""
Вспомогательные функции
"""

import re
import string
import hashlib
from typing import List, Dict, Set, Tuple, Any
from urllib.parse import urljoin, urlparse
from collections import defaultdict
import logging

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def normalize_url(base_url: str, url: str) -> str:
    """
    Нормализация URL
    """
    try:
        # Преобразование относительного URL в абсолютный
        absolute_url = urljoin(base_url, url)

        # Удаление фрагментов (якорей)
        parsed = urlparse(absolute_url)

        # Удаление параметров запроса для нормализации
        normalized_url = f"{parsed.scheme}://{parsed.netloc}{parsed.path}"

        # Удаление слеша в конце
        if normalized_url.endswith('/'):
            normalized_url = normalized_url[:-1]

        return normalized_url.lower()
    except Exception as e:
        logger.error(f"Error normalizing URL {url}: {e}")
        return url


def extract_links(html_content: str, base_url: str) -> List[str]:
    """
    Извлечение ссылок из HTML контента
    """
    import re

    # Регулярное выражение для поиска ссылок
    link_pattern = r'href=[\'"]?([^\'" >]+)'
    links = re.findall(link_pattern, html_content, re.IGNORECASE)

    # Нормализация ссылок
    normalized_links = []
    for link in links:
        # Пропускаем якорные ссылки и javascript
        if link.startswith('#') or link.startswith('javascript:'):
            continue

        normalized = normalize_url(base_url, link)
        if normalized:
            normalized_links.append(normalized)

    return list(set(normalized_links))  # Удаление дубликатов


def clean_text(text: str) -> str:
    """
    Очистка текста от лишних символов и приведение к нижнему регистру
    """
    # Удаление HTML тегов
    text = re.sub(r'<[^>]+>', ' ', text)

    # Удаление специальных символов
    text = re.sub(r'[^\w\s]', ' ', text)

    # Замена множественных пробелов на один
    text = re.sub(r'\s+', ' ', text)

    return text.strip().lower()


def tokenize(text: str, stop_words: Set[str] = None) -> List[str]:
    """
    Токенизация текста с удалением стоп-слов
    """
    if stop_words is None:
        stop_words = set()

    # Очистка текста
    text = clean_text(text)

    # Разделение на токены
    tokens = text.split()

    # Удаление стоп-слов
    tokens = [token for token in tokens if token not in stop_words]

    return tokens


def calculate_tf(tokens: List[str]) -> Dict[str, float]:
    """
    Расчет TF (Term Frequency)
    """
    if not tokens:
        return {}

    total_terms = len(tokens)
    tf = {}

    for token in tokens:
        tf[token] = tf.get(token, 0) + 1

    # Нормализация
    for token in tf:
        tf[token] = tf[token] / total_terms

    return tf


def generate_snippet(text: str, query_terms: List[str], max_length: int = 150) -> str:
    """
    Генерация сниппета с подсветкой найденных терминов
    """
    if not text:
        return ""

    # Находим позиции всех терминов запроса
    positions = []
    for term in query_terms:
        start = 0
        while True:
            pos = text.lower().find(term.lower(), start)
            if pos == -1:
                break
            positions.append((pos, pos + len(term)))
            start = pos + 1

    if not positions:
        # Если термины не найдены, берем начало текста
        return text[:max_length] + "..." if len(text) > max_length else text

    # Берем позицию первого найденного термина
    first_pos = min(positions, key=lambda x: x[0])[0]

    # Выделяем контекст вокруг первого термина
    start = max(0, first_pos - max_length // 2)
    end = min(len(text), start + max_length)

    snippet = text[start:end]

    # Добавляем многоточия если нужно
    if start > 0:
        snippet = "..." + snippet
    if end < len(text):
        snippet = snippet + "..."

    return snippet


class TextProcessor:
    """Класс для обработки текста"""

    def __init__(self, stop_words: Set[str] = None):
        self.stop_words = stop_words or set()
        self.punctuation = set(string.punctuation)

    def preprocess(self, text: str) -> List[str]:
        """Предобработка текста"""
        # Приведение к нижнему регистру
        text = text.lower()

        # Удаление пунктуации
        text = ''.join([ch for ch in text if ch not in self.punctuation])

        # Токенизация
        tokens = text.split()

        # Удаление стоп-слов
        tokens = [token for token in tokens if token not in self.stop_words]

        return tokens

    def create_ngrams(self, tokens: List[str], n: int = 2) -> List[str]:
        """Создание N-грамм"""
        ngrams = []
        for i in range(len(tokens) - n + 1):
            ngrams.append(' '.join(tokens[i:i + n]))
        return ngrams